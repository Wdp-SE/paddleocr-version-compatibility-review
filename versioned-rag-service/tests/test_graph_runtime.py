from typing import TypedDict

import pytest
from langgraph.graph import StateGraph, START, END

from src.graph_runtime import TaskGraphRuntime, UnsafeGraphReplay
from src.internal_tasks import TaskStore, fingerprint


class State(TypedDict):
    value: int


def make_store(tmp_path,request=None):
    clock=[1000.]
    store=TaskStore('sqlite:///'+str(tmp_path/'tasks.db'),clock=lambda:clock[0])
    row=store.submit('alice','document-ocr','a'*64,request or {})
    claim=store.claim('first')
    return store,claim,clock


def runtime(store,claim):
    return TaskGraphRuntime(store,claim,cancelled=lambda:False)


def graph(first,second,*,second_name='report'):
    def build(checkpointer=None):
        b=StateGraph(State)
        b.add_node('static_review',first);b.add_node(second_name,second)
        b.add_edge(START,'static_review');b.add_edge('static_review',second_name);b.add_edge(second_name,END)
        return b.compile(checkpointer=checkpointer)
    return build


def test_restart_resumes_checkpoint_without_rerunning_completed_node(tmp_path):
    store,claim,clock=make_store(tmp_path);calls=[]
    def first(s):calls.append('first');return {'value':s['value']+1}
    def crash(s):raise RuntimeError('untrusted secret exception')
    with pytest.raises(RuntimeError):runtime(store,claim).run('review',1,graph(first,crash),{'value':0})
    clock[0]+=61
    new=store.claim('replacement')
    result=runtime(store,new).run('review',1,graph(first,lambda s:{'value':s['value']+1}),{'value':0})
    assert result=={'value':2} and calls==['first']
    events=store.events(claim['id'])
    assert any(e['action']=='GRAPH_NODE_STARTED' for e in events)
    assert 'untrusted secret' not in str(events)


def test_old_lease_cannot_read_write_or_emit_checkpoint_events(tmp_path):
    store,claim,clock=make_store(tmp_path)
    digest=fingerprint({})
    store.graph_state(claim,'review',digest,payload={'checkpoint':None})
    clock[0]+=61;new=store.claim('replacement')
    for call in (lambda:store.graph_state(claim,'review',digest),
                 lambda:store.graph_state(claim,'review',digest,payload={}),
                 lambda:store.graph_event(claim,'review','report','STARTED')):
        with pytest.raises(PermissionError):call()
    assert store.graph_state(new,'review',digest)=={'checkpoint':None}


@pytest.mark.parametrize('change',['input','version','build'])
def test_checkpoint_identity_change_is_rejected(tmp_path,change):
    store,claim,clock=make_store(tmp_path,{'build_identity':{'code':'a'}})
    build=graph(lambda s:s,lambda s:s)
    runtime(store,claim).run('review',1,build,{'value':0})
    if change=='build':claim['request']['build_identity']['code']='changed'
    with pytest.raises(ValueError,match='identity changed'):
        runtime(store,claim).run('review',2 if change=='version' else 1,build,{'value':1 if change=='input' else 0})


def test_ambiguous_model_node_is_not_replayed(tmp_path):
    store,claim,clock=make_store(tmp_path,{'generate_advice':True});calls=[]
    def crash(s):calls.append('POST');raise RuntimeError('crashed after POST')
    build=graph(lambda s:s,crash,second_name='model_advice')
    with pytest.raises(RuntimeError):runtime(store,claim).run('review',1,build,{'value':0})
    clock[0]+=61;new=store.claim('replacement')
    with pytest.raises(UnsafeGraphReplay):runtime(store,new).run('review',1,build,{'value':0})
    assert calls==['POST']


def test_cancellation_fences_checkpoint_and_node_execution(tmp_path):
    store,claim,clock=make_store(tmp_path);calls=[]
    store.cancel(claim['id'],'alice')
    with pytest.raises(PermissionError):
        runtime(store,claim).run('review',1,graph(lambda s:calls.append(1),lambda s:s),{'value':0})
    assert not calls


def test_checkpoint_rejects_non_json_and_budget(tmp_path):
    store,claim,clock=make_store(tmp_path)
    with pytest.raises(TypeError):store.graph_state(claim,'review',fingerprint({}),payload={'client':object()})
    with pytest.raises(ValueError):store.graph_state(claim,'review',fingerprint({}),payload={'score':float('nan')})
    with pytest.raises(ValueError,match='budget'):
        store.graph_state(claim,'review',fingerprint({}),payload={'value':'x'*5_000_001})


def test_schema_one_requires_explicit_migration_preserving_jobs(tmp_path):
    store,claim,clock=make_store(tmp_path)
    with store._db() as db:
        db.execute('DROP TABLE review_graph_state')
        db.execute('UPDATE review_schema SET version=1 WHERE id=1')
    with pytest.raises(ValueError,match='migration required'):
        TaskStore(store.url,initialize=False)
    migrated=TaskStore(store.url)
    assert migrated.read(claim['id'])['request_hash']==claim['request_hash']
    assert migrated.health()['schema_version']==2
    assert TaskStore(store.url,initialize=False).health()['schema_version']==2


def test_lease_stolen_after_node_cannot_run_downstream_or_save_result(tmp_path):
    store,claim,clock=make_store(tmp_path);calls=[]
    def steal(s):
        clock[0]+=61
        assert store.claim('replacement')['id']==claim['id']
        return s
    def downstream(s):calls.append('should not execute');return s
    with pytest.raises(PermissionError):
        runtime(store,claim).run('review',1,graph(steal,downstream),{'value':0})
    assert not calls
    assert not store.finish(claim,{'untrusted':'stale worker'})


def test_completed_graph_restart_returns_saved_state_without_reexecution(tmp_path):
    store,claim,clock=make_store(tmp_path);calls=[]
    def node(s):calls.append(1);return {'value':s['value']+1}
    build=graph(node,node)
    runtime(store,claim).run('review',1,build,{'value':0})
    clock[0]+=61;new=store.claim('replacement')
    assert runtime(store,new).run('review',1,build,{'value':0})=={'value':2}
    assert calls==[1,1]


def test_worker_reports_ambiguous_replay_as_safe_distinct_error(tmp_path):
    from src.internal_worker import run_one
    store,claim,clock=make_store(tmp_path,{'generate_advice':True})
    def crash(s):raise RuntimeError('private provider detail')
    build=graph(lambda s:s,crash,second_name='model_advice')
    with pytest.raises(RuntimeError):runtime(store,claim).run('review',1,build,{'value':0})
    clock[0]+=61
    def execute(row,cancelled,progress):
        return progress.graph_runtime.run('review',1,build,{'value':0})
    assert run_one(store,'replacement',execute)
    row=store.read(claim['id'])
    assert row['state']=='FAILED' and row['error']=='GRAPH_REPLAY_UNSAFE'
    assert row['result']=={'runtime_verified':False}


def test_runtime_closure_credentials_do_not_enter_checkpoint_or_audit(tmp_path):
    store,claim,clock=make_store(tmp_path)
    secret='JWT-private-token-sentinel'
    checks=[]
    execution=runtime(store,claim)
    execution.authorize=lambda:checks.append(secret)
    execution.run('review',1,graph(lambda s:s,lambda s:s),{'value':0})
    with store._db() as db:
        payload=db.execute('SELECT payload FROM review_graph_state WHERE task=?',(claim['id'],)).fetchone()['payload']
    assert checks and secret not in payload and secret not in str(store.events(claim['id']))


def test_resumed_node_mutation_before_failure_does_not_corrupt_saved_checkpoint(tmp_path):
    store,claim,clock=make_store(tmp_path)
    def build(action):
        def compiled(checkpointer=None):
            b=StateGraph(dict)
            b.add_node('static_review',lambda s:s);b.add_node('report',action)
            b.add_edge(START,'static_review');b.add_edge('static_review','report');b.add_edge('report',END)
            return b.compile(checkpointer=checkpointer)
        return compiled
    def crash(s):raise RuntimeError('crash')
    with pytest.raises(RuntimeError):runtime(store,claim).run('review',1,build(crash),{'items':[]})
    clock[0]+=61;second=store.claim('second')
    def corrupt_then_crash(s):s['items'].append('partial');raise RuntimeError('crash')
    with pytest.raises(RuntimeError):runtime(store,second).run('review',1,build(corrupt_then_crash),{'items':[]})
    clock[0]+=61;third=store.claim('third')
    assert runtime(store,third).run('review',1,build(lambda s:s),{'items':[]})=={'items':[]}


def test_explicit_baseline_and_candidate_scopes_remain_independently_bound(tmp_path):
    store,claim,clock=make_store(tmp_path)
    execution=runtime(store,claim)
    build=graph(lambda s:s,lambda s:s)
    assert execution.scoped('baseline').run('review',1,build,{'value':0})=={'value':0}
    assert execution.scoped('candidate').run('review',1,build,{'value':1})=={'value':1}
    with pytest.raises(ValueError,match='identity changed'):
        execution.scoped('candidate').run('review',1,build,{'value':2})


def test_delayed_pending_write_for_committed_superstep_preserves_new_checkpoint(tmp_path):
    from src.graph_runtime import _LeaseSaver
    store,claim,clock=make_store(tmp_path)
    saver=_LeaseSaver(runtime(store,claim),'review',fingerprint({}),None)
    config={'configurable':{'thread_id':claim['id'],'checkpoint_ns':''}}
    first=saver.put(config,{'id':'first','channel_values':{'value':0}}, {}, {})
    second=saver.put(first,{'id':'second','channel_values':{'value':1}}, {}, {})
    # LangGraph can finish persistence of pending writes after the next complete
    # superstep checkpoint already durably incorporates those writes.
    saver.put_writes(first,[('value',0)],'completed-task')
    assert saver.get_tuple(second).checkpoint['channel_values']=={'value':1}
    assert saver.get_tuple(second).pending_writes==[]
    unknown={'configurable':{**config['configurable'],'checkpoint_id':'unknown'}}
    with pytest.raises(ValueError,match='sequence changed'):
        saver.put_writes(unknown,[('value',99)],'unknown-task')
    clock[0]+=61;store.claim('replacement')
    with pytest.raises(PermissionError):saver.put_writes(first,[('value',0)],'old-worker')
