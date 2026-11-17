import threading
import pytest

from src.internal_tasks import TaskStore, Conflict


@pytest.fixture
def store(tmp_path):
    now=[1000.0]
    return TaskStore('sqlite:///'+str(tmp_path/'tasks.db'),clock=lambda:now[0]),now


def submit(store,owner='alice',identity='a'*64):
    return store.submit(owner,'document-ocr',identity,{'application_commit':'b'*40})


def test_submission_is_deduplicated_and_read_does_not_interrupt_live_worker(store):
    s,now=store
    task=submit(s); assert submit(s)['id']==task['id']
    claimed=s.claim('worker-A',lease_seconds=30)
    assert s.read(task['id'])['state']=='RUNNING'
    assert s.claim('worker-B') is None
    assert s.read(task['id'])['state']=='RUNNING'
    assert claimed['attempts']==1


def test_expired_lease_can_be_taken_but_old_worker_cannot_finish(store):
    s,now=store;task=submit(s);a=s.claim('a',lease_seconds=10)
    now[0]+=11;b=s.claim('b',lease_seconds=10)
    assert b['lease_epoch']>a['lease_epoch']
    assert not s.finish(a,{'status':'OLD'})
    assert not s.heartbeat(a)
    assert s.finish(b,{'runtime_verified':False})
    assert s.read(task['id'])['result']=={'runtime_verified':False}


def test_static_preview_is_interim_and_cannot_be_written_by_expired_worker(store):
    s,now=store;task=submit(s);claim=s.claim('first',lease_seconds=10)
    preview={'runtime_verified':False,'risk_count':1,'finding_count':1,
             'findings':[{'title':'旧结果读取','path':'application/adapter.py','line':6}],
             'files':[{'path':'application/adapter.py','sha256':'a'*64}]}
    assert s.save_preview(claim,preview)
    row=s.read(task['id']);assert row['state']=='RUNNING' and row['result']['partial'] is True
    now[0]+=11;new=s.claim('second')
    assert s.read(task['id'])['result'] is None
    assert not s.save_preview(claim,preview)
    assert s.finish(new,{'runtime_verified':False,'final':True})
    assert s.read(task['id'])['result'].get('partial') is None


def test_cancel_and_expired_workers_cannot_write(store):
    s,now=store;task=submit(s);claim=s.claim('w',lease_seconds=10)
    s.cancel(task['id'],'alice')
    assert not s.finish(claim,{'x':1})
    assert s.read(task['id'])['state']=='CANCELLED'
    assert s.claim('w2') is None


def test_read_and_cancel_require_owner_and_record_safe_events(store):
    s,_=store;task=submit(s)
    with pytest.raises(PermissionError):s.read(task['id'],owner='bob')
    with pytest.raises(PermissionError):s.cancel(task['id'],'bob')
    s.cancel(task['id'],'alice')
    assert [r['action'] for r in s.events(task['id'])]==['SUBMITTED','CANCELLED']
    assert 'request' not in str(s.events(task['id']))


def test_equal_timestamp_audit_order_is_not_decided_by_random_uuid(store, monkeypatch):
    from types import SimpleNamespace
    import src.internal_tasks as tasks
    monkeypatch.setattr(tasks.time, 'time_ns', lambda: 1000)
    ids = iter(['f'*32, 'e'*32, 'd'*32])
    monkeypatch.setattr(tasks.uuid, 'uuid4', lambda: SimpleNamespace(hex=next(ids)))
    s, _ = store
    task = submit(s)
    s.cancel(task['id'], 'alice')
    assert [r['action'] for r in s.events(task['id'])] == ['SUBMITTED', 'CANCELLED']


def test_capacity_and_retry_are_bounded(store):
    s,now=store;task=submit(s)
    with pytest.raises(Conflict):s.submit('alice','document-ocr','c'*64,{},max_active=1)
    for i in range(3):
        claim=s.claim('w',lease_seconds=1)
        assert claim is not None
        now[0]+=2
    assert s.claim('w') is None
    assert s.read(task['id'])['state']=='FAILED'


def test_concurrent_claims_have_one_winner(store):
    s,_=store;submit(s);winners=[]
    threads=[threading.Thread(target=lambda i=i:winners.append(s.claim(str(i)))) for i in range(6)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert sum(row is not None for row in winners)==1


def test_store_survives_new_store_instance_and_closes_connections(store):
    s,_=store;task=submit(s);s.claim('A')
    other=TaskStore(s.url,clock=s.clock)
    assert other.read(task['id'])['state']=='RUNNING'


def test_heartbeat_only_updates_allowed_stage(store):
    s,now=store;submit(s);claim=s.claim('w')
    assert s.heartbeat(claim,stage='official_lookup')
    with pytest.raises(ValueError):s.heartbeat(claim,stage='untrusted raw source')


def test_changed_request_cannot_reuse_an_identity(store):
    s,_=store;submit(s)
    with pytest.raises(Conflict):s.submit('alice','document-ocr','a'*64,{'different':True})


def test_explicit_retry_is_bounded_and_old_claim_stays_invalid(store):
    s,_=store;task=submit(s);old=s.claim('old')
    s.cancel(task['id'],'alice')
    s.retry(task['id'],'alice')
    new=s.claim('new')
    assert new['attempts']==2 and not s.finish(old,{})
    assert s.finish(new,{})
    with pytest.raises(Conflict):s.retry(task['id'],'alice')


def test_background_heartbeat_does_not_reset_progress(store):
    s,_=store;task=submit(s);claim=s.claim('w')
    s.heartbeat(claim,stage='official_lookup');s.heartbeat(claim)
    assert s.read(task['id'])['stage']=='official_lookup'


def test_generation_capacity_is_global_and_released_with_expiry(store):
    s,now=store
    first=s.reserve_generation('alice','document-ocr',max_active=1)
    with pytest.raises(Conflict):s.reserve_generation('bob','document-ocr',max_active=1)
    s.release_generation(first)
    second=s.reserve_generation('bob','document-ocr',max_active=1)
    now[0]+=901
    assert s.reserve_generation('alice','document-ocr',max_active=1)!=second


def test_retry_uses_same_capacity_admission_as_submission(store):
    s,_=store
    cancelled=[]
    for i in range(3):
        row=s.submit('alice','document-ocr',f'{i:064x}',{},max_active=1,max_total_active=1)
        s.cancel(row['id'],'alice');cancelled.append(row['id'])
    s.retry(cancelled[0],'alice',max_active=1,max_total_active=1)
    with pytest.raises(Conflict):s.retry(cancelled[1],'alice',max_active=1,max_total_active=1)
    assert s.health()['states']['QUEUED']==1
