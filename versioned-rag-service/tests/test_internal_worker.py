import pytest
from src.internal_tasks import TaskStore
from src.internal_worker import run_one


def test_worker_executes_once_and_persists_result(tmp_path):
    store=TaskStore('sqlite:///'+str(tmp_path/'db'))
    task=store.submit('alice','document-ocr','a'*64,{'snapshot':{'head_commit':'b'*40}})
    calls=[]
    def execute(row,cancelled,progress):
        progress('static_review');calls.append(row['id'])
        assert not cancelled()
        return {'runtime_verified':False,'status':'needs_verification'}
    assert run_one(store,'worker',execute)
    assert not run_one(store,'other',execute)
    assert calls==[task['id']] and store.read(task['id'])['state']=='DONE'


def test_cancel_during_worker_does_not_publish_report(tmp_path):
    store=TaskStore('sqlite:///'+str(tmp_path/'db'));task=store.submit('alice','document-ocr','a'*64,{})
    def execute(row,cancelled,progress):
        store.cancel(task['id'],'alice')
        assert cancelled()
        return {'should_not_be_published':True}
    run_one(store,'worker',execute)
    assert store.read(task['id'])['state']=='CANCELLED'
    assert store.read(task['id'])['result'] is None


def test_worker_exception_records_only_safe_error_code(tmp_path):
    store=TaskStore('sqlite:///'+str(tmp_path/'db'));task=store.submit('alice','document-ocr','a'*64,{})
    def execute(*args):raise RuntimeError('secret source and credential')
    run_one(store,'worker',execute)
    row=store.read(task['id'])
    assert row['error']=='WORKER_EXECUTION_FAILED'
    assert 'credential' not in str(row['result'])


def test_configured_worker_rejects_different_local_analysis_build_before_http(tmp_path):
    import json
    from types import SimpleNamespace
    from src.internal_worker import configured_executor
    from src.internal_tasks import fingerprint
    policy={'principals':{'alice':{'document-ocr':['submit']}},
            'applications':{'document-ocr':{'classification':'internal','external_models':[]}}}
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy))
    execute=configured_executor(SimpleNamespace(policy_path=path),base_url='https://unused.invalid',token_file=tmp_path/'missing')
    row={'owner':'alice','application':'document-ocr','request':{
        'access_policy_hash':fingerprint(policy),'build_identity':{'review_code_fingerprint':'a'*64}}}
    with pytest.raises(ValueError,match='Worker analysis build changed'):execute(row,lambda:False,lambda _:None)


def test_worker_uses_refreshed_service_token_for_each_request(tmp_path):
    import requests
    from src.internal_worker import TokenFileAuth
    path=tmp_path/'service-token';path.write_text('first-token')
    session=requests.Session();session.auth=TokenFileAuth(path)
    try:
        first=session.prepare_request(requests.Request('GET','https://internal.invalid/public/workspace'))
        path.write_text('refreshed-token')
        second=session.prepare_request(requests.Request('GET','https://internal.invalid/public/workspace'))
        assert first.headers['Authorization']=='Bearer first-token'
        assert second.headers['Authorization']=='Bearer refreshed-token'
        path.write_text('invalid\r\nheader: value')
        with pytest.raises(ValueError):session.prepare_request(requests.Request('GET','https://internal.invalid/'))
    finally:session.close()


def test_expired_lease_is_cancelled_before_heartbeat_thread_notices(tmp_path):
    now=[1000.0];store=TaskStore('sqlite:///'+str(tmp_path/'jobs.db'),clock=lambda:now[0])
    task=store.submit('alice','document-ocr','a'*64,{})
    observed=[]
    def execute(row,cancelled,progress):
        now[0]+=61
        observed.append(cancelled())
        return {'should_not_publish':True}
    run_one(store,'worker',execute)
    assert observed==[True], 'An expired worker must not keep issuing requests'
    assert store.read(task['id'])['result'] is None
