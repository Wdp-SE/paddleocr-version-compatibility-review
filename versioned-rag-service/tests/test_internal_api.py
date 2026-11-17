import time
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal_api import install_internal
from src.internal_security import AccessPolicy,Principal
from src.internal_tasks import TaskStore


@pytest.fixture
def service(tmp_path):
    policy=AccessPolicy({'principals':{'alice':{'document-ocr':['read','submit']},'bob':{'document-ocr':['read']},
        'reviewer':{'document-ocr':['read','review']}},'applications':{'document-ocr':{'classification':'internal','external_models':[]}}})
    class Verifier:
        def verify(self,value):
            if value not in ('alice','bob','reviewer'):raise PermissionError()
            return Principal(value,{})
    store=TaskStore('sqlite:///'+str(tmp_path/'jobs.db'))
    app=FastAPI()
    snapshot={'application_id':'document-ocr','head_commit':'b'*40,'base_commit':None,
        'files':[{'path':'application/main.py','content':'pass\n'},
                 {'path':'application/pipeline.py','content':'def process(): pass\ndef process_target(): pass\n'},
                 {'path':'application/requirements-v2.txt','content':'paddleocr==2.9.1\n'},
                 {'path':'application/requirements-v3.txt','content':'paddleocr==3.0.0\n'}],
        'inventory':[{'path':'application/main.py','sha256':'c'*64}],
        'registration_fingerprint':'d'*64,'gaps':[]}
    install_internal(app,store=store,policy=policy,verifier=Verifier(),
        registry_loader=lambda:{'document-ocr':{'id':'document-ocr','name':'OCR','entrypoints':{
            'application/pipeline.py::process':'application/requirements-v2.txt',
            'application/pipeline.py::process_target':'application/requirements-v3.txt',
            'application/pipeline.py::missing':'application/requirements-v3.txt'}}},
        collector=lambda app,ref,**kw:{**snapshot,'base_commit':kw.get('base_commit')},
        build_identity=lambda:{'corpus_fingerprint':'e'*64},provider='deepseek')
    with TestClient(app) as client:yield client,store


def headers(user):return {'Authorization':'Bearer '+user}


def test_joint_query_uses_application_acl_and_external_egress_gate(service):
    client,_=service
    head={**headers('alice'),'X-Application-ID':'document-ocr'}
    mismatch=client.post('/public/application-query',headers=head,json={'application_id':'other'})
    assert mismatch.status_code==403 and mismatch.json()['detail']=='APPLICATION_SCOPE_MISMATCH'
    forbidden=client.post('/public/application-query',headers=head,json={'application_id':'document-ocr'})
    assert forbidden.status_code==403
    assert client.post('/public/application-query',json={}).status_code==401


def test_internal_knowledge_only_accepts_authorized_server_snapshots(service, monkeypatch):
    from src import public_api
    client, _ = service
    seen = []

    async def knowledge(payload, request, *, generate):
        seen.append((payload.application_commit, payload.files[0].content, generate))
        return {'status': 'EVIDENCE_ONLY'}

    monkeypatch.setattr(public_api, 'resolve_application_knowledge', knowledge, raising=False)
    url = '/internal/application-knowledge'
    data = {'application_id': 'document-ocr', 'head': 'HEAD', 'query': 'OCR 输出'}
    assert client.post(url, json=data).status_code == 401
    assert client.post(url, headers=headers('alice'), json={**data, 'application_id': 'other'}).status_code == 403
    assert client.post(url, headers=headers('alice'), json={**data, 'files': []}).status_code == 422
    assert client.post(url, headers=headers('alice'), json={**data, 'generate': True}).status_code == 403
    assert not seen
    assert client.post(url, headers=headers('alice'), json=data).json()['status'] == 'EVIDENCE_ONLY'
    assert seen == [('b'*40, 'pass\n', False)]


def test_auto_knowledge_version_is_bound_to_registered_entry_and_commit(service, monkeypatch):
    from src import public_api
    client, _ = service
    observed = []

    async def knowledge(payload, request, *, generate):
        observed.append(payload.version)
        return {'status': 'EVIDENCE_ONLY'}

    monkeypatch.setattr(public_api, 'resolve_application_knowledge', knowledge)
    data = {'application_id': 'document-ocr', 'head': 'HEAD', 'query': 'OCR 输出', 'version': 'auto'}

    def query(entry):
        return client.post('/internal/application-knowledge', headers=headers('alice'),
            json={**data, 'entrypoint': 'application/pipeline.py::' + entry}).json()

    assert query('process')['entrypoint_context']['declared_version'] == 'v2.9.1'
    assert query('process_target')['entrypoint_context']['declared_version'] == 'v3.0.0'
    assert query('missing')['status'] == 'NEEDS_ENTRYPOINT_CONTEXT'
    assert observed == ['v2.9.1', 'v3.0.0']


def test_exact_cache_repeat_does_not_spend_internal_model_budget(tmp_path, monkeypatch):
    from src import public_api as api, rag_graph
    app=FastAPI()
    policy=AccessPolicy({'principals':{'alice':{'document-ocr':['read']}},
        'applications':{'document-ocr':{'classification':'public','external_models':['deepseek'],'daily_model_requests':1}}})
    class Verifier:
        def verify(self,value):return Principal('alice',{})
    store=TaskStore('sqlite:///'+str(tmp_path/'budget.db'))
    install_internal(app,store=store,policy=policy,verifier=Verifier(),registry_loader=lambda:{},collector=lambda *a:None,
        build_identity=lambda:{},provider='deepseek')
    app.include_router(api.router)
    app.state.public_build_identity={k:'a'*64 for k in ('corpus_fingerprint','service_code_fingerprint',
        'retrieval_config_fingerprint','review_code_fingerprint','evaluation_fingerprint')}
    app.state.public_generator=SimpleNamespace()
    monkeypatch.setattr(api,'_index',lambda request:SimpleNamespace())
    monkeypatch.setattr(api,'_require_project_query_ready',lambda *a:None)
    calls=[]
    async def run(payload,request):
        calls.append(1)
        return {'status':'OK','answer':'supported'}
    monkeypatch.setattr(rag_graph,'run_query',run)
    with TestClient(app) as client:
        head={**headers('alice'),'X-Application-ID':'document-ocr','X-Demo-Session-ID':'same-session'}
        assert client.post('/public/query',json={'query':'OCR'},headers=head).status_code==200
        cached=client.post('/public/query',json={'query':'OCR'},headers=head)
        assert cached.status_code==200 and cached.json()['cache']['hit'] is True
        assert len(calls)==1
        assert client.post('/public/query',json={'query':'different OCR'},headers=head).status_code==429


def test_requires_auth_and_server_acl_before_collecting_application(service):
    c,s=service
    assert c.post('/internal/tasks',json={'application_id':'document-ocr','head':'HEAD'}).status_code==401
    assert c.post('/internal/tasks',headers=headers('bob'),json={'application_id':'document-ocr','head':'HEAD'}).status_code==403
    assert c.post('/internal/tasks',headers=headers('alice'),json={'application_id':'other','head':'HEAD'}).status_code in (403,404)


def test_submit_dedup_ownership_and_cancel(service):
    c,s=service;body={'application_id':'document-ocr','head':'HEAD'}
    r=c.post('/internal/tasks',headers=headers('alice'),json=body)
    assert r.status_code==202,r.text
    task=r.json()['id']
    assert c.post('/internal/tasks',headers=headers('alice'),json=body).json()['id']==task
    assert c.get('/internal/tasks/'+task,headers=headers('bob')).status_code==404
    assert c.get('/internal/tasks/'+task,headers=headers('reviewer')).status_code==200
    assert 'request' not in c.get('/internal/tasks/'+task,headers=headers('alice')).json()
    assert c.post('/internal/tasks/'+task+'/cancel',headers=headers('alice')).status_code==200
    assert s.read(task)['state']=='CANCELLED'


def test_egress_is_denied_by_default(service):
    c,s=service
    r=c.post('/internal/tasks',headers=headers('alice'),json={'application_id':'document-ocr','head':'HEAD','generate_advice':True})
    assert r.status_code==403
    assert s.health()['states']=={}


def test_configuration_submission_requires_baseline_and_persists_same_dependency(service):
    c,s=service
    body={'application_id':'document-ocr','head':'HEAD','workflow_type':'configuration_change'}
    assert c.post('/internal/tasks',headers=headers('alice'),json=body).status_code==422
    body['base']='b'*40
    submitted=c.post('/internal/tasks',headers=headers('alice'),json=body)
    assert submitted.status_code==202,submitted.text
    request=s.read(submitted.json()['id'])['request']
    assert request['workflow_type']=='configuration_change'
    assert request['source_version']==request['target_version']=='v2.9.1'
    assert request['generate_advice'] is False
    upgrade=c.post('/internal/tasks',headers=headers('alice'),json={**body,'workflow_type':'upgrade_review'})
    assert upgrade.status_code==202 and upgrade.json()['id']!=submitted.json()['id']


@pytest.mark.parametrize('changes',[{'workflow_type':'arbitrary_agent'},
    {'workflow_type':'configuration_change','base':'b'*40,'generate_advice':True}])
def test_rejects_unsupported_workflows_and_configuration_model_advice(service,changes):
    c,_=service
    response=c.post('/internal/tasks',headers=headers('alice'),json={'application_id':'document-ocr',**changes})
    assert response.status_code==422


def test_report_requires_completed_task_and_does_not_imply_runtime_verification(service):
    c,s=service
    task=c.post('/internal/tasks',headers=headers('alice'),json={'application_id':'document-ocr','head':'HEAD'}).json()['id']
    assert c.get('/internal/tasks/'+task+'/report',headers=headers('alice')).status_code==409
    claim=s.claim('worker');s.finish(claim,{'runtime_verified':False,'status':'needs_verification'})
    result=c.get('/internal/tasks/'+task+'/report',headers=headers('alice'))
    assert result.status_code==200 and result.json()['report']['runtime_verified'] is False


def test_liveness_is_public_but_operational_metrics_are_not(service):
    c,s=service
    assert c.get('/health/live').status_code==200
    assert c.get('/internal/metrics').status_code==401
    assert c.get('/internal/metrics',headers=headers('alice')).status_code==403


def test_readiness_fails_on_incomplete_build_identity(service):
    c,_=service
    assert c.get('/health/ready').status_code==503


def test_provider_gate_uses_running_generator_and_resolved_baseline(tmp_path):
    policy=AccessPolicy({'principals':{'alice':{'document-ocr':['read','submit']}},
        'applications':{'document-ocr':{'classification':'public','external_models':['deepseek']}}})
    class Verifier:
        def verify(self,value):return Principal('alice',{})
    seen=[]
    def collect(reg,ref,**kw):
        seen.append(ref)
        return {'head_commit':'a'*40 if ref=='a'*40 else 'b'*40,'base_commit':'a'*40 if kw.get('base_commit') else None,
                'files':[]}
    app=FastAPI();actual=['dashscope']
    @app.post('/public/query')
    def generate():raise AssertionError('provider gate must reject before generation')
    install_internal(app,store=TaskStore('sqlite:///'+str(tmp_path/'tasks.db')),policy=policy,
        verifier=Verifier(),registry_loader=lambda:{'document-ocr':{}},collector=collect,
        build_identity=lambda:{k:'c'*64 for k in ('corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint')},
        provider=lambda:actual[0],retrieval_ready=lambda:False)
    with TestClient(app) as c:
        h={**headers('alice'),'X-Application-ID':'document-ocr'}
        assert c.post('/public/query',headers=h,json={'question':'hello'}).status_code==403
        assert c.get('/health/ready').status_code==503
        assert c.post('/internal/tasks',headers=h,json={'application_id':'document-ocr','base':'main'}).status_code==202
        assert seen==['HEAD','a'*40]


def test_ci_endpoint_cryptographically_separates_employee_and_ci_identities(tmp_path):
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from src.internal_security import TokenVerifier
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    def signed(issuer,audience,**extra):
        return jwt.encode({'sub':'test-runner','iss':issuer,'aud':audience,'iat':int(time.time()),
            'exp':int(time.time())+60,**extra},key,algorithm='RS256')
    employee=TokenVerifier('https://employees.invalid','api',key_resolver=lambda _:key.public_key())
    ci=TokenVerifier('https://ci.invalid','ci-api',key_resolver=lambda _:key.public_key())
    workflow='company/ocr/.github/workflows/trusted.yml@refs/heads/main'
    policy=AccessPolicy({'principals':{'test-runner':{'document-ocr':['read','review']}},
        'applications':{'document-ocr':{'classification':'internal',
        'external_models':[],'ci':{'repository':'company/ocr','workflow_ref':workflow,'workflow_sha':'e'*40}}}})
    store=TaskStore('sqlite:///'+str(tmp_path/'db'))
    inventory=[{'path':'contract.json','sha256':'a'*64},{'path':'tests/test_contract.py','sha256':'b'*64}]
    task=store.submit('alice','document-ocr','c'*64,{'snapshot':{'head_commit':'d'*40,'inventory':inventory}})
    store.finish(store.claim('worker'),{'report_fingerprint':'f'*64,'runtime_verified':False,
        'compatibility_report':{'regression_requirements':[
            {'finding_id':'contract','checks':['JSON 消费','空页保留']}]}})
    payload={'application_commit':'d'*40,'report_fingerprint':'f'*64,'run_id':'123','status':'PASSED',
        'input_sha256':{r['path']:r['sha256'] for r in inventory},'environment':{
        'paddleocr':'3.0.0','paddlepaddle':'3.0.0','python':'3.11','runner_os':'linux',
        'model_sha256':'1'*64,'container_digest':'sha256:'+'2'*64}}
    claims={'repository':'company/ocr','sha':'d'*40,'job_workflow_ref':workflow,'job_workflow_sha':'e'*40,
            'run_id':'123','run_attempt':'1','run_number':'12',
            'workflow_ref':'company/ocr/.github/workflows/checks.yml@refs/heads/main'}
    app=FastAPI()
    install_internal(app,store=store,policy=policy,verifier=employee,ci_verifier=ci,
        registry_loader=lambda:{},collector=None,build_identity=lambda:{},provider=None)
    with TestClient(app) as client:
        task_url='/internal/tasks/'+task['id']
        endpoint=task_url+'/ci'
        employee_header={'Authorization':'Bearer '+signed('https://employees.invalid','api',**claims)}
        coverage=client.get(task_url+'/report',headers=employee_header).json()['regression_coverage']
        assert [row['status'] for row in coverage]==['NOT_RUN','NOT_RUN']
        payload['check_results']=[{'check_id':coverage[0]['check_id'],'status':'PASSED',
            'execution_kind':'REAL_OCR','artifact_sha256':'3'*64}]
        forged=client.post(endpoint,headers=employee_header,json=payload)
        assert forged.status_code==401 and store.records(task['id'])==[]
        trusted=client.post(endpoint,headers={'Authorization':'Bearer '+signed('https://ci.invalid','ci-api',**claims)},json=payload)
        assert trusted.status_code==200,trusted.text
        assert store.records(task['id'])[0]['payload']['run_attempt']==1
        assert trusted.json()['upgrade_approved'] is False
        coverage=client.get(task_url+'/report',headers=employee_header).json()['regression_coverage']
        assert [row['status'] for row in coverage]==['PASSED','NOT_RUN']
