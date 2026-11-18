"""Actual corpus/API/Agent/worker integration. Identity and HTTP transport are test adapters;
these tests do not claim a deployed IdP, network or OCR inference has been verified.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from src.internal_api import install_internal
from src.internal_security import AccessPolicy,TokenVerifier
from src.internal_tasks import TaskStore
from src.internal_worker import configured_executor,run_one
from src.public_server import create_app


def test_real_api_worker_reviews_three_cases_without_generation(tmp_path,monkeypatch):
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    monkeypatch.delenv('APP_ENV',raising=False)
    root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(root/'change-review-agent'),str(root/'demo-ui')]
    from services.interview_examples import DEMO_CASES,demo_application
    from services.configuration_workflow import configuration_demo_inputs
    from services.registered_application import collect_snapshot, load_registry
    before_config,after_config=configuration_demo_inputs()
    policy_value={'principals':{'alice':{'document-ocr':['read','submit']},'worker':{'document-ocr':['worker']}},
        'applications':{'document-ocr':{'classification':'internal','external_models':[]}}}
    policy=AccessPolicy(policy_value)
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy_value),encoding='utf8')
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    def signed(subject,**overrides):
        return jwt.encode({'sub':subject,'iss':'https://local-identity.invalid','aud':'internal-api',
            'exp':int(time.time())+600,'iat':int(time.time()),**overrides},key,algorithm='RS256')
    employee=signed('alice')
    token=tmp_path/'token';token.write_text(signed('worker'))
    verifier=TokenVerifier('https://local-identity.invalid','internal-api',key_resolver=lambda _:key.public_key())
    cases={str(i):label for i,label in enumerate(DEMO_CASES)}
    def collect(reg,ref,**kw):
        if ref == 'HEAD':
            return collect_snapshot(reg, ref, **kw)
        files={'config-before':before_config,'config-after':after_config,'4'*40:before_config}.get(ref)
        if files is None:files=demo_application(cases[ref])
        commit={'config-before':'4'*40,'4'*40:'4'*40,'config-after':'5'*40}.get(ref)
        if commit is None:commit=str(int(ref)+1)*40
        return {'application_id':'document-ocr','head_commit':commit,'base_commit':'4'*40 if kw.get('base_commit') else None,
            'inventory':[{'path':f['path'],'sha256':hashlib.sha256(f['content'].encode()).hexdigest()} for f in files],
            'files':files,'gaps':[],'coverage':{'status':'REGISTERED_SCOPE_COMPLETE'}}
    store=TaskStore('sqlite:///'+str(tmp_path/'db'));app=create_app()
    def identity():return {k:app.state.public_build_identity[k] for k in
        ('corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint','review_code_fingerprint')}
    install_internal(app,store=store,policy=policy,verifier=verifier,
        registry_loader=load_registry,collector=collect,
        build_identity=identity,provider=lambda:None,
        retrieval_ready=lambda:app.state.public_base_knowledge_index.ready)
    with TestClient(app) as api:
        assert api.get('/health/ready').status_code==200
        assert api.get('/internal/applications',headers={'Authorization':'Bearer '+signed('alice',aud='another-api')}).status_code==401
        knowledge = api.post('/internal/application-knowledge', headers={'Authorization':'Bearer '+employee},
            json={'application_id':'document-ocr', 'version':'auto',
                  'entrypoint':'application/pipeline.py::process',
                  'query':'PaddleOCR 2.9.1 的 ocr() 如何只识别已有裁剪图片？'})
        assert knowledge.status_code == 200, knowledge.text
        answer = knowledge.json()
        assert answer['entrypoint_context']['declared_version'] == 'v2.9.1'
        sdk_evidence = [row for row in answer['evidence'] if row['source_identity'] == 'official_component']
        assert sdk_evidence and {row['version'] for row in sdk_evidence} == {'v2.9.1'}
        class Session:
            def __init__(self):self.headers={}
            def request(self,method,url,**kwargs):
                kwargs.pop('timeout',None)
                prepared=SimpleNamespace(headers=dict(self.headers))
                self.auth(prepared)
                return api.request(method,url,headers=prepared.headers,**kwargs)
            def close(self):pass
        monkeypatch.setattr('requests.Session',Session)
        execute=configured_executor(SimpleNamespace(policy_path=path),base_url='http://testserver',token_file=token)
        reports=[]
        for ref in cases:
            submitted=api.post('/internal/tasks',headers={'Authorization':'Bearer '+employee},
                json={'application_id':'document-ocr','head':ref})
            assert submitted.status_code==202,submitted.text
            assert run_one(store,'integration-worker',execute)
            task=store.read(submitted.json()['id'])
            assert task['state']=='DONE',task['error']
            reports.append(task['result'])
            assert task['result']['runtime_verified'] is False
            assert task['result']['application_snapshot']['head_commit']==str(int(ref)+1)*40
        supported=lambda r:[f for f in r['compatibility_report']['findings'] if f['status']=='supported_risk']
        assert supported(reports[0])
        assert not supported(reports[1])
        assert reports[2]['compatibility_report']['gaps']
        assert not supported(reports[2])
        # A paired upgrade executes the same graph twice with different immutable
        # inputs. Baseline and candidate must have independent checkpoint scopes.
        paired=api.post('/internal/tasks',headers={'Authorization':'Bearer '+employee},json={
            'application_id':'document-ocr','head':'0','base':'4'*40})
        assert paired.status_code==202,paired.text
        assert run_one(store,'paired-upgrade-worker',execute)
        paired_task=store.read(paired.json()['id'])
        assert paired_task['state']=='DONE',paired_task['error']
        assert paired_task['result']['commit_comparison']
        with store._db() as db:
            scopes={r['graph'] for r in db.execute('SELECT graph FROM review_graph_state WHERE task=?',(paired_task['id'],)).fetchall()}
        assert scopes=={'paddleocr_compatibility_baseline','paddleocr_compatibility_candidate'}
        submitted=api.post('/internal/tasks',headers={'Authorization':'Bearer '+employee},json={
            'application_id':'document-ocr','head':'config-after','base':'config-before',
            'workflow_type':'configuration_change'})
        assert submitted.status_code==202,submitted.text
        assert run_one(store,'configuration-worker',execute)
        task=store.read(submitted.json()['id'])
        assert task['state']=='DONE',task['error']
        config_report=task['result']
        assert config_report['workflow_type']=='configuration_change'
        assert config_report['model_calls']==0 and config_report['runtime_verified'] is False
        assert config_report['source_version']==config_report['target_version']=='v2.9.1'
        assert config_report['application_snapshot']['head_commit']=='5'*40
        assert config_report['baseline_application_snapshot']['head_commit']=='4'*40
        changes=config_report['compatibility_report']['findings']
        assert len(changes)==1 and changes[0]['status']=='behavior_change'
        assert changes[0]['before_value']==0.5 and changes[0]['after_value']==0.8
        assert changes[0]['evidence']
        assert not run_one(store,'different-worker',execute)
