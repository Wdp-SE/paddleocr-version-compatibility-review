import uuid
from types import SimpleNamespace
from fastapi.testclient import TestClient
from src.public_server import create_app
from src import rag_graph


def test_exact_session_repeat_skips_provider_but_drift_and_other_sessions_do_not(monkeypatch):
    calls = []
    async def run(payload, request):
        calls.append(payload.query)
        return {'status':'OK', 'answer':'supported', 'sources':[], 'evidence':[],
                'generation':{'usage':{'total_tokens':100}},
                'workflow_usage':{'total_tokens':100,'reported_calls':2,'complete':True}}
    monkeypatch.setattr(rag_graph, 'run_query', run)
    with TestClient(create_app()) as client:
        client.app.state.public_generator = SimpleNamespace()
        headers={'X-Demo-Session-ID':uuid.uuid4().hex}
        body={'query':'OCR 3.0.0 返回什么？','version':'v3.0.0'}
        assert client.post('/public/query',json=body,headers=headers).json().get('cache',{}).get('hit') is False
        hit=client.post('/public/query',json=body,headers=headers).json()
        assert hit['cache']['hit'] is True and len(calls)==1
        assert hit['workflow_usage']['total_tokens']==0
        assert hit['cache']['historical_workflow_usage']['total_tokens']==100
        assert hit['generation']['usage'] is None
        client.post('/public/query',json=body,headers={'X-Demo-Session-ID':uuid.uuid4().hex})
        assert len(calls)==2
        client.app.state.public_build_identity['service_code_fingerprint']='b'*64
        client.post('/public/query',json=body,headers=headers)
        assert len(calls)==3


def test_missing_session_and_failed_result_are_never_cached(monkeypatch):
    calls=[]
    async def run(payload, request):
        calls.append(1)
        return {'status':'ABSTAINED'}
    monkeypatch.setattr(rag_graph,'run_query',run)
    with TestClient(create_app()) as client:
        client.app.state.public_generator=SimpleNamespace()
        for _ in range(2): client.post('/public/query',json={'query':'OCR'},headers={'X-Demo-Session-ID':'local-session'})
        assert len(calls)==2
