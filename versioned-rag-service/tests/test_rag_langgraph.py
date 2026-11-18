"""Behavioral LangGraph tests with local deterministic adapters only."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from src import public_api as api
from src import rag_graph


@pytest.fixture
def runtime(monkeypatch):
    index = SimpleNamespace(manifest={'workspace_id': 'test', 'workspace': 'local fixture'})
    request = SimpleNamespace(state=SimpleNamespace(request_id='graph-test'),
        app=SimpleNamespace(state=SimpleNamespace(public_generator=None, public_knowledge_index=index)))
    monkeypatch.setattr(api, '_index', lambda request: index)
    for name in ('_require_project_query_ready', '_validate_public_language',
                 '_validate_public_retrieval_policy', '_validate_edge_ai_facets'):
        monkeypatch.setattr(api, name, lambda *args: None)
    monkeypatch.setattr(api, '_paddleocr_version_clarification', lambda *args: None)
    monkeypatch.setattr(api, 'verified_consistency_notes', lambda *args: [])
    monkeypatch.setattr(api, 'is_out_of_scope_public_request', lambda query: False)
    hit = {'chunk_id': 'source-chunk', 'heading': 'OCR output', 'document_key': 'ocr.md',
           'version': 'v3.0.0', 'locale': 'zh', 'source_url': 'https://example.test/official',
           'content': 'OCR output has text.', 'source_type': 'official', 'module':'ocr'}
    def execution(hits=None):
        return {'results': [hit] if hits is None else hits, 'requirements': [],
            'requested_policy': 'bm25', 'actual_policy': 'bm25', 'route': 'DEFAULT',
            'rerank_status': 'NOT_RUN', 'rerank_diagnostics': {}, 'candidate_count': 1,
            'final_evidence_count': 1, 'retrieval_latency_ms': 1, 'rerank_latency_ms': 0}
    monkeypatch.setattr(api, '_execute_public_search', lambda *args, **kwargs: execution())
    return request, index, execution


def test_real_pass_graph_runs_evidence_generation_and_verification(runtime):
    request, _, _ = runtime
    class Generator:
        def generate(self, *, question, context):
            assert 'source=https://example.test/official' in context
            assert 'E1' in context
            return {'claims': [{'text': 'OCR output has text.', 'evidence_ids': ['E1']}]}
    request.app.state.public_generator = Generator()
    out = asyncio.run(api.query(api.SearchRequest(query='OCR output'), request))
    assert out['status'] == 'OK'
    assert out['claims'][0]['evidence_ids'] == ['source-chunk']
    assert out['sources'][0]['source_url'] == 'https://example.test/official'
    trace = out['graph_execution']
    assert trace['engine'] == 'langgraph'
    assert trace['passes'][0]['completed_nodes'] == list(rag_graph.PASS_NODES)
    assert trace['completed_nodes'] == ['first_pass', 'correction_gate', 'finish']
    assert trace['graph_version'] == rag_graph.GRAPH_VERSION
    json.dumps(out)  # No runtime dependencies leak into response/state trace.


def test_scope_guard_terminates_before_retrieval(runtime, monkeypatch):
    request, _, _ = runtime
    monkeypatch.setattr(api, 'is_out_of_scope_public_request', lambda query: True)
    monkeypatch.setattr(api, '_execute_public_search', lambda *args, **kwargs: pytest.fail('retrieval ran'))
    out = asyncio.run(api.query(api.SearchRequest(query='outside'), request))
    assert out['status'] == 'OUT_OF_SCOPE'
    assert out['graph_execution']['passes'][0]['completed_nodes'] == ['scope']


def test_empty_retrieval_terminates_before_provider(runtime, monkeypatch):
    request, _, execution = runtime
    monkeypatch.setattr(api, '_execute_public_search', lambda *args, **kwargs: execution([]))
    request.app.state.public_generator = SimpleNamespace(generate=lambda **kw: pytest.fail('provider ran'))
    out = asyncio.run(api.query(api.SearchRequest(query='OCR output'), request))
    assert out['status'] == 'NO_EVIDENCE'
    assert out['graph_execution']['passes'][0]['completed_nodes'] == ['scope', 'retrieve']


def test_normal_paddleocr_pass_uses_retrieval_state_not_optional_injected_execution(runtime):
    request,index,_=runtime
    index.manifest.update(workspace_id='paddleocr',versions=['v3.0.0'])
    index._version_members=lambda version: ['v3.0.0']
    calls=[]
    class Generator:
        def generate(self, *, question, context):
            calls.append(context)
            assert 'Evidence locations' in context
            return {'claims':[{'text':'OCR output has text.','evidence_ids':['E1']}], 'evidence_gaps':[]}
        def verify_claims_with_diagnostics(self, rows):
            return {'verdicts':[{'claim_index':0,'supported':True,'reason':'The text is explicit'}]}, {}
    request.app.state.public_generator=Generator()
    out=asyncio.run(api._query_once(api.SearchRequest(query='OCR 3.0.0 output text',version='v3.0.0'),request))
    assert calls
    assert out['status']=='OK', (out.get('generation'),out.get('claim_verification'),out.get('mechanical_verification'))


def test_provider_failure_stops_graph_without_verification_or_repair(runtime):
    request, _, _ = runtime
    class Generator:
        def generate(self, **kwargs):
            raise TimeoutError('local fixture')
    request.app.state.public_generator = Generator()
    out = asyncio.run(api.query(api.SearchRequest(query='OCR output'), request))
    assert out['status'] == 'GENERATION_PROVIDER_TIMEOUT'
    assert out['correction']['attempts'] == 0
    assert out['graph_execution']['passes'][0]['completed_nodes'] == [
        'scope', 'retrieve', 'prepare_context', 'generate']


def test_deadline_cancels_graph_before_generation(runtime, monkeypatch):
    request, _, execution = runtime
    entered, release = threading.Event(), threading.Event()
    calls = []
    def retrieve(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return execution()
    monkeypatch.setattr(api, '_execute_public_search', retrieve)
    request.app.state.public_generator = SimpleNamespace(generate=lambda **kw: calls.append('generate'))
    async def run():
        task = asyncio.create_task(api.query(api.SearchRequest(query='OCR output'), request))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, timeout=0.01)
            assert task.cancelled()
        finally:
            release.set()  # The already-running synchronous adapter is not forcibly killed.
    asyncio.run(run())
    assert calls == []


def test_inherited_langsmith_environment_cannot_export_knowledge(runtime, monkeypatch):
    from langsmith import tracing_context
    from langsmith.utils import tracing_is_enabled
    request, _, execution = runtime
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'true')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'local-test-key-never-sent')
    inherited_enabled = tracing_is_enabled()
    calls = []
    def retrieve(*args, **kwargs):
        assert tracing_is_enabled() is False
        calls.append('retrieve')
        return execution()
    monkeypatch.setattr(api, '_execute_public_search', retrieve)
    class Generator:
        def generate(self, **kwargs):
            assert tracing_is_enabled() is False
            calls.append('generate')
            return {'claims': [{'text': 'OCR output has text.', 'evidence_ids': ['E1']}]}
    request.app.state.public_generator = Generator()
    # Establish an explicit enabled caller context: LangSmith caches environment
    # lookups, so earlier tests may have observed tracing disabled already.
    with tracing_context(enabled=True):
        assert tracing_is_enabled() is True
        out = asyncio.run(api.query(api.SearchRequest(query='OCR output'), request))
        assert out['status'] == 'OK'
        assert calls == ['retrieve', 'generate']
        assert tracing_is_enabled() is True
    assert tracing_is_enabled() == inherited_enabled
