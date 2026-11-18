from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


@pytest.mark.parametrize('question',[
    '没有提交应用源码和 JSON 契约，只看 PaddleOCR 2.9.1 与 3.0.0 官方资料，能否保证我们升级后输出兼容？',
    '官方 OCR 3.0.0 文档能否证明我们服务器的模型已经成功下载并加载？',
])
def test_official_evidence_boundary_is_answered_without_inventing_company_facts(question):
    class NeverGenerate:
        provider,model='test','test'
        def generate(self,**kw):raise AssertionError('A request-scope boundary requires no model call')
    index=PublicKnowledgeIndex(Path(__file__).parents[1]/'public_corpus_paddleocr')
    with TestClient(create_app(index=index,generator=NeverGenerate())) as client:
        response=client.post('/public/query',json={'query':question,'version':'all','language':'zh'})
        assert response.status_code==200
        out=response.json()
        assert out['status']=='NEEDS_APPLICATION_CONTEXT'
        assert out['answer_type']=='REQUEST_SCOPE_EXPLANATION'
        assert out['required_inputs'] and out['evidence']==[]
        assert out['answer_completeness']=='BOUNDARY_EXPLAINED_NOT_APPLICATION_VERIFIED'
        assert out['workflow_usage']['reported_calls']==0


def test_component_usage_with_internal_contract_followup_is_not_wholly_refused():
    from src.public_scope import application_boundary_answer
    assert application_boundary_answer('PaddleOCR 3.0.0 的 save_to_json 如何使用？这是否等于我们应用的 JSON 契约？') is None
