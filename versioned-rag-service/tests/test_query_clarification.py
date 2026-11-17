from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


@pytest.mark.parametrize('policy', ['bm25','paddleocr_evidence'])
@pytest.mark.parametrize('question,version', [('OCR 3.0.0 的 text_rec_score_thresh 是什么？','v2.9.1'),
                                           ('PaddleOCR 9.9.9 的参数？','all')])
def test_version_conflict_is_clarification_not_failed_retrieval(policy, question, version):
    class NeverGenerate:
        provider, model = 'test', 'test'
        def generate(self, **kw): raise AssertionError('Version conflict must not call a model')
    index = PublicKnowledgeIndex(Path(__file__).resolve().parents[1]/'public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=NeverGenerate())) as client:
        for endpoint in ('query','search'):
            out = client.post('/public/'+endpoint, json={'query':question,'version':version,
                              'language':'zh','retrieval_policy':policy}).json()
            assert out['status'] == 'NEEDS_CLARIFICATION'
            assert out['clarification']['available_versions'] == ['v2.9.1','v3.0.0']
