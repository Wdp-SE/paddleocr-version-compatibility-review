import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'versioned-rag-service'))
from src.public_server import create_app
from src.public_knowledge import PublicKnowledgeIndex
from fastapi.testclient import TestClient
from app.paddleocr_configuration_review import review_configuration_change


def files(value):
    contract = (ROOT/'examples/paddleocr_document_app/application/contract.json').read_text(encoding='utf-8')
    return [{'path': 'application/ocr_client.py', 'content':
             f'from paddleocr import PaddleOCR\ndef create_engine():\n return PaddleOCR(lang="ch", drop_score={value})\n'},
            {'path': 'application/contract.json', 'content': contract},
            {'path': 'application/requirements-v2.txt', 'content': 'paddleocr==2.9.1\n'}]


class Gateway:
    def __init__(self, client): self.client = client
    def workspace(self): return self.client.get('/public/workspace').json()
    def configuration_review(self, before_files, after_files, *, dependency_version):
        response = self.client.post('/public/configuration-review', json=locals_payload(before_files, after_files, dependency_version))
        response.raise_for_status(); return response.json()
    def search(self, query, **kwargs):
        kwargs.pop('request_timeout', None)
        return self.client.post('/public/search', json={'query': query, **kwargs}).json()
    def document(self, document_id, **kwargs):
        return self.client.post('/public/document', json={'document_id': document_id}).json()['chunks']


def locals_payload(before_files, after_files, dependency_version):
    return dict(before_files=before_files, after_files=after_files, dependency_version=dependency_version)


def test_configuration_workflow_binds_real_official_definition_and_application_locations():
    index = PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=None)) as client:
        result = review_configuration_change(Gateway(client), files('.5'), files('.8'))
    report = result['compatibility_report']
    assert report['findings'][0]['status'] == 'behavior_change'
    assert report['findings'][0]['evidence'][0]['version'] == 'v2.9.1'
    assert report['application_contracts']
    assert result['model_calls'] == 0 and result['runtime_verified'] is False
    assert result['source_version'] == result['target_version']


def test_missing_or_forged_evidence_never_promotes_configuration_delta():
    index = PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    class Forged(Gateway):
        def search(self, *args, **kwargs):
            value = super().search(*args, **kwargs)
            for row in value['results']: row['content'] = 'drop_score 得分 过滤 Fake'
            return value
    with TestClient(create_app(index=index, generator=None)) as client:
        result = review_configuration_change(Forged(client), files('.5'), files('.8'))
    assert result['compatibility_report']['findings'][0]['status'] == 'needs_verification'
    assert any(g['code'] == 'parameter_evidence_missing' for g in result['compatibility_report']['gaps'])


def test_full_original_application_variants_reach_actionable_configuration_report():
    sys.path.append(str(ROOT/'demo-ui'))
    from services.configuration_workflow import configuration_demo_inputs
    before, after = configuration_demo_inputs()
    index = PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=None)) as client:
        result = review_configuration_change(Gateway(client), before, after)
    assert result['compatibility_report']['findings'][0]['status'] == 'behavior_change'
    assert not result['compatibility_report']['gaps']
    assert any('preserve_empty_pages' in x for x in result['compatibility_report']['regression_requirements'][0]['checks'])


def test_dependency_mismatch_cannot_be_reviewed_with_old_version_parameter_evidence():
    before, after = files('.5'), files('.8')
    after[-1]['content'] = 'paddleocr==3.0.0\n'
    index = PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=None)) as client:
        result = review_configuration_change(Gateway(client), before, after)
    assert result['compatibility_report']['findings'][0]['status'] == 'needs_verification'
    assert any(g['code'] == 'dependency_context_unverified' for g in result['compatibility_report']['gaps'])
