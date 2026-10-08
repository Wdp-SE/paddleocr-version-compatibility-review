"""Configuration API guards against the active PaddleOCR corpus."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


@pytest.fixture
def current_index():
    return PublicKnowledgeIndex(Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr')


def yaml_root(index):
    return next(c['document_id'] for c in index.chunks
                if c['version'] == 'v3.0.0' and c['document_path'].endswith(('.yaml', '.yml')))


def test_current_trace_does_not_claim_unsupported_config_graph(current_index):
    with TestClient(create_app(index=current_index)) as client:
        response = client.post('/public/config-trace', json={
            'document_ids': [yaml_root(current_index)], 'version': 'v3.0.0'})
    assert response.status_code == 422
    assert response.json()['detail'] == 'CONFIG_TRACE_NOT_SUPPORTED'


def test_current_trace_rejects_root_from_other_version(current_index):
    with TestClient(create_app(index=current_index)) as client:
        response = client.post('/public/config-trace', json={
            'document_ids': [yaml_root(current_index)], 'version': 'v2.9.1'})
    assert response.status_code == 422
    assert response.json()['detail'] == 'CONFIG_TRACE_NOT_SUPPORTED'


def test_current_trace_rejects_unbounded_roots(current_index):
    with TestClient(create_app(index=current_index)) as client:
        response = client.post('/public/config-trace', json={
            'document_ids': [yaml_root(current_index)] * 9, 'version': 'v3.0.0'})
    assert response.status_code == 422


def test_supported_trace_protocol_keeps_pinned_version_and_model_free(tmp_path):
    import hashlib
    import json
    from test_pphuman_corpus import _manifest
    from src.public_knowledge import build_index
    manifest = _manifest(tmp_path)
    for source in manifest['sources']:
        body = "_BASE_: ['./base.yml']\nnum_classes: 1\n".encode()
        (tmp_path / source['local_path']).write_bytes(body)
        source.update(path='deploy/pipeline/config/infer_cfg_pphuman.yml', document_path='deploy/pipeline/config/infer_cfg_pphuman.yml',
                      document_key='deploy/pipeline/config/infer_cfg_pphuman', sha256=hashlib.sha256(body).hexdigest(),
                      source_url=f"https://github.com/PaddlePaddle/PaddleDetection/blob/{source['commit']}/deploy/pipeline/config/infer_cfg_pphuman.yml")
    (tmp_path / 'corpus_manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    (tmp_path / 'retrieval_policy.json').write_text(
        '{"default_policy":"bm25","benchmark_corpus_sha256":"","index_artifacts_sha256":{}}', encoding='utf8')
    build_index(tmp_path)
    # Isolate this route protocol; startup/sidecar integrity has its own tests.
    app = create_app(index=PublicKnowledgeIndex(tmp_path))
    app.state.public_knowledge_index = PublicKnowledgeIndex(tmp_path)
    client = TestClient(app)
    try:
        result = client.post('/public/config-trace', json={
            'document_ids': ['v2.9.0:zh:deploy/pipeline/config/infer_cfg_pphuman'], 'version': 'latest'})
        wrong = client.post('/public/config-trace', json={
            'document_ids': ['v2.8.1:zh:deploy/pipeline/config/infer_cfg_pphuman'], 'version': 'v2.9.0'})
    finally:
        client.close()
    assert result.status_code == 200
    report = result.json()
    assert report['model_calls'] == 0
    assert report['version'] == 'v2.9.0'
    assert report['relations']
    assert all(r['version'] == 'v2.9.0' for r in report['relations'])
    assert report['impact_status'] == 'REQUIRES_HUMAN_REVIEW'
    assert wrong.status_code == 422
    assert wrong.json()['detail'] == 'INVALID_CONFIG_TRACE'
