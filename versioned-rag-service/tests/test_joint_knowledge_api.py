from fastapi.testclient import TestClient
from src import public_api as api
from src.public_server import create_app


def request_data():
    return {'files': [{'path': 'application/contract.json', 'content': '{"required_line_fields":["text","confidence","page_index"]}'}],
            'query': 'OCR 3.0.0 rec_texts 如何对应本应用 JSON 契约？',
            'application_id': 'document-ocr', 'application_commit': 'a'*40, 'version': 'v3.0.0'}


def test_joint_api_keeps_evidence_without_model_and_validates_commit(monkeypatch):
    monkeypatch.setattr(api, '_execute_public_search', lambda *args, **kwargs: {'results': [
        {'chunk_id': 'official', 'content': 'rec_texts rec_scores', 'version': 'v3.0.0'}]})
    with TestClient(create_app()) as client:
        client.app.state.public_generator = None
        response = client.post('/public/application-query', json=request_data())
        assert response.status_code == 200
        value = response.json()
        assert value['status'] == 'GENERATION_NOT_CONFIGURED'
        assert value['application_commit'] == 'a'*40
        assert len(value['evidence']) >= 2
        bad = {**request_data(), 'application_commit': 'z'*40}
        assert client.post('/public/application-query', json=bad).status_code == 422


def test_joint_api_version_conflict_never_searches(monkeypatch):
    monkeypatch.setattr(api, '_execute_public_search', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('searched')))
    with TestClient(create_app()) as client:
        value = client.post('/public/application-query', json={**request_data(), 'version':'v2.9.1'})
        assert value.status_code == 200
        assert value.json()['status'] == 'NEEDS_CLARIFICATION'
