import pytest
from src.application_knowledge import search_application


def test_original_contract_evidence_retains_commit_and_is_not_official():
    rows = [{'path': 'application/contract.json', 'content': '{"required_line_fields":["text","confidence","page_index"]}'},
            {'path': 'application/design.md', 'content': '测试规范不是已经执行的测试记录。'}]
    value = search_application(rows, 'confidence page_index', application_id='ocr', application_commit='a'*40)
    assert value['results'][0]['path'] == rows[0]['path']
    hit = value['results'][0]
    assert hit['application_commit'] == 'a'*40 and hit['source_identity'] == 'submitted_application'
    assert hit['line_start'] == 1 and hit['sha256']
    assert value['runtime_verified'] is False


def test_different_commit_and_no_match_do_not_fall_back_to_official_sources():
    value = search_application([{'path': 'contract.md', 'content': 'text confidence'}],
        'unrelated_parameter', application_id='ocr', application_commit='b'*40)
    assert value['results'] == []


def test_application_search_rejects_unbounded_or_ambiguous_input():
    with pytest.raises(ValueError):
        search_application([{'path': '../secret', 'content': 'secret'}], 'text', application_id='ocr', application_commit='a'*40)


def test_api_application_search_is_request_scoped_without_external_model():
    from fastapi.testclient import TestClient
    from src.public_server import create_app
    with TestClient(create_app()) as client:
        response = client.post('/public/application-search', json={
            'files': [{'path': 'contract.json', 'content': '{"required": ["confidence"]}'}],
            'query': 'confidence', 'application_id': 'document-ocr', 'application_commit': 'a'*40})
        assert response.status_code == 200
        assert response.json()['results'][0]['source_identity'] == 'submitted_application'
        invalid = client.post('/public/application-search', json={
            'files': [{'path': '../secret', 'content': 'secret'}], 'query': 'secret',
            'application_id': 'document-ocr', 'application_commit': 'a'*40})
        assert invalid.status_code == 422
        assert 'secret' not in invalid.text
