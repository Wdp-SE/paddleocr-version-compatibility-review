"""Task-bound clients reject responses from another service build."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from services.rag_client import RAGClient, ServiceError, service_identity_digest
from services.public_knowledge_client import PublicKnowledgeClient


IDENTITY = {
    'build_revision': 'a' * 40,
    'corpus_fingerprint': {'fingerprint_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64},
    'retrieval_config_fingerprint': 'd' * 64,
    'service_code_fingerprint': 'e' * 64,
    'review_code_fingerprint': 'f' * 64,
    'evaluation_fingerprint': '1' * 64,
}


class Response:
    def __init__(self, payload, identity=None):
        self.payload = payload
        self.headers = {} if identity is None else {'X-Service-Identity': identity}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_digest_matches_backend_full_identity_contract():
    path = Path(__file__).parents[2] / 'versioned-rag-service/src/build_identity.py'
    spec = importlib.util.spec_from_file_location('backend_identity_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert service_identity_digest(IDENTITY) == module.service_identity_digest(IDENTITY)
    assert service_identity_digest({}) == module.service_identity_digest({})


@pytest.mark.parametrize('field', list(IDENTITY))
def test_each_identity_field_is_pinned_and_cannot_be_rebound(field):
    client = RAGClient('http://unused.test')
    client.bind_service_identity(IDENTITY)
    changed = deepcopy(IDENTITY)
    changed[field] = 'changed'
    with pytest.raises(ServiceError) as error:
        client.bind_service_identity(changed)
    assert error.value.code == 'SERVICE_IDENTITY_CHANGED'
    assert client._service_identity_digest == service_identity_digest(IDENTITY)


@pytest.mark.parametrize('header', [None, 'changed'])
@pytest.mark.parametrize('endpoint', ['/public/search', '/public/workspace', '/public/review-advice', '/health'])
def test_bound_response_drift_and_missing_identity_fail_closed_without_retry(header, endpoint):
    session = Session([Response({'status': 'OK'}, header)])
    client = PublicKnowledgeClient('http://unused.test', session=session, retry_limit=3,
                                   session_id='credential-not-for-error-output')
    client.bind_service_identity(IDENTITY)
    with pytest.raises(ServiceError) as error:
        client._request('GET', endpoint)
    assert error.value.code == 'SERVICE_IDENTITY_CHANGED'
    assert session.calls == 1
    assert 'credential' not in str(error.value) + error.value.detail


def test_bound_workspace_recheck_validates_body_even_with_pinned_header():
    changed = {**IDENTITY, 'review_code_fingerprint': 'changed'}
    digest = service_identity_digest(IDENTITY)
    client = PublicKnowledgeClient('http://unused.test', session=Session([
        Response(IDENTITY, digest), Response(changed, digest)]))
    client.bind_service_identity(client.workspace())
    with pytest.raises(ServiceError) as error:
        client.workspace()
    assert error.value.code == 'SERVICE_IDENTITY_CHANGED'


def test_bound_matching_identity_accepts_all_calls_and_unbound_legacy_stays_compatible():
    digest = service_identity_digest(IDENTITY)
    client = PublicKnowledgeClient('http://unused.test', session=Session([
        Response(IDENTITY, digest), Response({'results': []}, digest), Response(IDENTITY, digest)]))
    client.bind_service_identity(client.workspace())
    assert client.search('OCR result', version='v3.0.0')['results'] == []
    assert client.workspace() == IDENTITY
    legacy = PublicKnowledgeClient('http://unused.test', session=Session([Response({'results': []})]))
    assert legacy.search('OCR result', version='v3.0.0')['results'] == []
