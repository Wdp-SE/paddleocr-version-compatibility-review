import socket
import ssl
from urllib import error, request

import pytest

from src.provider_network import ProviderConnectionError, open_provider_request


def test_direct_mode_does_not_inherit_a_stale_environment_or_windows_proxy(monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY', 'http://test-secret@127.0.0.1:1')
    captured = {}
    response = object()
    class Opener:
        def open(self, req, timeout):
            captured.update(request=req, timeout=timeout)
            return response
    def build(handler):
        captured['proxies'] = handler.proxies
        return Opener()
    monkeypatch.setattr(request, 'build_opener', build)
    monkeypatch.setattr(request, 'urlopen', lambda *a, **k: pytest.fail('Implicit proxy route used'))
    req = request.Request('https://api.deepseek.com/chat/completions', data=b'{}')
    assert open_provider_request(req, timeout=12, mode='direct') is response
    assert captured['proxies'] == {}
    assert captured['timeout'] == 12
    assert captured['request'] is req


@pytest.mark.parametrize('reason,code', [
    (socket.gaierror('private dns'), 'DNS_FAILURE'),
    (ConnectionRefusedError('private proxy secret'), 'CONNECTION_REFUSED'),
    (ConnectionResetError('private endpoint'), 'CONNECTION_RESET'),
    (ssl.SSLCertVerificationError('private certificate'), 'TLS_CERTIFICATE_ERROR'),
    (ssl.SSLError('private handshake'), 'TLS_FAILURE'),
])
def test_system_failures_keep_safe_reason_without_retrying_billable_post(monkeypatch, reason, code):
    calls = []
    def fail(req, timeout):
        calls.append(req)
        raise error.URLError(reason)
    monkeypatch.setattr(request, 'urlopen', fail)
    with pytest.raises(ProviderConnectionError) as caught:
        open_provider_request(object(), timeout=8, mode='system')
    assert caught.value.reason_code == code
    assert caught.value.network_mode == 'system'
    assert str(caught.value) == 'generation provider is unreachable'
    assert len(calls) == 1


def test_http_authentication_error_is_not_misclassified_or_retried(monkeypatch):
    def fail(req, timeout):
        raise error.HTTPError('https://api.deepseek.com/chat/completions',401,'secret',None,None)
    monkeypatch.setattr(request, 'urlopen', fail)
    with pytest.raises(error.HTTPError) as caught:
        open_provider_request(object(), timeout=8, mode='system')
    assert caught.value.code == 401


def test_unknown_mode_fails_before_network_access():
    with pytest.raises(ValueError, match='network mode'):
        open_provider_request(object(), timeout=8, mode='silent-fallback')


def test_other_provider_sdk_does_not_advertise_direct_transport_control(monkeypatch):
    from src.answer_generation import StructuredAnswerGenerator
    monkeypatch.setenv('DASHSCOPE_API_KEY','fake-test-key')
    monkeypatch.setenv('RD_V2_GENERATION_NETWORK_MODE','direct')
    assert StructuredAnswerGenerator(provider='dashscope',model='test-model').network_mode is None
