"""Explicit model transport routing, with safe failure labels and no POST retry."""
import socket
import ssl
from urllib import error, request

NETWORK_MODES = {'system', 'direct'}
NETWORK_FAILURE_REASONS = {
    'DNS_FAILURE', 'CONNECTION_REFUSED', 'CONNECTION_RESET',
    'TLS_CERTIFICATE_ERROR', 'TLS_FAILURE', 'CONNECTION_FAILED',
}


class ProviderConnectionError(ConnectionError):
    def __init__(self, reason_code, network_mode):
        super().__init__('generation provider is unreachable')
        self.reason_code = reason_code if reason_code in NETWORK_FAILURE_REASONS else 'CONNECTION_FAILED'
        self.network_mode = network_mode if network_mode in NETWORK_MODES else 'system'


def open_provider_request(req, *, timeout, mode):
    if mode not in NETWORK_MODES:
        raise ValueError('model network mode must be system or direct')
    try:
        if mode == 'direct':
            # Explicit operator choice: no inherited transient proxy, no global
            # opener mutation, and Python's default HTTPS certificate validation.
            return request.build_opener(request.ProxyHandler({})).open(req, timeout=timeout)
        return request.urlopen(req, timeout=timeout)
    except error.HTTPError:
        raise  # Preserve authentication, billing and rate-limit classification.
    except error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise TimeoutError('generation provider request timed out') from None
        code = ('DNS_FAILURE' if isinstance(reason, socket.gaierror) else
                'TLS_CERTIFICATE_ERROR' if isinstance(reason, ssl.SSLCertVerificationError) else
                'TLS_FAILURE' if isinstance(reason, ssl.SSLError) else
                'CONNECTION_REFUSED' if isinstance(reason, ConnectionRefusedError) else
                'CONNECTION_RESET' if isinstance(reason, ConnectionResetError) else 'CONNECTION_FAILED')
        raise ProviderConnectionError(code, mode) from None
