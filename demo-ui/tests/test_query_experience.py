import pytest

from services.query_experience import cached_workspace, interview_queries


class Client:
    base_url = 'http://one'

    def __init__(self):
        self.calls = 0
        self.value = {'rag_ready': True}

    def workspace(self):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return dict(self.value)


def test_navigation_reuses_metadata_but_submission_forces_fresh_state():
    client, state = Client(), {}
    assert cached_workspace(state, client, now=0)['rag_ready']
    assert cached_workspace(state, client, now=5)['rag_ready']
    assert client.calls == 1
    client.value = {'rag_ready': False}
    assert not cached_workspace(state, client, now=6, force=True)['rag_ready']
    assert client.calls == 2


def test_cache_expires_and_failure_cannot_reuse_old_ready_state():
    client, state = Client(), {}
    cached_workspace(state, client, now=0)
    client.value = RuntimeError('offline')
    with pytest.raises(RuntimeError):
        cached_workspace(state, client, now=31)
    client.value = {'rag_ready': False}
    assert not cached_workspace(state, client, now=32)['rag_ready']
    assert client.calls == 3


def test_metadata_is_session_and_backend_bound():
    client, state = Client(), {}
    cached_workspace(state, client, now=0)
    client.base_url = 'http://two'
    cached_workspace(state, client, now=1)
    cached_workspace({}, client, now=2)
    assert client.calls == 3


def test_interview_copy_keeps_version_and_business_scope_without_mutating_manifest():
    original = [
        'PaddleOCR 2.9.1 的 OCR 调用如何只识别文字而不检测？',
        'PaddleOCR 3.0.0 的 OCR 结果中 rec_texts 和 rec_scores 分别表示什么？',
        'PaddleOCR 2.9.1 升级到 3.0.0，OCR 旧结果读取方式为什么需要调整？',
        '仅凭官方资料，能否确认我们应用升级后下游 JSON 输出兼容？',
    ]
    result = interview_queries({'workspace_id': 'paddleocr', 'domain_profile': {'example_queries': original}})
    assert len(result) == 4
    assert result[0] == original[0] and result[2] == original[2]
    assert '归一化' in result[1] and '3.0.0' in result[1]
    assert '我们应用' in result[3] and '测试' in result[3]
    assert '分别表示' in original[1]
