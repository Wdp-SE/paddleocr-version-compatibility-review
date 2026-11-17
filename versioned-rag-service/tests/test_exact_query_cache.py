from src.exact_query_cache import ExactQueryCache


def test_cache_requires_exact_scope_and_is_copy_isolated():
    cache = ExactQueryCache(max_entries=2, ttl=5, clock=lambda: 1)
    value = {'status': 'OK', 'answer': 'supported', 'workflow_usage': {'total_tokens': 123}}
    assert cache.put({'scope': 'alice', 'commit': 'a'}, value)
    value['answer'] = 'mutated'
    hit = cache.get({'scope': 'alice', 'commit': 'a'})
    assert hit['answer'] == 'supported'
    hit['answer'] = 'mutated again'
    assert cache.get({'scope': 'alice', 'commit': 'a'})['answer'] == 'supported'
    assert cache.get({'scope': 'bob', 'commit': 'a'}) is None
    assert cache.get({'scope': 'alice', 'commit': 'b'}) is None


def test_cache_expires_evicts_and_never_retains_failures():
    now = [0]
    cache = ExactQueryCache(max_entries=1, ttl=5, clock=lambda: now[0])
    assert not cache.put({'q': 'failure'}, {'status': 'GENERATION_PROVIDER_TIMEOUT'})
    assert cache.put({'q': 'a'}, {'status': 'OK'})
    cache.put({'q': 'b'}, {'status': 'OK'})
    assert cache.get({'q': 'a'}) is None
    now[0] = 5
    assert cache.get({'q': 'b'}) is None


def test_cache_rejects_oversized_entries():
    cache = ExactQueryCache(max_bytes=128)
    assert not cache.put({'q': 'large'}, {'status': 'OK', 'answer': 'x'*200})
