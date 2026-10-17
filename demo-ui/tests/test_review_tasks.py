import pytest
from services.review_tasks import ReviewTaskStore, CachedReviewGateway


def test_durable_result_requires_possession_token_and_current_identity(tmp_path):
    path = tmp_path / 'tasks.sqlite3'
    store = ReviewTaskStore(path)
    ticket = store.create('fingerprint')
    store.finish(ticket, {'status': 'static', 'resume_state': {'x': 1}})
    restored = ReviewTaskStore(path).load(ticket)
    assert restored['state'] == 'DONE' and restored['result']['resume_state']['x'] == 1
    with pytest.raises(ValueError): store.load({**ticket, 'token': 'not-the-token'})
    with pytest.raises(ValueError): store.load(ticket, identity='other')


def test_successful_tool_calls_survive_restart_but_not_identity_changes(tmp_path):
    store = ReviewTaskStore(tmp_path / 'tasks.sqlite3'); ticket = store.create('scope-a')
    calls = []
    class Gateway:
        def search(self, query, **scope):
            calls.append(query); return {'results': [{'chunk_id': 'public-1'}]}
    gateway = CachedReviewGateway(Gateway(), store, ticket)
    assert gateway.search('OCR', version='v3.0.0', request_timeout=20)['results']
    restarted = CachedReviewGateway(Gateway(), ReviewTaskStore(store.path), ticket)
    restarted.search('OCR', version='v3.0.0', request_timeout=10)
    assert calls == ['OCR']
    another = CachedReviewGateway(Gateway(), store, store.create('scope-b'))
    another.search('OCR', version='v3.0.0')
    assert len(calls) == 2


def test_failures_are_not_cached_and_task_error_has_no_source_content(tmp_path):
    store = ReviewTaskStore(tmp_path / 'tasks.sqlite3'); ticket = store.create('scope')
    class Gateway:
        def search(self, *args, **kwargs): raise ValueError('sensitive content')
    with pytest.raises(ValueError): CachedReviewGateway(Gateway(), store, ticket).search('x')
    store.fail(ticket, 'ValueError')
    assert 'sensitive' not in str(store.load(ticket))


def test_restarted_worker_is_interrupted_and_backend_identity_is_checked(tmp_path):
    import sqlite3
    store = ReviewTaskStore(tmp_path / 'tasks.sqlite3'); ticket = store.create('scope')
    store.start(ticket)
    with sqlite3.connect(store.path) as db:
        db.execute('UPDATE application_tasks SET process=?', ('previous-process',))
    assert ReviewTaskStore(store.path).load(ticket)['state'] == 'INTERRUPTED'
    class Gateway:
        def workspace(self): return {'workspace_id': 'paddleocr', 'corpus_fingerprint': 'changed'}
    guarded = CachedReviewGateway(Gateway(), store, ticket, expected_workspace={'workspace_id': 'paddleocr', 'corpus_fingerprint': 'old'})
    with pytest.raises(ValueError, match='身份'): guarded.workspace()
