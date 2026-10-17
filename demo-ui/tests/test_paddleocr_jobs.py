from threading import Event
import pytest

from services.paddleocr_jobs import JobManager, snapshot, cancel_job, job_diagnostics


def test_duplicate_same_owner_reuses_work_but_other_session_never_reuses_result():
    manager = JobManager(max_workers=1, max_pending=3)
    entered, release = Event(), Event()
    calls = []
    def work(cancelled, update):
        calls.append(1); entered.set(); release.wait(3); return 'done'
    try:
        first = manager.start(work, 'files', owner='session-a')
        assert entered.wait(2)
        assert manager.start(work, 'files', owner='session-a') is first
        other = manager.start(work, 'files', owner='session-b')
        assert other is not first
        with pytest.raises(ValueError, match='已有'):
            manager.start(work, 'changed', owner='session-a')
        release.set()
        assert first['future'].result(3) == 'done'
        assert other['future'].result(3) == 'done'
        assert len(calls) == 2
    finally:
        release.set(); manager.close()


def test_bounded_admission_and_cancelled_queued_job_do_not_call_provider():
    manager = JobManager(max_workers=1, max_pending=2)
    entered, release = Event(), Event()
    queued_calls = []
    def work(cancelled, update):
        entered.set(); release.wait(3)
    try:
        running = manager.start(work, 'a', owner='one'); assert entered.wait(2)
        queued = manager.start(lambda *_: queued_calls.append(1), 'b', owner='two')
        with pytest.raises(ValueError, match='繁忙'):
            manager.start(work, 'c', owner='three')
        cancel_job(queued)
        release.set(); running['future'].result(3)
        assert queued['future'].cancelled()
        assert not queued_calls
    finally:
        release.set(); manager.close()


def test_diagnostics_are_bounded_do_not_contain_source_and_release_slot_after_failure():
    manager = JobManager(max_workers=1, max_pending=1)
    def fail(cancelled, update):
        update({'stage': 'official_lookup', 'raw_source': 'PRIVATE SOURCE'})
        raise RuntimeError('PRIVATE ERROR')
    try:
        job = manager.start(fail, 'fingerprint', owner='session')
        with pytest.raises(RuntimeError): job['future'].result(3)
        details = job_diagnostics(job)
        assert details['state'] == 'FAILED'
        assert details['error_type'] == 'RuntimeError'
        assert 'PRIVATE' not in str(details)
        assert snapshot(job)['done']
        next_job = manager.start(lambda *_: 'ok', 'new', owner='session')
        assert next_job['future'].result(3) == 'ok'
    finally:
        manager.close()
