"""Bounded, session-owned read-only jobs. Workers never access Streamlit state."""
from concurrent.futures import ThreadPoolExecutor, CancelledError
from threading import Event, Lock
import time
import uuid


class JobManager:
    def __init__(self, max_workers=4, max_pending=8):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ocr-evidence')
        self._limit = max_pending
        self._jobs = {}
        self._lock = Lock()

    def start(self, call, identity, *, owner):
        with self._lock:
            active = self._jobs.get(owner)
            if active and not active['future'].done():
                if active['identity'] == identity and not active['cancel'].is_set():
                    return active
                raise ValueError('当前会话已有任务正在结束，请稍后再试。')
            self._jobs = {k: v for k, v in self._jobs.items() if not v['future'].done()}
            if len(self._jobs) >= self._limit:
                raise ValueError('审查服务繁忙，排队容量已满，请稍后再试。')
            job = {'cancel': Event(), 'lock': Lock(), 'identity': identity,
                   'job_id': uuid.uuid4().hex, 'started': time.monotonic(),
                   'progress': {}, 'state': 'QUEUED', 'finished': None, 'error_type': None}

            def update(value):
                values = value if isinstance(value, dict) else {'stage': value}
                bounded = {k: v for k, v in values.items() if k in ('processed', 'round', 'checks')
                           and type(v) is int and 0 <= v <= 10000}
                if values.get('stage') in ('static_review', 'evidence_validation', 'official_lookup', 'model_advice'):
                    bounded['stage'] = values['stage']
                with job['lock']:
                    job['progress'].update(bounded)

            def run():
                try:
                    if job['cancel'].is_set():
                        raise CancelledError()
                    with job['lock']:
                        job['state'] = 'RUNNING'
                    value = call(job['cancel'].is_set, update)
                    with job['lock']:
                        job['state'] = 'CANCELLED' if job['cancel'].is_set() else 'DONE'
                    return value
                except BaseException as exc:
                    with job['lock']:
                        job['state'] = 'CANCELLED' if isinstance(exc, CancelledError) else 'FAILED'
                        job['error_type'] = type(exc).__name__
                    raise
                finally:
                    with job['lock']:
                        job['finished'] = time.monotonic()

            job['future'] = self._pool.submit(run)
            self._jobs[owner] = job
            return job

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)


_MANAGER = JobManager()


def start_job(call, identity, *, owner=None):
    return _MANAGER.start(call, identity, owner=owner or uuid.uuid4().hex)


def cancel_job(job):
    job['cancel'].set()
    if job['future'].cancel():
        with job['lock']:
            job['state'] = 'CANCELLED'
            job['finished'] = time.monotonic()


def snapshot(job):
    with job['lock']:
        return {**job['progress'], 'state': job['state'],
                'elapsed_seconds': (job['finished'] or time.monotonic()) - job['started'],
                'done': job['future'].done()}


def job_diagnostics(job):
    state = snapshot(job)
    return {'job_id': job['job_id'], 'state': state['state'],
            'stage': state.get('stage'), 'elapsed_seconds': round(state['elapsed_seconds'], 3),
            'error_type': job['error_type']}
