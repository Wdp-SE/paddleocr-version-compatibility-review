"""Session-owned read-only jobs. Workers never access Streamlit state."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event,Lock
import time

_POOL=ThreadPoolExecutor(max_workers=4,thread_name_prefix='ocr-evidence')


def start_job(call,identity):
    cancel=Event();lock=Lock();progress={}
    def update(value):
        with lock:progress.update(value if isinstance(value,dict) else {'stage':value})
    future=_POOL.submit(call,cancel.is_set,update)
    return {'future':future,'cancel':cancel,'identity':identity,'started':time.monotonic(),'progress':progress,'lock':lock}


def snapshot(job):
    with job['lock']:stage=dict(job['progress'])
    return {**stage,'elapsed_seconds':time.monotonic()-job['started'],'done':job['future'].done()}
