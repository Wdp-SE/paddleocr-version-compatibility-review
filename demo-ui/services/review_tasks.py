"""Local durable task records and completed-call cache. Capability tokens are not enterprise identity."""
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import uuid

PROCESS = uuid.uuid4().hex


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ReviewTaskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS application_tasks (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, identity TEXT NOT NULL, process TEXT NOT NULL, state TEXT NOT NULL, result TEXT, error_type TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP)')
            db.execute('CREATE TABLE IF NOT EXISTS application_tool_cache (task TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(task,key))')

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, identity):
        ticket = {'task_id': uuid.uuid4().hex, 'token': secrets.token_urlsafe(32)}
        with self._db() as db:
            if db.execute('SELECT COUNT(*) FROM application_tasks').fetchone()[0] >= 1000:
                raise ValueError('本地任务记录已达容量上限，请由管理员清理过期记录。')
            db.execute('INSERT INTO application_tasks(id,token_hash,identity,process,state) VALUES(?,?,?,?,?)',
                       (ticket['task_id'], _hash(ticket['token']), identity, PROCESS, 'QUEUED'))
        return ticket

    def load(self, ticket, *, identity=None):
        if not isinstance(ticket, dict) or not isinstance(ticket.get('token'), str) or len(ticket['token']) > 200:
            raise ValueError('任务不存在或恢复凭据无效。')
        with self._db() as db:
            row = db.execute('SELECT * FROM application_tasks WHERE id=? AND token_hash=?',
                             (ticket.get('task_id'), _hash(ticket['token']))).fetchone()
            if row is None: raise ValueError('任务不存在或恢复凭据无效。')
            if identity is not None and row['identity'] != identity:
                raise ValueError('输入、提交、语料或策略已变化，不能复用旧任务。')
            row = dict(row)
            if row['state'] in ('QUEUED', 'RUNNING') and row['process'] != PROCESS:
                db.execute("UPDATE application_tasks SET state='INTERRUPTED',updated=CURRENT_TIMESTAMP WHERE id=?", (row['id'],))
                row['state'] = 'INTERRUPTED'
        row.pop('token_hash'); row.pop('process')
        row['result'] = json.loads(row['result']) if row['result'] else None
        return row

    def start(self, ticket):
        self.load(ticket)
        with self._db() as db:
            db.execute("UPDATE application_tasks SET state='RUNNING',process=?,updated=CURRENT_TIMESTAMP WHERE id=?", (PROCESS, ticket['task_id']))

    def finish(self, ticket, result):
        self.load(ticket)
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode()) > 5_000_000: raise ValueError('审查结果超过持久化预算。')
        state = 'CANCELLED' if (result.get('investigation') or {}).get('stop_reason') == 'CANCELLED' else 'DONE'
        with self._db() as db:
            db.execute("UPDATE application_tasks SET state=?,result=?,error_type=NULL,updated=CURRENT_TIMESTAMP WHERE id=?", (state, encoded, ticket['task_id']))

    def fail(self, ticket, error_type):
        self.load(ticket)
        safe = error_type if error_type in ('ValueError', 'ServiceError', 'CancelledError', 'OSError') else 'TaskError'
        with self._db() as db:
            db.execute("UPDATE application_tasks SET state='FAILED',error_type=?,updated=CURRENT_TIMESTAMP WHERE id=?", (safe, ticket['task_id']))

    def cached(self, ticket, key):
        self.load(ticket)
        with self._db() as db:
            row = db.execute('SELECT value FROM application_tool_cache WHERE task=? AND key=?', (ticket['task_id'], key)).fetchone()
            return json.loads(row[0]) if row else None

    def cache(self, ticket, key, value):
        self.load(ticket)
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded.encode()) > 1_000_000: return
        with self._db() as db:
            if db.execute('SELECT COUNT(*) FROM application_tool_cache WHERE task=?', (ticket['task_id'],)).fetchone()[0] >= 100: return
            db.execute('INSERT OR REPLACE INTO application_tool_cache VALUES(?,?,?)', (ticket['task_id'], key, encoded))


class CachedReviewGateway:
    """Only completed immutable-scope calls are cached; workspace identity is always fetched live."""
    def __init__(self, gateway, store, ticket, *, expected_workspace=None):
        self.gateway, self.store, self.ticket = gateway, store, ticket
        self.expected_workspace = expected_workspace

    def __getattr__(self, name):
        return getattr(self.gateway, name)

    def workspace(self):
        value = self.gateway.workspace()
        if self.expected_workspace and any(value.get(k) != self.expected_workspace.get(k) for k in (
                'workspace_id', 'corpus_fingerprint', 'service_code_fingerprint', 'retrieval_config_fingerprint')):
            raise ValueError('知识服务身份已变化，请重新加载资料后发起新审查。')
        return value

    def _call(self, name, args, kwargs):
        scope = {k: v for k, v in kwargs.items() if k != 'request_timeout'}
        key = _hash(json.dumps([name, args, scope], sort_keys=True, ensure_ascii=False))
        cached = self.store.cached(self.ticket, key)
        if cached is not None: return cached
        value = getattr(self.gateway, name)(*args, **kwargs)
        if name == 'document' or (isinstance(value, dict) and ('results' in value or value.get('status') == 'OK')) or (name == 'plan_investigation' and isinstance(value, list)):
            self.store.cache(self.ticket, key, value)
        return value

    def search(self, query, *, request_timeout=None, **scope):
        return self._call('search', [query], {**scope, **({'request_timeout': request_timeout} if request_timeout else {})})

    def document(self, document_id, *, request_timeout=None):
        return self._call('document', [document_id], {'request_timeout': request_timeout} if request_timeout else {})

    def plan_investigation(self, *, unresolved, request_timeout):
        return self._call('plan_investigation', [], {'unresolved': unresolved, 'request_timeout': request_timeout})

    def review_advice_for_version(self, summary, ids, *, version, request_timeout=None):
        return self._call('review_advice_for_version', [summary, ids], {'version': version, 'request_timeout': request_timeout})
