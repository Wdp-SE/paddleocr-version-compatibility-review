"""Durable leased review jobs. Reads never decide whether another worker is alive."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid


class Conflict(ValueError):
    pass


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class _DB:
    def __init__(self,connection,postgres):
        self.connection,self.postgres=connection,postgres

    def execute(self,sql,values=()):
        return self.connection.execute(sql.replace('?', '%s') if self.postgres else sql,values)


class TaskStore:
    stages={'queued','static_review','evidence_validation','official_lookup','model_advice','completed'}

    def __init__(self,url,*,clock=time.time,initialize=True):
        self.url,self.clock=url,clock
        self.postgres=url.startswith(('postgresql://','postgres://'))
        if not self.postgres and not url.startswith('sqlite:///'):
            raise ValueError('Unsupported task database')
        if not self.postgres:
            self.path=Path(url[len('sqlite:///'):]).resolve()
            self.path.parent.mkdir(parents=True,exist_ok=True)
        if not initialize:
            with self._db() as db:
                row=db.execute('SELECT version FROM review_schema WHERE id=1').fetchone()
                if row is None or row['version']!=2:raise ValueError('Task schema migration required')
                db.execute('SELECT task FROM review_graph_state LIMIT 0')
            return
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS review_schema (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)')
            db.execute('INSERT INTO review_schema(id,version) VALUES(1,1) ON CONFLICT(id) DO NOTHING')
            if db.execute('SELECT version FROM review_schema WHERE id=1').fetchone()['version'] not in (1,2):
                raise ValueError('Unsupported task schema version')
            db.execute('CREATE TABLE IF NOT EXISTS review_jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, application TEXT NOT NULL, identity TEXT NOT NULL, request_hash TEXT NOT NULL, request TEXT NOT NULL, state TEXT NOT NULL, stage TEXT NOT NULL, worker TEXT, lease_token TEXT, lease_epoch INTEGER NOT NULL DEFAULT 0, lease_until DOUBLE PRECISION NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT, created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL, UNIQUE(owner,application,identity))')
            db.execute('CREATE INDEX IF NOT EXISTS review_jobs_queue ON review_jobs(state,lease_until,created)')
            db.execute('CREATE TABLE IF NOT EXISTS review_capacity (id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL)')
            db.execute('INSERT INTO review_capacity(id,epoch) VALUES(1,0) ON CONFLICT(id) DO NOTHING')
            db.execute('CREATE TABLE IF NOT EXISTS review_events (id TEXT PRIMARY KEY, task TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, details TEXT NOT NULL, created DOUBLE PRECISION NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS review_records (id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL, actor TEXT NOT NULL, identity TEXT NOT NULL, payload TEXT NOT NULL, created DOUBLE PRECISION NOT NULL, UNIQUE(task,kind,identity))')
            db.execute('CREATE TABLE IF NOT EXISTS generation_leases (id TEXT PRIMARY KEY, owner TEXT NOT NULL, application TEXT NOT NULL, active INTEGER NOT NULL, created DOUBLE PRECISION NOT NULL, expires DOUBLE PRECISION NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS review_graph_state (task TEXT NOT NULL, graph TEXT NOT NULL, binding TEXT NOT NULL, payload TEXT NOT NULL, updated DOUBLE PRECISION NOT NULL, PRIMARY KEY(task,graph))')
            db.execute('UPDATE review_schema SET version=2 WHERE id=1')

    @contextmanager
    def _db(self):
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row
            connection=psycopg.connect(self.url,autocommit=True,row_factory=dict_row,connect_timeout=5)
            connection.execute('BEGIN')
        else:
            connection=sqlite3.connect(self.path,timeout=10,isolation_level=None)
            connection.row_factory=sqlite3.Row
            connection.execute('BEGIN IMMEDIATE')
        try:
            yield _DB(connection,self.postgres)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _event(self,db,task,actor,action,details=None):
        # Row locking serializes per-task event ordinals on PostgreSQL;
        # SQLite's BEGIN IMMEDIATE already serializes writers. Physical clocks
        # may repeat, so random UUID order must not decide the audit sequence.
        if self.postgres:
            db.execute('SELECT id FROM review_jobs WHERE id=? FOR UPDATE', (task,)).fetchone()
        previous = db.execute('SELECT id FROM review_events WHERE task=? ORDER BY id DESC LIMIT 1', (task,)).fetchone()
        prefix = previous['id'].partition('-')[0] if previous else ''
        ordinal = max(time.time_ns(), int(prefix)+1 if len(prefix) == 20 and prefix.isdecimal() else 0)
        db.execute('INSERT INTO review_events(id,task,actor,action,details,created) VALUES(?,?,?,?,?,?)',
                   (f'{ordinal:020d}-'+uuid.uuid4().hex,task,actor,action,canonical(details or {}),self.clock()))

    @staticmethod
    def _row(row):
        if row is None:raise KeyError('Task not found')
        value=dict(row)
        value['request']=json.loads(value['request'])
        value['result']=json.loads(value['result']) if value['result'] else None
        return value

    def submit(self,owner,application,identity,request,*,max_active=8,max_total_active=64):
        if (not isinstance(owner,str) or not 1<=len(owner)<=256 or not re.fullmatch('[a-z0-9-]{1,64}',application)
                or not re.fullmatch('[a-f0-9]{64}',identity) or not 1<=max_active<=64 or not 1<=max_total_active<=1024):
            raise ValueError('Invalid task identity or budget')
        body=canonical(request)
        if len(body.encode())>500_000:raise ValueError('Task snapshot too large')
        now=self.clock();digest=hashlib.sha256(body.encode()).hexdigest()
        # A unique key resolves same-input races; retries read the winner in a fresh transaction.
        for attempt in range(2):
            try:
                with self._db() as db:
                    # Serialize admission accounting across API replicas, including PostgreSQL.
                    db.execute('UPDATE review_capacity SET epoch=epoch+1 WHERE id=1')
                    prior=db.execute('SELECT * FROM review_jobs WHERE owner=? AND application=? AND identity=?',
                                     (owner,application,identity)).fetchone()
                    if prior:
                        if prior['request_hash']!=digest:raise Conflict('Identity reused for different request')
                        return self._row(prior)
                    active=db.execute("SELECT COUNT(*) AS n FROM review_jobs WHERE owner=? AND state IN ('QUEUED','RUNNING')",(owner,)).fetchone()['n']
                    total=db.execute("SELECT COUNT(*) AS n FROM review_jobs WHERE state IN ('QUEUED','RUNNING')").fetchone()['n']
                    if active>=max_active or total>=max_total_active:raise Conflict('Task capacity reached')
                    task=uuid.uuid4().hex
                    db.execute('INSERT INTO review_jobs(id,owner,application,identity,request_hash,request,state,stage,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)',
                               (task,owner,application,identity,digest,body,'QUEUED','queued',now,now))
                    self._event(db,task,owner,'SUBMITTED',{'identity':identity})
                    return self._row(db.execute('SELECT * FROM review_jobs WHERE id=?',(task,)).fetchone())
            except Exception as exc:
                # Never retry validation or policy errors. Only a concurrent unique-key winner.
                duplicate=isinstance(exc,sqlite3.IntegrityError) or getattr(exc,'sqlstate',None)=='23505'
                if not duplicate or attempt:raise

    def read(self,task,*,owner=None):
        with self._db() as db:
            row=self._row(db.execute('SELECT * FROM review_jobs WHERE id=?',(task,)).fetchone())
            if owner is not None and row['owner']!=owner:raise PermissionError('Task not found')
            return row

    def claim(self,worker,*,lease_seconds=60):
        if not isinstance(worker,str) or not 1<=len(worker)<=128 or not 1<=lease_seconds<=300:
            raise ValueError('Invalid lease')
        now=self.clock()
        with self._db() as db:
            exhausted=db.execute("SELECT id FROM review_jobs WHERE state='RUNNING' AND lease_until<=? AND attempts>=3",(now,)).fetchall()
            for row in exhausted:
                changed=db.execute("UPDATE review_jobs SET state='FAILED',error='RETRY_EXHAUSTED',updated=? WHERE id=? AND state='RUNNING' AND lease_until<=?",(now,row['id'],now)).rowcount
                if changed:self._event(db,row['id'],worker,'FAILED',{'reason':'RETRY_EXHAUSTED'})
            rows=db.execute("SELECT id FROM review_jobs WHERE (state='QUEUED' OR (state='RUNNING' AND lease_until<=?)) AND attempts<3 ORDER BY created,id LIMIT 16",(now,)).fetchall()
            for row in rows:
                token=uuid.uuid4().hex
                changed=db.execute("UPDATE review_jobs SET state='RUNNING',worker=?,lease_token=?,lease_epoch=lease_epoch+1,lease_until=?,attempts=attempts+1,result=NULL,updated=? WHERE id=? AND (state='QUEUED' OR (state='RUNNING' AND lease_until<=?)) AND attempts<3",
                                   (worker,token,now+lease_seconds,now,row['id'],now)).rowcount
                if changed:
                    self._event(db,row['id'],worker,'CLAIMED')
                    return self._row(db.execute('SELECT * FROM review_jobs WHERE id=?',(row['id'],)).fetchone())
        return None

    def heartbeat(self,claim,*,stage=None,lease_seconds=60):
        if stage is not None and stage not in self.stages:raise ValueError('Invalid stage')
        if not 1<=lease_seconds<=300:raise ValueError('Invalid lease')
        now=self.clock()
        with self._db() as db:
            return bool(db.execute("UPDATE review_jobs SET lease_until=?,stage=COALESCE(?,stage),updated=? WHERE id=? AND lease_token=? AND lease_epoch=? AND state='RUNNING' AND lease_until>?",
                (now+lease_seconds,stage,now,claim['id'],claim['lease_token'],claim['lease_epoch'],now)).rowcount)

    def save_preview(self,claim,preview):
        if (not isinstance(preview,dict) or preview.get('runtime_verified') is not False
                or any(type(preview.get(k)) is not int or not 0<=preview[k]<=100 for k in ('risk_count','finding_count'))
                or not isinstance(preview.get('findings'),list) or len(preview['findings'])>5
                or not isinstance(preview.get('files'),list) or len(preview['files'])>12):
            raise ValueError('Invalid static preview')
        body=canonical({'partial':True,'static_preview':preview})
        if len(body.encode())>12000:raise ValueError('Static preview budget')
        now=self.clock()
        with self._db() as db:
            return bool(db.execute("UPDATE review_jobs SET result=?,updated=? WHERE id=? AND lease_token=? AND lease_epoch=? AND state='RUNNING' AND lease_until>?",
                (body,now,claim['id'],claim['lease_token'],claim['lease_epoch'],now)).rowcount)

    def graph_state(self,claim,graph,binding,*,payload=None):
        """Read/write a bounded graph checkpoint under the authoritative task lease.

        This is internal execution state, never a second queue or a public task field.
        A deployment upgrading an existing database must run initialization first.
        """
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',graph) or not re.fullmatch('[a-f0-9]{64}',binding):
            raise ValueError('Invalid graph identity')
        body=canonical(payload) if payload is not None else None
        if body is not None and len(body.encode())>5_000_000:raise ValueError('Graph checkpoint budget')
        with self._db() as db:
            # The update locks the job for the entire checkpoint transaction on both DBs.
            live=db.execute("UPDATE review_jobs SET updated=updated WHERE id=? AND lease_token=? AND lease_epoch=? AND request_hash=? AND state='RUNNING' AND lease_until>?",
                (claim['id'],claim['lease_token'],claim['lease_epoch'],claim['request_hash'],self.clock())).rowcount
            if not live:raise PermissionError('Graph lease lost')
            row=db.execute('SELECT binding,payload FROM review_graph_state WHERE task=? AND graph=?',(claim['id'],graph)).fetchone()
            if row and row['binding']!=binding:raise ValueError('Graph execution identity changed')
            if body is not None:
                db.execute('INSERT INTO review_graph_state(task,graph,binding,payload,updated) VALUES(?,?,?,?,?) ON CONFLICT(task,graph) DO UPDATE SET payload=excluded.payload,updated=excluded.updated',
                    (claim['id'],graph,binding,body,self.clock()))
            return json.loads(row['payload']) if row else None

    def graph_event(self,claim,graph,node,event):
        if (not re.fullmatch('[a-zA-Z0-9_-]{1,100}',graph) or not re.fullmatch('[a-zA-Z0-9_-]{1,100}',node)
                or event not in ('STARTED','COMPLETED')):raise ValueError('Invalid graph event')
        with self._db() as db:
            live=db.execute("UPDATE review_jobs SET updated=updated WHERE id=? AND lease_token=? AND lease_epoch=? AND state='RUNNING' AND lease_until>?",
                (claim['id'],claim['lease_token'],claim['lease_epoch'],self.clock())).rowcount
            if not live:raise PermissionError('Graph lease lost')
            self._event(db,claim['id'],claim['worker'],'GRAPH_NODE_'+event,{'graph':graph,'node':node,'lease_epoch':claim['lease_epoch']})

    def finish(self,claim,result,*,error=None):
        body=canonical(result)
        if len(body.encode())>5_000_000:raise ValueError('Report too large')
        if error is not None and not re.fullmatch('[A-Z0-9_]{1,80}',error):raise ValueError('Unsafe error code')
        now=self.clock();state='FAILED' if error else 'DONE'
        with self._db() as db:
            changed=db.execute("UPDATE review_jobs SET state=?,stage='completed',result=?,error=?,updated=? WHERE id=? AND lease_token=? AND lease_epoch=? AND state='RUNNING' AND lease_until>?",
                (state,body,error,now,claim['id'],claim['lease_token'],claim['lease_epoch'],now)).rowcount
            if changed:self._event(db,claim['id'],claim['worker'],state,{'result_hash':fingerprint(result)})
            return bool(changed)

    def cancel(self,task,owner):
        now=self.clock()
        with self._db() as db:
            row=self._row(db.execute('SELECT * FROM review_jobs WHERE id=?',(task,)).fetchone())
            if row['owner']!=owner:raise PermissionError('Task not found')
            changed=db.execute("UPDATE review_jobs SET state='CANCELLED',updated=? WHERE id=? AND state IN ('QUEUED','RUNNING')",(now,task)).rowcount
            if changed:self._event(db,task,owner,'CANCELLED')
            return bool(changed)

    def events(self,task):
        with self._db() as db:
            return [dict(row) for row in db.execute('SELECT actor,action,details,created FROM review_events WHERE task=? ORDER BY created,id',(task,)).fetchall()]

    def retry(self,task,owner,*,max_active=8,max_total_active=64):
        if not 1<=max_active<=64 or not 1<=max_total_active<=1024:raise ValueError('Invalid capacity')
        with self._db() as db:
            db.execute('UPDATE review_capacity SET epoch=epoch+1 WHERE id=1')
            row=self._row(db.execute('SELECT * FROM review_jobs WHERE id=?',(task,)).fetchone())
            if row['owner']!=owner:raise PermissionError('Task not found')
            active=db.execute("SELECT COUNT(*) AS n FROM review_jobs WHERE owner=? AND state IN ('QUEUED','RUNNING')",(owner,)).fetchone()['n']
            total=db.execute("SELECT COUNT(*) AS n FROM review_jobs WHERE state IN ('QUEUED','RUNNING')").fetchone()['n']
            if active>=max_active or total>=max_total_active:raise Conflict('Task capacity reached')
            changed=db.execute("UPDATE review_jobs SET state='QUEUED',stage='queued',lease_token=NULL,lease_until=0,error=NULL,updated=? WHERE id=? AND state IN ('FAILED','CANCELLED') AND attempts<3",(self.clock(),task)).rowcount
            if not changed:raise Conflict('Task not retryable or attempt budget exhausted')
            self._event(db,task,owner,'RETRIED')

    def reserve_generation(self,owner,application,*,max_active=4,daily_budget=None):
        if type(max_active) is not int or not 1<=max_active<=64:raise ValueError('Invalid generation capacity')
        if daily_budget is not None and (type(daily_budget) is not int or not 1<=daily_budget<=100000):raise ValueError('Invalid generation budget')
        now=self.clock()
        with self._db() as db:
            db.execute('UPDATE review_capacity SET epoch=epoch+1 WHERE id=1')
            count=db.execute('SELECT COUNT(*) AS n FROM generation_leases WHERE active=1 AND expires>?',(now,)).fetchone()['n']
            used=db.execute('SELECT COUNT(*) AS n FROM generation_leases WHERE application=? AND created>=?',(application,now-now%86400)).fetchone()['n']
            if count>=max_active or daily_budget is not None and used>=daily_budget:raise Conflict('Generation capacity or configured budget exhausted')
            token=uuid.uuid4().hex
            db.execute('INSERT INTO generation_leases(id,owner,application,active,created,expires) VALUES(?,?,?,?,?,?)',
                       (token,owner,application,1,now,now+900))
            return token

    def release_generation(self,token):
        with self._db() as db:db.execute('UPDATE generation_leases SET active=0 WHERE id=?',(token,))

    def record(self,task,kind,actor,identity,payload):
        if kind not in ('CI','DECISION'):raise ValueError('Invalid record kind')
        body=canonical(payload)
        if len(body.encode())>200_000:raise ValueError('Record too large')
        with self._db() as db:
            db.execute('INSERT INTO review_records(id,task,kind,actor,identity,payload,created) VALUES(?,?,?,?,?,?,?) ON CONFLICT(task,kind,identity) DO NOTHING',
                       (uuid.uuid4().hex,task,kind,actor,identity,body,self.clock()))
            self._event(db,task,actor,kind+'_RECORDED',{'identity':identity})

    def records(self,task):
        with self._db() as db:
            return [{**dict(r),'payload':json.loads(r['payload'])} for r in db.execute('SELECT kind,actor,identity,payload,created FROM review_records WHERE task=? ORDER BY created,id',(task,)).fetchall()]

    def health(self):
        with self._db() as db:
            if db.execute('SELECT version FROM review_schema WHERE id=1').fetchone()['version']!=2:
                raise ValueError('Unsupported schema')
            counts={r['state']:r['n'] for r in db.execute('SELECT state,COUNT(*) AS n FROM review_jobs GROUP BY state').fetchall()}
            row=db.execute("SELECT MIN(created) AS oldest FROM review_jobs WHERE state='QUEUED'").fetchone()
            active=db.execute("SELECT COUNT(*) AS n FROM review_jobs WHERE state='RUNNING' AND lease_until>?",(self.clock(),)).fetchone()['n']
        return {'database':'ready','schema_version':2,'states':counts,'active_worker_leases':active,
                'oldest_queued_seconds':max(0,self.clock()-row['oldest']) if row['oldest'] is not None else 0}
