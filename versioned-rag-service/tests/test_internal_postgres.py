"""Optional integration against a disposable database named review_test only."""
import os
import threading
import uuid
import pytest
from src.internal_tasks import TaskStore,Conflict


def test_postgres_admission_concurrent_claim_and_stale_worker():
    url=os.environ.get('INTERNAL_TEST_DATABASE_URL')
    if not url:pytest.skip('Disposable PostgreSQL integration database not configured')
    import psycopg
    from psycopg.conninfo import conninfo_to_dict
    from urllib.parse import urlsplit,urlunsplit,parse_qsl,quote,urlencode
    from psycopg import sql
    if conninfo_to_dict(url).get('dbname')!='review_test':pytest.fail('Only disposable review_test is allowed')
    schema='review_test_'+uuid.uuid4().hex
    with psycopg.connect(url,autocommit=True) as db:
        db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    # Disposable CI container owns cleanup; never issue DROP against operator databases.
    parts=urlsplit(url)
    options = '-c search_path=' + schema
    query = urlencode([*parse_qsl(parts.query), ('options', options)], quote_via=quote)
    scoped = urlunsplit(parts._replace(query=query))
    assert conninfo_to_dict(scoped)['options'] == options
    now=[1000.0];store=TaskStore(scoped,clock=lambda:now[0])
    winners=[];errors=[]
    def submit(i):
        try:winners.append(store.submit('alice','document-ocr',f'{i:064x}',{},max_active=1))
        except Conflict:pass
        except Exception as exc:errors.append(type(exc).__name__)
    threads=[threading.Thread(target=submit,args=(i,)) for i in range(8)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert not errors and len(winners)==1
    claims=[]
    def claim():claims.append(store.claim(uuid.uuid4().hex,lease_seconds=1))
    threads=[threading.Thread(target=claim) for _ in range(6)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert sum(c is not None for c in claims)==1
    old=next(c for c in claims if c);now[0]+=2;new=store.claim('replacement')
    assert new and not store.finish(old,{'stale':True})
    assert store.finish(new,{'runtime_verified':False})
