from src.internal_tasks import TaskStore
from src.internal_operations import sqlite_backup, sqlite_restore


def test_backup_restores_only_into_new_database_and_keeps_records(tmp_path):
    import pytest
    source=tmp_path/'source.db';s=TaskStore('sqlite:///'+str(source))
    row=s.submit('alice','document-ocr','a'*64,{'commit':'b'*40})
    backup=tmp_path/'snapshot.db';sqlite_backup(source,backup)
    restored=tmp_path/'scratch.db';sqlite_restore(backup,restored)
    value=TaskStore('sqlite:///'+str(restored),initialize=False).read(row['id'])
    assert value['identity']==row['identity']
    with pytest.raises(ValueError):sqlite_restore(backup,source)


def test_backup_explicitly_closes_both_connections(tmp_path,monkeypatch):
    import sqlite3
    import src.internal_operations as operations
    source=tmp_path/'source.db';TaskStore('sqlite:///'+str(source))
    opened=[]
    class Tracking(sqlite3.Connection):
        was_closed=False
        def close(self):
            self.was_closed=True
            return super().close()
    real=sqlite3.connect
    def connect(*args,**kwargs):
        value=real(*args,**kwargs,factory=Tracking);opened.append(value);return value
    monkeypatch.setattr(operations.sqlite3,'connect',connect)
    try:
        sqlite_backup(source,tmp_path/'backup.db')
        assert len(opened)==2 and all(value.was_closed for value in opened)
    finally:
        for value in opened:value.close()
