import hashlib
from src.paddleocr_internal_release import frozen_lock_matches


def test_publication_cannot_rehash_changed_source_and_keep_old_measurement(tmp_path):
    path=tmp_path/'engine.py';path.write_text('original')
    old=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen={'files': {'engine.py':old}}
    assert frozen_lock_matches(tmp_path,frozen,{'files':{'engine.py':old}})
    path.write_text('changed')
    new=hashlib.sha256(path.read_bytes()).hexdigest()
    assert not frozen_lock_matches(tmp_path,frozen,{'files':{'engine.py':new}})


def test_missing_or_escaped_frozen_files_fail_closed(tmp_path):
    assert not frozen_lock_matches(tmp_path,{'files':{'missing.py':'0'*64}}, {'files':{}})
    assert not frozen_lock_matches(tmp_path,{'files':{'../outside.py':'0'*64}}, {'files':{'../outside.py':'0'*64}})
