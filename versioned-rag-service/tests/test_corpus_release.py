import json
from pathlib import Path
import pytest
from src.corpus_release import tree_identity, stage, activate, resolve


def test_release_requires_matching_gate_and_uses_compare_and_swap(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'a.txt').write_text('original')
    root=tmp_path/'releases';root.mkdir()
    gate={'passed':True,'tree_sha256':tree_identity(source),'service_code_fingerprint':'a'*64,
          'evaluation_sha256':'b'*64}
    validate=lambda path:None
    release=stage(source,root,gate,code_identity='a'*64,validate=validate)
    pointer=root/'active.json'
    activate(root,pointer,release,expected=None,code_identity='a'*64,validate=validate)
    assert resolve(pointer,code_identity='a'*64,validate=validate)==root/release/'corpus'
    with pytest.raises(ValueError):activate(root,pointer,release,expected='stale',code_identity='a'*64,validate=validate)
    (root/release/'corpus/a.txt').write_text('modified')
    with pytest.raises(ValueError):resolve(pointer,code_identity='a'*64,validate=validate)


def test_failed_or_wrong_build_gate_cannot_publish(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'x').write_text('x')
    root=tmp_path/'releases';root.mkdir()
    gate={'passed':False,'tree_sha256':tree_identity(source),'service_code_fingerprint':'a'*64,'evaluation_sha256':'b'*64}
    with pytest.raises(ValueError):stage(source,root,gate,code_identity='a'*64,validate=lambda _:None)
    gate['passed']=True
    with pytest.raises(ValueError):stage(source,root,gate,code_identity='c'*64,validate=lambda _:None)


def test_atomic_release_can_roll_back_without_rewriting_old_corpus(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'x').write_text('original')
    releases=tmp_path/'releases';releases.mkdir();pointer=releases/'active.json'
    def publish(expected):
        gate={'passed':True,'tree_sha256':tree_identity(source),'service_code_fingerprint':'a'*64,'evaluation_sha256':'b'*64}
        name=stage(source,releases,gate,code_identity='a'*64,validate=lambda _:None)
        activate(releases,pointer,name,expected=expected,code_identity='a'*64,validate=lambda _:None)
        return name
    first=publish(None)
    (source/'x').write_text('candidate')
    second=publish(first)
    assert (resolve(pointer,code_identity='a'*64,validate=lambda _:None)/'x').read_text()=='candidate'
    activate(releases,pointer,first,expected=second,code_identity='a'*64,validate=lambda _:None)
    assert (resolve(pointer,code_identity='a'*64,validate=lambda _:None)/'x').read_text()=='original'


def test_actual_pinned_corpus_release_can_boot_api(tmp_path,monkeypatch):
    """Fixture gate tests integrity/control flow, not an invented answer-quality pass."""
    import hashlib
    from fastapi.testclient import TestClient
    from src.build_identity import content_fingerprint
    from src.public_server import create_app
    service=Path(__file__).resolve().parents[1]
    source=service/'public_corpus_paddleocr'
    code=content_fingerprint({p.name:p for p in (service/'src').glob('*.py')})['fingerprint_sha256']
    releases=tmp_path/'releases';releases.mkdir()
    gate={'passed':True,'tree_sha256':tree_identity(source),'service_code_fingerprint':code,
        'evaluation_sha256':hashlib.sha256(b'integration fixture: integrity only').hexdigest()}
    name=stage(source,releases,gate,code_identity=code)
    activate(releases,releases/'active.json',name,expected=None,code_identity=code)
    monkeypatch.setenv('APP_ENV','local')
    monkeypatch.setenv('RAG_CORPUS_RELEASE_POINTER',str(releases/'active.json'))
    monkeypatch.delenv('RAG_PUBLIC_RETRIEVAL_CONFIG',raising=False)
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    with TestClient(create_app()) as client:
        assert client.get('/health').json()['rag_ready'] is True
        assert client.get('/public/workspace').json()['workspace_id']=='paddleocr'
