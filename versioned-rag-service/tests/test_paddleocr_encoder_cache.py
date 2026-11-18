import numpy as np
import pytest


def test_asset_identity_streams_weights_without_reading_entire_file(tmp_path, monkeypatch):
    from pathlib import Path
    import hashlib
    from src.paddleocr_quality import asset_identity
    body=b'local test weights'
    (tmp_path/'model.safetensors').write_bytes(body)
    monkeypatch.setattr(Path,'read_bytes',lambda path:pytest.fail('weights read into one allocation'))
    expected=hashlib.sha256(('model.safetensors'+hashlib.sha256(body).hexdigest()).encode()).hexdigest()
    assert asset_identity(tmp_path)==expected


def test_cache_reuses_only_identical_texts_and_model(tmp_path):
    from src.paddleocr_quality import CachedEncoder
    calls=[]
    def encode(texts):
        calls.append(texts)
        return np.ones((len(texts),3),dtype=np.float32)
    first=CachedEncoder(encode,identity='model-a',cache_root=str(tmp_path))
    assert first(['one','two']).shape==(2,3)
    assert first(['one','two']).shape==(2,3)
    assert len(calls)==1
    first(['one','changed']);assert len(calls)==2
    CachedEncoder(encode,identity='model-b',cache_root=str(tmp_path))(['one','two'])
    assert len(calls)==3


def test_cache_rejects_nonfinite_or_corrupted_values(tmp_path):
    from src.paddleocr_quality import CachedEncoder,ModelUnavailable
    bad=CachedEncoder(lambda t:np.full((len(t),2),np.nan),identity='a',cache_root=str(tmp_path))
    with pytest.raises(ModelUnavailable):bad(['a','b'])
    assert not list(tmp_path.glob('*.npy'))
    good=CachedEncoder(lambda t:np.ones((len(t),2)),identity='a',cache_root=str(tmp_path))
    good(['a','b'])
    np.save(next(tmp_path.glob('*.npy')),np.full((2,2),np.inf))
    with pytest.raises(ModelUnavailable):good(['a','b'])
