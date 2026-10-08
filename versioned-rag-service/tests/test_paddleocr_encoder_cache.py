import numpy as np
import pytest


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
