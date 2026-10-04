"""Quality strategy must preserve the source boundary, not just improve ranks."""
import numpy as np
import pytest
from src.public_knowledge import PublicKnowledgeIndex
from pathlib import Path


def test_semantic_candidates_are_filtered_before_ranking():
    from src.paddleocr_quality import QualityRetriever
    index = PublicKnowledgeIndex(Path(__file__).parents[1] / 'public_corpus_paddleocr')
    matrix = np.zeros((len(index.chunks), 2), dtype=np.float32)
    old = next(i for i,c in enumerate(index.chunks) if c['version']=='v2.9.1')
    new = next(i for i,c in enumerate(index.chunks) if c['version']=='v3.0.0')
    matrix[old] = [1,0]
    matrix[new] = [.8,.6]
    retriever = QualityRetriever(index, encoder=lambda texts: np.array([[1.,0.] for _ in texts]), vectors=matrix)
    hits = retriever.search('接口迁移', version='v3.0.0', strategy='semantic', top_k=5)
    assert hits[0]['chunk_id']==index.chunks[new]['chunk_id']
    assert all(c['version']=='v3.0.0' for c in hits)


def test_reranker_scores_are_not_probabilities_and_preserve_evidence():
    from src.paddleocr_quality import QualityRetriever
    index = PublicKnowledgeIndex(Path(__file__).parents[1] / 'public_corpus_paddleocr')
    retriever=QualityRetriever(index, scorer=lambda pairs: np.arange(len(pairs),dtype=float))
    result=retriever.search('PPStructureV3',version='v3.0.0',strategy='bm25_rerank',top_k=3)
    assert len(result)==3
    assert result[0]['rerank_score']>result[1]['rerank_score']
    assert result[0]['score_type']=='cross_encoder_logit'
    assert 'confidence' not in result[0]
    assert result[0]['content']==next(c['content'] for c in index.chunks if c['chunk_id']==result[0]['chunk_id'])


def test_missing_models_and_invalid_scores_are_not_silent_success():
    from src.paddleocr_quality import QualityRetriever, ModelUnavailable
    index=PublicKnowledgeIndex(Path(__file__).parents[1] / 'public_corpus_paddleocr')
    with pytest.raises(ModelUnavailable):
        QualityRetriever(index).search('迁移',strategy='hybrid_rerank',version='v3.0.0')
    with pytest.raises(ModelUnavailable):
        QualityRetriever(index,scorer=lambda pairs:[float('nan')]*len(pairs)).search('PaddleOCR',strategy='bm25_rerank',version='v3.0.0')


def test_quality_strategy_rejects_unknown_version_and_language():
    from src.paddleocr_quality import QualityRetriever
    index=PublicKnowledgeIndex(Path(__file__).parents[1] / 'public_corpus_paddleocr')
    retriever=QualityRetriever(index,encoder=lambda texts:np.ones((len(texts),2)),vectors=np.ones((len(index.chunks),2)))
    with pytest.raises(ValueError): retriever.search('API',strategy='semantic',version='v9')
    with pytest.raises(ValueError): retriever.search('API',strategy='semantic',language='xx')


def test_zero_lexical_candidates_never_trigger_reranking():
    from src.paddleocr_quality import QualityRetriever
    index=PublicKnowledgeIndex(Path(__file__).parents[1] / 'public_corpus_paddleocr')
    def scorer(pairs):
        pytest.fail('zero lexical matches must not be reranked')
    retriever=QualityRetriever(index,scorer=scorer)
    assert retriever.search('zzzzqvxx999xyz',version='v3.0.0',strategy='bm25')==[]
    assert retriever.search('zzzzqvxx999xyz',version='v3.0.0',strategy='bm25_rerank')==[]
def test_asset_identity_does_not_exclude_a_model_under_home_cache(tmp_path):
    from src.paddleocr_quality import asset_identity
    model=tmp_path/'.cache'/'models'/'snapshot';model.mkdir(parents=True)
    weights=model/'model.bin';weights.write_bytes(b'weights')
    first=asset_identity(str(model))
    assert first!='NOT_CONFIGURED'
    weights.write_bytes(b'changed weights')
    assert asset_identity(str(model))!=first

def test_exact_score_cache_reuses_only_identical_pairs_and_valid_outputs():
    from src.paddleocr_quality import ExactScoreCache,ModelUnavailable
    calls=[]
    def predict(pairs):
        calls.append(pairs)
        return [float(len(q)+len(t)) for q,t in pairs]
    scorer=ExactScoreCache(predict,max_pairs=3)
    assert scorer([('q','one'),('q','one'),('q','two')])==[4.,4.,4.]
    assert calls==[[('q','one'),('q','two')]]
    assert scorer([('q','two'),('other','one')])==[4.,8.]
    assert calls[-1]==[('other','one')]
    broken=ExactScoreCache(lambda pairs:[float('nan')])
    with pytest.raises(ModelUnavailable):broken([('q','one')])
    assert not broken.cache
