from src.paddleocr_internal_release import neural_execution_succeeded


def test_neural_measurement_requires_actual_reranker_on_every_row():
    good={'diagnostics':{'actual_strategy':'contextual_rerank','rerank_calls':1}}
    assert neural_execution_succeeded([good])
    fallback={'diagnostics':{'actual_strategy':'contextual_bm25','rerank_calls':0,'fallback_reason':'not available'}}
    assert not neural_execution_succeeded([good,fallback])
    assert not neural_execution_succeeded([])
    assert not neural_execution_succeeded([{'diagnostics':{'actual_strategy':'contextual_rerank','rerank_calls':1,'fallback_reason':'failed'}}])
