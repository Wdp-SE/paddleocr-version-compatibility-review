import pytest
from src.paddleocr_retrieval_views import build_views


class Index:
    def __init__(self):
        self.chunks=[{'chunk_id':f'c{i}','source_id':f's{i}','version':v,'language':'zh',
                      'namespace':n,'source_sha256':str(i),'document_path':'docs/ocr.md',
                      'line_start':1,'line_end':1,'content':text,'heading':''}
                     for i,(v,n,text) in enumerate([
                         ('v3.0.0','project_primary','page_num 默认 42'),
                         ('v2.9.1','project_primary','page_num 默认 99'),
                         ('v3.0.0','dependency_reference','page_num 默认 999')])]


def test_search_filters_version_and_namespace_before_scoring():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index(); views=build_views(idx.chunks,token_count=len)
    out=EvidenceSearch(idx,views).search(plan_query('page_num 默认',versions=('v3.0.0',)),top_k=20)
    assert [h['chunk_id'] for h in out['results']]==['c0']
    assert out['diagnostics']['eligible_windows']==1


def test_missing_reranker_does_not_claim_success():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index(); views=build_views(idx.chunks,token_count=len)
    out=EvidenceSearch(idx,views).search(plan_query('page_num',versions=('v3.0.0',)),strategy='contextual_rerank')
    assert out['diagnostics']['actual_strategy']=='contextual_bm25'
    assert out['diagnostics']['rerank_calls']==0
    assert out['diagnostics']['fallback_reason']=='RERANK_MODEL_NOT_CONFIGURED'


def test_empty_candidates_do_not_call_scorer_and_budget_is_validated():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index(); calls=[]
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),scorer=lambda pairs:calls.append(pairs))
    plan=plan_query('不存在的zzzzxyz',versions=('v3.0.0',))
    out=search.search(plan,strategy='contextual_rerank')
    assert out['results']==[] and calls==[]
    with pytest.raises(ValueError): search.search(plan,candidate_budget=999)
def test_lexical_rerank_fallback_reports_actual_scoring_method():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index()
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),scorer=lambda pairs:[1]*len(pairs))
    out=search.search(plan_query('page_num',versions=('v3.0.0',)),strategy='contextual_rrf_rerank')
    assert out['diagnostics']['actual_strategy']=='contextual_rerank'
    assert out['diagnostics']['rerank_calls']==1
