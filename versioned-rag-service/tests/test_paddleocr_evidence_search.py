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


def test_configured_window_budget_is_bounded_and_cache_distinguishes_it():
    from src.paddleocr_evidence_search import configured_evidence_search
    idx = Index(); idx.manifest = {'workspace_id': 'paddleocr'}
    narrow = configured_evidence_search(idx, window_budget=256)
    wide = configured_evidence_search(idx, window_budget=384)
    assert narrow is not wide
    with pytest.raises(ValueError):
        configured_evidence_search(idx, window_budget=99999)


def test_legacy_ocr_guide_keeps_pipeline_scope_without_repeating_class_name():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index();idx.chunks=[{**idx.chunks[0], 'version':'v2.9.1',
        'document_path':'doc/doc_ch/whl.md','content':'result = ocr.ocr(image, det=False)'}]
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len))
    result=search.search(plan_query('OCR 如何 det=False 只识别？',versions=('v2.9.1',)))
    assert len(result['results'])==1


def test_old_mixed_api_tests_classify_by_definition_not_whole_file():
    from src.paddleocr_retrieval_views import module_for
    path='tests/test_paddleocr_api.py'
    assert module_for({'document_path':path,'content':'def test_ocr_function(ocr_engine):\n    result = ocr_engine.ocr(image)'})=='ocr'
    assert module_for({'document_path':path,'content':'def test_structure_function(structure_engine):\n    result = structure_engine(image)'})=='structure'
    assert module_for({'document_path':'docs/unknown.md','content':'result = ocr.ocr(image)'})=='general'


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
