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


def test_structured_windows_preserve_parameter_rows_and_original_offsets():
    from src.paddleocr_retrieval_views import build_structure_views, validate_views, materialize_evidence, evidence_text
    idx=Index()
    body='<table>\n<tr><th>参数</th><th>说明</th></tr>\n<tr>\n<td>batch_size</td>\n<td>一次处理的图片数量</td>\n</tr>\n</table>'
    idx.chunks=[{**idx.chunks[0], 'content':body,'line_end':7}]
    views=build_structure_views(idx.chunks, max_chars=256)
    validate_views(idx.chunks,views)
    selected=[v for v in views if 'batch_size' in v['content']]
    assert selected and '一次处理的图片数量' in selected[0]['content']
    assert '<td>' not in selected[0]['retrieval_text']
    hits=materialize_evidence(idx.chunks,selected)
    assert '<td>batch_size</td>' in evidence_text(hits[0])


def test_candidate_diagnostics_separate_semantic_and_lexical_channels():
    import numpy as np
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index()
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),encoder=lambda texts:np.ones((len(texts),2)))
    out=search.search(plan_query('page_num',versions=('v3.0.0',)),strategy='contextual_rrf')
    assert out['diagnostics']['semantic_candidates'] == 1
    assert out['diagnostics']['lexical_candidates'] == 1
    assert out['diagnostics']['actual_strategy'] == 'contextual_rrf'


def test_migration_keeps_both_callable_contracts_even_when_reranker_prefers_examples():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index()
    rows=[('v2.9.1','paddleocr.py','ocr_res.append([box.tolist(), res])\nreturn ocr_res'),
          ('v3.0.0','paddleocr/_pipelines/ocr.py','def ocr(self, img, **kwargs):\n    return self.predict(img, **kwargs)'),
          ('v2.9.1','doc/doc_ch/whl.md','OCR 结果读取升级 print(result) 坐标输出'),
          ('v3.0.0','docs/pipeline_usage/ocr.md','OCR 结果读取升级 rec_texts rec_scores')]
    idx.chunks=[{**idx.chunks[0],'chunk_id':f'c{i}','source_id':f's{i}',
                 'version':v,'document_path':path,'content':text,'line_end':text.count('\n')+1}
                for i,(v,path,text) in enumerate(rows)]
    scorer=lambda pairs:[0 if ('ocr_res.append' in text or 'def ocr(' in text) else 10 for _,text in pairs]
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),scorer=scorer)
    result=search.search(plan_query('OCR 结果读取升级有什么变化？',versions=('v2.9.1','v3.0.0')),
                         top_k=3,strategy='contextual_rerank')
    assert {'c0','c1'} <= {h['chunk_id'] for h in result['results']}


def test_migration_reserves_official_field_access_assertions_before_printed_samples():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index()
    rows=[('v2.9.1','paddleocr.py','ocr_res.append([box.tolist(), res])\nreturn ocr_res'),
          ('v3.0.0','paddleocr/_pipelines/ocr.py','def ocr(self, img, **kwargs):\n    return self.predict(img, **kwargs)'),
          ('v3.0.0','tests/pipelines/test_ocr.py','res = result[0]\nassert isinstance(res["rec_texts"], list)'),
          ('v3.0.0','docs/pipeline_usage/ocr.md','OCR 升级打印结果 rec_texts rec_scores')]
    idx.chunks=[{**idx.chunks[0],'chunk_id':f'c{i}','source_id':f's{i}','version':v,
                 'document_path':path,'content':text,'line_end':text.count('\n')+1}
                for i,(v,path,text) in enumerate(rows)]
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),scorer=lambda pairs:[10 if '打印' in t else 0 for _,t in pairs])
    result=search.search(plan_query('OCR 结果读取升级有什么变化？',versions=('v2.9.1','v3.0.0')),top_k=3,strategy='contextual_rerank')
    assert {'c0','c1','c2'}=={h['chunk_id'] for h in result['results']}


def test_migration_prefers_return_container_implementation_over_deprecated_alias():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    idx=Index()
    rows=[('v2.9.1','paddleocr.py','ocr_res.append([box.tolist(), res])\nreturn ocr_res'),
          ('v3.0.0','paddleocr/_pipelines/ocr.py','def ocr(self, img, **kwargs):\n    return self.predict(img, **kwargs)'),
          ('v3.0.0','paddleocr/_pipelines/ocr.py','def predict(self, input):\n    return list(self.predict_iter(input))'),
          ('v3.0.0','tests/pipelines/test_ocr.py','res = result[0]\nassert isinstance(res["rec_texts"], list)')]
    idx.chunks=[{**idx.chunks[0],'chunk_id':f'c{i}','source_id':f's{i}','version':v,
                 'document_path':path,'content':text,'line_end':text.count('\n')+1}
                for i,(v,path,text) in enumerate(rows)]
    search=EvidenceSearch(idx,build_views(idx.chunks,token_count=len),scorer=lambda pairs:[1]*len(pairs))
    result=search.search(plan_query('OCR 结果读取升级有什么变化？',versions=('v2.9.1','v3.0.0')),top_k=3,strategy='contextual_rerank')
    assert {'c0','c2','c3'}=={h['chunk_id'] for h in result['results']}
