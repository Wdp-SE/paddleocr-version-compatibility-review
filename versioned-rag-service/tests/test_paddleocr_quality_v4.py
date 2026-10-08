from src.paddleocr_query_plan import query_module
from src.paddleocr_retrieval_views import generation_evidence_text as evidence_text
from src.public_api import _claim_scope_mismatches


def test_product_name_does_not_force_ocr_pipeline_scope():
    assert query_module('PaddleOCR 3.0 如何保存 JSON 结果？') is None
    assert query_module('PPStructureV3 的结果如何保存？') == 'structure'
    assert query_module('OCR 产线的 device 参数') == 'ocr'


def test_structure_api_accepts_structure_evidence_but_not_other_pipeline():
    hit = {'chunk_id': 's', 'module': 'structure', 'version': 'v3.0.0',
           'content': 'PPStructureV3 的结果调用 save_to_json', 'document_path': 'docs/structure.md'}
    claims = [{'text': 'PaddleOCR 的 PPStructureV3 结果可保存 JSON', 'evidence_ids': ['s']}]
    requirements = [{'id': 'r1', 'module': 'structure', 'version': 'v3.0.0'}]
    assert _claim_scope_mismatches(claims, [hit], requirements) == set()
    hit['module'] = 'text_detection'
    assert _claim_scope_mismatches(claims, [hit], requirements) == {0}


def test_selected_table_row_keeps_canonical_header_for_default_check():
    text = '参数说明\n| 参数 | 默认值 | 说明 |\n| --- | --- | --- |\n| cpu_threads | 10 | CPU线程 |\n'
    lo = text.index('| cpu_threads')
    hit = {'content': text, 'line_start': 8, 'line_end': 11,
           'evidence_spans': [{'char_start': lo, 'char_end': len(text),
                               'line_start': 11, 'line_end': 11}]}
    expanded = evidence_text(hit)
    assert '| 参数 | 默认值 | 说明 |' in expanded
    assert '| cpu_threads | 10 | CPU线程 |' in expanded


def test_html_parameter_row_retains_default_column_and_full_description():
    text = '<table>\n<thead>\n<tr><th>参数</th><th>说明</th><th>默认值</th></tr>\n</thead>\n<tbody>\n<tr>\n<td>cpu_threads</td>\n<td>CPU线程数</td>\n<td>10</td>\n</tr>\n</tbody>\n</table>'
    lo = text.index('<td>cpu_threads')
    hi = lo+len('<td>cpu_threads</td>')
    hit = {'content': text, 'line_start': 1, 'line_end': 12,
           'evidence_spans': [{'char_start': lo, 'char_end': hi, 'line_start': 7, 'line_end': 7}]}
    expanded = evidence_text(hit)
    assert '<th>默认值</th>' in expanded
    assert '<td>10</td>' in expanded


def test_context_expansion_remains_bounded_and_preserves_selected_fact():
    text = '无关内容\n' * 2000 + '必要事实\n'
    lo = text.index('必要事实')
    hit = {'content': text, 'line_start': 1, 'line_end': 2001,
           'evidence_spans': [{'char_start': lo, 'char_end': len(text),
                               'line_start': 2001, 'line_end': 2001}]}
    result = evidence_text(hit)
    assert '必要事实' in result
    assert len(result) <= 6000


def test_compound_query_reserves_evidence_for_each_requested_module():
    from types import SimpleNamespace
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_retrieval_views import build_views
    from src.paddleocr_query_plan import plan_query
    chunks = [{'chunk_id': f'c{i}', 'source_id': f's{i}', 'source_sha256': str(i),
               'version': 'v3.0.0', 'language': 'zh', 'namespace': 'project_primary',
               'document_path': path, 'heading': '', 'line_start': 1, 'line_end': 1,
               'content': text} for i, (path, text) in enumerate([
                   ('docs/module_usage/text_detection.md', '文本检测 阈值 阈值 阈值'),
                   ('docs/module_usage/text_detection.md', '文本检测 阈值 详细说明'),
                   ('docs/module_usage/text_recognition.md', '文本识别 模型选择')])]
    search = EvidenceSearch(SimpleNamespace(chunks=chunks), build_views(chunks, token_count=len),
                            scorer=lambda pairs: [3 if '文本检测' in text else 1 for _, text in pairs])
    result = search.search(plan_query('文本检测的阈值；文本识别的模型选择', versions=('v3.0.0',)),
                           top_k=2, strategy='contextual_rerank')
    assert {h['module'] for h in result['results']} == {'text_detection', 'text_recognition'}


def test_failed_support_gets_one_correction_and_still_cannot_release_hallucination(monkeypatch):
    from fastapi.testclient import TestClient
    from src.public_server import create_app
    monkeypatch.setenv('APP_ENV', 'public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION', 'false')
    class Generator:
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return {'claims': [{'text': '所有应用升级均不需要修改', 'evidence_ids': ['E1']}], 'relevant_sources': []}
        def verify_claims_with_diagnostics(self, payload):
            return {'verdicts': [{'claim_index': i, 'supported': False, 'reason': '没有兼容性保证'}
                                 for i in range(len(payload))]}, {}
    app = create_app()
    with TestClient(app) as client:
        gen = Generator(); app.state.public_generator = gen
        result = client.post('/public/query', json={'query': 'PPStructureV3 迁移', 'version': 'v3.0.0'}).json()
        assert gen.calls == 2
        assert result['status'] == 'ABSTAINED' and result['answer'] == 'N/A'
        assert result['correction']['attempts'] == 1


def test_checker_outage_does_not_trigger_paid_generation_retry(monkeypatch):
    from fastapi.testclient import TestClient
    from src.public_server import create_app
    monkeypatch.setenv('APP_ENV', 'public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION', 'false')
    class Generator:
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return {'claims': [{'text': '可以迁移至 PPStructureV3', 'evidence_ids': ['E1']}], 'relevant_sources': []}
        def verify_claims_with_diagnostics(self, payload): raise TimeoutError()
    app = create_app()
    with TestClient(app) as client:
        gen = Generator(); app.state.public_generator = gen
        result = client.post('/public/query', json={'query': 'PPStructureV3 迁移', 'version': 'v3.0.0'}).json()
        assert gen.calls == 1
        assert result['status'] == 'ABSTAINED'
        assert result['correction']['attempts'] == 0


def test_model_ranker_cannot_invent_evidence_ids():
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    from src.paddleocr_retrieval_views import build_views
    from test_paddleocr_evidence_search import Index
    idx = Index()
    search = EvidenceSearch(idx, build_views(idx.chunks, token_count=len))
    result = search.search(plan_query('page_num', versions=('v3.0.0',)),
                           strategy='contextual_llm_rerank', ranker=lambda **kw: (['invented'], {}))
    assert result['diagnostics']['actual_strategy'] == 'contextual_bm25'
    assert [h['chunk_id'] for h in result['results']] == ['c0']


def test_identical_field_names_in_other_pipelines_cannot_crowd_out_selected_scope():
    from types import SimpleNamespace
    from src.paddleocr_evidence_search import EvidenceSearch
    from src.paddleocr_query_plan import plan_query
    from src.paddleocr_retrieval_views import build_views
    chunks = [{'chunk_id': f'c{i}', 'source_id': f's{i}', 'source_sha256': str(i),
               'version': 'v3.0.0', 'language': 'zh', 'namespace': 'project_primary',
               'document_path': path, 'heading': '', 'line_start': 1, 'line_end': 1,
               'content': text} for i,(path,text) in enumerate([
                   ('docs/pipeline_usage/OCR.md','OCR rec_texts 与 rec_scores 按阈值过滤'),
                   *[('docs/pipeline_usage/table_recognition_v2.md','rec_texts rec_scores rec_texts rec_scores')]*50])]
    search = EvidenceSearch(SimpleNamespace(chunks=chunks), build_views(chunks, token_count=len))
    result = search.search(plan_query('OCR rec_texts rec_scores', versions=('v3.0.0',)),candidate_budget=40)
    assert {h['chunk_id'] for h in result['results']} == {'c0'}


def test_html_default_column_rejects_wrong_numeric_default():
    from src.paddleocr_fact_checks import check_mechanical_facts
    hit={'chunk_id':'c','version':'v3.0.0','module':'ocr','document_path':'docs/OCR.md',
         'content':'<table><thead><tr><th>参数</th><th>说明</th><th>默认值</th></tr></thead>'
                   '<tbody><tr><td>cpu_threads</td><td>CPU线程数</td><td>8</td></tr></tbody></table>'}
    result=check_mechanical_facts([{'text':'OCR cpu_threads 默认值为 10','evidence_ids':['c']}],[hit],
        requirements=[{'id':'r1','module':'ocr','version':'v3.0.0','symbols':['cpu_threads']}])
    assert result['claims'][0]['status']=='CONTRADICTED'


def test_usage_counts_generation_verification_and_ranking_across_both_passes():
    from src.public_api import _workflow_usage
    results=[{'generation':{'usage':{'total_tokens':100}},
              'claim_verification':{'diagnostics':{'usage':{'total_tokens':25}}},
              'rerank_diagnostics':{'llm_ranking':{'usage':{'total_tokens':50}}}},
             {'generation':{'usage':{'total_tokens':200}},
              'claim_verification':{'diagnostics':{'usage':{'total_tokens':20}}}}]
    assert _workflow_usage(results)['total_tokens']==395


def test_result_description_does_not_change_its_pipeline_to_recognition_module():
    from src.paddleocr_fact_checks import check_mechanical_facts
    hits=[{'chunk_id':'c','module':'ocr','version':'v3.0.0','content':'rec_texts 是文本识别结果列表',
           'document_path':'docs/OCR.md'}]
    claims=[{'text':'rec_texts 是文本识别结果列表','evidence_ids':['c']}]
    reqs=[{'id':'r1','module':'ocr','version':'v3.0.0','symbols':['rec_texts']}]
    assert _claim_scope_mismatches(claims,hits,reqs)==set()
    assert check_mechanical_facts(claims,hits,requirements=reqs)['claims'][0]['status']!='SCOPE_MISMATCH'


def test_general_upgrade_guide_can_support_structure_api_migration():
    from src.paddleocr_fact_checks import check_mechanical_facts
    hits=[{'chunk_id':'c','module':'general','version':'v3.0.0','content':'使用 PPStructureV3 接口',
           'document_path':'docs/upgrade_notes.md'}]
    claims=[{'text':'使用 PPStructureV3 接口','evidence_ids':['c']}]
    reqs=[{'id':'r1','module':'structure','version':'v3.0.0','symbols':['PPStructure']}]
    assert check_mechanical_facts(claims,hits,requirements=reqs)['claims'][0]['status']!='SCOPE_MISMATCH'


def test_ocr_component_question_keeps_owner_pipeline_scope():
    assert query_module('OCR 产线的文档预处理如何关闭？')=='ocr'


def test_table_pipeline_is_classified_as_its_own_evidence_scope():
    from src.paddleocr_retrieval_views import module_for
    assert module_for({'document_path':'docs/version3.x/pipeline_usage/table_recognition_v2.md'})=='table_recognition_v2'


def test_long_line_context_is_bounded_without_dropping_selected_span():
    text='a'*8000+'必要事实'+'b'*8000
    hit={'content':text,'line_start':1,'line_end':1,
         'evidence_spans':[{'char_start':8000,'char_end':8004,'line_start':1,'line_end':1}]}
    value=evidence_text(hit)
    assert '必要事实' in value and len(value)<=6000


def test_partial_correction_union_retains_both_windows_of_same_source():
    from src.public_api import _merge_evidence
    from src.paddleocr_retrieval_views import evidence_text as exact_text
    original={'chunk_id':'c','content':'FIRST_FACT\nSECOND_FACT','line_start':1,'line_end':2}
    first={**original,'evidence_spans':[{'line_start':1,'line_end':1}]}
    second={**original,'evidence_spans':[{'line_start':2,'line_end':2}]}
    merged=_merge_evidence([first],[second])
    assert len(merged)==1 and exact_text(merged[0])=='FIRST_FACT\nSECOND_FACT'


def test_auto_policy_uses_validated_release_and_configured_ranker(monkeypatch):
    from fastapi.testclient import TestClient
    from src.public_server import create_app
    import src.public_api as api
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    monkeypatch.setattr(api,'load_rag_release',lambda corpus:{'selected_strategy':'contextual_llm_rerank','candidate_budget':40})
    class Ranker:
        def rerank_candidate_ids(self,*,task,candidates):
            return [c['chunk_id'] for c in candidates],{'provider':'test','usage':{'total_tokens':1}}
    app=create_app()
    with TestClient(app) as client:
        app.state.public_generator=Ranker()
        result=client.post('/public/search',json={'query':'OCR cpu_threads','version':'v3.0.0',
            'retrieval_policy':'paddleocr_evidence','evidence_strategy':'auto'}).json()
        assert result['actual_policy']=='contextual_llm_rerank'
        assert result['rerank_diagnostics']['candidate_budget']==40
        assert result['rerank_diagnostics']['scored_windows']<=40


def test_auto_repair_keeps_structured_selection_and_expands_candidates(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import src.public_api as api
    calls=[]
    async def once(payload,request,**kw):
        calls.append(payload)
        return {'status':'OK','answer_completeness':'PARTIAL_SUPPORTED' if len(calls)==1 else 'COMPLETE',
                'actual_policy':'contextual_rerank','claim_verification':{'status':'SUPPORTED'}}
    monkeypatch.setattr(api,'_query_once',once)
    monkeypatch.setattr(api,'_index',lambda request:SimpleNamespace(manifest={'workspace_id':'paddleocr'}))
    payload=api.SearchRequest(query='OCR parameters',retrieval_policy='paddleocr_evidence',evidence_strategy='auto')
    asyncio.run(api.query(payload,None))
    assert calls[1].evidence_strategy=='auto'
    assert calls[1].candidate_budget==120


def test_repair_uses_release_windows_without_overriding_expanded_budget(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    import src.public_api as api
    import src.paddleocr_evidence_search as es
    captured=[]
    class Search:
        def search(self,plan,**options):
            captured.append(options)
            return {'results':[],'requirements':plan['requirements'],'diagnostics':{'actual_strategy':'contextual_rerank','rerank_calls':1,'fallback_reason':None}}
    def configured(index,strategy,**options):
        assert options['structure_windows'] is True
        return Search()
    monkeypatch.setattr(es,'configured_evidence_search',configured)
    monkeypatch.setattr(api,'load_rag_release',lambda corpus:{'selected_strategy':'contextual_rerank','candidate_budget':80,'structure_windows':True})
    monkeypatch.setattr(api,'load_impact_release',lambda corpus:None)
    index=SimpleNamespace(root=Path('.'),manifest={'workspace_id':'paddleocr','versions':['v3.0.0']},_version_members=lambda v:{'v3.0.0'})
    payload=api.SearchRequest(query='OCR usage',version='v3.0.0',retrieval_policy='paddleocr_evidence',evidence_strategy='auto',candidate_budget=120)
    api._execute_public_search(index,payload,None,top_k=8,correction=True)
    assert captured[0]['candidate_budget']==120
