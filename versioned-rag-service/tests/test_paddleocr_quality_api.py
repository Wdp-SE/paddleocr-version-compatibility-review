from fastapi.testclient import TestClient
from src.public_server import create_app

def test_workspace_hides_release_missing_presentation_metadata(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import src.public_api as api
    monkeypatch.setattr(api,'load_impact_release',lambda corpus:{'top_k':5,'retrieval':{}})
    assert api._impact_summary(SimpleNamespace(root=tmp_path)) is None

def test_explicit_claim_version_rejects_another_versions_citation():
    from src.public_api import _claim_scope_mismatches
    hits=[{'chunk_id':'old','module':'ocr','version':'v2.9.1','content':'PaddleOCR'}]
    claims=[{'text':'v3.0.0 OCR 可调用该接口','evidence_ids':['old']}]
    req=[{'id':v,'module':'ocr','version':v} for v in ('v2.9.1','v3.0.0')]
    assert _claim_scope_mismatches(claims,hits,req)=={0}


def test_non_numeric_claim_cannot_use_a_different_module_as_ocr_evidence():
    from src.public_api import _claim_scope_mismatches
    hits=[{'chunk_id':'table','module':'table_recognition_v2','version':'v3.0.0','content':'save_to_json'}]
    claims=[{'text':'调用 save_to_json 即可保存结果','evidence_ids':['table']}]
    req=[{'id':'r1','module':'ocr','version':'v3.0.0','symbols':[]}]
    assert _claim_scope_mismatches(claims,hits,req)=={0}


def test_auto_policy_preserves_baseline_when_validation_rejects_candidate(monkeypatch):
    import src.public_api as api
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    monkeypatch.setattr(api,'load_impact_release',lambda corpus:{'promotion_passed':False,'selected_strategy':'bm25',
        'development_candidate':'window_rerank_120','configs':{'bm25':['bm25',80],'window_rerank_120':['contextual_rerank',120]}})
    with TestClient(create_app()) as client:
        result=client.post('/public/search',json={'query':'cpu_threads','version':'v3.0.0',
              'retrieval_policy':'paddleocr_evidence','evidence_strategy':'auto'}).json()
        assert result['actual_policy']=='bm25'


def test_window_evidence_search_returns_verified_spans_and_actual_strategy(monkeypatch):
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    with TestClient(create_app()) as client:
        response=client.post('/public/search',json={'query':'v3.0.0 OCR cpu_threads 默认值',
            'version':'v3.0.0','retrieval_policy':'paddleocr_evidence','evidence_strategy':'contextual_bm25','candidate_budget':80})
        assert response.status_code==200
        result=response.json()
        assert result['actual_policy']=='contextual_bm25'
        assert result['results'] and all(h['evidence_spans'] for h in result['results'])
        assert result['requirements']
        invalid=client.post('/public/search',json={'query':'OCR','retrieval_policy':'paddleocr_evidence','candidate_budget':900})
        assert invalid.status_code==422


def test_review_advice_requires_support_assessment(monkeypatch):
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    class Generator:
        def generate_review(self,**kwargs):
            return {'change_interpretation':'升级接口','impact_candidates':[{'evidence_chunk_id':self.cid,
                'reason':'目标接口无条件兼容','suggested_action':'无需测试'}],'evidence_gaps':[],
                'version_ambiguities':[],'reviewer_actions':[],'review_status':'REQUIRES_HUMAN_REVIEW'}
        def verify_claims_with_diagnostics(self,payload):
            return {'verdicts':[{'claim_index':0,'supported':False,'reason':'来源未作兼容保证'}]},{}
    app=create_app()
    with TestClient(app) as client:
        gen=Generator(); gen.cid=client.post('/public/search',json={'query':'PPStructureV3','version':'v3.0.0'}).json()['results'][0]['chunk_id']
        app.state.public_generator=gen
        result=client.post('/public/review-advice',json={'change_summary':'升级接口','version':'v3.0.0','evidence_chunk_ids':[gen.cid]}).json()
        assert result['status']=='ABSTAINED'
        assert result['claim_verification']['status']=='UNSUPPORTED'


def test_quality_unavailable_explicitly_falls_back_without_version_leak(monkeypatch):
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    monkeypatch.delenv('PADDLEOCR_EMBEDDING_PATH',raising=False)
    monkeypatch.delenv('PADDLEOCR_RERANKER_PATH',raising=False)
    with TestClient(create_app()) as client:
        response=client.post('/public/search',json={'query':'PPStructureV3','version':'v3.0.0','retrieval_policy':'paddleocr_quality'})
        assert response.status_code==200
        result=response.json()
        assert result['actual_policy']=='bm25'
        assert result['rerank_status']=='QUALITY_FALLBACK'
        assert all(r['version']=='v3.0.0' for r in result['results'])


def test_quality_nonfinite_model_output_is_a_diagnosed_fallback(monkeypatch):
    from src import paddleocr_quality
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    class Broken:
        def search(self,*args,**kwargs):raise paddleocr_quality.ModelUnavailable('RERANK_INVALID_OUTPUT')
    monkeypatch.setattr(paddleocr_quality,'configured_quality_retriever',lambda index:Broken())
    with TestClient(create_app()) as client:
        result=client.post('/public/search',json={'query':'PPStructureV3','version':'v3.0.0','retrieval_policy':'paddleocr_quality'}).json()
        assert result['rerank_status']=='QUALITY_FALLBACK'
        assert result['rerank_diagnostics']['failure_reason']=='RERANK_INVALID_OUTPUT'
        assert result['actual_policy']=='bm25'


def test_empty_quality_results_do_not_report_a_rerank_call(monkeypatch):
    from src import paddleocr_quality
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    monkeypatch.setenv('PADDLEOCR_QUALITY_STRATEGY','bm25_rerank')
    class Empty:
        def search(self,*args,**kwargs):return []
    monkeypatch.setattr(paddleocr_quality,'configured_quality_retriever',lambda index:Empty())
    with TestClient(create_app()) as client:
        result=client.post('/public/search',json={'query':'zzzzqvxx999xyz','version':'v3.0.0','retrieval_policy':'paddleocr_quality'}).json()
        assert result['rerank_status']=='SKIPPED_NO_CANDIDATES'
        assert result['rerank_diagnostics']['rerank_calls']==0


def test_public_generation_does_not_release_failed_support_check(monkeypatch):
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    class Generator:
        def generate(self,**kwargs):
            return {'claims':[{'text':'升级后保证所有应用无须修改','evidence_ids':['E1']}],'relevant_sources':[]}
        def verify_claims_with_diagnostics(self,payload):
            return {'verdicts':[{'claim_index':0,'supported':False,'reason':'原文没有无条件兼容保证'}]},{}
    app=create_app()
    with TestClient(app) as client:
        app.state.public_generator=Generator()
        result=client.post('/public/query',json={'query':'PPStructureV3 迁移','version':'v3.0.0'}).json()
        assert result['status']=='ABSTAINED'
        assert result['answer']=='N/A'
        assert result['evidence']
        assert result['generation']['failure_reason']=='CLAIM_SUPPORT_REJECTED'
        assert result['claim_verification']['status']=='UNSUPPORTED'


def test_partial_supported_claims_keep_only_their_cited_sources(monkeypatch):
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('RD_V2_ALLOW_EXTERNAL_GENERATION','false')
    class Generator:
        def generate(self,**kwargs):
            return {'claims':[{'text':'迁移到PPStructureV3','evidence_ids':['E1']},
                              {'text':'所有文档识别百分百正确','evidence_ids':['E2']}],'relevant_sources':[]}
        def verify_claims_with_diagnostics(self,payload):
            return {'verdicts':[{'claim_index':0,'supported':True,'reason':'支持'},
                                {'claim_index':1,'supported':False,'reason':'无保证'}]},{}
    app=create_app()
    with TestClient(app) as client:
        app.state.public_generator=Generator()
        result=client.post('/public/query',json={'query':'PPStructureV3 迁移','version':'v3.0.0'}).json()
        assert result['status']=='OK'
        assert len(result['claims'])==1
        assert len(result['sources'])==1
        assert result['answer_completeness']=='PARTIAL_SUPPORTED'
