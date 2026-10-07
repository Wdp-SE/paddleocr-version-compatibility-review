from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services.public_knowledge_client import PublicKnowledgeClient
from services.public_workspace_profile import clear_workspace_bound_results, public_workspace_mismatch
from services.review_audit import SQLiteReviewAudit


APP = Path(__file__).parents[1] / "app.py"
COMMITS = {"v2.9.1": "07603421c20a96bb94bb87d0c4211032527ae706", "v3.0.0": "a8474288ad53c0f439c272b786c5fa6240f0cf27"}
SOURCES = [{"source_id": f"ocr-{version}", "document_id": f"{version}:contract:paddleocr",
            "version": version, "repository": "PaddlePaddle/PaddleOCR", "commit": commit,
            "path": "paddleocr.py" if version == "v2.9.1" else "paddleocr/__init__.py",
            "sha256": ("a" if version == "v2.9.1" else "b") * 64,
            "source_url": f"https://github.com/PaddlePaddle/PaddleOCR/blob/{commit}/" + ("paddleocr.py" if version == "v2.9.1" else "paddleocr/__init__.py")}
           for version, commit in COMMITS.items()]
CHUNKS = [{**row, "chunk_id": row["source_id"] + ":1", "document_path": row["path"],
           "document_key": "paddleocr-contract", "document_title": "OCR 接口契约", "heading": "OCR 接口",
           "locale": "zh-CN", "language": "zh", "line_start": 1, "line_end": 30,
           "content": "固定提交接口与结果说明", "source_type": "official_code_contract", "retrieval_score": 1.0}
          for row in SOURCES]
WORKSPACE = {"workspace_id": "paddleocr", "workspace": "PaddleOCR 文档处理应用研发知识",
             "domain_profile": {"id": "paddleocr", "example_queries": ["PaddleOCR 3.0.0 如何调用 OCR？"]},
             "repository": "PaddlePaddle/PaddleOCR", "repositories": ["PaddlePaddle/PaddleOCR"],
             "languages": ["zh"], "current_version": "v3.0.0", "baseline_version": "v2.9.1",
             "available_versions": ["v3.0.0", "v2.9.1"], "source_registry": SOURCES, "rag_ready": True,
             "retrieval_policy": "bm25", "source_count": 2, "chunk_count": 2, "generation_available": False}


def static_report(files):
    content = files[0]["content"]
    return {"schema_version": 1, "project_id": "paddleocr", "workspace_id": "paddleocr", "source_version": "v2.9.1",
            "target_version": "v3.0.0", "runtime_verified": False, "status": "supported_risk",
            "files": [{"path": row["path"], "sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                       "line_count": len(row["content"].splitlines())} for row in files],
            "findings": [{"finding_id": "removed-structure", "rule_id": "removed_structure_api", "status": "supported_risk",
                          "title": "旧版结构接口需迁移", "explanation": "目标版本的官方导出接口变化。",
                          "application": {"path": files[0]["path"], "line": 1, "end_line": 1, "snippet": content.splitlines()[0]},
                          "evidence": [{"source_id": row["source_id"], "chunk_id": row["chunk_id"], "version": row["version"],
                                        "sha256": row["sha256"], "commit": row["commit"], "url": row["source_url"],
                                        "path": row["path"], "line_start": 5, "line_end": 7, "title": "OCR 接口契约"} for row in CHUNKS],
                          "desired_check": "运行应用样本并核对结果字段"}],
            "gaps": [{"code": "runtime_not_verified", "detail": "真实扫描文档的 OCR 推理尚未验证"}],
            "verification_steps": ["在目标版本环境用实际扫描文档回归"],
            "summary": {"finding_count": 1, "risk_count": 1, "gap_count": 1}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("DEMO_AUDIT_DB_PATH", str(tmp_path / "audit.sqlite3"))
    calls = []
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: copy.deepcopy(WORKSPACE))
    monkeypatch.setattr(PublicKnowledgeClient, "documents", lambda self: copy.deepcopy(CHUNKS))
    monkeypatch.setattr(PublicKnowledgeClient, "document", lambda self, document_id: copy.deepcopy(CHUNKS))
    def compatibility(self, files, **scope):
        calls.append((copy.deepcopy(files), scope))
        return static_report(files)
    monkeypatch.setattr(PublicKnowledgeClient, "compatibility_review", compatibility)
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "results": [copy.deepcopy(row) for row in CHUNKS if row["version"] == scope["version"]], "retrieval_policy": "bm25"})
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "status": "GENERATION_NOT_CONFIGURED", "answer": "N/A", "sources": [],
        "evidence": [copy.deepcopy(row) for row in CHUNKS if row["version"] == scope["version"]]})
    return calls, tmp_path


def visible(app):
    return "\n".join(str(item.value) for item in list(app.markdown) + list(app.caption) + list(app.info) + list(app.warning))


def test_interview_examples_keep_cross_version_question_and_select_scope(client, monkeypatch):
    workspace=copy.deepcopy(WORKSPACE)
    workspace['domain_profile']['example_queries']=[
        'PaddleOCR 2.9.1 的 OCR 调用如何只识别文字而不检测？',
        'PaddleOCR 3.0.0 的 OCR 结果中 rec_texts 和 rec_scores 分别表示什么？',
        'PaddleOCR 2.9.1 升级到 3.0.0，旧结果读取方式为什么需要调整？',
        '仅凭官方资料，能否确认我们应用升级后下游 JSON 输出兼容？']
    monkeypatch.setattr(PublicKnowledgeClient,'workspace',lambda self:copy.deepcopy(workspace))
    app=AppTest.from_file(APP,default_timeout=30).run()
    app.button(key='nav_版本检索与问答').click().run()
    assert len(app.selectbox(key='official_example').options)==4
    app.selectbox(key='official_example').select(workspace['domain_profile']['example_queries'][2]).run()
    assert app.selectbox(key='official_version').value=='all'
    assert not app.exception


def test_navigation_cache_and_query_submission_refresh(client, monkeypatch):
    workspace_calls, query_calls = [], []
    def workspace(self):
        workspace_calls.append(self)
        return copy.deepcopy(WORKSPACE)
    monkeypatch.setattr(PublicKnowledgeClient, 'workspace', workspace)
    monkeypatch.setattr(PublicKnowledgeClient, 'query_official',
                        lambda self, question, **kwargs: query_calls.append(kwargs) or {'status': 'NO_EVIDENCE'})
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key='nav_版本检索与问答').click().run()
    app.slider(key='official_top_k').set_value(8).run()
    assert len(workspace_calls) == 1
    app.button(key='knowledge_generate').click().run()
    assert not app.exception
    assert len(workspace_calls) == 2 and len(query_calls) == 1
    assert query_calls[0]['top_k'] == 8
    assert workspace_calls[0] is workspace_calls[1]


def test_submission_blocks_when_cached_ready_backend_becomes_unavailable(client, monkeypatch):
    current, submitted = {'ready': True}, []
    monkeypatch.setattr(PublicKnowledgeClient, 'workspace',
                        lambda self: {**copy.deepcopy(WORKSPACE), 'rag_ready': current['ready']})
    monkeypatch.setattr(PublicKnowledgeClient, 'query_official',
                        lambda *args, **kwargs: submitted.append(True))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key='nav_版本检索与问答').click().run()
    current['ready'] = False
    app.button(key='knowledge_generate').click().run()
    assert not app.exception and not submitted
    assert '本次请求未提交' in visible(app)


def test_internal_fact_refusal_is_not_mislabeled_as_an_empty_search(client, monkeypatch):
    monkeypatch.setattr(PublicKnowledgeClient, 'query_official',
                        lambda *args, **kwargs: {'status': 'OUT_OF_SCOPE', 'evidence': []})
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key='nav_版本检索与问答').click().run()
    app.button(key='knowledge_generate').click().run()
    assert not app.exception
    assert '输出契约和目标环境的回归记录' in visible(app)
    assert '未找到可直接支持答案' not in visible(app)


def open_review():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="nav_新建变更审查").click().run()
    if any(box.key=='ocr_input_mode' for box in app.selectbox):
        app.selectbox(key='ocr_input_mode').select('规则片段').run()
    return app


def test_full_application_is_default_and_edits_clear_review(client):
    calls,_=client
    app=AppTest.from_file(APP,default_timeout=30).run()
    app.button(key='nav_新建变更审查').click().run()
    assert app.selectbox(key='ocr_input_mode').value=='完整文档处理应用'
    app.button(key='ocr_run_review').click().run()
    assert not app.exception
    assert len(calls[0][0])>4
    app.text_area(key='ocr_bundle_content').input('from paddleocr import PaddleOCR\n').run()
    assert not app.session_state.get('ocr_compatibility_review')


def test_identical_input_rerun_preserves_active_background_job(client,monkeypatch):
    from threading import Event
    release=Event();original=PublicKnowledgeClient.compatibility_review
    def slow(self,*args,**kwargs):
        release.wait(20)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(PublicKnowledgeClient,'compatibility_review',slow)
    app=open_review()
    try:
        app.button(key='ocr_run_review').click().run()
        job=app.session_state['ocr_job']
        app.run()
        assert app.session_state['ocr_job']['identity']==job['identity']
        assert not job['cancel'].is_set()
    finally:release.set()


def test_home_shows_latest_but_query_follows_application_dependency(client):
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any(item.value == "PaddleOCR 文档处理应用研发工作台" for item in app.title)
    assert "PP-Human" not in visible(app)
    assert "v3.0.0" in visible(app) and "当前应用依赖" in visible(app)
    app.button(key="nav_版本检索与问答").click().run()
    assert not app.exception
    assert app.selectbox(key="official_version").value == "v2.9.1"
    assert app.selectbox(key='official_version').disabled
    app.radio(key='internal_query_mode').set_value('target').run()
    assert app.selectbox(key='official_version').value == 'v3.0.0'
    app.radio(key='internal_query_mode').set_value('manual').run()
    app.selectbox(key='official_version').select('v2.9.1').run()
    app.run()
    assert app.selectbox(key='official_version').value == 'v2.9.1'
    assert app.selectbox(key="official_retrieval_policy").options == ["BM25 基线（默认）"]
    assert "PP-Human" not in visible(app)


def test_current_impact_report_does_not_claim_evaluation_is_missing(client, monkeypatch):
    workspace = copy.deepcopy(WORKSPACE)
    workspace['impact_evaluation'] = {
        'retrieval': {'bm25': {'dev': {'metrics': {
            'complete_count': 1, 'count': 2, 'fact_hit_count': 1, 'fact_count': 2,
            'wrong_version_count': 0, 'wrong_module_count': 1}}}},
        'selected_strategy': 'bm25', 'promotion_reason': 'validation gate not passed',
        'review': [], 'limitations': ['Internal labels, not business accuracy'],
    }
    monkeypatch.setattr(PublicKnowledgeClient, 'workspace', lambda self: copy.deepcopy(workspace))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key='nav_检索评测').click().run()
    assert not app.exception
    assert '当前服务尚未返回与 PaddleOCR 活动语料绑定的评测记录' not in visible(app)
    assert any(item.value == '当前语料与实现的冻结评测' for item in app.subheader)


def test_static_report_workflow_shows_bound_locations_both_versions_and_human_review(client):
    calls, _ = client
    app = open_review()
    assert not app.exception
    assert app.selectbox(key="ocr_source_version").value == "v2.9.1"
    assert app.selectbox(key="ocr_target_version").value == "v3.0.0"
    app.button(key="ocr_run_review").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert set(calls[0][1]) == {"source_version", "target_version",'request_timeout'}
    assert 0<calls[0][1]['request_timeout']<=240
    page = visible(app)
    assert "应用位置：app/removed_structure_api.py" in page
    assert "当前依赖证据 · v2.9.1" in page and "升级目标证据 · v3.0.0" in page
    assert "实际 OCR 推理未验证" in page and "真实扫描文档" in page
    assert "模型核查建议（待人工确认）" not in page
    for key in ("nav_可能相关资料", "nav_修改前后对照", "nav_人工审核"):
        app.button(key=key).click().run()
        assert not app.exception
        assert app.session_state["ocr_compatibility_review"]["workspace_id"] == "paddleocr"


def test_human_review_persists_versions_files_and_exact_static_facts(client):
    _, temp = client
    app = open_review()
    app.button(key="ocr_run_review").click().run()
    next(button for button in app.button if button.label == "确认已审阅本次静态兼容性报告").click().run()
    assert not app.exception
    result = app.session_state["ocr_compatibility_review"]
    records = SQLiteReviewAudit(temp / "audit.sqlite3").list_recent(session_id=app.session_state["official_session_id"])
    assert len(records) == 1
    record = records[0]
    assert record["source_version"] == "v2.9.1" and record["target_version"] == "v3.0.0"
    assert record["application_files"] == result["compatibility_report"]["files"]
    assert record["compatibility_report"] == result["compatibility_report"]
    assert record["runtime_verified"] is False
    assert "不代表实际推理兼容性已验证" in record["human_review_scope"]
    assert "人工审核已记录" in visible(app)
    assert "等待人工审核" not in visible(app)


def test_model_suggestion_records_show_actual_provider_model_keys(client):
    app = open_review()
    app.button(key="ocr_run_review").click().run()
    result = copy.deepcopy(app.session_state["ocr_compatibility_review"])
    result.update(model_status="OK", model_suggestions=[{
        "reason": "核对新接口", "suggested_action": "执行扫描样本回归", "evidence": CHUNKS[1], "verified": False,
    }], model_generation={"requested_model": "deepseek-v4-flash", "returned_model": "deepseek-flash",
                         "usage": {"total_tokens": 100}})
    app.session_state["ocr_compatibility_review"] = result
    app.run()
    assert not app.exception
    assert "请求模型：deepseek-v4-flash" in visible(app)
    assert "实际模型：deepseek-flash" in visible(app)
    result.update(model_status="GENERATION_RESPONSE_INVALID", model_suggestions=[])
    app.session_state["ocr_compatibility_review"] = result
    app.run()
    assert not app.exception
    assert "建议结构或引用未通过校验" in visible(app)
    assert "实际模型：deepseek-flash" in visible(app)
    assert "实际 OCR 推理未验证" in visible(app)


def test_editing_application_invalidates_previous_report_and_decision(client):
    app = open_review()
    app.button(key="ocr_run_review").click().run()
    next(button for button in app.button if button.label == "确认已审阅本次静态兼容性报告").click().run()
    app.text_area(key="ocr_application_content").set_value("from paddleocr import PaddleOCR\nocr = PaddleOCR()\n").run()
    assert not app.exception
    assert "ocr_compatibility_review" not in app.session_state
    assert "official_review_decision" not in app.session_state


def test_ocr_benchmark_does_not_reuse_other_project_metrics(client, monkeypatch):
    stale = {**WORKSPACE, "retrieval_evaluation_status": "edge_ai_retrieval_v2_validated",
             "development_evaluation": {"rag": {"bm25@5": {"mean_required_source_recall": 0.98}}}}
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: stale)
    app = AppTest.from_file(APP).run()
    app.button(key="nav_检索评测").click().run()
    assert not app.exception
    assert "0.98" not in visible(app)
    assert "PP-Human" not in visible(app)
    assert "静态规则验收不代表实际 OCR" in visible(app)


def test_current_source_derived_metrics_show_span_hits_and_static_cases_without_accuracy_claim(client, monkeypatch):
    report_path = APP.parents[1] / "evaluation" / "paddleocr_compatibility_v1" / "report.json"
    report = {**json.loads(report_path.read_text(encoding="utf-8")), "project_id": "paddleocr",
              "assessment_type": "source_derived_acceptance"}
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: {**WORKSPACE, "retrieval_evaluation": report})
    app = AppTest.from_file(APP).run()
    app.button(key="nav_检索评测").click().run()
    assert not app.exception
    assert len(app.dataframe) == 2
    assert "不能替代回答事实性" in visible(app)
    assert "保留 BM25" in visible(app)


def test_workspace_switch_clears_manual_question_and_static_review():
    state = {"official_workspace": {"workspace_id": "pphuman", "repository": "PaddlePaddle/PaddleDetection"},
             "official_question": "old field question", "official_review_decision": "reviewed",
             "ocr_compatibility_review": {"workspace_id": "other"}}
    assert clear_workspace_bound_results(state, WORKSPACE)
    assert "official_question" not in state and "ocr_compatibility_review" not in state
    assert public_workspace_mismatch(WORKSPACE, public_demo=True) is None
    assert public_workspace_mismatch({**WORKSPACE, "workspace_id": "pphuman"}, public_demo=True)


def test_quality_comparison_and_original_runtime_are_separate_from_uploaded_app(client,monkeypatch):
    quality={'selected_on_dev':'bm25_rerank','top_k':5,'strategies':{
        'bm25_rerank':{'dev':{'count':12,'complete_rate':.5,'fact_recall':.6,'wrong_version_count':0},
                       'holdout':{'count':12,'complete_rate':.4,'fact_recall':.5,'wrong_version_count':0}}}}
    runtime={'scope':'原创三页回归','reports':[
        {'paddleocr_version':'2.9.1','legacy_consumer_success':True,'sample_regression_passed':True},
        {'paddleocr_version':'3.0.0','legacy_consumer_success':False,'sample_regression_passed':True}]}
    monkeypatch.setattr(PublicKnowledgeClient,'workspace',lambda self:{**WORKSPACE,'quality_comparison':quality,'upgrade_demo':runtime})
    app=AppTest.from_file(APP).run()
    app.button(key='nav_检索评测').click().run()
    assert not app.exception
    assert '开发集选型' in visible(app)
    assert '不代表用户上传应用已运行' in visible(app)


def test_disconnect_preserves_identity_until_new_workspace_invalidates_evidence():
    previous = {"workspace_id": "pphuman", "repository": "PaddlePaddle/PaddleDetection"}
    state = {"official_workspace": previous, "official_result": {"answer": "old corpus answer"}}
    assert clear_workspace_bound_results(state, None) is False
    assert state["official_workspace"] == previous
    assert clear_workspace_bound_results(state, WORKSPACE) is True
    assert "official_result" not in state


def test_gateway_posts_only_bounded_request_body_and_keeps_session_header():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"status": "needs_verification"}
    class Session:
        def __init__(self):
            self.calls = []
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return Response()
    session = Session()
    client = PublicKnowledgeClient("http://localhost:8765", session=session, session_id="session_12345678", retry_limit=2)
    files = [{"path": "app/ocr.py", "content": "from paddleocr import PaddleOCR"}]
    client.compatibility_review(files)
    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "POST" and url.endswith("/public/compatibility-review")
    assert kwargs["json"] == {"source_version": "v2.9.1", "target_version": "v3.0.0", "files": files}
    assert kwargs["headers"]["X-Demo-Session-ID"] == "session_12345678"


def test_risk_disposition_hold_is_persisted_without_runtime_approval(client):
    _, temp=client
    app=open_review()
    app.button(key='ocr_run_review').click().run()
    assert not app.exception
    continuation=next(b for b in app.button if b.label=='人工记录：继续升级流程')
    assert continuation.disabled  # no disposition or regression logs submitted
    next(b for b in app.button if b.label=='人工记录：暂缓升级').click().run()
    assert not app.exception
    records=SQLiteReviewAudit(temp/'audit.sqlite3').list_recent(session_id=app.session_state['official_session_id'])
    assert records[0]['event_type']=='upgrade_disposition'
    assert records[0]['upgrade_decision']=='hold'
    assert records[0]['runtime_verified'] is False
    assert 'ocr_disposition_state' in app.session_state
    app.text_area(key='ocr_application_content').set_value('from paddleocr import PaddleOCR\n').run()
    assert not app.exception
    assert 'ocr_disposition_state' not in app.session_state


def test_application_context_change_invalidates_bound_review(client):
    app=open_review()
    app.button(key='ocr_run_review').click().run()
    assert not app.exception
    assert app.session_state['ocr_compatibility_review']['application_context']['application_version'] is None
    app.text_input(key='ocr_change_reason').set_value('替换底层 OCR 依赖，保持 JSON 输出契约').run()
    assert not app.exception
    assert 'ocr_compatibility_review' not in app.session_state


def test_benchmark_labels_partial_neural_execution_as_fallback(client,monkeypatch):
    metrics={'complete_count':15,'count':24,'wrong_module_count':52,'wrong_version_count':0}
    release={'selected_strategy':'contextual_bm25','limitations':'混合执行不是稳定重排成绩',
        'retrieval':{'bge_rerank':{'dev':{'metrics':metrics,'execution':'重排 22/24；回退 2'}}}}
    monkeypatch.setattr(PublicKnowledgeClient,'workspace',lambda self:{**WORKSPACE,'rag_quality_evaluation':release})
    app=AppTest.from_file(APP).run()
    app.button(key='nav_检索评测').click().run()
    assert not app.exception
    assert '回退 2' in app.dataframe[0].value.to_string(index=False)
