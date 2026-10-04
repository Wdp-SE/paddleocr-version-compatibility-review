from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from components.evidence_view import render_query_sources, render_retrieval_results


UI_ROOT = Path(__file__).parents[1]


def _hit(rank: int) -> dict:
    return {
        "rank": rank,
        "document_id": f"doc-{rank}",
        "document_title": f"研发文档{rank}",
        "version_id": f"v{rank}",
        "version_label": "V2.0",
        "version_status": "ACTIVE",
        "project_id": "PAYMENT",
        "document_type": "REQUIREMENT",
        "section_id": f"REQ-{rank:03}",
        "section_path": ["容量"],
        "page_number": rank,
        "chunk_id": f"chunk-{rank}",
        "text": f"REQ-{rank:03} 第 {rank} 条真实内容",
        "similarity": 0.8,
    }


@pytest.mark.skip(reason="历史合成案例 tab 页面已退役；当前版本/中文导航由 test_paddleocr_workbench 验收")
def test_knowledge_defaults_to_current_case_and_uses_chinese_navigation(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()

    assert not app.exception
    assert app.radio(key="scope_mode").value == "当前案例"
    assert [tab.label for tab in app.tabs][:5] == [
        "工作台", "变更分析", "修改审核", "版本发布", "知识服务"
    ]
    page = "\n".join(item.value for item in app.markdown)
    assert "版本可信 RAG" in page
    assert "变更审查 Agent" in page


def test_case_scope_never_searches_other_project_without_explicit_global_choice():
    from types import SimpleNamespace

    from components.product_experience import case_knowledge_scope

    case_a = SimpleNamespace(project_id="PAYMENT")
    case_b = SimpleNamespace(project_id="AUDIT")
    assert case_knowledge_scope("当前案例", case_a) == {
        "project_ids": ["PAYMENT"], "active_only": True
    }
    assert case_knowledge_scope("当前案例", case_b) == {
        "project_ids": ["AUDIT"], "active_only": True
    }
    assert case_knowledge_scope("全部资料", case_a) == {"active_only": True}


def test_prebuilt_case_titles_use_business_chinese():
    from services.demo_cases import load_demo_cases

    cases = load_demo_cases()
    assert cases["case-a"].title.startswith("演示案例 A")
    assert cases["case-b"].title.startswith("演示案例 B")


def test_ranked_evidence_shows_three_first_and_folds_remaining_results():
    hits = [_hit(rank) for rank in range(1, 5)]
    def render_hits(value):
        from components.evidence_view import render_retrieval_results

        render_retrieval_results(value, {})

    app = AppTest.from_function(render_hits, args=(hits,)).run()

    assert not app.exception
    assert any(item.label == "查看更多结果（1）" for item in app.expander)
    rendered = "\n".join(item.value for item in list(app.markdown) + list(app.caption))
    assert rendered.index("第 1 条来源") < rendered.index("第 2 条来源") < rendered.index("第 3 条来源")
    assert "工程编号：REQ-001" in rendered
    assert "检索得分" in rendered
    assert "不代表内容真实性" in rendered


def test_query_citation_shows_actual_retrieved_content():
    source = _hit(1)
    source["content"] = source["text"]
    def render_sources(value):
        from components.evidence_view import render_query_sources

        render_query_sources(value, {})

    app = AppTest.from_function(render_sources, args=([source],)).run()

    assert not app.exception
    rendered = "\n".join(item.value for item in list(app.markdown) + list(app.caption))
    assert "引用依据 1" in rendered
    assert "REQ-001 第 1 条真实内容" in rendered
    assert "工程编号：REQ-001" in rendered
    assert "当前版本" in rendered


@pytest.mark.skip(reason="历史 PAYMENT/AUDIT 目录页面已退役；当前固定官方版本由 test_paddleocr_workbench 验收")
def test_knowledge_catalog_uses_case_versions_not_frozen_runtime(monkeypatch, tmp_path):
    from services.change_impact_client import ChangeImpactClient
    from services.rag_client import RAGClient

    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(RAGClient, "health", lambda self: {"service_status": "READY"})
    monkeypatch.setattr(RAGClient, "artifact_status", lambda self: {})
    monkeypatch.setattr(RAGClient, "documents", lambda self: {"documents": [{
        "document_id": "frozen-only", "title": "旧冻结语料",
        "project_id": "OTHER", "document_type": "OTHER", "versions": [],
    }]})
    monkeypatch.setattr(ChangeImpactClient, "ensure_documents", lambda self: None)
    monkeypatch.setattr(ChangeImpactClient, "_version_documents", lambda self: [{
        "document_id": "requirements", "title": "requirements_v1",
        "project_id": "PAYMENT", "document_type": "需求规格说明书",
        "active_version": {"version_id": "requirements-v2", "version_label": "V2.0",
                           "source_name": "requirements_v2.docx", "status": "ACTIVE"},
        "versions": [],
    }])
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60)
    app.session_state["product_nav"] = "知识服务"
    app.run()

    assert not app.exception
    labels = [item.label for item in app.expander]
    assert any("当前案例资料" in label for label in labels)
    shown = "\n".join(item.value for item in app.markdown)
    assert "需求规格说明书" in shown
    assert "requirements_v2.docx" in shown
    assert "旧冻结语料" not in shown
