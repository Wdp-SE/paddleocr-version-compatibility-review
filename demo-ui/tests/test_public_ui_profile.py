from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest


UI_ROOT = Path(__file__).parents[1]

# This module only tests removed synthetic-case screens and their former
# session budget. Current public-source generation is covered by the active
# workbench tests; the old UI must not be reactivated to satisfy these tests.
pytestmark = pytest.mark.skip(reason="历史合成案例及会话预算页面已退役；活动入口由 test_public_official_workbench/test_paddleocr_workbench 验收")


def test_public_ui_shows_case_selector_and_cold_start_reconnect(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()

    assert not app.exception
    selector = app.selectbox(key="demo_case_selector")
    assert len(selector.options) == 2
    assert selector.value == "case-a"
    app.radio(key="rag_mode").set_value("RAG 问答").run()
    page = "\n".join(
        [item.value for item in app.markdown]
        + [item.value for item in app.info]
        + [item.value for item in app.caption]
    )
    assert "公开演示" in page
    assert "公共免费演示后端可能正在启动，请稍后重试" in page
    assert app.button(key="reconnect_backend").label == "重新连接"
    assert "当前环境暂未启用生成式回答" in page
    assert "剩余在线模型调用额度" not in page
    assert app.button(key="query_button").disabled is True


def test_public_query_switch_enables_budgeted_answer_when_backend_is_ready(monkeypatch, tmp_path):
    from services.rag_client import RAGClient

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("DEMO_ALLOW_RAG_QUERY", "true")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_SESSION", "3")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(RAGClient, "health", lambda self: {"service_status": "READY"})
    monkeypatch.setattr(RAGClient, "artifact_status", lambda self: {})
    monkeypatch.setattr(RAGClient, "documents", lambda self: {"documents": []})
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()
    app.radio(key="rag_mode").set_value("RAG 问答").run()

    assert not app.exception
    assert app.button(key="query_button").disabled is False
    assert any("本次会话剩余生成次数：3" in item.value for item in app.caption)


def test_public_query_updates_remaining_session_budget_after_answer(monkeypatch, tmp_path):
    from services.change_impact_client import ChangeImpactClient
    from services.rag_client import RAGClient

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("DEMO_ALLOW_RAG_QUERY", "true")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_SESSION", "3")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(RAGClient, "health", lambda self: {"service_status": "READY"})
    monkeypatch.setattr(RAGClient, "artifact_status", lambda self: {})
    monkeypatch.setattr(RAGClient, "documents", lambda self: {"documents": []})
    monkeypatch.setattr(RAGClient, "query_candidate_versions", lambda self, question, scope: {
        "status": "OK", "answer": "有依据的回答", "sources": [{
            "document_id": "requirements", "page_number": 1, "content": "REQ-023 原文",
        }],
    })
    monkeypatch.setattr(ChangeImpactClient, "ensure_documents", lambda self: None)
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()
    app.radio(key="rag_mode").set_value("RAG 问答").run()
    app.button(key="query_button").click().run()

    assert not app.exception
    assert any("本次会话剩余生成次数：2" in item.value for item in app.caption)


def test_query_without_render_generator_explains_retrieval_fallback(monkeypatch, tmp_path):
    from services.change_impact_client import ChangeImpactClient
    from services.rag_client import RAGClient

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("DEMO_ALLOW_RAG_QUERY", "true")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(RAGClient, "health", lambda self: {"service_status": "READY"})
    monkeypatch.setattr(RAGClient, "artifact_status", lambda self: {})
    monkeypatch.setattr(RAGClient, "documents", lambda self: {"documents": []})
    monkeypatch.setattr(RAGClient, "query_candidate_versions", lambda self, question, scope: {
        "status": "GENERATION_NOT_CONFIGURED", "answer": "N/A", "sources": [],
    })
    monkeypatch.setattr(RAGClient, "search_candidate_versions", lambda self, question, top_k, scope: {
        "results": [{"rank": 1, "document_id": "requirements", "page_number": 1,
                     "text": "REQ-023 当前最大并发为 1000", "version_status": "ACTIVE"}],
    })
    monkeypatch.setattr(ChangeImpactClient, "ensure_documents", lambda self: None)
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()
    app.radio(key="rag_mode").set_value("RAG 问答").run()
    app.button(key="query_button").click().run()

    assert not app.exception
    rendered = "\n".join(item.value for item in list(app.info) + list(app.markdown))
    assert "当前服务未配置在线回答" in rendered
    assert "REQ-023 当前最大并发为 1000" in rendered
