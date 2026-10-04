from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_always_runs_the_public_profile_workbench(monkeypatch):
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("DEMO_LEGACY_FIXTURES", "true")

    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()

    assert not app.exception
    visible = "\n".join(item.value for item in list(app.title) + list(app.markdown))
    assert "PaddleOCR 文档处理应用研发工作台" in visible
    assert "研发知识与变更审查工作台" not in visible
    assert "DEMO_LEGACY_FIXTURES" not in visible
    assert not app.tabs
    assert any(button.label == "版本化知识检索" for button in app.button)


def test_public_profile_workbench_handles_unavailable_rag(monkeypatch):
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("DEMO_LEGACY_FIXTURES", "false")

    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()

    assert not app.exception
    assert any("暂未连接" in item.value or "等待连接" in item.value for item in app.markdown)
