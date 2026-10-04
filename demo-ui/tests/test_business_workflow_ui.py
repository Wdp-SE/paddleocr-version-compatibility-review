from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


UI_ROOT = Path(__file__).parents[1]


def test_agent_adapter_imports_only_facade_from_document_workflow():
    source = (UI_ROOT / "services" / "agent_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app.document_workflow"):
            imports.append((node.module, tuple(alias.name for alias in node.names)))
    assert imports == [("app.document_workflow", ("DocumentWorkflowFacade",))]


def test_active_workbench_exposes_retrieval_and_change_review_pages_without_rag(monkeypatch):
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.2")
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()
    assert not app.exception
    button_labels = {item.label for item in app.button}
    assert {"总览", "版本化知识检索", "版本与历史", "资料来源", "发起兼容性审查"} <= button_labels
    assert "加载旗舰模板" not in button_labels


def test_ui_copy_uses_final_product_name_and_no_legacy_research_entry():
    source = (UI_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'page_title="PaddleOCR 版本知识与应用兼容性审查"' in source
    assert "from public_workbench import render" in source
    for forbidden in ("Knowledge Research", "Candidate Knowledge", "Source Discovery", "网页研究"):
        assert forbidden not in source

