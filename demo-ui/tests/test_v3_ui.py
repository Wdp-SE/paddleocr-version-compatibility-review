from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from services.rag_client import RAGClient


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url, kwargs))
        return Response(self.responses.pop(0))


def test_client_catalog_scope_and_diff_contracts():
    document = {
        "document_id": "REQ", "title": "需求", "project_id": "P",
        "document_type": "requirements", "active_version": {"version_id": "REQ@2"},
        "versions": [],
    }
    hit = {
        "rank": 1, "document_id": "REQ", "version_id": "REQ@2",
        "version_label": "V2", "version_status": "ACTIVE", "page_number": 2,
        "section_path": ["容量"], "similarity": 0.9, "content": "并发 1000",
        "chunk_id": "REQ@2:c1",
    }
    diff = {
        "document_id": "REQ", "from_version": "REQ@1", "to_version": "REQ@2",
        "summary": {"MODIFIED": 1}, "sections": [],
    }
    session = Session([
        {"documents": [document]},
        {"query": "并发", "results": [hit]},
        diff,
    ])
    client = RAGClient("http://localhost:8765", session=session)
    assert client.documents()["documents"][0]["title"] == "需求"
    scope = {"document_ids": ["REQ"], "active_only": True}
    assert client.retrieve("并发", 3, scope)["results"][0]["version_id"] == "REQ@2"
    assert session.calls[1][2]["json"]["scope"] == scope
    assert client.diff("REQ", "REQ@1", "REQ@2")["summary"]["MODIFIED"] == 1
    assert session.calls[2][2]["params"] == {
        "from_version_id": "REQ@1", "to_version_id": "REQ@2",
    }


def test_streamlit_exposes_current_edge_knowledge_and_review_controls(monkeypatch):
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.2")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()
    assert not app.exception
    navigation = {item.label for item in app.button}
    assert {"版本化知识检索", "版本与历史", "发起兼容性审查"} <= navigation

    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    assert not app.exception
    selectors = {item.label for item in app.selectbox}
    assert {"版本范围", "资料语言", "示例问题（选择后可编辑）"} <= selectors
    assert any(item.label == "只想核对原文？" for item in app.expander)



def test_evidence_components_show_version_status_and_freshness(monkeypatch):
    from components import evidence_view

    rendered = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(evidence_view.st, "markdown", lambda value, **kwargs: rendered.append(str(value)))
    monkeypatch.setattr(evidence_view.st, "caption", lambda value, **kwargs: rendered.append(str(value)))
    monkeypatch.setattr(evidence_view.st, "write", lambda value, **kwargs: rendered.append(str(value)))
    technical = []
    monkeypatch.setattr(evidence_view.st, "divider", lambda: None)
    monkeypatch.setattr(evidence_view.st, "json", lambda value, **kwargs: technical.append(value))
    monkeypatch.setattr(evidence_view.st, "expander", lambda *args, **kwargs: Context())

    evidence_view.render_retrieval_results([
        {
            "rank": 1, "document_id": "REQ", "version_id": "REQ@2",
            "version_label": "V2.0", "version_status": "ACTIVE",
            "page_number": 2, "section_path": ["容量"], "similarity": 0.9,
            "content": "最大并发 1000", "chunk_id": "REQ@2:c1",
        }
    ], {"REQ": "需求规格说明书"})
    evidence_view.render_agent_evidence([
        {
            "evidence_id": "ev_1", "document_id": "REQ", "version_id": "REQ@2",
            "version_label": "V2.0", "freshness": "FRESH",
            "page_number": 2, "section_path": ["容量"],
            "content": "最大并发 1000",
        }
    ], {"REQ": "需求规格说明书"})
    combined = "\n".join(rendered)
    assert "《需求规格说明书》" in combined
    assert "版本：V2.0" in combined and "章节：容量" in combined and "页码：第 2 页" in combined
    assert "当前版本" in combined and "有效" in combined
    assert "REQ@2" not in combined and "ev_1" not in combined
    assert any(item.get("version_id") == "REQ@2" for item in technical)

