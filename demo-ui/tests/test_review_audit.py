from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


MODULE_PATH = Path(__file__).parents[2] / "demo-ui" / "services" / "review_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("review_audit", MODULE_PATH)
    assert spec and spec.loader, "SQLite review audit repository is not implemented"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(task_id: str, decision: str) -> dict:
    return {
        "schema_version": 3,
        "task_id": task_id,
        "request_fingerprint": f"fingerprint-{task_id}",
        "request_mode": "natural_language",
        "request_summary": "调整全局参数并核对默认值。",
        "request_plan": {"change_type": "parameter_config", "queries": []},
        "retrieval_trace": {"queries": [{"query": "默认值", "status": "candidate_found"}]},
        "evidence_sources": [{"chunk_id": "wiki-1eadc6584f96:zh:guide/parameter/global:1"}],
        "human_decision": decision,
        "decided_at_utc": "2026-09-30T12:00:00+00:00",
    }


def test_review_decisions_survive_repository_recreation_and_remain_append_only(tmp_path):
    repository_type = _module().SQLiteReviewAudit
    database = tmp_path / "audit.sqlite3"
    first = repository_type(database)
    first_event = first.record(_report("task-a", "reviewed"), session_id="session-1")
    second_event = first.record(_report("task-a", "rejected"), session_id="session-1")

    reopened = repository_type(database)
    events = reopened.list_recent(session_id="session-1")

    assert first_event["event_id"] != second_event["event_id"]
    assert [event["human_decision"] for event in events] == ["rejected", "reviewed"]
    assert events[0]["task_id"] == "task-a"
    assert events[0]["actor_session_id"] == "session-1"
    assert events[0]["request_plan"]["change_type"] == "parameter_config"
    assert reopened.list_recent(session_id="session-2") == []


def test_review_audit_rejects_invalid_decisions_and_limits_history_query(tmp_path):
    repository = _module().SQLiteReviewAudit(tmp_path / "audit.sqlite3")

    try:
        repository.record(_report("task-a", "auto_apply"), session_id="session-1")
    except ValueError as exc:
        assert "human_decision" in str(exc)
    else:
        raise AssertionError("unsupported review decisions must be rejected")

    repository.record(_report("task-a", "reviewed"), session_id="session-1")
    repository.record(_report("task-b", "rejected"), session_id="session-2")
    assert len(repository.list_recent(session_id="session-1", limit=1)) == 1


def test_workbench_review_callback_persists_decision_and_reports_event(monkeypatch, tmp_path):
    demo_ui = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(demo_ui))
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = {}
    streamlit_stub.fragment = lambda **kwargs: lambda fn: fn
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    spec = importlib.util.spec_from_file_location(
        "public_workbench_under_test", demo_ui / "public_workbench.py"
    )
    assert spec and spec.loader
    workbench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workbench)
    repository = _module().SQLiteReviewAudit(tmp_path / "ui-audit.sqlite3")
    monkeypatch.setattr(workbench, "_audit_repository", lambda: repository)
    result = {
        "task_id": "task-ui",
        "request_fingerprint": "fingerprint-ui",
        "request_mode": "natural_language",
        "request_summary": "核对参数优先级。",
        "request_plan": {"change_type": "parameter_config", "queries": []},
        "retrieval_trace": {"queries": []},
        "retrieved_results": [{
            "chunk_id": "chunk-ui", "source_url": "https://github.com/Seeed-Studio/wiki-documents/doc.md",
        }],
        "review_advice": {"status": "NO_EVIDENCE", "sources": []},
        "evidence_gaps": [],
    }

    workbench._set_review_decision("reviewed", "task-ui", result, "session-ui")

    saved = repository.list_recent(session_id="session-ui")
    assert streamlit_stub.session_state["official_review_decision"] == "reviewed"
    assert streamlit_stub.session_state["official_review_audit_event_id"] == saved[0]["event_id"]
    assert saved[0]["evidence_sources"][0]["chunk_id"] == "chunk-ui"
