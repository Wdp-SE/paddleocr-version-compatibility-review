from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from streamlit.testing.v1 import AppTest

from components.product_experience import build_status_cards, build_workflow_progress


UI_ROOT = Path(__file__).parents[1]


@pytest.mark.skip(reason="历史合成案例 tab 入口已退役；当前应用审查流程由 test_paddleocr_workbench 验收")
def test_workbench_starts_existing_analysis_flow_and_tracks_case(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    app = AppTest.from_file(UI_ROOT / "app.py", default_timeout=60).run()

    assert not app.exception
    assert app.session_state["product_nav"] == "工作台"
    assert app.button(key="start_demo").label == "开始预置案例"
    app.selectbox(key="demo_case_selector").set_value("case-b").run()
    assert not app.exception
    assert app.session_state["demo_case_selector"] == "case-b"
    assert any("AUDIT" in item.value for item in app.markdown)

    app.button(key="start_analysis").click().run()
    assert not app.exception
    assert app.session_state["product_nav"] == "变更分析"
    assert app.button(key="v4_prepare").label == "运行变更分析"


def test_status_cards_use_live_catalog_values_instead_of_case_hardcodes() -> None:
    case = SimpleNamespace(project_id="AUDIT")
    catalog = [{
        "project_id": "AUDIT",
        "document_type": "REQUIREMENT",
        "active_version": {"version_id": "public-b-req-v2", "version_label": "V2.0"},
        "versions": [{"version_id": "public-b-req-v1"}, {"version_id": "public-b-req-v2"}],
    }, {
        "project_id": "PAYMENT", "document_type": "REQUIREMENT",
        "active_version": {"version_id": "public-a-req-v2", "version_label": "V2.0"},
        "versions": [{"version_id": "public-a-req-v2"}],
    }]
    cards = build_status_cards(
        case, catalog, {"service_status": "READY"},
        {"artifact_status": "COMPLETE"}, {"ready": True},
    )

    assert cards == {
        "project": "AUDIT",
        "current_version": "V2.0",
        "current_version_id": "public-b-req-v2",
        "document_count": 1,
        "version_count": 2,
        "system_status": "就绪",
    }
    assert build_status_cards(case, [], None, None, {"ready": True})["current_version"] == "待连接"
    degraded = build_status_cards(
        case, catalog, {"service_status": "DEGRADED"},
        {"artifact_status": "COMPLETE"}, {"ready": True},
    )
    assert degraded["system_status"] == "待连接"


def test_progress_only_marks_stages_confirmed_by_existing_workflow_state() -> None:
    initial = dict(build_workflow_progress(None))
    assert set(initial.values()) == {"待执行"}

    prepared = {
        "changes": [{"change_type": "MODIFIED"}],
        "evidence_selection": {"selected_evidence_ids": ["ev-1"]},
        "state": {
            "status": "REVIEW_REQUIRED",
            "impacts": [{"impacted_item_id": "design"}],
            "patches": [{"review_status": "PENDING"}],
        },
    }
    progress = dict(build_workflow_progress(prepared))
    assert progress["生成修改建议"] == "已完成"
    assert progress["人工审核"] == "待人工审核"
    assert progress["安全发布新版本"] == "待执行"

    prepared["state"]["patches"][0]["review_status"] = "APPROVED"
    prepared["state"]["status"] = "COMPLETED"
    prepared["state"]["candidate_version_record"] = {"status": "ACTIVE"}
    completed = dict(build_workflow_progress(prepared))
    assert completed["人工审核"] == "已完成"
    assert completed["生成候选版本"] == "已完成"
    assert completed["安全发布新版本"] == "已完成"
