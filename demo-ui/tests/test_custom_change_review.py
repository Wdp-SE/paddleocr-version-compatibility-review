from __future__ import annotations

import hashlib
import json
import sys

import pytest

from config import DemoConfig, RAG_ROOT
from services.change_impact_client import ChangeImpactClient
from services.demo_cases import load_demo_cases


sys.path.insert(0, str(RAG_ROOT))
from src.engineering_change import (  # noqa: E402
    EngineeringImpactService, EngineeringItem, TraceLink, compare_engineering_items,
)


class LocalEngineeringRag:
    """Keep the real domain algorithms; only replace network and embedding calls."""

    def __init__(self):
        self.search_queries = []

    def diff_engineering_items(self, old_items, new_items):
        changes = compare_engineering_items(
            [EngineeringItem.model_validate(item) for item in old_items],
            [EngineeringItem.model_validate(item) for item in new_items],
        )
        return {"changes": [item.model_dump(mode="json") for item in changes]}

    def search_candidate_versions(self, query, scope, *, top_k):
        assert scope["active_only"] is True
        self.search_queries.append(query)
        return {"results": []}

    def discover_engineering_impacts(
        self, changed_item_id, items, trace_links, *, dense_item_ids, evidence_by_item
    ):
        impacts = EngineeringImpactService().discover(
            changed_item_id=changed_item_id,
            items=[EngineeringItem.model_validate(item) for item in items],
            trace_links=[TraceLink.model_validate(item) for item in trace_links],
            dense_item_ids=dense_item_ids,
            evidence_by_item=evidence_by_item,
        )
        return {"impacts": [item.model_dump(mode="json") for item in impacts]}


def test_custom_requirement_reuses_diff_impact_and_human_review_in_isolated_sessions(tmp_path):
    case = load_demo_cases()["case-a"]
    inventory = json.loads(case.inventory_path.read_text(encoding="utf-8"))
    current = next(
        item for item in inventory["items"]
        if item["version_id"] == case.requirement_new_version_id
        and item["external_identifier"] == case.changed_external_identifier
    )
    edited = current["content"].replace("1000", "1500")
    assert edited != current["content"]
    source = case.data_root / case.patch_document_filename
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    clients = []
    results = []
    for session_id in ("session_11111111", "session_22222222"):
        config = DemoConfig(runtime_root=tmp_path).for_session(session_id, case.case_id)
        client = ChangeImpactClient(config, case)
        client.rag = LocalEngineeringRag()
        client._ensure_seeded = lambda _: None
        result = client.prepare_custom(
            case.changed_external_identifier, edited,
            "general", "批处理容量",
        )
        clients.append(client)
        results.append(result)

    for result in results:
        modified = next(
            change for change in result["changes"]
            if change["external_identifier"] == case.changed_external_identifier
        )
        assert modified["change_type"] == "MODIFIED"
        assert "1000" in modified["old_content"]
        assert "1500" in modified["new_content"]
        assert result["state"]["status"] == "REVIEW_REQUIRED"
        assert result["state"]["impacts"]
        assert result["state"]["patches"]
        assert result["evidence"]
        assert result["custom_change"]["source"] == "用户输入"
        assert result["custom_change"]["change_type"] == "general"
        assert result["custom_change"]["impact_scope"] == "批处理容量"
        assert "批处理容量" in result["custom_change"]["retrieval_query"]
        assert "批处理容量" in clients[results.index(result)].rag.search_queries[0]
    assert results[0]["state"]["task_id"] != results[1]["state"]["task_id"]
    assert results[0]["state"]["source_path"] != results[1]["state"]["source_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash

    patch = results[0]["state"]["patches"][0]
    reviewed = clients[0].review_patch(
        results[0]["state"]["task_id"], patch["patch_id"],
        action="APPROVE", reviewer="reviewer", comment="已核对本次输入",
    )
    assert reviewed["status"] == "APPLY_READY"
    assert clients[1].facade.get(results[1]["state"]["task_id"]).status == "REVIEW_REQUIRED"


@pytest.mark.skip(reason="历史合成 PAYMENT 页面已退役；当前应用文本入口由 test_paddleocr_workbench 验收")
def test_custom_change_entry_shows_current_requirement_and_clear_action(monkeypatch, tmp_path):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()

    assert not app.exception
    assert app.radio(key="change_mode").value == "预置演示案例"
    app.radio(key="change_mode").set_value("自定义变更").run()
    assert not app.exception
    assert "REQ-023" in app.text_area(key="custom_requirement_content").value
    assert app.button(key="custom_prepare").label == "开始变更审查"


@pytest.mark.skip(reason="历史合成 PAYMENT 页面已退役；当前人审与不写回边界由 test_paddleocr_workbench 验收")
def test_custom_review_does_not_offer_unvalidated_candidate_publication(monkeypatch, tmp_path):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DEMO_REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()
    app.session_state["v4_change_result"] = {
        "custom_change": {"source": "用户输入", "publication_available": False},
        "state": {"task_id": "cir_custom", "status": "REVIEW_REQUIRED", "impacts": [], "patches": []},
        "changes": [], "evidence": {},
    }
    app.radio(key="change_mode").set_value("自定义变更").run()

    assert not app.exception
    assert not any(button.key and button.key.startswith("v4_apply_") for button in app.button)
    assert not any(button.key and button.key.startswith("v4_publish_") for button in app.button)
    assert not any(button.key == "v4_prepare" for button in app.button)
    rendered = "\n".join(item.value for item in list(app.info) + list(app.markdown))
    assert "自定义变更只到人工审核" in rendered


@pytest.mark.skip(reason="历史合成 PAYMENT 页面已退役；当前提交及结果导航由 test_paddleocr_workbench 验收")
def test_custom_submit_navigates_without_streamlit_session_state_error(monkeypatch, tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from services.rag_client import RAGClient

    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(RAGClient, "health", lambda self: {"service_status": "READY"})
    monkeypatch.setattr(RAGClient, "artifact_status", lambda self: {})
    monkeypatch.setattr(RAGClient, "documents", lambda self: {"documents": []})
    monkeypatch.setattr(ChangeImpactClient, "prepare_custom", lambda self, req, content, *_args: {
        "state": {"task_id": "custom-task", "status": "REVIEW_REQUIRED", "impacts": [], "patches": []},
        "changes": [], "evidence": {}, "custom_change": {"source": "用户输入"},
    })
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=60).run()
    app.radio(key="change_mode").set_value("自定义变更").run()
    editor = app.text_area(key="custom_requirement_content")
    editor.set_value(editor.value.replace("1000", "1500")).run()
    app.button(key="custom_prepare").click().run()

    assert not app.exception
    assert app.session_state["product_nav"] == "变更分析"
    assert app.session_state["v4_change_result"]["state"]["task_id"] == "custom-task"
