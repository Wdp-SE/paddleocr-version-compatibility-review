from __future__ import annotations

import pytest

from app.public_review import PublicReviewAgent, _official_hit


SNAPSHOT = "wiki-1eadc6584f96"
REPOSITORY = "Seeed-Studio/wiki-documents"
SOURCE = {
    "chunk_id": f"{SNAPSHOT}:zh:industrial-deployment:1",
    "document_id": f"{SNAPSHOT}:zh:industrial-deployment",
    "document_key": "industrial-deployment",
    "version": SNAPSHOT,
    "source_snapshot": SNAPSHOT,
    "source_id": "seeed-industrial-deployment",
    "repository": REPOSITORY,
    "document_path": "sites/zh-CN/docs/recomputer_industrial/deployment.md",
    "source_url": "https://wiki.seeedstudio.com/cn/recomputer_industrial_deployment/",
    "language": "zh",
    "locale": "zh-CN",
    "heading": "边缘推理部署与软件基线",
    "content": "部署前核对设备型号、JetPack、L4T 与驱动版本；升级后按目标工作负载验证。",
    "retrieval_score": 1.25,
    "device_model": ["reComputer Industrial J4012"],
    "module_sku": ["P3767-0000"],
    "carrier_board": ["J401"],
    "software_baselines": ["JetPack 6.2"],
}
REGISTRY = [{
    "source_id": SOURCE["source_id"],
    "source_url": SOURCE["source_url"],
    "repository": REPOSITORY,
    "source_snapshot": SNAPSHOT,
    "commit": "1eadc6584f962b6efdbdb3e49b2b4ce30c85be08",
    "sha256": "a" * 64,
}]


class EdgeGateway:
    def __init__(self, *, results=None, fail_search=False, foreign_workspace=False, invalid_advice=False):
        self.results = [SOURCE] if results is None else results
        self.fail_search = fail_search
        self.foreign_workspace = foreign_workspace
        self.invalid_advice = invalid_advice
        self.calls = []

    def workspace(self):
        self.calls.append(("workspace",))
        if self.foreign_workspace:
            return {"workspace_id": "other", "current_version": "old"}
        return {
            "workspace_id": "edge_ai_device",
            "workspace": "reComputer 工程知识",
            "domain_profile": {"id": "edge_ai_device"},
            "repository": REPOSITORY,
            "repositories": [REPOSITORY],
            "current_version": SNAPSHOT,
            "available_versions": [SNAPSHOT],
            "version_scopes": {SNAPSHOT: {"versions": [SNAPSHOT]}},
            "languages": ["zh"],
            "hardware_models": ["reComputer Industrial J4012"],
            "module_skus": ["P3767-0000"],
            "carrier_boards": ["J401"],
            "software_baselines": ["JetPack 6.2", "JetPack 7.2 (L4T 39.2.0)"],
            "source_registry": REGISTRY,
            "snapshots": [{"version": SNAPSHOT, "commit": REGISTRY[0]["commit"]}],
        }

    def search(
        self, question, *, version, language, top_k=5,
        device_model=None, module_sku=None, carrier_board=None, software_baseline=None,
        retrieval_policy=None,
    ):
        self.calls.append((
            "search", question, version, language, top_k,
            device_model, module_sku, carrier_board, software_baseline, retrieval_policy,
        ))
        if self.fail_search:
            raise ConnectionError("retrieval unavailable")
        return {"retrieval_policy": "bm25", "results": self.results[:top_k]}

    def review_advice_for_version(
        self, summary, evidence_chunk_ids, *, version,
        device_model=None, module_sku=None, carrier_board=None, software_baseline=None,
    ):
        self.calls.append(("review", version, tuple(evidence_chunk_ids), device_model, module_sku, carrier_board, software_baseline))
        cited_id = "unretrieved-chunk" if self.invalid_advice else evidence_chunk_ids[0]
        return {
            "status": "OK",
            "answer": "请按目标设备和软件基线复核部署步骤。",
            "sources": [SOURCE],
            "review": {
                "impact_candidates": [{
                    "evidence_chunk_id": cited_id,
                    "reason": "该资料列出设备和软件基线的适用条件。",
                    "suggested_action": "核对升级路径并执行目标设备回归验证。",
                }],
                "evidence_gaps": ["未提供目标项目的实测记录。"],
                "version_ambiguities": [],
                "reviewer_actions": ["由设备负责人确认兼容性并记录验证结果。"],
                "review_status": "REQUIRES_HUMAN_REVIEW",
            },
        }


def test_agent_uses_pinned_chinese_source_and_forwards_confirmed_device_scope():
    gateway = EdgeGateway()
    result = PublicReviewAgent(gateway).analyze_request(
        "将 J4012 的 JetPack 6.2 升级到 7.2，核对刷写、L4T 版本和运行验证。",
        device_model="reComputer Industrial J4012",
        module_sku="P3767-0000",
        carrier_board="J401",
        software_baseline="JetPack 6.2",
    )

    searches = [call for call in gateway.calls if call[0] == "search"]
    assert result["request_plan"]["domain_profile_id"] == "edge_ai_device"
    assert result["request_plan"]["change_type"] == "software_baseline"
    assert result["retrieval_trace"]["languages_per_check"] == ["zh"]
    assert 0 < len(searches) <= 4
    assert all(call[2:4] == (SNAPSHOT, "zh") for call in searches)
    assert all(call[5:9] == (
        "reComputer Industrial J4012", "P3767-0000", "J401", "JetPack 6.2",
    ) for call in searches)
    assert result["retrieved_results"] == [SOURCE]
    assert result["impacts"][0]["evidence"]["chunk_id"] == SOURCE["chunk_id"]
    assert result["review_advice"]["review"]["review_status"] == "REQUIRES_HUMAN_REVIEW"
    assert result["sandbox_only"] is True
    assert result["public_baseline_written"] is False


def test_agent_rejects_pphuman_only_policy_for_edge_workspace():
    with pytest.raises(ValueError, match="当前知识空间"):
        PublicReviewAgent(EdgeGateway()).analyze_request(
            "将 J4012 的 JetPack 6.2 升级到 7.2，核对刷写、L4T 版本和运行验证。",
            retrieval_policy="bm25_pphuman_term_expansion_rrf",
        )


def test_agent_skips_generation_when_current_corpus_has_no_evidence():
    gateway = EdgeGateway(results=[])
    result = PublicReviewAgent(gateway).analyze_request("核对当前资料未覆盖的 JetPack 部署问题。")

    assert result["stage_status"] == {"planning": "OK", "retrieval": "EMPTY", "generation": "SKIPPED"}
    assert result["review_advice"]["status"] == "NO_EVIDENCE"
    assert result["retrieved_results"] == []
    assert result["impacts"] == []
    assert result["evidence_gap_details"]
    assert not any(call[0] == "review" for call in gateway.calls)


def test_agent_distinguishes_retrieval_failure_from_a_successful_empty_search():
    gateway = EdgeGateway(fail_search=True)
    result = PublicReviewAgent(gateway).analyze_request("核对 JetPack 升级步骤。")

    assert result["stage_status"] == {"planning": "OK", "retrieval": "FAILED", "generation": "SKIPPED"}
    assert result["review_advice"]["status"] == "RETRIEVAL_UNAVAILABLE"
    assert any("检索服务" in gap for gap in result["evidence_gaps"])
    assert not any(call[0] == "review" for call in gateway.calls)


def test_agent_stops_private_company_request_before_workspace_or_model_calls():
    gateway = EdgeGateway()
    result = PublicReviewAgent(gateway).analyze_request("查询本公司的内部 API 和私有工单。")

    assert result["scope_status"] == "OUT_OF_SCOPE"
    assert result["stage_status"]["generation"] == "OUT_OF_SCOPE"
    assert gateway.calls == []


def test_agent_rejects_unsupported_language_workspace_and_unknown_hardware():
    with pytest.raises(ValueError, match="仅收录中文"):
        PublicReviewAgent(EdgeGateway()).analyze_request("检查 JetPack 升级。", language_mode="en")

    with pytest.raises(ValueError, match="仅支持已配置"):
        PublicReviewAgent(EdgeGateway(foreign_workspace=True)).analyze_request("检查 JetPack 升级。")

    gateway = EdgeGateway()
    with pytest.raises(ValueError, match="不在当前知识空间"):
        PublicReviewAgent(gateway).analyze_request("检查未知设备 JetPack 升级。", device_model="Unknown Board")
    assert not any(call[0] == "search" for call in gateway.calls)


def test_agent_discards_model_impact_that_cites_unretrieved_evidence():
    result = PublicReviewAgent(EdgeGateway(invalid_advice=True)).analyze_request("核对 J4012 JetPack 部署基线。")

    assert result["retrieved_results"] == [SOURCE]
    assert result["impacts"] == []
    assert result["evidence_gap_details"]
    assert result["public_baseline_written"] is False


def test_official_source_must_match_the_manifest_registry_exactly():
    assert _official_hit(SOURCE, SNAPSHOT, {REPOSITORY}, allowed_sources=REGISTRY)
    assert not _official_hit({**SOURCE, "source_url": "https://attacker.example/forged"}, SNAPSHOT,
                             {REPOSITORY}, allowed_sources=REGISTRY)
    assert not _official_hit({**SOURCE, "language": "en"}, SNAPSHOT, {REPOSITORY}, allowed_sources=REGISTRY)
    assert not _official_hit({**SOURCE, "version": "old-snapshot"}, SNAPSHOT,
                             {REPOSITORY}, allowed_sources=REGISTRY)
