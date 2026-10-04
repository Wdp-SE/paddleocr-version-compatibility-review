from __future__ import annotations

import json
from pathlib import Path

from app.change_request import build_request_plan, classify_change_type
from app.domain_profile import load_change_profile
from app.public_review import PublicReviewAgent


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "pphuman_change_profile.json"
VERSION = "v2.9.0"
COMMIT = "b25522a0f4bde8c80603f3ba5e3472059972e3b5"
REPOSITORY = "PaddlePaddle/PaddleDetection"
SOURCE = {
    "chunk_id": f"{VERSION}:zh:deploy/pipeline/docs/tutorials/pphuman_mot:1",
    "document_id": f"{VERSION}:zh:deploy/pipeline/docs/tutorials/pphuman_mot",
    "document_key": "deploy/pipeline/docs/tutorials/pphuman_mot",
    "document_title": "行人跟踪部署",
    "version": VERSION,
    "source_snapshot": VERSION,
    "source_id": "pphuman-source-v2-9-mot",
    "repository": REPOSITORY,
    "document_path": "deploy/pipeline/docs/tutorials/pphuman_mot.md",
    "source_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/deploy/pipeline/docs/tutorials/pphuman_mot.md",
    "commit": COMMIT,
    "source_sha256": "a" * 64,
    "language": "zh",
    "locale": "zh-CN",
    "heading": "行人跟踪部署",
    "content": "变更跟踪器配置后，需要核对检测输入、跟踪参数和部署流程。",
    "retrieval_score": 1.25,
}
REGISTRY = [{
    "source_id": SOURCE["source_id"],
    "source_url": SOURCE["source_url"],
    "repository": REPOSITORY,
    "version": VERSION,
    "source_snapshot": VERSION,
    "commit": COMMIT,
    "path": SOURCE["document_path"],
    "sha256": SOURCE["source_sha256"],
}]


def test_active_profile_and_rules_are_specific_to_pphuman():
    profile = load_change_profile(PROFILE_PATH)

    assert profile["id"] == "pphuman"
    assert profile["languages"] == ["zh"]
    assert {row["id"] for row in profile["change_types"]} == {
        "model_config", "behavior_pipeline", "tracking", "deployment", "general",
    }
    assert classify_change_type("调整行人跟踪器配置并检查跨镜跟踪流程") == "tracking"
    assert classify_change_type("更换行为识别模型并核对行为分析配置") == "behavior_pipeline"


def test_request_plan_does_not_invent_hardware_scope_for_software_docs():
    profile = load_change_profile(PROFILE_PATH)
    plan = build_request_plan(
        "变更行人跟踪的配置，核查相关部署步骤和验证项。",
        profile=profile,
        target_snapshot=VERSION,
    )

    assert plan["domain_profile_id"] == "pphuman"
    assert plan["target_version"] == VERSION
    assert plan["device_scope"] == {}
    assert plan["scope_warnings"] == []
    assert plan["manual_review_required"] is True


class PPHumanGateway:
    def __init__(self, reported_policy=None, omit_reported_policy=False):
        self.calls = []
        self.reported_policy = reported_policy
        self.omit_reported_policy = omit_reported_policy
        self.results = [SOURCE]

    def workspace(self):
        return {
            "workspace_id": "pphuman",
            "workspace": "PP-Human 行人分析工程知识",
            "domain_profile": {"id": "pphuman"},
            "repository": REPOSITORY,
            "repositories": [REPOSITORY],
            "current_version": VERSION,
            "available_versions": [VERSION, "v2.8.1"],
            "version_scopes": {"latest": {"versions": [VERSION]}},
            "languages": ["zh"],
            "source_registry": REGISTRY,
        }

    def search(self, question, *, version, language, top_k=5,
               device_model=None, module_sku=None, carrier_board=None, software_baseline=None,
               retrieval_policy=None):
        self.calls.append((question, version, language, top_k, retrieval_policy))
        reported_policy = self.reported_policy
        if isinstance(reported_policy, list):
            reported_policy = reported_policy.pop(0) if reported_policy else None
        result = {"results": self.results[:top_k]}
        if not self.omit_reported_policy:
            result["retrieval_policy"] = reported_policy or retrieval_policy or "bm25"
        return result

    def engineering_diff(self, old_items, new_items):
        return {"changes": [{"change_type": "MODIFIED"}]}

    def engineering_impacts(self, payload):
        related = payload.get("items", [])[1:]
        return {"impacts": [{
            "impacted_item_id": item["item_id"],
            "review_status": "SUGGESTED",
        } for item in related]}

    def review_advice(self, summary, evidence_chunk_ids, *, version="current", **_scope):
        return self.review_advice_for_version(summary, evidence_chunk_ids, version=version)

    def review_advice_for_version(self, summary, evidence_chunk_ids, *, version, **_scope):
        return {
            "status": "OK",
            "answer": "跟踪配置变更需要人工核对输入、参数和部署验证。",
            "sources": [SOURCE],
            "review": {
                "impact_candidates": [{
                    "evidence_chunk_id": evidence_chunk_ids[0],
                    "reason": "该来源说明跟踪器配置与部署步骤。",
                    "suggested_action": "检查相关配置并执行回归验证。",
                }],
                "evidence_gaps": [],
                "version_ambiguities": [],
                "reviewer_actions": ["由研发人员确认并记录结论。"],
                "review_status": "REQUIRES_HUMAN_REVIEW",
            },
        }


def test_agent_runs_version_scoped_pphuman_review_with_manifest_evidence():
    gateway = PPHumanGateway()
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
    )

    assert result["request_plan"]["domain_profile_id"] == "pphuman"
    assert result["request_plan"]["change_type"] == "tracking"
    assert gateway.calls
    assert all(call[1:3] == (VERSION, "zh") for call in gateway.calls)
    assert result["retrieved_results"] == [SOURCE]
    assert result["impacts"][0]["evidence"]["source_id"] == SOURCE["source_id"]
    assert result["sandbox_only"] is True
    assert result["public_baseline_written"] is False


class ConfigurationGateway(PPHumanGateway):
    def __init__(self, *, trace_version=VERSION, fail_trace=False):
        super().__init__()
        self.trace_version = trace_version
        self.fail_trace = fail_trace
        self.trace_calls = []

    def config_trace(self, document_ids, *, version):
        self.trace_calls.append((document_ids, version))
        if self.fail_trace:
            raise TimeoutError("configuration service unavailable")
        row = {
            "source_id": SOURCE["source_id"], "document_id": SOURCE["document_id"],
            "path": SOURCE["document_path"], "version": self.trace_version,
            "source_url": SOURCE["source_url"],
            "relation_type": "document_references", "direction": "outgoing",
            "target_path": "deploy/pipeline/config/tracker_config.yml",
            "target_document_id": None, "target_source_id": None,
            "target_status": "not_indexed",
            "evidence": {"line_start": 2, "line_end": 2, "text": "configs/tracker_config.yml"},
        }
        return {
            "version": version, "status": "partial", "relations": [row], "gaps": [dict(row, kind="not_indexed")],
            "roots": [{key: row[key] for key in ("source_id", "document_id", "path", "version", "source_url")}],
            "bounds": {"depth": 1, "truncated": False}, "latency_ms": 2.0,
        }


def test_agent_attaches_explicit_reference_and_missing_target_without_confirming_impact():
    gateway = ConfigurationGateway()
    result = PublicReviewAgent(gateway).analyze_request("调整 PP-Human 跟踪器配置")
    assert gateway.trace_calls == [([SOURCE["document_id"]], VERSION)]
    assert result["configuration_trace"]["relations"][0]["target_status"] == "not_indexed"
    assert result["configuration_trace"]["impact_status"] == "REQUIRES_HUMAN_REVIEW"
    assert any(row["gap_type"] == "CONFIG_REFERENCE_NOT_INDEXED" for row in result["evidence_gap_details"])
    assert result["public_baseline_written"] is False


def test_agent_drops_cross_version_configuration_report_and_keeps_rag_evidence():
    gateway = ConfigurationGateway(trace_version="v2.8.1")
    result = PublicReviewAgent(gateway).analyze_request("调整 PP-Human 跟踪器配置")
    assert result["configuration_trace"]["status"] == "INVALID_RESPONSE"
    assert result["configuration_trace"]["relations"] == []
    assert result["retrieved_results"] == [SOURCE]
    assert any(row["gap_type"] == "CONFIG_TRACE_UNAVAILABLE" for row in result["evidence_gap_details"])


def test_agent_keeps_generation_available_when_reference_tool_times_out():
    result = PublicReviewAgent(ConfigurationGateway(fail_trace=True)).analyze_request("调整 PP-Human 跟踪器配置")
    assert result["configuration_trace"]["status"] == "UNAVAILABLE"
    assert result["review_advice"]["status"] == "OK"


def test_named_change_target_limits_trace_roots_instead_of_tracing_all_semantic_hits():
    class NamedTargetGateway(ConfigurationGateway):
        def workspace(self):
            workspace = super().workspace()
            workspace["source_registry"] = [*REGISTRY, {
                **REGISTRY[0], "source_id": "target-tracker",
                "path": "deploy/pipeline/config/tracker_config.yml",
                "source_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/deploy/pipeline/config/tracker_config.yml",
            }]
            return workspace
    gateway = NamedTargetGateway()
    PublicReviewAgent(gateway).analyze_request(
        "调整 deploy/pipeline/config/tracker_config.yml 的 camera_motion，核对跟踪参数"
    )
    assert gateway.trace_calls[0][0] == ["v2.9.0:zh:deploy/pipeline/config/tracker_config"]


def test_agent_can_opt_in_to_term_expansion_for_every_pphuman_rag_check():
    gateway = PPHumanGateway()
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )

    assert gateway.calls
    assert all(call[-1] == "bm25_pphuman_term_expansion_rrf" for call in gateway.calls)
    assert result["retrieval_policy"] == "bm25_pphuman_term_expansion_rrf"
    assert result["retrieval_policy_requested"] == "bm25_pphuman_term_expansion_rrf"
    assert result["retrieval_policy_status"] == "CONFIRMED"


def test_agent_does_not_mark_mixed_or_different_backend_policy_as_confirmed():
    gateway = PPHumanGateway(reported_policy="bm25")
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )

    assert result["retrieval_policy"] == "bm25"
    assert result["retrieval_policy_requested"] == "bm25_pphuman_term_expansion_rrf"
    assert result["retrieval_policy_status"] == "MISMATCH"


def test_agent_marks_mixed_backend_policies_as_mismatch():
    gateway = PPHumanGateway(reported_policy=["bm25", "bm25_pphuman_term_expansion_rrf"])
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )

    assert result["retrieval_policy"] == "mixed"
    assert result["retrieval_policy_status"] == "MISMATCH"


def test_agent_does_not_claim_policy_confirmation_when_backend_does_not_report_it():
    gateway = PPHumanGateway(omit_reported_policy=True)
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )

    assert result["retrieval_policy"] == "not_confirmed"
    assert result["retrieval_policy_requested"] == "bm25_pphuman_term_expansion_rrf"
    assert result["retrieval_policy_status"] == "UNCONFIRMED"


class BatchGateway(PPHumanGateway):
    def __init__(self, *, batch_error=None):
        super().__init__()
        self.batch_calls = []
        self.legacy_search_calls = []
        self.batch_error = batch_error

    def search_batch(self, summary, *, checks, version, language, top_k=5,
                     retrieval_policy=None, **scope):
        self.batch_calls.append({
            "summary": summary, "checks": checks, "version": version,
            "language": language, "top_k": top_k,
            "retrieval_policy": retrieval_policy, **scope,
        })
        if self.batch_error:
            raise self.batch_error
        return {
            "checks": [{
                "check_index": check["check_index"],
                "query": check["query"],
                "results": [SOURCE],
            } for check in checks],
            "candidates": [{**SOURCE, "check_indices": [check["check_index"] for check in checks]}],
            "results": [SOURCE],
            "ranked_ids": [SOURCE["chunk_id"]],
            "requested_policy": retrieval_policy,
            "actual_policy": "task_adaptive_rerank",
            "retrieval_policy": retrieval_policy,
            "rerank_status": "OK",
            "rerank_diagnostics": {"rerank_calls": 1},
            "candidate_count": 1,
            "final_evidence_count": 1,
            "stage_latency_ms": {"retrieval": 100, "rerank": 250, "total": 350},
        }

    def search(self, question, **kwargs):
        self.legacy_search_calls.append((question, kwargs))
        return super().search(question, **kwargs)


def test_task_adaptive_agent_uses_one_batch_and_exposes_rerank_trace():
    gateway = BatchGateway()
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="task_adaptive_rerank",
    )

    assert len(gateway.batch_calls) == 1
    assert gateway.legacy_search_calls == []
    assert gateway.batch_calls[0]["retrieval_policy"] == "task_adaptive_rerank"
    assert result["retrieved_results"] == [SOURCE]
    assert result["retrieval_trace"]["rerank_status"] == "OK"
    assert result["retrieval_trace"]["http_request_count"] == 1
    assert result["retrieval_trace"]["subquery_count"] >= 1


def test_task_adaptive_batch_connection_failure_falls_back_to_bounded_bm25():
    gateway = BatchGateway(batch_error=ConnectionError("batch endpoint unavailable"))
    result = PublicReviewAgent(gateway).analyze_request(
        "调整 PP-Human 行人跟踪器配置，核对相关部署和回归验证。",
        target_version=VERSION,
        retrieval_policy="task_adaptive_rerank",
    )

    assert len(gateway.batch_calls) == 1
    assert 0 < len(gateway.legacy_search_calls) <= 4
    assert all(call[1].get("retrieval_policy", "bm25") == "bm25" for call in gateway.legacy_search_calls)
    assert result["retrieval_trace"]["rerank_status"] == "NOT_AVAILABLE_FALLBACK"
    assert result["retrieval_policy"] == "bm25"


def test_selected_document_review_accepts_task_adaptive_policy_for_pphuman():
    gateway = PPHumanGateway()
    related = {**SOURCE, "chunk_id": SOURCE["chunk_id"] + ":related",
               "content": "相关跟踪验证应覆盖目标配置。"}
    gateway.results = [related]
    result = PublicReviewAgent(gateway).analyze(
        SOURCE, "改用新的跟踪器参数", retrieval_policy="task_adaptive_rerank",
    )

    searches = [call for call in gateway.calls if isinstance(call, tuple) and len(call) == 5]
    assert len(searches) == 1
    assert searches[0][-1] == "task_adaptive_rerank"
    assert result["retrieval_policy_requested"] == "task_adaptive_rerank"
