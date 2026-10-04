"""Validate offline evaluation reports against the exact public corpus build."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.build_identity import edge_evaluation_fingerprint


ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_ROOT = ROOT / "evaluation" / "edge_ai_retrieval_v2"
AGENT_ROOT = ROOT / "evaluation" / "edge_ai_change_review_v2"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_ratio(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and 0 <= value <= 1
    )


def validate_pphuman_dev_diagnostic(corpus_root: Path) -> dict | None:
    """Expose matching DEV observations without marking a strategy as validated."""
    folder = ROOT / "evaluation" / "pphuman_release_candidate_v1"
    report = _read(folder / "report.json")
    if (
        not report or report.get("schema_version") != 1
        or report.get("workspace_id") != "pphuman"
        or report.get("assessment_type") != "dev_diagnostic_only"
        or report.get("holdout_executed") is not False
        or report.get("default_promotion_eligible") is not False
    ):
        return None
    paths = {
        "corpus_manifest_sha256": corpus_root / "corpus_manifest.json",
        "chunks_sha256": corpus_root / "chunks.json",
        "retrieval_policy_sha256": corpus_root / "retrieval_policy.json",
        "runtime_config_sha256": corpus_root / "public_retrieval_runtime.json",
        "rag_cases_sha256": ROOT / "evaluation/pphuman_quality_v1/rag_cases.jsonl",
        "agent_cases_sha256": ROOT / "evaluation/pphuman_quality_v1/agent_cases.jsonl",
        "diagnostic_runner_sha256": folder / "run_dev_diagnostic.py",
    }
    for relative in (
        "evaluation/pphuman_quality_v1/run_evaluation.py",
        "evaluation/pphuman_quality_v1/evaluation.py",
        "versioned-rag-service/src/public_knowledge.py",
        "versioned-rag-service/src/public_retrieval_runtime.py",
        "versioned-rag-service/src/retrieval_fusion.py",
        "change-review-agent/app/change_request.py",
        "change-review-agent/app/domain_profile.py",
        "change-review-agent/config/pphuman_change_profile.json",
    ):
        paths[f"code:{relative}"] = ROOT / relative
    inputs = report.get("input_fingerprints")
    if not isinstance(inputs, dict) or any(
        (actual := _sha256(path)) is None or inputs.get(key) != actual
        for key, path in paths.items()
    ):
        return None
    denominators = report.get("denominators")
    if not isinstance(denominators, dict) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in denominators.values()
    ):
        return None
    slim = {}
    for group, metrics in (
        ("rag", ("mean_required_source_recall", "complete_required_source_set_rate", "mean_ndcg_at_k")),
        ("agent", ("required_evidence_source_recall", "complete_required_evidence_set_rate")),
    ):
        values = report.get(group)
        if not isinstance(values, dict) or not values:
            return None
        if any(not isinstance(row, dict) or any(not _valid_ratio(row.get(key)) for key in metrics)
               for row in values.values()):
            return None
        slim[group] = {
            name: {key: row.get(key) for key in (*metrics, "top_k", "query_count", "answerable_case_count",
                "case_count", "in_scope_case_count", "retrieval_top_k_per_query", "wrong_version_result_count",
                "wrong_language_result_count", "unanswerable_candidate_query_count")}
            for name, row in values.items()
        }
    return {
        "name": "pphuman_release_candidate_v1", "assessment_type": "dev_diagnostic_only",
        "default_promotion_eligible": False, "generated_at_utc": report.get("generated_at_utc"),
        "denominators": denominators, "limitations": report.get("limitations", []),
        "input_fingerprints": {key: inputs[key] for key in ("corpus_manifest_sha256", "chunks_sha256")},
        **slim,
    }


def _expected_inputs(corpus_root: Path) -> dict[str, str | None]:
    return {
        "corpus_manifest_sha256": _sha256(corpus_root / "corpus_manifest.json"),
        "chunks_sha256": _sha256(corpus_root / "chunks.json"),
        "dense_vectors_sha256": _sha256(corpus_root / "dense_vectors.npy"),
        "retrieval_policy_sha256": _sha256(corpus_root / "retrieval_policy.json"),
        "runtime_config_sha256": _sha256(corpus_root / "public_retrieval_runtime.json"),
        "cases_sha256": _sha256(RETRIEVAL_ROOT / "cases.jsonl"),
        "code:evaluation/edge_ai_retrieval_v2/run_evaluation.py": _sha256(RETRIEVAL_ROOT / "run_evaluation.py"),
        "code:versioned-rag-service/src/public_knowledge.py": _sha256(ROOT / "versioned-rag-service" / "src" / "public_knowledge.py"),
        "code:versioned-rag-service/src/public_retrieval_runtime.py": _sha256(ROOT / "versioned-rag-service" / "src" / "public_retrieval_runtime.py"),
        "code:versioned-rag-service/src/retrieval_fusion.py": _sha256(ROOT / "versioned-rag-service" / "src" / "retrieval_fusion.py"),
    }


def _retrieval_release(corpus_root: Path, expected: dict[str, str | None]) -> dict | None:
    lock_path = RETRIEVAL_ROOT / "split_lock.json"
    lock = _read(lock_path)
    if not lock or lock.get("dataset_id") != "edge_ai_retrieval_v2":
        return None
    inputs = lock.get("input_fingerprints")
    if not isinstance(inputs, dict) or any(inputs.get(key) != value for key, value in expected.items()):
        return None
    if any(value is None for value in expected.values()):
        return None
    split_counts = lock.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {"dev", "holdout"}:
        return None
    reports = {}
    for split in ("dev", "holdout"):
        path = RETRIEVAL_ROOT / f"{split}_report.json"
        report = _read(path)
        if (
            not report or report.get("status") != "MEASURED"
            or report.get("dataset_id") != "edge_ai_retrieval_v2"
            or report.get("input_fingerprints") != inputs
            or report.get("split_lock_sha256") != _sha256(lock_path)
            or set((report.get("splits") or {})) != {split}
        ):
            return None
        strategies = report["splits"][split]
        if set(strategies) != set(lock.get("strategy_allowlist", [])) or not strategies:
            return None
        for values in strategies.values():
            if (
                not isinstance(values, dict)
                or values.get("query_count") != split_counts[split]
                or not _valid_ratio(values.get("mean_required_source_recall"))
                or not _valid_ratio(values.get("complete_required_source_set_rate"))
            ):
                return None
        reports[split] = strategies
    return {"lock": lock, "reports": reports}


def _agent_release(expected_retrieval_inputs: dict[str, str | None]) -> dict | None:
    lock_path = AGENT_ROOT / "split_lock.json"
    lock = _read(lock_path)
    if not lock or lock.get("dataset_id") != "edge_ai_change_review_v2":
        return None
    inputs = lock.get("input_fingerprints")
    if not isinstance(inputs, dict):
        return None
    for key, value in expected_retrieval_inputs.items():
        if inputs.get(key) != value:
            return None
    expected_agent = {
        "agent_cases_sha256": _sha256(AGENT_ROOT / "cases.jsonl"),
        "code:evaluation/edge_ai_change_review_v2/run_evaluation.py": _sha256(AGENT_ROOT / "run_evaluation.py"),
        "code:change-review-agent/app/change_request.py": _sha256(ROOT / "change-review-agent" / "app" / "change_request.py"),
        "code:change-review-agent/app/domain_profile.py": _sha256(ROOT / "change-review-agent" / "app" / "domain_profile.py"),
        "code:change-review-agent/config/edge_ai_device_change_profile.json": _sha256(ROOT / "change-review-agent" / "config" / "edge_ai_device_change_profile.json"),
    }
    if any(value is None or inputs.get(key) != value for key, value in expected_agent.items()):
        return None
    split_counts = lock.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {"dev", "holdout"}:
        return None
    reports = {}
    for split in ("dev", "holdout"):
        report = _read(AGENT_ROOT / f"{split}_report.json")
        if (
            not report or report.get("status") != "MEASURED"
            or report.get("dataset_id") != "edge_ai_change_review_v2"
            or report.get("input_fingerprints") != inputs
            or report.get("split_lock_sha256") != _sha256(lock_path)
            or set((report.get("splits") or {})) != {split}
        ):
            return None
        metrics = report["splits"][split]
        if (
            metrics.get("case_count") != split_counts[split]
            or any(not _valid_ratio(metrics.get(name)) for name in (
                "expected_change_type_accuracy", "scope_gap_detection_accuracy",
                "manual_review_boundary_accuracy",
            ))
            or not isinstance(metrics.get("human_quality_scoring"), dict)
            or metrics["human_quality_scoring"].get("status") != "not_scored"
        ):
            return None
        reports[split] = metrics
    return {"lock": lock, "reports": reports}


def validate_public_evaluation_release(
    manifest: dict, corpus_root: Path, build_identity: dict,
) -> dict | None:
    """Return compact measured summaries only when every frozen input still matches."""
    if manifest.get("workspace_id") != "edge_ai_device":
        return None
    corpus_root = Path(corpus_root)
    expected = _expected_inputs(corpus_root)
    retrieval = _retrieval_release(corpus_root, expected)
    if retrieval is None:
        return None
    agent = _agent_release(expected)
    if agent is None:
        return None
    corpus_identity = build_identity.get("corpus_fingerprint", {})
    if (
        not isinstance(corpus_identity, dict)
        or corpus_identity.get("manifest_sha256") != expected["corpus_manifest_sha256"]
        or corpus_identity.get("chunks_sha256") != expected["chunks_sha256"]
        or build_identity.get("evaluation_fingerprint") != edge_evaluation_fingerprint(ROOT)
    ):
        return None
    policy = "bm25"
    if any(values.get(policy) is None for values in retrieval["reports"].values()):
        return None
    return {
        "status": "validated",
        "dataset_id": "edge_ai_retrieval_v2",
        "case_count": retrieval["lock"]["case_count"],
        "case_split_counts": retrieval["lock"]["split_counts"],
        "selected_policy": policy,
        "selection_reason": "DEV/HOLDOUT 上 BM25 Faceted RRF 未提高已标注来源召回或完整来源集率，保留更简单的 BM25 基线。",
        "retrieval": {
            split: {
                name: {
                    key: values[key]
                    for key in (
                        "query_count", "answerable_query_count", "unanswerable_query_count",
                        "mean_required_source_recall", "complete_required_source_set_rate",
                        "wrong_scope_result_count", "wrong_language_result_count",
                        "wrong_snapshot_result_count", "unanswerable_candidate_rate", "latency_ms",
                    )
                }
                for name, values in retrieval["reports"][split].items()
            }
            for split in ("dev", "holdout")
        },
        "change_review": {
            "dataset_id": "edge_ai_change_review_v2",
            "case_count": agent["lock"]["case_count"],
            "case_split_counts": agent["lock"]["split_counts"],
            "splits": {
                split: {
                    key: value for key, value in metrics.items()
                    if key not in {"cases", "human_quality_scoring"}
                } | {"human_quality_scoring": metrics["human_quality_scoring"]}
                for split, metrics in agent["reports"].items()
            },
        },
        "corpus_manifest_sha256": expected["corpus_manifest_sha256"],
        "evaluation_fingerprint": build_identity.get("evaluation_fingerprint"),
    }
