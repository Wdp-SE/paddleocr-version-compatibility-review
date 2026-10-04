"""Verified source and public-asset fingerprints for deployment diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def git_revision(root: Path, *, timeout_seconds: float = 1.0) -> str | None:
    """Return the platform deploy commit or a clean Git HEAD; otherwise unknown."""
    if os.getenv("RENDER", "").casefold() == "true":
        render_revision = os.getenv("RENDER_GIT_COMMIT", "").strip().casefold()
        if _SHA40.fullmatch(render_revision):
            return render_revision
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False, capture_output=True, text=True, timeout=timeout_seconds,
        )
        if revision.returncode != 0:
            return None
        value = revision.stdout.strip().casefold()
        if not _SHA40.fullmatch(value):
            return None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=False, capture_output=True, text=True, timeout=timeout_seconds,
        )
        if status.returncode != 0 or status.stdout.strip():
            return None
        return value
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def content_fingerprint(paths: dict[str, Path]) -> dict[str, str | None]:
    """Hash named files without returning any file contents."""
    files = {name: _sha256(path) for name, path in sorted(paths.items())}
    if any(value is None for value in files.values()):
        aggregate = "unknown"
    else:
        serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        aggregate = hashlib.sha256(serialized).hexdigest()
    return {**files, "fingerprint_sha256": aggregate}


def public_build_identity(
    *, repo_root: Path, corpus_root: Path, retrieval_config_path: Path,
) -> dict:
    corpus_fingerprint = content_fingerprint({
        "manifest_sha256": corpus_root / "corpus_manifest.json",
        "chunks_sha256": corpus_root / "chunks.json",
        "document_relationships_sha256": corpus_root / "document_relations.json",
        "reviewed_figure_sidecar_sha256": corpus_root / "figure_evidence_reviewed.json",
    })
    retrieval_fingerprint = content_fingerprint({
        "runtime_config_sha256": retrieval_config_path,
        "base_policy_sha256": corpus_root / "retrieval_policy.json",
        "figure_lock_sha256": corpus_root / "figure_evidence_reviewed.lock.json",
    })["fingerprint_sha256"]
    try:
        manifest = json.loads((corpus_root / "corpus_manifest.json").read_text(encoding="utf-8"))
        project_id = manifest.get("project_id")
        workspace_id = manifest.get("workspace_id")
    except (OSError, json.JSONDecodeError, AttributeError):
        project_id = None
        workspace_id = None
    evaluation_fingerprint = (
        industrial_inspection_evaluation_fingerprint(repo_root)
        if project_id == "industrial-inspection"
        else pphuman_evaluation_fingerprint(repo_root)
        if workspace_id == "pphuman"
        else paddleocr_evaluation_fingerprint(repo_root)
        if workspace_id == "paddleocr"
        else edge_evaluation_fingerprint(repo_root)
    )
    return {
        "build_revision": git_revision(repo_root) or "unknown",
        "corpus_fingerprint": corpus_fingerprint,
        "retrieval_config_fingerprint": retrieval_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
    }


def paddleocr_evaluation_fingerprint(repo_root: Path) -> str:
    """Keep application compatibility acceptance separate from past-domain scores."""
    root = Path(repo_root) / "evaluation" / "paddleocr_compatibility_v1"
    paths = {
        "cases": root / "cases.json",
        "runner": root / "run_evaluation.py",
        "report": root / "report.json",
        "compatibility_tool": Path(repo_root) / "versioned-rag-service" / "src" / "paddleocr_compatibility.py",
        "retrieval_fusion": Path(repo_root) / "versioned-rag-service" / "src" / "retrieval_fusion.py",
    }
    if any(not path.is_file() for path in paths.values()):
        return "pending_paddleocr_evaluation"
    return content_fingerprint(paths)["fingerprint_sha256"]


def edge_evaluation_fingerprint(repo_root: Path) -> str:
    """Bind the current public evaluation identity to its inputs and measured outputs."""
    evaluation_root = repo_root / "evaluation"
    retrieval_root = evaluation_root / "edge_ai_retrieval_v2"
    agent_root = evaluation_root / "edge_ai_change_review_v2"
    return content_fingerprint({
        "retrieval_cases": retrieval_root / "cases.jsonl",
        "retrieval_split_lock": retrieval_root / "split_lock.json",
        "retrieval_runner": retrieval_root / "run_evaluation.py",
        "retrieval_dev_report": retrieval_root / "dev_report.json",
        "retrieval_holdout_report": retrieval_root / "holdout_report.json",
        "agent_cases": agent_root / "cases.jsonl",
        "agent_split_lock": agent_root / "split_lock.json",
        "agent_runner": agent_root / "run_evaluation.py",
        "agent_dev_report": agent_root / "dev_report.json",
        "agent_holdout_report": agent_root / "holdout_report.json",
        "agent_change_planner": repo_root / "change-review-agent" / "app" / "change_request.py",
        "agent_domain_profile_loader": repo_root / "change-review-agent" / "app" / "domain_profile.py",
        "agent_domain_profile": repo_root / "change-review-agent" / "config" / "edge_ai_device_change_profile.json",
    })["fingerprint_sha256"]


def industrial_inspection_evaluation_fingerprint(repo_root: Path) -> str:
    """Return only this project's evaluation identity; never reuse legacy-domain scores."""
    evaluation_root = Path(repo_root) / "evaluation"
    retrieval_root = evaluation_root / "industrial_inspection_retrieval_v1"
    review_root = evaluation_root / "industrial_inspection_change_review_v1"
    paths = {
        "retrieval_cases": retrieval_root / "cases.jsonl",
        "retrieval_split_lock": retrieval_root / "split_lock.json",
        "retrieval_runner": retrieval_root / "run_evaluation.py",
        "retrieval_dev_report": retrieval_root / "dev_report.json",
        "retrieval_holdout_report": retrieval_root / "holdout_report.json",
        "review_cases": review_root / "cases.jsonl",
        "review_split_lock": review_root / "split_lock.json",
        "review_runner": review_root / "run_evaluation.py",
        "review_dev_report": review_root / "dev_report.json",
        "review_holdout_report": review_root / "holdout_report.json",
    }
    if any(not path.is_file() for path in paths.values()):
        return "pending_project_evaluation"
    return content_fingerprint(paths)["fingerprint_sha256"]


def pphuman_evaluation_fingerprint(repo_root: Path) -> str:
    """Bind evaluation status only to PP-Human cases, code and measured reports."""
    evaluation_root = Path(repo_root) / "evaluation" / "pphuman_v1"
    paths = {
        "retrieval_cases": evaluation_root / "retrieval_cases.jsonl",
        "retrieval_split_lock": evaluation_root / "retrieval_split_lock.json",
        "retrieval_runner": evaluation_root / "run_retrieval_evaluation.py",
        "retrieval_dev_report": evaluation_root / "retrieval_dev_report.json",
        "retrieval_holdout_report": evaluation_root / "retrieval_holdout_report.json",
        "agent_cases": evaluation_root / "agent_cases.jsonl",
        "agent_split_lock": evaluation_root / "agent_split_lock.json",
        "agent_runner": evaluation_root / "run_agent_evaluation.py",
        "agent_dev_report": evaluation_root / "agent_dev_report.json",
        "agent_holdout_report": evaluation_root / "agent_holdout_report.json",
        "agent_change_planner": Path(repo_root) / "change-review-agent" / "app" / "change_request.py",
        "agent_domain_profile_loader": Path(repo_root) / "change-review-agent" / "app" / "domain_profile.py",
        "agent_domain_profile": Path(repo_root) / "change-review-agent" / "config" / "pphuman_change_profile.json",
    }
    if any(not path.is_file() for path in paths.values()):
        return "pending_pphuman_evaluation"
    return content_fingerprint(paths)["fingerprint_sha256"]
