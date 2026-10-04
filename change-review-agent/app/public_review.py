"""Evidence-based review of hypothetical changes to official public knowledge.

The Agent orchestrates RAG HTTP search, existing engineering Diff/Impact APIs,
and human review. It never calls candidate activation or writes upstream data.
The public workbench persists anonymous review decisions in a local SQLite store.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from app.change_request import (
    build_request_plan,
    is_out_of_scope_public_request,
    request_queries,
)
from app.domain_profile import load_change_profile


PPHUMAN_CHANGE_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "pphuman_change_profile.json"
EDGE_CHANGE_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "edge_ai_device_change_profile.json"
_CHANGE_PROFILE_PATHS = {
    "pphuman": PPHUMAN_CHANGE_PROFILE_PATH,
    # Kept for isolated legacy tests and local tooling; the public demo only
    # accepts the PP-Human workspace.
    "edge_ai_device": EDGE_CHANGE_PROFILE_PATH,
}


def _profile_for_workspace(workspace: dict) -> dict:
    workspace_id = workspace.get("workspace_id")
    path = _CHANGE_PROFILE_PATHS.get(workspace_id)
    if path is None:
        raise ValueError("当前工作台仅支持已配置的公开中文知识空间")
    profile = load_change_profile(path)
    declared_profile_id = (workspace.get("domain_profile") or {}).get("id")
    if profile["id"] != workspace_id or declared_profile_id != profile["id"]:
        raise ValueError("Agent 领域配置与当前知识空间不匹配")
    if workspace_id == "edge_ai_device":
        profile["scope_options"] = {
            "device_model": list(workspace.get("hardware_models", [])),
            "module_sku": list(workspace.get("module_skus", [])),
            "carrier_board": list(workspace.get("carrier_boards", [])),
            "software_baseline": list(workspace.get("software_baselines", [])),
        }
    return profile


class PublicKnowledgeGateway(Protocol):
    def workspace(self) -> dict: ...
    def document(self, document_id: str) -> list[dict]: ...
    def config_trace(self, document_ids: list[str], *, version: str) -> dict: ...
    def search(
        self, question: str, *, version: str, language: str, top_k: int = 5,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
        retrieval_policy: str | None = None,
    ) -> dict: ...
    def search_batch(
        self, summary: str, *, checks: list[dict], version: str, language: str,
        top_k: int = 5, retrieval_policy: str | None = None,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
    ) -> dict: ...
    def review_advice(self, change_summary: str, evidence_chunk_ids: list[str]) -> dict: ...
    def review_advice_for_version(
        self, change_summary: str, evidence_chunk_ids: list[str], *, version: str,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
    ) -> dict: ...
    def engineering_diff(self, old_items: list[dict], new_items: list[dict]) -> dict: ...
    def engineering_impacts(self, payload: dict) -> dict: ...


def _normalized_hash(content: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", content)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _request_queries(summary: str) -> list[str]:
    """Search the whole change and up to three distinct clauses of a compound request."""
    return request_queries(summary)


def _scope_versions(workspace: dict, selected_version: str) -> set[str]:
    scopes = workspace.get("version_scopes")
    definition = scopes.get(selected_version) if isinstance(scopes, dict) else None
    members = definition.get("versions") if isinstance(definition, dict) else None
    if isinstance(members, list) and members and all(isinstance(item, str) for item in members):
        return set(members)
    return {selected_version}


def _workspace_repositories(workspace: dict) -> set[str]:
    values = workspace.get("repositories")
    if isinstance(values, list):
        repositories = {
            value.strip() for value in values
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value.strip())
        }
    else:
        repositories = set()
    repository = workspace.get("repository")
    if isinstance(repository, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        repositories.add(repository)
    return repositories


def _official_hit(
    row: dict, current_version: str, repository: str | set[str], *,
    allowed_versions: set[str] | None = None,
    allowed_sources: list[dict] | None = None,
) -> bool:
    """Accept only current Chinese chunks present in the pinned source registry."""
    if not isinstance(allowed_sources, list):
        return False
    score = row.get("retrieval_score")
    parsed = urlsplit(str(row.get("source_url", "")))
    repositories = {repository} if isinstance(repository, str) else set(repository)
    source_id = row.get("source_id")
    trusted = next((item for item in allowed_sources if item.get("source_id") == source_id), None)
    if not isinstance(trusted, dict):
        return False
    trusted_url = str(trusted.get("source_url") or "")
    trusted_parsed = urlsplit(trusted_url)
    trusted_repository = str(trusted.get("repository") or "")
    trusted_snapshot = str(trusted.get("source_snapshot") or "")
    trusted_commit = str(trusted.get("commit") or "")
    trusted_hash = str(trusted.get("sha256") or "")
    trusted_path = str(trusted.get("path") or trusted.get("document_path") or "")
    url_is_pinned_to_registered_blob = False
    if trusted_parsed.scheme == "https" and trusted_parsed.netloc == "github.com":
        expected_path = f"/{trusted_repository}/blob/{trusted_commit}/{trusted_path}"
        url_is_pinned_to_registered_blob = bool(trusted_path and trusted_parsed.path == expected_path)
    elif trusted_parsed.scheme == "https" and trusted_parsed.netloc == "wiki.seeedstudio.com":
        url_is_pinned_to_registered_blob = trusted_parsed.path.startswith("/cn/")
    row_commit = row.get("commit")
    row_hash = row.get("source_sha256") or row.get("sha256")
    return bool(
        row.get("version") in (allowed_versions or {current_version})
        and row.get("source_snapshot") == trusted_snapshot
        and trusted_snapshot == row.get("version")
        and row.get("source_url") == trusted_url
        and parsed.scheme == "https"
        and trusted_parsed.scheme == "https"
        and url_is_pinned_to_registered_blob
        and trusted_repository in repositories
        and row.get("repository") == trusted_repository
        and re.fullmatch(r"[0-9a-f]{40}", trusted_commit)
        and re.fullmatch(r"[0-9a-f]{64}", trusted_hash)
        and (row_commit is None or row_commit == trusted_commit)
        and (row_hash is None or row_hash == trusted_hash)
        and (not trusted_path or row.get("document_path") == trusted_path)
        and row.get("language") == "zh"
        and bool(row.get("chunk_id"))
        and (score is None or isinstance(score, (int, float)) and score > 0)
    )


_SOURCE_OVERLAP_STOP_WORDS = {
    "guide", "task", "docs", "official", "check", "verify", "and", "the",
    "what", "does", "how", "is", "for", "with", "parent", "child",
}


def _source_path_overlap(trace: dict, row: dict) -> int:
    """Use query-specific heading/path token overlap as a tie-break among RAG hits."""
    query = str(trace.get("search_query") or trace.get("query") or "").casefold()
    source = " ".join((
        str(row.get("document_key") or ""),
        str(row.get("heading") or ""),
    )).casefold()
    normalized_source = re.sub(r"[^a-z0-9]+", "", source)
    terms = {
        term for term in re.findall(r"[a-z][a-z0-9]*", query)
        if len(term) > 2 and term not in _SOURCE_OVERLAP_STOP_WORDS
    }
    return sum(
        re.sub(r"[^a-z0-9]+", "", term) in normalized_source
        for term in terms
    )


def _select_request_evidence(
    searches: list[tuple[dict, list[dict]]], *, max_items: int = 8,
) -> list[dict]:
    """Reserve query- and language-aligned sources, then fill a bounded evidence set."""
    selected: dict[str, dict] = {}
    selected_documents: set[tuple[str, str]] = set()
    for trace, rows in searches:
        if len(selected) >= max_items:
            break
        eligible = [
            row for row in rows
            if (
                str(trace.get("language") or "all"),
                row.get("document_key") or row["chunk_id"],
            ) not in selected_documents
        ]
        if eligible:
            best = max(eligible, key=lambda row: _source_path_overlap(trace, row))
            selected[best["chunk_id"]] = best
            selected_documents.add((str(trace.get("language") or "all"), best.get("document_key") or best["chunk_id"]))
    # Prefer another previously unseen document within each language branch.
    for trace, rows in searches:
        for row in rows:
            if len(selected) >= max_items:
                return list(selected.values())
            language = str(trace.get("language") or "all")
            document_key = (language, row.get("document_key") or row["chunk_id"])
            if (
                row["chunk_id"] not in selected
                and document_key not in selected_documents
                and _source_path_overlap(trace, row) > 0
            ):
                selected[row["chunk_id"]] = row
                selected_documents.add(document_key)
    for _trace, rows in searches:
        for row in rows:
            if len(selected) >= max_items:
                return list(selected.values())
            selected.setdefault(row["chunk_id"], row)
    return list(selected.values())


def _select_ranked_request_evidence(
    searches: list[tuple[dict, list[dict]]], ranked_ids: list[str], *, max_items: int = 8,
) -> list[dict]:
    """Preserve reranker order within each check while rotating evidence coverage."""
    rank = {chunk_id: index for index, chunk_id in enumerate(ranked_ids)}
    ordered = [
        (trace, sorted(rows, key=lambda row: rank.get(row.get("chunk_id"), len(rank))))
        for trace, rows in searches
    ]
    selected: dict[str, dict] = {}
    selected_documents: set[tuple[str, str]] = set()
    max_rounds = max((len(rows) for _trace, rows in ordered), default=0)
    for row_index in range(max_rounds):
        for trace, rows in ordered:
            language = str(trace.get("language") or "all")
            row = next((
                item for item in rows
                if item.get("chunk_id") not in selected
                and (language, item.get("document_key") or item.get("chunk_id")) not in selected_documents
            ), None)
            if row is None:
                continue
            selected[row["chunk_id"]] = row
            selected_documents.add((language, row.get("document_key") or row["chunk_id"]))
            if len(selected) >= max_items:
                return list(selected.values())
    for _trace, rows in ordered:
        for row in rows:
            if len(selected) >= max_items:
                return list(selected.values())
            selected.setdefault(row["chunk_id"], row)
    return list(selected.values())


def _batch_endpoint_unavailable(exc: Exception) -> bool:
    """Allow bounded BM25 fallback only for transport or endpoint-compatibility failures."""
    if type(exc).__name__ in {"Timeout", "TimeoutError"}:
        return False
    if isinstance(exc, ConnectionError):
        return True
    if getattr(exc, "code", None) == "RAG_UNAVAILABLE":
        return True
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    detail = f"{getattr(exc, 'detail', '')} {exc}"
    return status in {404, 422} or any(
        marker in detail for marker in ("HTTPStatus=404", "HTTPStatus=422", "HTTP 404", "HTTP 422")
    )


def _retrieval_policy_unsupported(exc: Exception) -> bool:
    """Identify an older RAG API rejecting the experimental policy; avoid retrying timeouts."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    detail = f"{getattr(exc, 'detail', '')} {exc}"
    return status in {404, 422} or any(
        marker in detail for marker in ("HTTPStatus=404", "HTTPStatus=422", "HTTP 404", "HTTP 422")
    )


def _retrieval_gaps(searches: list[tuple[dict, list[dict]]]) -> list[str]:
    return [row["message"] for row in _retrieval_gap_details(searches)]


def _review_coverage(
    plan: dict, searches: list[tuple[dict, list[dict]]], candidates: list[dict],
    languages: list[str], *, evidence_budget: int = 8,
) -> dict:
    checks = list(range(len(plan.get("queries", []))))
    by_check_language = {
        (trace.get("check_index"), trace.get("language")): rows
        for trace, rows in searches
    }
    covered_checks = sum(
        all(by_check_language.get((check_index, language)) for language in languages)
        for check_index in checks
    )
    covered_languages = sorted({
        trace.get("language") for trace, rows in searches
        if rows and trace.get("language") in {"zh", "en"}
    })
    search_failure_count = sum(
        trace.get("status") == "search_unavailable" for trace, _rows in searches
    )
    complete = (
        bool(checks) and covered_checks == len(checks)
        and set(covered_languages) == set(languages)
        and search_failure_count == 0
        and 0 < len(candidates) <= evidence_budget
    )
    reasons = []
    if search_failure_count:
        reasons.append(f"{search_failure_count} 次语言检索未完成，不能据此判定无影响")
    missing_languages = sorted(set(languages) - set(covered_languages))
    if missing_languages:
        labels = {"zh": "中文", "en": "英文"}
        reasons.append("未召回" + "、".join(labels.get(item, item) for item in missing_languages) + "证据")
    if covered_checks < len(checks):
        reasons.append(f"{len(checks) - covered_checks} 项变更检查未同时覆盖所有指定语种")
    if not candidates:
        reasons.append("没有可供审查的有效证据")
    return {
        "required_check_count": len(checks),
        "covered_check_count": covered_checks,
        "attempted_languages": list(languages),
        "covered_languages": covered_languages,
        "search_failure_count": search_failure_count,
        "evidence_budget": evidence_budget,
        "selected_evidence_count": len(candidates),
        "complete": complete,
        "incomplete_reason": "；".join(dict.fromkeys(reasons)),
    }


def _retrieval_gap_details(
    searches: list[tuple[dict, list[dict]]], plan: dict | None = None,
) -> list[dict]:
    labels = {
        "no_retrieval_match": ("NO_REQUIRED_SOURCE", "NO_MATCH", "当前版本未检索到匹配资料"),
        "candidate_outside_evidence_budget": (
            "NO_REQUIRED_SOURCE", "EVIDENCE_BUDGET", "检索命中未纳入本次模型证据上限",
        ),
        "search_unavailable": ("RETRIEVAL_FAILED", "SEARCH_UNAVAILABLE", "检索服务未完成"),
    }
    material = (plan or {}).get("expected_materials") or "当前设备与软件范围内的公开工程资料"
    action = (plan or {}).get("gap_action") or "请按已选设备型号和软件基线补充检索，并由工程师核实。"
    details = []
    seen = set()
    for trace, _rows in searches:
        definition = labels.get(trace["status"])
        if definition is None:
            continue
        key = (trace.get("check_index"), trace.get("language"), trace.get("query"))
        if key in seen:
            continue
        seen.add(key)
        gap_type, legacy_gap_code, label = definition
        suggested_query = trace["query"]
        language_label = "中文" if trace.get("language") == "zh" else ""
        suffix = f"（{language_label}检索）" if language_label else ""
        details.append({
            "gap_type": gap_type,
            "legacy_gap_code": legacy_gap_code,
            "message": f"{label}{suffix}：{trace['query']}",
            "description": f"{label}{suffix}：{trace['query']}",
            "query": trace["query"],
            "suggested_query": suggested_query,
            "missing_source_type": material,
            "expected_version": (plan or {}).get("target_version"),
            "expected_materials": material,
            "suggested_action": action,
            "requires_human_review": True,
        })
    return details


def _model_evidence_gaps(advice: dict) -> list[str]:
    review = advice.get("review")
    if not isinstance(review, dict):
        return []
    gaps = review.get("evidence_gaps")
    return [gap.strip() for gap in gaps if isinstance(gap, str) and gap.strip()] if isinstance(gaps, list) else []


def _model_evidence_gap_details(advice: dict, *, expected_version: str | None = None) -> list[dict]:
    review = advice.get("review")
    if not isinstance(review, dict):
        return []
    details = []
    for gap in _model_evidence_gaps(advice):
        details.append({
            "gap_type": "NO_REQUIRED_SOURCE",
            "message": gap,
            "description": gap,
            "query": None,
            "suggested_query": None,
            "missing_source_type": None,
            "expected_version": expected_version,
            "expected_materials": None,
            "suggested_action": "模型提示尚未核验；请审核人根据原文确认是否确实缺少该资料。",
            "requires_human_review": True,
        })
    for ambiguity in review.get("version_ambiguities", []) or []:
        if not isinstance(ambiguity, str) or not ambiguity.strip():
            continue
        details.append({
            "gap_type": "VERSION_AMBIGUITY",
            "message": ambiguity.strip(),
            "description": ambiguity.strip(),
            "query": None,
            "suggested_query": None,
            "missing_source_type": "版本依据或版本间差异",
            "expected_version": expected_version,
            "expected_materials": "适用版本的官方资料及必要的历史版本",
            "suggested_action": "请人工逐版本对照原文，确认当前适用版本及差异原因。",
            "requires_human_review": True,
        })
    return details


def _invalid_citation_gap_details(count: int, *, expected_version: str | None = None) -> list[dict]:
    if not count:
        return []
    description = f"模型建议中有 {count} 项引用未出现在本次检索证据中，已移除这些候选。"
    return [{
        "gap_type": "INVALID_CITATION",
        "message": description,
        "description": description,
        "query": None,
        "suggested_query": None,
        "missing_source_type": "本次检索证据中的有效引用 ID",
        "expected_version": expected_version,
        "expected_materials": None,
        "suggested_action": "只审核仍绑定本次检索证据的候选；如需补充依据，请重新检索并核对原文。",
        "requires_human_review": True,
    }]


def _item(source: dict, content: str) -> dict:
    repository = str(source.get("repository") or "public-materials")
    owner, _, repository_name = repository.partition("/")
    return {
        "item_id": source["chunk_id"],
        "item_type": "API" if "/api/" in source["document_key"] else "DESIGN",
        "organization_id": owner or "public-materials",
        "project_id": repository_name or repository,
        "document_id": source["document_id"],
        "version_id": source["version"],
        "section_id": source["heading"] or source["chunk_id"],
        "external_identifier": source["chunk_id"],
        "title": source["heading"] or source["document_key"],
        "content": content,
        "content_hash": _normalized_hash(content),
        "metadata": {"source_url": source["source_url"]},
    }


def _normalize_review_advice(
    advice: dict, allowed_sources: dict[str, dict], *, expected_version: str | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """Keep only structured model suggestions tied to this request's RAG evidence."""
    review = advice.get("review")
    if (
        advice.get("status") == "ABSTAINED"
        and isinstance(review, dict)
        and review.get("review_status") == "REQUIRES_HUMAN_REVIEW"
        and review.get("impact_candidates") == []
    ):
        return {**advice, "answer": "N/A", "sources": [], "review": review}, [], []
    if advice.get("status") != "OK":
        return {**advice, "sources": [], "review": None}, [], []
    candidates = review.get("impact_candidates") if isinstance(review, dict) else None
    if (
        not isinstance(candidates, list)
        or review.get("review_status") != "REQUIRES_HUMAN_REVIEW"
        or not candidates
    ):
        return {
            **advice, "status": "ABSTAINED", "answer": "N/A",
            "sources": [], "review": None,
        }, [], []

    impacts = []
    sources = []
    valid_candidates = []
    seen = set()
    invalid_citation_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            invalid_citation_count += 1
            continue
        chunk_id = candidate.get("evidence_chunk_id")
        reason = candidate.get("reason")
        action = candidate.get("suggested_action")
        if (
            not isinstance(chunk_id, str) or chunk_id not in allowed_sources
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(action, str) or not action.strip()
        ):
            invalid_citation_count += 1
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        source = allowed_sources[chunk_id]
        sources.append(source)
        valid_candidates.append(candidate)
        impacts.append({
            "status": "SUGGESTED", "relation": "suggested", "reason": reason,
            "suggested_action": action, "evidence": source,
        })
    validation_gaps = _invalid_citation_gap_details(
        invalid_citation_count, expected_version=expected_version,
    )
    if not valid_candidates:
        safe_review = dict(review) if isinstance(review, dict) else {}
        safe_review["impact_candidates"] = []
        safe_review.pop("change_interpretation", None)
        return {
            **advice, "status": "ABSTAINED", "answer": "N/A", "sources": [], "review": safe_review,
        }, [], validation_gaps
    safe_review = dict(review)
    safe_review["impact_candidates"] = valid_candidates
    if invalid_citation_count:
        # Do not present a free-form summary that may have relied on a rejected citation.
        safe_review.pop("change_interpretation", None)
        advice = {**advice, "answer": "N/A"}
    return {**advice, "sources": sources, "review": safe_review}, impacts, validation_gaps


def _generation_stage_status(advice_status: str) -> str:
    if advice_status == "OK":
        return "OK"
    if advice_status in {"ABSTAINED", "NO_EVIDENCE"}:
        return "EMPTY"
    if advice_status == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    if advice_status in {"NOT_CALLED", "NOT_CALLED_OUT_OF_SCOPE"}:
        return "SKIPPED"
    return "FAILED"


def _trace_request_configuration(gateway, workspace: dict, candidates: list[dict], version: str, summary: str) -> dict:
    """One bounded, read-only tool call; fail closed on provenance mismatches."""
    empty = {"version": version, "relations": [], "gaps": [], "roots": [],
             "impact_status": "REQUIRES_HUMAN_REVIEW", "http_request_count": 0}
    if workspace.get("workspace_id") != "pphuman":
        return {**empty, "status": "NOT_APPLICABLE"}
    # A named file is the change target, not every semantically related RAG hit.
    # Resolve only registered same-version paths; never open the user-supplied path.
    named_paths = set(re.findall(r"(?<![A-Za-z0-9_./-])(?:configs|deploy|docs)/[A-Za-z0-9_./-]+\.(?:ya?ml|md)(?![A-Za-z0-9_.-])", summary))
    explicit_roots = [f"{version}:zh:{row['path'].rsplit('.', 1)[0]}"
        for row in workspace.get("source_registry", [])
        if row.get("version") == version and row.get("path") in named_paths]
    roots = list(dict.fromkeys(explicit_roots or [row["document_id"] for row in candidates if row.get("document_id")]))[:8]
    method = getattr(gateway, "config_trace", None)
    if not roots:
        return {**empty, "status": "NO_RETRIEVAL_EVIDENCE"}
    if not callable(method):
        return {**empty, "status": "NOT_AVAILABLE"}
    try:
        report = method(roots, version=version)
    except Exception:
        return {**empty, "status": "UNAVAILABLE", "http_request_count": 1}
    registry = {row.get("source_id"): row for row in workspace.get("source_registry", [])}

    def bound_source(row):
        if not isinstance(row, dict):
            return False
        source = registry.get(row.get("source_id"))
        path = source.get("path") if isinstance(source, dict) else None
        return bool(source and row.get("version") == version
            and source.get("version") == version and row.get("path") == path
            and row.get("source_url") == source.get("source_url")
            and row.get("document_id") == f"{version}:zh:{path.rsplit('.', 1)[0]}")

    try:
        if (not isinstance(report, dict) or report.get("version") != version
                or report.get("status") not in {"ok", "partial", "no_relations"}
                or not isinstance(report.get("roots"), list)
                or {row.get("document_id") for row in report["roots"] if isinstance(row, dict)} != set(roots)
                or not all(bound_source(row) for row in report["roots"])):
            raise ValueError("untrusted configuration roots")
        relations, gaps = report.get("relations"), report.get("gaps")
        if not isinstance(relations, list) or not isinstance(gaps, list) or len(relations) > 80 or len(gaps) > 80:
            raise ValueError("unbounded configuration response")
        for row in [*relations, *gaps]:
            evidence = row.get("evidence") if isinstance(row, dict) else None
            if (not bound_source(row) or not isinstance(evidence, dict)
                    or not isinstance(evidence.get("text"), str) or len(evidence["text"]) > 600
                    or not isinstance(evidence.get("line_start"), int) or isinstance(evidence["line_start"], bool) or evidence["line_start"] < 1
                    or not isinstance(evidence.get("line_end"), int) or isinstance(evidence["line_end"], bool) or evidence["line_end"] < evidence["line_start"]
                    or row.get("document_id") not in roots and row.get("target_document_id") not in roots):
                raise ValueError("untrusted configuration evidence")
            if row.get("target_status") == "indexed":
                target = registry.get(row.get("target_source_id"))
                if (not isinstance(target, dict) or target.get("version") != version
                        or row.get("target_path") != target.get("path")
                        or row.get("target_document_id") != f"{version}:zh:{target['path'].rsplit('.', 1)[0]}"):
                    raise ValueError("cross-version configuration target")
        for row in relations:
            if (row.get("relation_type") not in {"config_inherits", "config_references", "document_references"}
                    or row.get("target_status") not in {"indexed", "not_indexed", "different_version", "external_repository", "invalid_path"}
                    or row.get("direction") != ("outgoing" if row.get("document_id") in roots else "incoming")):
                raise ValueError("invalid configuration relation")
    except (ValueError, TypeError, KeyError):
        return {**empty, "status": "INVALID_RESPONSE", "http_request_count": 1}
    return {**report, "impact_status": "REQUIRES_HUMAN_REVIEW", "http_request_count": 1}


def _configuration_gap_details(report: dict, version: str) -> list[dict]:
    status = report.get("status")
    if status in {"NOT_AVAILABLE", "UNAVAILABLE", "INVALID_RESPONSE"}:
        return [{"gap_type": "CONFIG_TRACE_UNAVAILABLE", "expected_version": version,
            "message": "显式配置关联工具未可用或未通过来源校验；本次无法确认资料引用链。",
            "suggested_action": "请人工核对当前版本的配置引用关系。", "requires_human_review": True}]
    details = []
    for row in report.get("gaps", []):
        target = row.get("target_path")
        not_indexed = row.get("kind") == "not_indexed"
        details.append({
            "gap_type": "CONFIG_REFERENCE_NOT_INDEXED" if not_indexed else "CONFIG_REFERENCE_UNRESOLVED",
            "expected_version": version, "source_id": row.get("source_id"),
            "source_url": row.get("source_url"), "target_path": target,
            "message": f"资料显式引用未收录目标：{target}" if not_indexed else f"资料引用无法在当前版本解析：{target or row.get('kind')}",
            "suggested_action": "核对相同版本的引用文件；缺少资料不能解释为无影响。",
            "requires_human_review": True,
        })
    if (report.get("bounds") or {}).get("truncated"):
        details.append({"gap_type": "CONFIG_TRACE_BUDGET", "expected_version": version,
            "message": "显式关联超过查询预算，结果已截断。",
            "suggested_action": "缩小到具体配置文件并人工核对剩余引用。", "requires_human_review": True})
    return details


class PublicReviewAgent:
    def __init__(self, gateway: PublicKnowledgeGateway):
        self.gateway = gateway

    def analyze_request(
        self,
        change_summary: str,
        *,
        change_type: str | None = None,
        impact_scope: str | None = None,
        target_version: str | None = None,
        objective: str | None = None,
        constraints: str | None = None,
        validation_plan: str | None = None,
        language_mode: str = "zh",
        device_model: str | None = None,
        module_sku: str | None = None,
        carrier_board: str | None = None,
        software_baseline: str | None = None,
        retrieval_policy: str = "bm25",
    ) -> dict:
        """Find current-version candidates from a natural-language change request."""
        summary = change_summary.strip()
        if not summary or len(summary) > 4000:
            raise ValueError("变更描述应为 1 到 4000 字")
        if language_mode != "zh":
            raise ValueError("当前知识空间仅收录中文资料，请使用中文审查")
        if retrieval_policy not in {
            "bm25", "bm25_pphuman_term_expansion_rrf", "task_adaptive_rerank",
        }:
            raise ValueError("不支持的检索策略")
        languages = ["zh"]

        plan = build_request_plan(summary, change_type=change_type, impact_scope=impact_scope)
        plan["language_mode"] = language_mode

        if is_out_of_scope_public_request(summary):
            task_id = uuid.uuid4().hex
            fingerprint = _normalized_hash(f"out_of_scope_public:{summary}")[:20]
            explanation = (
                "本工作台当前只检索已配置知识空间的官方公开资料，未接入公司内部 API、"
                "Jira、通讯录或内部制度。为避免把企业私有问题误交给公开语料或模型，已在检索前停止。"
            )
            gap = {
                "gap_type": "OUT_OF_SCOPE_PUBLIC_CORPUS",
                "message": explanation,
                "description": explanation,
                "query": summary,
                "suggested_query": None,
                "missing_source_type": "经授权并脱敏的企业内部资料",
                "expected_version": None,
                "expected_materials": "经授权并脱敏的企业内部资料",
                "suggested_action": "请改问公开文档可回答的问题；企业场景需先接入经授权的知识源并配置访问控制。",
                "requires_human_review": True,
            }
            return {
                "task_id": task_id,
                "request_fingerprint": fingerprint,
                "request_mode": "natural_language",
                "request_summary": summary,
                "request_plan": plan,
                "stage_status": {
                    "planning": "OK", "retrieval": "OUT_OF_SCOPE", "generation": "OUT_OF_SCOPE",
                },
                "scope_status": "OUT_OF_SCOPE",
                "retrieval_policy": "not_run_scope_guard",
                "retrieval_policy_requested": retrieval_policy,
                "retrieval_policy_status": "NOT_RUN",
                "retrieval_trace": {
                    "queries": [],
                    "uncovered_queries": [summary],
                    "model_status": "NOT_CALLED_OUT_OF_SCOPE",
                    "query_limit": plan["query_limit"],
                    "languages_per_check": list(languages),
                    "search_call_limit": plan["query_limit"] * len(languages),
                    "scope_status": "OUT_OF_SCOPE",
                },
                "coverage": {
                    "required_check_count": len(plan["queries"]),
                    "covered_check_count": 0,
                    "attempted_languages": list(languages),
                    "covered_languages": [],
                    "search_failure_count": 0,
                    "evidence_budget": 8,
                    "selected_evidence_count": 0,
                    "complete": False,
                    "incomplete_reason": explanation,
                },
                "retrieved_results": [],
                "impacts": [],
                "review_advice": {
                    "status": "OUT_OF_SCOPE",
                    "answer": "未执行检索或模型生成。",
                    "sources": [],
                    "message": explanation,
                },
                "evidence_gaps": [explanation],
                "evidence_gap_details": [gap],
                "sandbox_only": True,
                "public_baseline_written": False,
            }

        workspace = self.gateway.workspace()
        if retrieval_policy != "bm25" and workspace.get("workspace_id") != "pphuman":
            raise ValueError("当前知识空间不支持该实验检索策略")
        current_version = workspace["current_version"]
        repositories = _workspace_repositories(workspace)
        active_profile = _profile_for_workspace(workspace)
        source_registry = workspace.get("source_registry")
        if not isinstance(source_registry, list) or not source_registry:
            raise ValueError("当前知识空间未提供可校验的来源清单")
        scope_values = {
            "device_model": device_model,
            "module_sku": module_sku,
            "carrier_board": carrier_board,
            "software_baseline": software_baseline,
        }

        available_versions = [
            str(value) for value in workspace.get("available_versions", [current_version])
        ]
        selected_version = current_version if not target_version or target_version == "current" else target_version
        if selected_version not in available_versions:
            raise ValueError("目标版本不在当前知识空间的已收录版本中")
        plan = build_request_plan(
            summary, change_type=change_type, impact_scope=impact_scope,
            profile=active_profile, device_model=device_model, module_sku=module_sku,
            carrier_board=carrier_board, software_baseline=software_baseline,
            target_snapshot=selected_version,
        )
        plan["language_mode"] = "zh"
        allowed_versions = _scope_versions(workspace, selected_version)
        context_values = {}
        for name, raw_value in (
            ("objective", objective),
            ("constraints", constraints),
            ("validation_plan", validation_plan),
        ):
            value = (raw_value or "").strip()
            if len(value) > 800:
                raise ValueError("提案补充信息每项不超过 800 字")
            context_values[name] = value or None
        missing_fields = [name for name, value in context_values.items() if value is None]
        request_context = {
            **context_values,
            "target_version": selected_version,
            "context_status": "needs_confirmation" if missing_fields else "provided",
            "missing_fields": missing_fields,
        }
        plan["target_version"] = selected_version
        plan["proposal_context_status"] = request_context["context_status"]
        task_id = uuid.uuid4().hex
        requested_retrieval_policy = retrieval_policy
        fingerprint = _normalized_hash(
            json.dumps({
                "version": selected_version, "change_type": plan["change_type"],
                "impact_scope": plan["impact_scope"], "summary": summary,
                "language_mode": "zh",
                "device_scope": plan.get("device_scope"),
                "retrieval_policy": requested_retrieval_policy,
                **context_values,
            }, ensure_ascii=False, sort_keys=True)
        )[:20]
        searches: list[tuple[dict, list[dict]]] = []
        observed_retrieval_policies: set[str] = set()
        retrieval_started = time.perf_counter()
        batch_response = None
        batch_attempted = False
        batch_fallback_reason = None
        batch_failure = None
        batch_rerank_status = "NOT_REQUESTED"
        batch_ranked_ids: list[str] = []
        batch_subquery_count: int | None = None
        batch_http_request_count = 0
        batch_rerank_calls = 0
        if requested_retrieval_policy == "task_adaptive_rerank":
            batch_method = getattr(self.gateway, "search_batch", None)
            if callable(batch_method):
                batch_attempted = True
                batch_http_request_count = 1
                try:
                    batch_response = batch_method(
                        summary,
                        checks=[{
                            "check_index": index,
                            "query": step.get("search_query", step["query"]),
                        } for index, step in enumerate(plan["queries"])],
                        version=selected_version, language="zh", top_k=5,
                        retrieval_policy=requested_retrieval_policy,
                        **scope_values,
                    )
                    if not isinstance(batch_response, dict) or not isinstance(batch_response.get("checks"), list):
                        raise ValueError("批量检索响应缺少 checks 列表")
                    batch_by_index = {
                        row.get("check_index"): row
                        for row in batch_response["checks"] if isinstance(row, dict)
                    }
                    if any(index not in batch_by_index for index in range(len(plan["queries"]))):
                        raise ValueError("批量检索响应未覆盖全部检查项")
                    diagnostics = batch_response.get("rerank_diagnostics")
                    if isinstance(diagnostics, dict):
                        batch_rerank_calls = int(diagnostics.get("rerank_calls") or 0)
                    batch_rerank_status = str(batch_response.get("rerank_status") or "UNKNOWN")
                    batch_ranked_ids = [
                        value for value in batch_response.get("ranked_ids", [])
                        if isinstance(value, str)
                    ]
                    batch_subquery_count = int(
                        batch_response.get("retrieval_call_count") or len(plan["queries"])
                    )
                    observed_policy = batch_response.get("actual_policy") or batch_response.get("retrieval_policy")
                    if isinstance(observed_policy, str) and observed_policy:
                        observed_retrieval_policies.add(observed_policy)
                except Exception as exc:
                    batch_response = None
                    if _batch_endpoint_unavailable(exc):
                        batch_fallback_reason = type(exc).__name__
                        batch_rerank_status = "NOT_AVAILABLE_FALLBACK"
                    else:
                        batch_failure = exc
                        batch_rerank_status = "FAILED"
            else:
                batch_fallback_reason = "gateway_missing_batch_method"
                batch_rerank_status = "NOT_AVAILABLE_FALLBACK"

        if batch_response is not None:
            batch_by_index = {row["check_index"]: row for row in batch_response["checks"]}
            for check_index, query_step in enumerate(plan["queries"]):
                query = query_step["query"]
                search_query = query_step.get("search_query", query)
                check_result = batch_by_index[check_index]
                trace = {
                    **query_step, "check_index": check_index, "language": "zh",
                    "status": "no_retrieval_match",
                    "retrieval_policy": next(iter(observed_retrieval_policies), "not_confirmed"),
                    "top_chunk_ids": [], "selected_chunk_ids": [],
                }
                raw_rows = check_result.get("results")
                if not isinstance(raw_rows, list):
                    raw_rows = []
                    trace["status"] = "search_unavailable"
                rows = [
                    row for row in raw_rows if isinstance(row, dict)
                    and _official_hit(
                        row, selected_version, repositories, allowed_versions=allowed_versions,
                        allowed_sources=source_registry,
                    )
                    and str(row.get("language") or row.get("locale") or "").casefold().startswith("zh")
                ]
                rows.sort(key=lambda row: batch_ranked_ids.index(row["chunk_id"])
                          if row.get("chunk_id") in batch_ranked_ids else len(batch_ranked_ids))
                trace["top_chunk_ids"] = [row["chunk_id"] for row in rows]
                searches.append((trace, rows))
        else:
            if batch_attempted:
                batch_http_request_count = 1
            for check_index, query_step in enumerate(plan["queries"]):
                query = query_step["query"]
                search_query = query_step.get("search_query", query)
                for language in languages:
                    trace = {
                        **query_step,
                        "check_index": check_index,
                        "language": language,
                        "status": "no_retrieval_match",
                        "retrieval_policy": "not_confirmed",
                        "top_chunk_ids": [],
                        "selected_chunk_ids": [],
                    }
                    try:
                        if batch_failure is not None:
                            raise batch_failure
                        search_options = {
                            "version": selected_version, "language": language, "top_k": 5,
                            **scope_values,
                        }
                        # A missing/old batch endpoint must fall back to honest BM25.
                        if requested_retrieval_policy == "task_adaptive_rerank":
                            search_options["retrieval_policy"] = "bm25"
                        elif requested_retrieval_policy != "bm25":
                            search_options["retrieval_policy"] = requested_retrieval_policy
                        batch_http_request_count += 1
                        search_result = self.gateway.search(search_query, **search_options)
                        observed_policy = search_result.get("retrieval_policy")
                        if isinstance(observed_policy, str) and observed_policy:
                            observed_retrieval_policies.add(observed_policy)
                            trace["retrieval_policy"] = observed_policy
                        rows = [
                            row for row in search_result.get("results", [])
                            if _official_hit(
                                row, selected_version, repositories, allowed_versions=allowed_versions,
                                allowed_sources=source_registry,
                            )
                            and str(row.get("language") or row.get("locale") or "").casefold().startswith(language)
                        ]
                        trace["top_chunk_ids"] = [row["chunk_id"] for row in rows]
                    except Exception:
                        # A failed search is distinct from a successful search without matches.
                        rows = []
                        trace["status"] = "search_unavailable"
                    searches.append((trace, rows))
            if requested_retrieval_policy == "task_adaptive_rerank" and batch_fallback_reason:
                # The compatibility path is deliberately BM25-only and visibly unreranked.
                observed_retrieval_policies.add("bm25")
                batch_subquery_count = len(searches)
        if requested_retrieval_policy != "task_adaptive_rerank":
            batch_rerank_status = "NOT_REQUESTED"

        retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
        retrieval_policy = (
            next(iter(observed_retrieval_policies)) if len(observed_retrieval_policies) == 1 else
            "mixed" if observed_retrieval_policies else "not_confirmed"
        )
        any_search_failure = any(trace["status"] == "search_unavailable" for trace, _rows in searches)
        retrieval_policy_status = (
            "NOT_AVAILABLE_FALLBACK" if batch_fallback_reason else
            "MISMATCH" if observed_retrieval_policies and observed_retrieval_policies != {requested_retrieval_policy} else
            "PARTIAL" if observed_retrieval_policies == {requested_retrieval_policy} and any_search_failure else
            "CONFIRMED" if observed_retrieval_policies == {requested_retrieval_policy} else "UNCONFIRMED"
        )

        candidates = (
            _select_ranked_request_evidence(searches, batch_ranked_ids, max_items=8)
            if batch_response is not None else _select_request_evidence(searches, max_items=8)
        )
        candidate_ids = {row["chunk_id"] for row in candidates}
        for trace, rows in searches:
            trace["selected_chunk_ids"] = [row["chunk_id"] for row in rows if row["chunk_id"] in candidate_ids]
            if trace["status"] != "search_unavailable" and rows:
                trace["status"] = "candidate_found" if trace["selected_chunk_ids"] else "candidate_outside_evidence_budget"
        retrieval_trace = {
            "queries": [trace for trace, _rows in searches],
            "uncovered_queries": list(dict.fromkeys(
                trace["query"] for trace, _rows in searches
                if trace["status"] != "candidate_found"
            )),
            "model_status": "NOT_CALLED",
            "query_limit": plan["query_limit"],
            "languages_per_check": list(languages),
            "search_call_limit": plan["query_limit"] * len(languages),
            "retrieval_policies": sorted(observed_retrieval_policies),
            "rerank_status": batch_rerank_status,
            "rerank_calls": batch_rerank_calls,
            "http_request_count": batch_http_request_count,
            "subquery_count": (
                batch_subquery_count if batch_subquery_count is not None else
                len(searches) if not batch_attempted else None
            ),
            "batch_fallback_reason": batch_fallback_reason,
            "route": "CHANGE_REVIEW",
            "candidate_count": (
                batch_response.get("candidate_count") if batch_response is not None else
                len({row.get("chunk_id") for _trace, rows in searches for row in rows})
            ),
            "final_evidence_count": len(candidates),
            "stage_latency_ms": (
                batch_response.get("stage_latency_ms", {}) if batch_response is not None else {}
            ),
            "rerank_diagnostics": (
                batch_response.get("rerank_diagnostics", {}) if batch_response is not None else {}
            ),
        }
        coverage = _review_coverage(plan, searches, candidates, languages, evidence_budget=8)
        evidence_gap_details = _retrieval_gap_details(searches, plan)
        evidence_gaps = [row["message"] for row in evidence_gap_details]
        base_advice = {"status": "NO_EVIDENCE", "answer": "N/A", "sources": []}
        if not candidates:
            if all(trace["status"] == "search_unavailable" for trace, _rows in searches):
                base_advice["status"] = "RETRIEVAL_UNAVAILABLE"
            return {
                "task_id": task_id,
                "request_fingerprint": fingerprint,
                "request_mode": "natural_language",
                "request_summary": summary,
                "request_plan": plan,
                "request_context": request_context,
                "stage_status": {
                    "planning": "OK",
                "retrieval": "FAILED" if any(
                    trace["status"] == "search_unavailable" for trace, _rows in searches
                ) else "EMPTY",
                    "generation": "SKIPPED",
                },
                "retrieval_policy": retrieval_policy,
                "retrieval_policy_requested": requested_retrieval_policy,
                "retrieval_policy_status": retrieval_policy_status,
                "retrieval_latency_ms": retrieval_latency_ms,
                "retrieval_trace": retrieval_trace,
                "coverage": coverage,
                "retrieved_results": [],
                "impacts": [],
                "review_advice": base_advice,
                "evidence_gaps": evidence_gaps,
                "evidence_gap_details": evidence_gap_details,
                "sandbox_only": True,
                "public_baseline_written": False,
            }

        configuration_trace = _trace_request_configuration(self.gateway, workspace, candidates, selected_version, summary)
        configuration_gaps = _configuration_gap_details(configuration_trace, selected_version)
        evidence_gap_details.extend(configuration_gaps)
        evidence_gaps.extend(row["message"] for row in configuration_gaps)

        try:
            advice_summary = (
                f"知识空间：{workspace.get('workspace', next(iter(repositories), '公开知识空间'))}\n目标版本：{selected_version}"
                f"\n变更描述：{summary}\n变更类型：{plan['change_type_label']}"
                f"（{plan['classification_source']}）\n影响范围：{plan['impact_scope'] or '待补充'}"
                f"\n变更目标：{context_values['objective'] or '待补充'}"
                f"\n约束条件：{context_values['constraints'] or '待补充'}"
                f"\n验证计划：{context_values['validation_plan'] or '待补充'}"
                f"\n检索关注点：{plan['retrieval_focus']}"
                f"\n必须核对的检查项：{'；'.join(plan['checklist_items'])}"
                f"\n检索覆盖状态：{'完整' if coverage['complete'] else '不完整'}"
                f"\n审查边界：{coverage['incomplete_reason'] or '仅把引用资料列为待核对候选，最终由人工确认'}"
            )
            scope = plan["device_scope"]
            if scope:
                scope_text = "；".join(f"{key}={value or '未指定'}" for key, value in scope.items())
                advice_summary += f"\n适用范围：{scope_text}"
            advice_summary += (
                "\n安全约束：只列有本次证据支持的待核对候选；不得将缺少证据解释为无影响或已通过验证；"
                "资料未明确覆盖的内容必须作为证据缺口交由工程师确认。"
            )
            evidence_ids = [row["chunk_id"] for row in candidates]
            versioned_review = getattr(self.gateway, "review_advice_for_version", None)
            if callable(versioned_review):
                advice = versioned_review(
                    advice_summary, evidence_ids, version=selected_version,
                    **scope_values,
                )
            else:
                advice_method = self.gateway.review_advice
                advice = advice_method(
                    advice_summary, evidence_ids, version=selected_version, **scope_values,
                )
        except Exception:
            # Model assistance is optional; the underlying RAG candidates remain visible.
            advice = {"status": "GENERATION_PROVIDER_UNAVAILABLE", "answer": "N/A", "sources": []}
        allowed = {row["chunk_id"]: row for row in candidates}
        advice, grounded_impacts, validation_gap_details = _normalize_review_advice(
            advice, allowed, expected_version=selected_version,
        )
        retrieval_trace["model_status"] = advice.get("status", "UNKNOWN")
        model_gap_details = _model_evidence_gap_details(advice, expected_version=selected_version)
        evidence_gap_details.extend(model_gap_details)
        evidence_gap_details.extend(validation_gap_details)
        evidence_gaps.extend(row["message"] for row in [*model_gap_details, *validation_gap_details])

        return {
            "task_id": task_id,
            "request_fingerprint": fingerprint,
            "request_mode": "natural_language",
            "request_summary": summary,
            "request_plan": plan,
            "request_context": request_context,
            "stage_status": {
                "planning": "OK",
                "retrieval": "FAILED" if any(
                    trace["status"] == "search_unavailable" for trace, _rows in searches
                ) else "OK",
                "generation": _generation_stage_status(str(advice.get("status", "UNKNOWN"))),
            },
            "retrieval_policy": retrieval_policy,
            "retrieval_policy_requested": requested_retrieval_policy,
            "retrieval_policy_status": retrieval_policy_status,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieval_trace": retrieval_trace,
            "coverage": coverage,
            "configuration_trace": configuration_trace,
            "retrieved_results": candidates,
            "impacts": grounded_impacts,
            "review_advice": advice,
            "evidence_gaps": evidence_gaps,
            "evidence_gap_details": evidence_gap_details,
            "sandbox_only": True,
            "public_baseline_written": False,
        }

    def analyze(
        self, selected: dict, proposed_text: str, *,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
        retrieval_policy: str = "bm25",
    ) -> dict:
        workspace = self.gateway.workspace()
        if retrieval_policy not in {
            "bm25", "bm25_pphuman_term_expansion_rrf", "task_adaptive_rerank",
        }:
            raise ValueError("不支持的检索策略")
        if retrieval_policy != "bm25" and workspace.get("workspace_id") != "pphuman":
            raise ValueError("当前知识空间不支持该实验检索策略")
        current_version = workspace["current_version"]
        repositories = _workspace_repositories(workspace)
        allowed_versions = _scope_versions(workspace, current_version)
        active_profile = _profile_for_workspace(workspace)
        source_registry = workspace.get("source_registry")
        if not isinstance(source_registry, list) or not source_registry:
            raise ValueError("当前知识空间未提供可校验的来源清单")
        source_is_trusted = _official_hit(
            selected, current_version, repositories, allowed_versions=allowed_versions,
            allowed_sources=source_registry,
        )
        if selected.get("version") not in allowed_versions:
            raise ValueError("只能选择当前知识空间最新已收录范围内的资料")
        if not source_is_trusted:
            raise ValueError("资料来源与当前知识空间的官方来源清单不匹配")
        proposed = proposed_text.strip()
        if not proposed or len(proposed) > 4000:
            raise ValueError("假设性变更内容应为 1 到 4000 字")
        old_item, new_item = _item(selected, selected["content"]), _item(selected, proposed)
        if old_item["content_hash"] == new_item["content_hash"]:
            raise ValueError("修改后内容与原文相同")
        changes = self.gateway.engineering_diff([old_item], [new_item])["changes"]
        change = changes[0]
        if change["change_type"] == "UNCHANGED":
            raise ValueError("修改后内容与原文相同")
        task_id = uuid.uuid4().hex
        fingerprint = _normalized_hash(
            f"exact:{current_version}:{selected['chunk_id']}:{old_item['content_hash']}:{new_item['content_hash']}:{retrieval_policy}"
        )[:20]
        related_query = (selected["heading"] + " " + proposed)[:1000]
        retrieval_failed = False
        all_scope_values = {
            "device_model": workspace.get("hardware_models", []),
            "module_sku": workspace.get("module_skus", []),
            "carrier_board": workspace.get("carrier_boards", []),
            "software_baseline": workspace.get("software_baselines", []),
        }
        supplied_scope_values = {
            "device_model": device_model,
            "module_sku": module_sku,
            "carrier_board": carrier_board,
            "software_baseline": software_baseline,
        }
        declared_scope_fields = active_profile.get(
            "scope_fields", ["device_model", "module_sku", "carrier_board", "software_baseline"],
        )
        if any(value is not None and field not in declared_scope_fields
               for field, value in supplied_scope_values.items()):
            raise ValueError("当前领域配置不支持所填的设备范围字段")
        scope_values = {
            field: supplied_scope_values[field]
            for field in declared_scope_fields
        }
        scope_options = active_profile.get("scope_options", {})
        for field, value in scope_values.items():
            if value is not None and value not in scope_options[field]:
                raise ValueError(f"{field} 不在当前知识空间的可选范围中")
        retrieval_started = time.perf_counter()
        actual_retrieval_policy = None
        selected_rerank_status = "NOT_REQUESTED"
        selected_http_request_count = 0
        try:
            search_options = {
                "version": current_version, "language": "zh", "top_k": 12,
                **scope_values,
            }
            if retrieval_policy != "bm25":
                search_options["retrieval_policy"] = retrieval_policy
            selected_http_request_count = 1
            try:
                search_result = self.gateway.search(related_query, **search_options)
            except Exception as exc:
                if retrieval_policy != "task_adaptive_rerank" or not _retrieval_policy_unsupported(exc):
                    raise
                # Only retry an explicit unsupported-policy response, never timeouts/provider failures.
                search_options["retrieval_policy"] = "bm25"
                search_result = self.gateway.search(related_query, **search_options)
                selected_http_request_count += 1
                selected_rerank_status = "NOT_AVAILABLE_FALLBACK"
            retrieved = search_result["results"]
            actual_retrieval_policy = search_result.get("retrieval_policy")
            if isinstance(search_result.get("rerank_status"), str):
                selected_rerank_status = search_result["rerank_status"]
            if retrieval_policy == "task_adaptive_rerank" and selected_rerank_status == "NOT_REQUESTED":
                selected_rerank_status = "OK" if actual_retrieval_policy == "task_adaptive_rerank" else "UNCONFIRMED"
        except Exception:
            retrieved = []
            retrieval_failed = True
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
        related = [
            row for row in retrieved
            if row["chunk_id"] != selected["chunk_id"]
            and _official_hit(
                row, current_version, repositories, allowed_versions=allowed_versions,
                allowed_sources=source_registry,
            )
        ][:5]
        impacts = self.gateway.engineering_impacts({
            "changed_item_id": selected["chunk_id"],
            "items": [old_item] + [_item(row, row["content"]) for row in related],
            "trace_links": [],
            # Existing API field name is historical; candidates come from the
            # measured public BM25 retrieval policy, not a Dense claim.
            "dense_item_ids": [row["chunk_id"] for row in related],
            "evidence_by_item": {row["chunk_id"]: [row["chunk_id"]] for row in related},
        })["impacts"]
        by_id = {row["chunk_id"]: row for row in related}
        review_advice = {"status": "NO_EVIDENCE", "answer": "N/A", "sources": []}
        if related:
            change_summary = (
                f"资料章节：{selected['heading']}\n变更类型：{change['change_type']}\n"
                f"原文：{selected['content'][:1400]}\n"
                f"拟议内容：{proposed[:1400]}"
            )
            try:
                review_advice = self.gateway.review_advice(
                    change_summary, [row["chunk_id"] for row in related],
                    version=current_version, **scope_values,
                )
            except Exception:
                # Optional model advice must never block the deterministic review flow.
                review_advice = {"status": "GENERATION_PROVIDER_UNAVAILABLE", "answer": "N/A", "sources": []}
            review_advice, _grounded_impacts, invalid_citation_gaps = _normalize_review_advice(
                review_advice, by_id, expected_version=current_version,
            )
        else:
            invalid_citation_gaps = []
        model_candidates = {
            row["evidence"]["chunk_id"]: row
            for row in _grounded_impacts
        } if related else {}
        retrieval_trace = {
            "queries": [{
                "query": related_query,
                "status": (
                    "search_unavailable" if retrieval_failed else
                    "candidate_found" if related else "no_retrieval_match"
                ),
                "top_chunk_ids": [row["chunk_id"] for row in related],
                "selected_chunk_ids": [row["chunk_id"] for row in related],
                "retrieval_policy": actual_retrieval_policy or "not_confirmed",
                "rerank_status": selected_rerank_status,
            }],
            "uncovered_queries": [] if related else [related_query],
            "model_status": review_advice.get("status", "NOT_CALLED") if related else "NOT_CALLED",
            "http_request_count": selected_http_request_count,
            "subquery_count": 1 if selected_http_request_count else 0,
        }
        model_gap_details = _model_evidence_gap_details(
            review_advice, expected_version=current_version,
        )
        exact_retrieval_gap_details = [] if related else [{
            "gap_type": "RETRIEVAL_FAILED" if retrieval_failed else "NO_REQUIRED_SOURCE",
            "legacy_gap_code": "SEARCH_UNAVAILABLE" if retrieval_failed else "NO_RELATED_MATERIAL",
            "message": "关联资料检索服务未完成。" if retrieval_failed else "当前版本未检索到其他需要核对的资料。",
            "description": "关联资料检索服务未完成。" if retrieval_failed else "当前版本未检索到其他需要核对的资料。",
            "query": related_query,
            "suggested_query": related_query,
            "missing_source_type": "当前版本相关官方资料",
            "expected_version": current_version,
            "expected_materials": "当前版本相关官方资料",
            "suggested_action": (
                "请检查知识服务连接后重试；当前仅保留所选原文，不把未完成的检索当作无命中。"
                if retrieval_failed else "请人工确认是否需要扩大检索范围或补充资料。"
            ),
            "requires_human_review": True,
        }]
        all_gap_details = [
            *exact_retrieval_gap_details, *model_gap_details, *invalid_citation_gaps,
        ]
        return {
            "task_id": task_id,
            "request_fingerprint": fingerprint,
            "stage_status": {
                "planning": "SKIPPED",
                "retrieval": "FAILED" if retrieval_failed else "OK" if related else "EMPTY",
                "generation": _generation_stage_status(
                    str(review_advice.get("status", "NOT_CALLED")) if related else "NOT_CALLED",
                ),
            },
            "retrieval_policy": actual_retrieval_policy or "not_confirmed",
            "retrieval_policy_requested": retrieval_policy,
            "retrieval_policy_status": (
                "NOT_AVAILABLE_FALLBACK" if selected_rerank_status == "NOT_AVAILABLE_FALLBACK" else
                "CONFIRMED" if actual_retrieval_policy == retrieval_policy else
                "MISMATCH" if actual_retrieval_policy else "UNCONFIRMED"
            ),
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieval_trace": retrieval_trace,
            "change": change,
            "selected_source": selected,
            "impacts": [{
                "status": row["review_status"],
                "relation": "suggested",
                "reason": model_candidates.get(row["impacted_item_id"], {}).get(
                    "reason", "RAG 检索到主题相关候选；模型未将其列为优先核对项，仍需人工确认。"
                ),
                "suggested_action": model_candidates.get(row["impacted_item_id"], {}).get(
                    "suggested_action", "人工确认该资料是否涉及本次变更。"
                ),
                "evidence": by_id[row["impacted_item_id"]],
            } for row in impacts],
            "confirmed_relations": [],
            "patch_candidate": {
                "target_chunk_id": selected["chunk_id"],
                "before": selected["content"], "proposed_after": proposed,
                "status": "REQUIRES_HUMAN_REVIEW",
            },
            "review_advice": review_advice,
            "evidence_gaps": [row["message"] for row in all_gap_details],
            "evidence_gap_details": all_gap_details,
            "sandbox_only": True,
            "public_baseline_written": False,
        }
