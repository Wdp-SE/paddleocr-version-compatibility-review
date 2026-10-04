"""Task-aware candidate preparation for the public PP-Human RAG service.

The helpers in this module are deterministic. They never create or alter
source evidence; the model reranker receives only bounded excerpts and may
return only the IDs supplied to it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .retrieval_fusion import fuse_ranked_hits


_CONCEPTUAL_MARKERS = (
    "为什么", "为何", "原理", "原因", "如何", "怎么", "怎样", "流程", "步骤",
    "影响", "区别", "比较", "适用", "用途", "作用", "验证", "检查", "建议",
    "what does", "why", "how", "explain", "impact", "compare", "difference",
    "workflow", "process", "validate", "verify", "recommend",
)
_EXPLICIT_REVIEW_MARKERS = (
    "变更审查", "影响审查", "变更影响", "影响分析", "change review",
    "impact review", "change impact", "audit change",
)
_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z0-9_./-]+\.(?:ya?ml|json|py|sh|xml|toml|ini)|"
    r"--?[A-Za-z][A-Za-z0-9_-]*|"
    r"\b[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+\b|"
    r"\b(?=[A-Za-z0-9_]*[A-Z])[A-Za-z][A-Za-z0-9_]*\b|"
    r"\bv?\d+\.\d+(?:\.\d+)?\b)"
)


def route_query(query: str, *, task_type: str) -> str:
    """Choose a retrieval shape without inferring user identity or permission."""
    normalized = query.strip().casefold() if isinstance(query, str) else ""
    task = task_type.strip().casefold() if isinstance(task_type, str) else ""
    if task in {"change_review", "review", "impact_review", "change_impact"}:
        return "CHANGE_REVIEW"
    if any(marker in normalized for marker in _EXPLICIT_REVIEW_MARKERS):
        return "CHANGE_REVIEW"
    if any(marker in normalized for marker in _CONCEPTUAL_MARKERS):
        return "CONCEPTUAL_LOOKUP"
    # Direct lookup requires a technical identifier, path, option or version;
    # a bare natural-language question stays on the general route.
    if _IDENTIFIER_RE.search(query if isinstance(query, str) else ""):
        return "DIRECT_LOOKUP"
    return "GENERAL_LOOKUP"


def build_candidate_pool(
    bm25_hits: list[dict],
    expanded_hits: list[dict],
    *,
    limit: int,
    bm25_anchor: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse lexical and expanded rankings while keeping a BM25 prefix."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(bm25_anchor, int) or isinstance(bm25_anchor, bool) or bm25_anchor < 0:
        raise ValueError("bm25_anchor must be a non-negative integer")
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")

    def valid_rows(rows: Iterable[dict]) -> list[dict]:
        return [
            dict(row)
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("chunk_id"), str)
            and row["chunk_id"].strip()
        ]

    lexical = valid_rows(bm25_hits)
    expanded = valid_rows(expanded_hits)
    anchor_ids = []
    if bm25_anchor:
        for row in lexical:
            chunk_id = row["chunk_id"]
            if chunk_id not in anchor_ids:
                anchor_ids.append(chunk_id)
                if len(anchor_ids) >= min(bm25_anchor, limit):
                    break
    anchors = {}
    for row in lexical:
        anchors.setdefault(row["chunk_id"], row)

    # Ask the existing RRF implementation for the full unique ranking, then
    # place the BM25 prefix first and use the remaining budget by fused rank.
    fused = fuse_ranked_hits([lexical, expanded], top_k=max(limit, len(lexical) + len(expanded)), rrf_k=rrf_k)
    ordered: list[dict] = [anchors[chunk_id] for chunk_id in anchor_ids]
    seen = set(anchor_ids)
    for row in fused:
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            continue
        ordered.append(row)
        seen.add(chunk_id)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def bound_rerank_excerpts(
    candidates: list[dict], *, per_item_chars: int = 600, total_chars: int = 12000,
) -> list[dict]:
    """Return copied candidates with fair, hard-bounded excerpts."""
    if not isinstance(per_item_chars, int) or isinstance(per_item_chars, bool) or per_item_chars < 1:
        raise ValueError("per_item_chars must be a positive integer")
    if not isinstance(total_chars, int) or isinstance(total_chars, bool) or total_chars < 1:
        raise ValueError("total_chars must be a positive integer")
    if not candidates:
        return []

    count = len(candidates)
    base_budget = min(per_item_chars, total_chars // count)
    remaining = total_chars - (base_budget * count)
    result = []
    for candidate in candidates:
        row = dict(candidate)
        content = row.get("content", row.get("text", row.get("excerpt", "")))
        text = content if isinstance(content, str) else str(content or "")
        extra = min(max(0, per_item_chars - base_budget), remaining)
        budget = base_budget + extra
        remaining -= extra
        excerpt = text[:budget]
        row["excerpt"] = excerpt
        row["excerpt_truncated"] = len(text) > len(excerpt)
        result.append(row)
    return result


def validate_ranked_ids(response: object, candidate_ids: list[str]) -> list[str] | None:
    """Accept only a complete, exact permutation of candidate IDs."""
    if not isinstance(response, dict) or set(response) != {"ranked_ids"}:
        return None
    if not isinstance(candidate_ids, list) or any(not isinstance(item, str) or not item for item in candidate_ids):
        return None
    if len(set(candidate_ids)) != len(candidate_ids):
        return None
    ranked = response.get("ranked_ids")
    if not isinstance(ranked, list) or any(not isinstance(item, str) or not item for item in ranked):
        return None
    if len(ranked) != len(candidate_ids) or len(set(ranked)) != len(ranked):
        return None
    if set(ranked) != set(candidate_ids):
        return None
    return list(ranked)


def select_coverage_ranked(
    searches: list[tuple[dict, list[dict]]],
    ranked_ids: list[str],
    *,
    max_items: int,
) -> list[dict]:
    """Round-robin across checks, then fill remaining evidence by global rank."""
    if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    rank_position = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids) if isinstance(chunk_id, str)}
    by_check: list[list[dict[str, Any]]] = []
    merged: dict[str, dict[str, Any]] = {}
    for ordinal, (check, hits) in enumerate(searches):
        check_info = dict(check) if isinstance(check, dict) else {}
        raw_index = check_info.get("check_index", ordinal)
        check_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else ordinal
        rows = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = hit.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            row = dict(hit)
            row["check_index"] = check_index
            row["check_indices"] = [check_index]
            if chunk_id in merged:
                existing = merged[chunk_id]
                if check_index not in existing["check_indices"]:
                    existing["check_indices"].append(check_index)
                continue
            merged[chunk_id] = row
            rows.append(row)
        rows.sort(key=lambda row: (rank_position.get(row["chunk_id"], len(rank_position)), row["chunk_id"]))
        by_check.append(rows)

    selected: list[dict] = []
    seen: set[str] = set()
    # One candidate per check in input order before considering second choices.
    for rows in by_check:
        candidate = next((row for row in rows if row["chunk_id"] not in seen), None)
        if candidate is not None:
            selected.append(candidate)
            seen.add(candidate["chunk_id"])
            if len(selected) >= max_items:
                return selected

    for chunk_id in ranked_ids:
        if chunk_id in merged and chunk_id not in seen:
            selected.append(merged[chunk_id])
            seen.add(chunk_id)
            if len(selected) >= max_items:
                break
    return selected
