"""Deterministic change-request context and bounded query planning.

The rules here make requests easier to inspect and evaluate. They do not replace
RAG or claim to understand every engineering phrase; unknown requests stay in the
general category and retain the user's original wording.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.domain_profile import categories_by_id, load_change_profile


_CLAUSE_SPLIT = re.compile(
    r"[。！？?；;\n]+|[,，](?=\s*(?:同时|并且|并|然后|随后|接着|核对|检查|确认|验证|评估|also\b|and\b|then\b|check\b|verify\b|confirm\b|validate\b))",
    re.IGNORECASE,
)
_CLAUSE_PREFIX = re.compile(r"^(?:同时|并且|并|然后|随后|接着|also\s+|and\s+then\s+|and\s+|then\s+)", re.IGNORECASE)


PPHUMAN_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "pphuman_change_profile.json"
AUTO_CHANGE_TYPE = "auto"


def _default_profile() -> dict[str, Any]:
    return load_change_profile(PPHUMAN_PROFILE_PATH)


CHANGE_TYPES: dict[str, dict[str, Any]] = categories_by_id(_default_profile())

_PRIVATE_ORG_CONTEXT = re.compile(
    r"(?:本公司|公司|企业|组织|客户|本单位).{0,14}(?:内部|专属|私有|专有|自有)"
    r"|(?:内部|专属|私有|专有|自有).{0,10}(?:公司|企业|组织|客户|工单|通讯录)",
    re.IGNORECASE,
)
_PRIVATE_DATA_SUBJECT = re.compile(
    r"(?:api|接口|工单|审批|流程|制度|通讯录|手机号|授权名单|权限|密级|质量|配置|映射|数据|测试|结果|负责人|签字)",
    re.IGNORECASE,
)
_PRIVATE_ORG_ENGLISH = re.compile(
    r"\b(?:internal|private|proprietary|company[- ]specific|organization[- ]specific)"
    r"\s+(?:company\s+)?(?:api|endpoint|ticket|workflow|policy|directory|phone|permission|data|test|result)\b",
    re.IGNORECASE,
)


def is_out_of_scope_public_request(text: str) -> bool:
    """Fail closed for explicitly private-company requests in the public-only demo."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return False
    return bool(
        (_PRIVATE_ORG_CONTEXT.search(normalized) and _PRIVATE_DATA_SUBJECT.search(normalized))
        or _PRIVATE_ORG_ENGLISH.search(normalized)
    )


def _contains_term(text: str, term: str) -> bool:
    if re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", term, flags=re.IGNORECASE):
        return re.search(
            rf"(?<![a-z0-9_]){re.escape(term.casefold())}(?![a-z0-9_])",
            text,
            flags=re.IGNORECASE,
        ) is not None
    return term.casefold() in text


def classify_change_type(text: str) -> str:
    """Classify a request using the active PP-Human engineering profile."""
    return _classify_change_type(text, _default_profile()["change_types"])


def _classify_change_type(text: str, categories: list[dict[str, Any]]) -> str:
    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(text or "") if part.strip()]
    if clauses:
        primary = _classify_profile_change_type(clauses[0], categories)
        if primary != "general":
            return primary
    return _classify_profile_change_type(text or "", categories)


def resolve_change_type(text: str, requested: str | None = None) -> tuple[str, str]:
    if requested and requested != AUTO_CHANGE_TYPE:
        if requested not in CHANGE_TYPES:
            raise ValueError("不支持的当前领域变更类型")
        return requested, "user_selected"
    inferred = classify_change_type(text)
    return inferred, "rule_inferred" if inferred != "general" else "unclassified"


def _classify_profile_change_type(text: str, categories: list[dict[str, Any]]) -> str:
    normalized = (text or "").casefold()
    ranked: list[tuple[int, int, str]] = []
    for order, category in enumerate(categories):
        category_id = str(category["id"])
        if category_id == "general":
            continue
        hits = sum(_contains_term(normalized, term) for term in category.get("keywords", []))
        if hits:
            ranked.append((hits, -order, category_id))
    return max(ranked, default=(0, 0, "general"))[2]


def _build_profile_request_plan(
    summary: str,
    *,
    change_type: str | None,
    impact_scope: str | None,
    profile: dict[str, Any],
    device_model: str | None,
    module_sku: str | None,
    carrier_board: str | None,
    software_baseline: str | None,
    target_snapshot: str,
) -> dict[str, Any]:
    categories = categories_by_id(profile)
    normalized_scope = (impact_scope or "").strip()
    if len(normalized_scope) > 160:
        raise ValueError("影响范围不超过 160 字")
    if not isinstance(target_snapshot, str) or not target_snapshot.strip() or len(target_snapshot) > 128:
        raise ValueError("资料快照不能为空且不能超过 128 字")
    requested = (change_type or "auto").strip()
    if requested in {"", "auto"}:
        resolved_type = _classify_change_type(summary, profile["change_types"])
        classification_source = "rule_inferred" if resolved_type != "general" else "unclassified"
    elif requested in categories:
        resolved_type = requested
        classification_source = "user_selected"
    else:
        raise ValueError("变更类型不在当前领域配置中")

    declared_scope_fields = profile.get(
        "scope_fields", ["device_model", "module_sku", "carrier_board", "software_baseline"],
    )
    all_scope_values = {
        "device_model": (device_model or "").strip() or None,
        "module_sku": (module_sku or "").strip() or None,
        "carrier_board": (carrier_board or "").strip() or None,
        "software_baseline": (software_baseline or "").strip() or None,
    }
    unsupported_scope = [
        field for field, value in all_scope_values.items()
        if value is not None and field not in declared_scope_fields
    ]
    if unsupported_scope:
        raise ValueError("当前领域配置不支持所填的设备范围字段")
    device_scope = {field: all_scope_values[field] for field in declared_scope_fields}
    declared_options = profile.get("scope_options", {})
    for field, value in device_scope.items():
        if value is None:
            continue
        if len(value) > 128:
            raise ValueError(f"{field} 不超过 128 字")
        allowed = declared_options.get(field)
        if isinstance(allowed, list) and value not in allowed:
            raise ValueError(f"{field} 不在当前知识空间的可选范围中")

    clauses = []
    for part in _CLAUSE_SPLIT.split(summary):
        cleaned = _CLAUSE_PREFIX.sub("", part.strip()).strip(" ,，;；")
        if cleaned:
            clauses.append(cleaned)
    if len(clauses) > 3:
        clauses = [*clauses[:2], " ".join(clauses[2:])]
    queries = list(dict.fromkeys([summary, *clauses])) if len(clauses) > 1 else [summary]
    category = categories[resolved_type]
    if normalized_scope:
        queries[0] = f"{normalized_scope} {category['focus']} {summary}"
    max_queries = int(profile["max_queries"])
    queries = list(dict.fromkeys(queries))[:max_queries]
    missing_scope = [field for field, value in device_scope.items() if value is None]
    scope_warnings = []
    if missing_scope:
        labels = {
            "device_model": "设备型号", "module_sku": "模组 SKU",
            "carrier_board": "载板", "software_baseline": "软件基线",
        }
        scope_warnings.append(
            "未指定" + "、".join(labels[field] for field in missing_scope)
            + "；检索结果只能作为候选线索，不能据此确认设备兼容或不受影响。"
        )
    query_terms = list(category.get("query_terms", []))
    return {
        "domain_profile_id": profile["id"],
        "original_request": summary,
        "change_type": resolved_type,
        "change_type_label": category["label"],
        "classification_source": classification_source,
        "impact_scope": normalized_scope,
        "retrieval_focus": category["focus"],
        "expected_materials": category["materials"],
        "gap_action": category["action"],
        "checklist_items": list(category["checklist"]),
        "manual_review_required": True,
        "planning_gap": category["action"] if resolved_type == "general" else None,
        "query_limit": max_queries,
        "target_snapshot": target_snapshot,
        "target_version": target_snapshot,
        "device_scope": device_scope,
        "scope_warnings": scope_warnings,
        "languages": list(profile["languages"]),
        "queries": [
            {
                "query": query,
                "search_query": " ".join([query, *query_terms]).strip(),
                "kind": "full_request" if index == 0 else "change_clause",
                "change_type": (
                    _classify_profile_change_type(query, profile["change_types"])
                    if _classify_profile_change_type(query, profile["change_types"]) != "general"
                    else resolved_type
                ),
            }
            for index, query in enumerate(queries)
        ],
    }



def build_request_plan(
    summary: str,
    *,
    change_type: str | None = None,
    impact_scope: str | None = None,
    profile: dict[str, Any] | None = None,
    device_model: str | None = None,
    module_sku: str | None = None,
    carrier_board: str | None = None,
    software_baseline: str | None = None,
    target_snapshot: str = "current",
) -> dict[str, Any]:
    """Build a bounded plan from the active profile; never switch business domains implicitly."""
    active_profile = profile or _default_profile()
    return _build_profile_request_plan(
        summary,
        change_type=change_type,
        impact_scope=impact_scope,
        profile=active_profile,
        device_model=device_model,
        module_sku=module_sku,
        carrier_board=carrier_board,
        software_baseline=software_baseline,
        target_snapshot=target_snapshot,
    )

def request_queries(summary: str) -> list[str]:
    """Backward-compatible helper used by callers that only need query text."""
    return [row["query"] for row in build_request_plan(summary)["queries"]]
