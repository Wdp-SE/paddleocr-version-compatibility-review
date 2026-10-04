"""Public-demo identity and readiness checks for the active public workspace."""

from __future__ import annotations


PROJECT_ID = "paddleocr"
PROJECT_PROFILE_ID = "paddleocr"
PROJECT_REPOSITORY = "PaddlePaddle/PaddleOCR"
_ALLOWED_LANGUAGES = {"zh"}
_WORKSPACE_RESULT_KEYS = (
    "official_result", "official_result_top_k", "official_review",
    "official_request_review", "official_review_decision",
    "official_review_decision_target", "official_review_decision_at",
    "official_review_audit_event_id", "official_review_audit_persisted_at",
    "official_review_audit_error", "official_exact_scope",
    "official_document_titles", "official_change_chunk",
    "official_change_document", "official_draft_source_id",
    "official_review_draft", "official_proposed_text",
    "ocr_compatibility_review", "ocr_review_input_fingerprint",
    "ocr_review_decision", "ocr_application_content", "ocr_application_path",
    "official_question", "official_change_request", "official_change_request_draft",
)


def public_workspace_mismatch(workspace: dict | None, *, public_demo: bool) -> str | None:
    """Reject a public backend that mixes another project or repository."""
    if not public_demo or not workspace:
        return None
    repository = str(workspace.get("repository") or "").strip()
    repositories = sorted(
        str(value).strip() for value in workspace.get("repositories") or [repository]
    )
    profile = workspace.get("domain_profile") or {}
    languages = {str(value).strip() for value in workspace.get("languages") or []}
    if (
        workspace.get("workspace_id") != PROJECT_ID
        or not isinstance(profile, dict)
        or profile.get("id") != PROJECT_PROFILE_ID
        or repository != PROJECT_REPOSITORY
        or repositories != [PROJECT_REPOSITORY]
        or languages != _ALLOWED_LANGUAGES
    ):
        return "当前连接的知识空间与 PaddleOCR 官方中文资料不匹配，已停止检索。请检查后端资料配置。"
    return None


def workspace_readiness_message(workspace: dict | None) -> str | None:
    """Explain why a connected workspace is not yet allowed to retrieve content."""
    if not workspace or workspace.get("rag_ready") is not False:
        return None
    status = str(
        workspace.get("activation_block_reason")
        or workspace.get("source_status")
        or "project_corpus_inactive"
    )
    if status == "pending_redistribution_license":
        return (
            "目标 GitHub 仓库未声明内容再分发许可。当前仅显示项目元数据；"
            "正文检索与变更审查暂不可用，待许可核实并重新构建语料后开放。"
        )
    if status == "pending_project_evaluation":
        return "PaddleOCR 语料尚未完成评测与激活，正文检索与兼容性审查暂不可用。"
    return "PaddleOCR 语料尚未通过激活校验，正文检索与兼容性审查暂不可用。"


def workspace_page_readiness_notice(workspace: dict | None, fallback: str) -> str | None:
    """Avoid repeating the global blocker while retaining a local connection fallback."""
    return fallback if workspace_readiness_message(workspace) is None else None


def workspace_identity_changed(previous: dict | None, current: dict | None) -> bool:
    """Detect a workspace or corpus revision change that invalidates cached evidence."""
    if not previous or not current:
        return False
    def identity(workspace: dict) -> tuple[object, ...]:
        fingerprint = workspace.get("corpus_fingerprint") or {}
        return (
            workspace.get("workspace_id"),
            workspace.get("repository"),
            workspace.get("current_version"),
            fingerprint.get("fingerprint_sha256") if isinstance(fingerprint, dict) else None,
        )
    return identity(previous) != identity(current)


def clear_workspace_bound_results(state, current: dict | None) -> bool:
    """Clear evidence and review decisions when the active corpus identity changes."""
    previous = state.get("official_workspace") or {}
    # A temporary disconnect must not erase the identity used to invalidate
    # old evidence when a different workspace reconnects.
    if not current:
        return False
    changed = workspace_identity_changed(previous, current or {})
    if changed:
        for key in _WORKSPACE_RESULT_KEYS:
            state.pop(key, None)
    state["official_workspace"] = current or {}
    return changed


def workspace_snapshot(workspace: dict) -> str:
    unique_documents = workspace.get("unique_document_count")
    sources = workspace.get("source_count")
    chunks = workspace.get("chunk_count", 0)
    if workspace.get("rag_ready") is False:
        reason = (
            "许可待确认" if workspace.get("source_status") == "pending_redistribution_license"
            else "语料未激活"
        )
        return f"可检索语料：{reason} · {sources or 0} 份来源 · {chunks or 0} 个检索片段"
    if unique_documents is not None and sources is not None:
        return (
            f"资料规模：{unique_documents} 个主题 · {sources} 条版本/语言来源 · "
            f"{chunks} 个检索片段"
        )
    return f"资料规模：{sources or 0} 条来源 · {chunks} 个检索片段"
