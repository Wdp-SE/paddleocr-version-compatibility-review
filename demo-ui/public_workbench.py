"""Public knowledge and hypothetical change-review workbench."""

from __future__ import annotations

import json
import hashlib
import os
import posixpath
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from html import escape
from urllib.parse import unquote, urljoin, urlsplit

import streamlit as st

from build_identity import ui_build_revision
from components.public_theme import PUBLIC_CSS
from services.public_knowledge_client import PublicKnowledgeClient
from services.query_experience import cached_workspace, interview_queries
from services.review_audit import SQLiteReviewAudit
from services.retrieval_diagnostics import (
    retrieval_diagnostic_lines as _retrieval_diagnostic_lines,
    retrieval_policy_options,
)
from services.rag_client import ServiceError
from services.public_workspace_profile import (
    public_workspace_mismatch,
    workspace_readiness_message,
    workspace_page_readiness_notice,
    workspace_snapshot,
    clear_workspace_bound_results,
    workspace_identity_changed,
)


CSS = PUBLIC_CSS


def _is_paddleocr(workspace: dict | None) -> bool:
    return not workspace or workspace.get("workspace_id") == "paddleocr"


def _page_label(page: str, workspace: dict | None) -> str:
    if _is_paddleocr(workspace):
        return {"新建变更审查": "发起兼容性审查", "可能相关资料": "影响与证据",
                "修改前后对照": "升级审查报告"}.get(page, NAV_PAGE_LABELS.get(page, page))
    return NAV_PAGE_LABELS.get(page, page)


def _module_heading(label: str) -> None:
    st.markdown(f'<span class="module-label">{escape(label)}</span>', unsafe_allow_html=True)


def _setting(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return str(st.secrets.get(name, fallback))
    except Exception:
        return fallback


def _workspace_name(workspace: dict | None) -> str:
    return str((workspace or {}).get("workspace") or "PaddleOCR 文档处理应用研发知识")


def _published_versions(workspace: dict | None) -> list[str]:
    """Offer only versions the connected, published corpus declares."""
    if not workspace:
        return ["等待连接"]
    declared = workspace.get("available_versions")
    versions = list(dict.fromkeys(str(version) for version in declared or [] if version))
    current = workspace.get("current_version")
    if current:
        versions = [str(current), *[version for version in versions if version != str(current)]]
    if not versions:
        versions = list(dict.fromkeys(
            str(version) for version in (current, workspace.get("baseline_version")) if version
        ))
    return versions or ["等待连接"]


def _confirmed_current_version(workspace: dict | None) -> str | None:
    if not workspace:
        return None
    version = workspace.get("current_version")
    return str(version).strip() if version else None


def _review_version_selector(workspace: dict | None) -> tuple[list[str], int]:
    versions = _published_versions(workspace)
    current = _confirmed_current_version(workspace)
    if not current or current not in versions:
        raise ValueError("知识空间当前版本不在可检索版本列表中")
    return versions, versions.index(current)


def _retrieval_policy_options(workspace: dict | None) -> list[dict[str, str | bool]]:
    return retrieval_policy_options(workspace)


def _retrieval_policy_selector(workspace: dict | None, *, key: str, label: str) -> str:
    options = _retrieval_policy_options(workspace)
    policy_ids = [str(row["id"]) for row in options]
    if st.session_state.get(key) not in policy_ids:
        st.session_state[key] = policy_ids[0]
    policy_labels = {str(row["id"]): str(row["label"]) for row in options}
    selected = st.selectbox(
        label, policy_ids, key=key,
        format_func=lambda value: policy_labels.get(value, value),
        help="实验策略只用于对照试用，提升检索命中不等于答案已证明更正确。",
    )
    if selected == 'paddleocr_quality':
        st.caption('质量优先：在所选版本内召回并重排。排序模型在本机运行；模型分数不代表答案正确概率。')
    elif selected=='paddleocr_evidence':
        st.caption('按版本筛选、拆分证据窗口后扩大候选；模型未配置时明确回退窗口 BM25。分数不是答案正确概率。')
    elif selected != "bm25":
        st.caption(
            "实验策略：先按版本和来源范围筛选，再扩展候选并由模型重排；"
            "查询与公开资料摘录会发送至已配置的模型服务，可能增加一次模型调用。"
            "系统不生成置信度分数，结果仍需核验引用和原文。"
        )
    return selected


def _request_context_matches(
    result: dict | None, *, summary: str, target_version: str,
    objective: str, constraints: str, validation_plan: str,
    selected_type_code: str | None, impact_scope: str,
    device_scope: dict[str, str | None] | None = None,
    retrieval_policy: str = "bm25",
) -> bool:
    """Avoid stale results while keeping pre-RAG scope rejections visible."""
    if not result or result.get("request_summary") != summary.strip():
        return False
    plan = result.get("request_plan") or {}
    plan_scope = plan.get("device_scope") or {}
    if any(plan_scope.get(key) != value for key, value in (device_scope or {}).items()):
        return False
    if result.get("scope_status") == "OUT_OF_SCOPE":
        return True
    if result.get("retrieval_policy_requested", result.get("retrieval_policy", "bm25")) != retrieval_policy:
        return False
    context = result.get("request_context") or {}
    return (
        (plan.get("target_version") or context.get("target_version")) == target_version
        and context.get("objective") == (objective.strip() or None)
        and context.get("constraints") == (constraints.strip() or None)
        and context.get("validation_plan") == (validation_plan.strip() or None)
        and plan.get("impact_scope", "") == impact_scope.strip()
        and (
            plan.get("change_type") == selected_type_code
            and plan.get("classification_source") == "user_selected"
            if selected_type_code else plan.get("classification_source") != "user_selected"
        )
    )


def _published_language_options(workspace: dict | None) -> list[tuple[str, str]]:
    locales = set((workspace or {}).get("languages") or [])
    if locales.intersection({"zh", "zh-CN", "zh-TW"}):
        return [("zh", "中文")]
    if locales.intersection({"en", "en-US", "en-GB"}):
        return [("en", "English")]
    return [("all", "语言元数据未声明")]


def _sync_workspace_version(workspace: dict | None) -> str | None:
    """Follow a newly published workspace version while preserving user scope otherwise."""
    current = _confirmed_current_version(workspace)
    clear_workspace_bound_results(st.session_state, workspace)
    for widget_key in ("official_version", "source_version", "agent_target_version"):
        default_key = f"{widget_key}_default"
        confirmed_key = f"{default_key}_confirmed"
        previous = st.session_state.get(default_key)
        previous_confirmed = st.session_state.get(confirmed_key, False)
        if current and (current != previous or not previous_confirmed):
            st.session_state[widget_key] = current
        if current:
            st.session_state[default_key] = current
        st.session_state[confirmed_key] = bool(current)
    st.session_state["official_current_version"] = current
    return current


def _version_option_label(version: str, workspace: dict | None) -> str:
    current = _confirmed_current_version(workspace)
    if version == "all":
        return "全部已收录版本"
    labels = (workspace or {}).get("version_labels") or {}
    if version in labels:
        label = str(labels[version])
        return label if version in label else f"{version} · {label}"
    if not current:
        return f"{version} · 离线回退配置（未确认）"
    if version == current:
        return f"{version} · 最新已收录"
    if version == workspace.get("baseline_version"):
        return f"{version} · 历史基线"
    return f"{version} · 历史版本"


def _scope_members(version: str, workspace: dict | None) -> set[str] | None:
    if version == "all":
        return None
    scopes = (workspace or {}).get("version_scopes") or {}
    definition = scopes.get(version) if isinstance(scopes, dict) else None
    members = definition.get("versions") if isinstance(definition, dict) else None
    return set(members) if isinstance(members, list) else {version}


def _format_snapshot_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _document_family_label(value: object) -> str:
    raw = str(value or "工程资料")
    labels = {"deployment": "部署方式", "inference_config": "推理配置", "installation": "安装指南",
              "model_config": "模型配置", "ocr_api": "OCR 接口契约", "ocr_pipeline": "OCR 使用说明",
              "structure_api": "版面解析接口", "structure_pipeline": "版面与表格解析",
              "upgrade_notes": "升级说明", "quick_start": "快速开始", "python_package": "Python 包接口"}
    return labels.get(raw, raw.replace("_", " "))


def _source_coverage_text(workspace: dict | None) -> str | None:
    if not workspace:
        return None
    rows = workspace.get("source_breakdown") or []
    counts = [row for row in rows if isinstance(row, dict) and int(row.get("count", 0)) > 0]
    if not counts:
        return None
    families = sorted({_document_family_label(row.get("document_family")) for row in counts})
    source_count = int(workspace.get("source_count") or sum(int(row.get("count", 0)) for row in counts))
    snapshot = str(workspace.get("current_version") or "当前快照")
    return f"知识空间共收录 {source_count} 份资料；默认版本 {snapshot}。覆盖：{'、'.join(families)}。"


_DEVICE_SCOPE_FIELDS = (
    ("device_model", "hardware_models", "设备型号"),
    ("module_sku", "module_skus", "模组 SKU"),
    ("carrier_board", "carrier_boards", "载板"),
    ("software_baseline", "software_baselines", "软件基线"),
)


def _device_scope_options(workspace: dict | None) -> dict[str, list[str]]:
    workspace = workspace or {}
    return {
        key: list(dict.fromkeys(str(value) for value in workspace.get(source, []) if value))
        for key, source, _label in _DEVICE_SCOPE_FIELDS
    }


def _device_scope_controls(workspace: dict | None, *, prefix: str) -> dict[str, str | None]:
    """Render manifest-backed hardware/software filters and return confirmed values."""
    options = _device_scope_options(workspace)
    selected = {key: None for key, _source, _label in _DEVICE_SCOPE_FIELDS}
    available = [field for field in _DEVICE_SCOPE_FIELDS if options[field[0]]]
    if not available:
        return selected
    with st.expander("可选：限定设备与软件环境", expanded=False):
        st.caption("选择后会作为硬过滤条件；未指定的维度不会被推断为兼容。")
        columns = st.columns(min(4, len(available)), gap="small")
        for index, (key, _source, label) in enumerate(available):
            widget_key = f"{prefix}_{key}"
            choices = ["不限定", *options[key]]
            if st.session_state.get(widget_key) not in choices:
                st.session_state[widget_key] = "不限定"
            with columns[index % len(columns)]:
                value = st.selectbox(label, choices, key=widget_key)
            selected[key] = value if value != "不限定" else None
    return selected


def _client() -> PublicKnowledgeClient:
    session_id = st.session_state.setdefault("official_session_id", uuid.uuid4().hex)
    options = (_setting("RAG_API_BASE_URL", "http://127.0.0.1:8765"),
               float(_setting("DEMO_REQUEST_TIMEOUT_SECONDS", "45")),
               session_id, int(_setting("RAG_RETRY_LIMIT", "1")))
    cached = st.session_state.get('official_http_client')
    if not cached or cached[0] != options:
        cached = (options, PublicKnowledgeClient(options[0], timeout=options[1],
                  session_id=options[2], retry_limit=options[3]))
        st.session_state['official_http_client'] = cached
    return cached[1]


def _request_workspace(client, *, force=False):
    return _request(lambda: cached_workspace(st.session_state, client, force=force),
                    fallback="知识服务暂未连接，页面仍可浏览。")


def _validate_submission_workspace(client, workspace):
    fresh = _request_workspace(client, force=True)
    if not fresh or fresh.get('rag_ready', True) is False:
        st.warning('当前知识服务不可用，本次请求未提交。请恢复连接后重试。')
        return False
    if workspace_identity_changed(workspace, fresh) or public_workspace_mismatch(
        fresh, public_demo=_setting('APP_ENV', 'local').casefold() == 'public_demo',
    ):
        st.session_state['navigation_workspace_cache'] = None
        st.rerun(scope='app')
        return False
    return True


def _audit_repository() -> SQLiteReviewAudit:
    configured_path = _setting("DEMO_AUDIT_DB_PATH", "").strip()
    if configured_path:
        database_path = Path(configured_path).expanduser()
    else:
        runtime_root = Path(_setting(
            "DEMO_RUNTIME_ROOT", str(Path(__file__).resolve().parent / "runtime")
        )).expanduser()
        database_path = runtime_root / "review_audit.sqlite3"
    return SQLiteReviewAudit(database_path)


def _request(call, *, fallback: str):
    try:
        return call()
    except ServiceError as exc:
        st.warning(f"{fallback} {exc.public_message}")
    except Exception:
        st.warning(fallback)
    return None


def _scroll_to_results() -> None:
    """Move the Streamlit page to the knowledge results after a request."""
    _scroll_to_anchor('knowledge-results-anchor')


def _scroll_to_anchor(anchor: str) -> None:
    script = """
    <script>
    (() => {
      const scroll = () => {
        const target = document.getElementById("knowledge-results-anchor");
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      };
      requestAnimationFrame(() => requestAnimationFrame(scroll));
    })();
    </script>
    """
    script = script.replace('knowledge-results-anchor', anchor)
    html_renderer = getattr(st, "html", None)
    if html_renderer is not None:
        html_renderer(script, unsafe_allow_javascript=True)
    else:
        st.components.v1.html(script, height=1, scrolling=False)


def _remember_document_titles(docs: list[dict]) -> None:
    titles = st.session_state.setdefault("official_document_titles", {})
    titles.update({row["document_id"]: row.get("title") or row.get("document_key", "")
                   for row in docs if row.get("document_id")})


def _replace_markdown_images(content: str) -> str:
    """State the image evidence boundary without pretending alt text is OCR."""
    def figure_notice(label: str) -> str:
        return f"（原文配图「{label}」，图中文字未纳入检索；可在固定来源查看）"

    content = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda match: figure_notice(match.group(1).strip() or "未命名配图"),
        content,
    )
    def describe_html_image(match: re.Match[str]) -> str:
        alt = re.search(r"\balt\s*=\s*(['\"])(.*?)\1", match.group(0), flags=re.I | re.S)
        return figure_notice(alt.group(2).strip() if alt and alt.group(2).strip() else "未命名配图")

    content = re.sub(r"<img\b[^>]*>", describe_html_image, content, flags=re.I)
    return re.sub(r"<p\b[^>]*>\s*(（原文配图「[^<]+?）)\s*</p>", r"\1", content, flags=re.I)


def _rewrite_relative_source_links(content: str, source_url: str) -> str:
    """Resolve links only while they remain in the pinned repository or docs locale."""
    source = urlsplit(source_url)
    pinned_prefix = re.match(
        r"^/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/blob/([0-9a-f]{40})/", source.path
    )
    wiki_source = source.scheme == "https" and source.netloc == "wiki.seeedstudio.com"
    github_source = source.scheme == "https" and source.netloc == "github.com" and pinned_prefix
    if not wiki_source and not github_source:
        return content
    pinned_root = (
        f"https://github.com/{pinned_prefix.group(1)}/blob/{pinned_prefix.group(2)}/"
        if pinned_prefix else None
    )
    wiki_locale_root = None
    if wiki_source:
        locale = re.match(r"^/(cn|en)/", source.path)
        if locale:
            wiki_locale_root = f"https://{source.netloc}/{locale.group(1)}/"

    def replace(match: re.Match[str]) -> str:
        destination = match.group(2).strip()
        parsed = urlsplit(destination)
        if (
            not destination or any(char.isspace() for char in destination)
            or parsed.scheme or parsed.netloc
            or (not wiki_source and parsed.path.startswith("/"))
        ):
            return match.group(0)
        resolved = urljoin(source_url, destination)
        resolved_parts = urlsplit(resolved)
        if resolved_parts.scheme != "https" or resolved_parts.netloc != source.netloc:
            return match.group(0)
        if pinned_root and not resolved.startswith(pinned_root):
            return match.group(1)
        if wiki_locale_root:
            safe_root_path = urlsplit(wiki_locale_root).path
            resolved_path = posixpath.normpath(unquote(resolved_parts.path))
            normalized_root = posixpath.normpath(safe_root_path)
            if resolved_path != normalized_root and not resolved_path.startswith(f"{normalized_root}/"):
                return match.group(1)
        return f"[{match.group(1)}]({resolved})"

    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", replace, content)


def _document_relationship_caption(row: dict) -> str:
    relationships = row.get("document_relationships")
    if not isinstance(relationships, list):
        return ""
    labels = []
    for relation in relationships:
        if not isinstance(relation, dict):
            continue
        relation_type = relation.get("relation_type")
        state = relation.get("verification_status")
        if relation_type == "translation_of":
            label = (
                "中英文对应关系待核验（不据此判断同步差异）"
                if state != "verified" else "译文关系已核验"
            )
        elif relation_type == "localized_variant_of":
            label = "本地化变体待核验" if state != "verified" else "本地化变体关系已核验"
        elif relation_type == "references_or_depends_on":
            label = "显式工程关联已确认" if state == "verified" else "工程关联待核验"
        elif relation_type == "supersedes":
            label = "替代关系已确认" if state == "verified" else "替代关系待核验"
        else:
            continue
        if label not in labels:
            labels.append(label)
    return " · ".join(labels)


def _source_card(row: dict, *, index: int, key_prefix: str = "evidence", query: str = '') -> None:
    section = row.get("heading") or "正文"
    document_title = st.session_state.get("official_document_titles", {}).get(row.get("document_id")) or row.get("document_key") or "公开资料"
    version = row.get("version", "")
    current_version = st.session_state.get("official_current_version")
    latest_members = _scope_members(current_version or "", st.session_state.get("official_workspace")) if current_version else set()
    version_label = (
        "最新资料范围" if current_version and version in (latest_members or set())
        else "历史资料快照" if current_version else "版本状态未确认"
    )
    source_label = {
        "official_documentation": "官方文档",
        "community_translation": "中文社区译本",
        "github_release": "官方 Release",
        "github_issue": "官方 Issue",
        "github_pull_request": "官方 PR",
    }.get(row.get("source_type"), "官方公开资料")
    if row.get("modality") == "image_ocr" and not _verified_image_citation(row):
        st.warning(f"[{index}] 截图证据未通过来源校验，已隐藏识别文本。")
        return
    with st.container(border=True, key=f"source_card_{key_prefix}_{index}"):
        is_image = row.get("modality") == "image_ocr"
        if is_image:
            st.markdown(f"**[{index}] 截图 OCR 证据 · {document_title}**")
            st.caption(f"章节：{section}　｜　派生证据　｜　{version_label} {version}　｜　{row.get('locale', '')}")
            st.info("人工校对的 OCR 派生证据，请对照原图。")
            st.write(row.get("content", ""))
            st.markdown(f"[查看原图]({row['raw_url']})")
        else:
            st.markdown(f"**[{index}] {document_title}**")
            st.caption(f"章节：{section}　｜　{version_label}　｜　{_version_option_label(version, st.session_state.get('official_workspace'))}　｜　{row.get('locale', '')}　｜　{source_label}")
            relation_caption = _document_relationship_caption(row)
            if relation_caption:
                st.caption(relation_caption)
            from services.evidence_span_display import display_spans
            try:
                spans=display_spans(row)
            except (ValueError,TypeError,AttributeError):
                st.warning('本次证据范围与原文不一致，已隐藏片段；请核对原始来源。')
                spans=[]
            if spans:
                from components.reading_layout import evidence_excerpt
                excerpt = evidence_excerpt([_replace_markdown_images(span['content']) for span in spans], query)
                st.markdown(f'<p class="evidence-preview">{escape(excerpt)}</p>', unsafe_allow_html=True)
                st.caption('原文节选（省略排版格式）；完整检索片段见下方。')
                with st.expander(f"查看原文片段 · {len(spans)} 段"):
                    for span in spans:
                        if span['line_start'] is not None:
                            st.caption(f"原文第 {span['line_start']}–{span['line_end']} 行")
                        content = _replace_markdown_images(span['content'])
                        content = _rewrite_relative_source_links(content, row.get("source_url", ""))
                        st.write(content)
            if row.get("source_type") == "community_translation":
                st.caption("社区维护的中文译本；关键参数请结合固定提交来源核对。")
                if row.get("rendered_url"):
                    st.markdown(f"[阅读中文社区资料]({row['rendered_url']})")
        if row.get("source_url"):
            st.markdown(f"[阅读原始页面]({row['source_url']})")
        with st.expander("技术详情"):
            st.code(f"document_key={row.get('document_key', '')}\nchunk_id={row.get('chunk_id', '')}\npolicy={row.get('retrieval_policy', '')}")
            if is_image:
                st.code(f"figure_id={row.get('figure_id', '')}\ncommit={row.get('commit', '')}\nsha256={row.get('sha256', '')}")


def _verified_image_citation(row: dict) -> bool:
    commit = row.get("commit", "")
    sha = row.get("sha256", "")
    repository = str(row.get("repository") or "")
    if row.get("review_status") != "approved" or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not str(row.get("content", "")).strip():
        return False
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        return False
    workspace = st.session_state.get("official_workspace") or {}
    allowed_repositories = set(workspace.get("repositories") or [workspace.get("repository", "")])
    pinned_commits = {
        str(item.get("commit"))
        for item in [*(workspace.get("source_registry") or []), *(workspace.get("snapshots") or [])]
        if isinstance(item, dict) and item.get("commit")
    }
    if repository not in allowed_repositories or commit not in pinned_commits:
        return False
    raw_prefix = f"https://raw.githubusercontent.com/{repository}/{commit}/"
    source_prefix = f"https://github.com/{repository}/blob/{commit}/"
    return str(row.get("raw_url", "")).startswith(raw_prefix) and str(row.get("source_url", "")).startswith(source_prefix)


def _evidence(hits: list[dict], *, heading: str = "引用依据", query: str = '') -> None:
    st.subheader(heading)
    if not hits:
        st.info("当前范围内未找到可直接支持答案的资料。请调整问题或检索范围。")
        return
    key_prefix = re.sub(r"[^\w-]+", "_", heading)
    for index, row in enumerate(hits[:3], 1):
        _source_card(row, index=index, key_prefix=key_prefix, query=query)
    if len(hits) > 3:
        with st.expander(f"查看更多结果（{len(hits) - 3}）"):
            for index, row in enumerate(hits[3:], 4):
                _source_card(row, index=index, key_prefix=key_prefix, query=query)


def _consistency(notes: list[dict], *, primary: dict | None = None) -> None:
    if not notes:
        return
    st.caption("版本差异提醒")
    if primary:
        st.markdown(
            f"**当前回答引用依据之一：** {primary.get('heading') or primary.get('document_key')} "
            f"（{primary.get('version', '版本未标注')}，{primary.get('locale', '语言未标注')}）。"
        )
    else:
        st.caption("本次未生成可核验回答；下列差异只说明资料文字或明确参数值不同。")
    st.caption("资料差异不自动等于事实冲突；请对照来源版本，人工确认相关资料是否已同步。")
    for note in notes:
        with st.container(border=True):
            st.markdown(f"**{note.get('heading') or note.get('document_key')}**")
            st.write(note.get("message", "请核对固定版本的官方原文。"))
            if note.get("kind") == "verified_literal_value_difference":
                values = " / ".join(str(value) for value in note.get("values", []))
                st.markdown(f"明确字面值：`{note.get('parameter', '参数')}` → **{values}**")
            for source in note.get("sources", []):
                st.markdown(
                    f"- [{source.get('version', '版本未标注')} · {source.get('locale', '语言未标注')} · 原始来源]"
                    f"({source.get('source_url', '')})"
                )


NAV_GROUPS = (
    ("知识服务", ("总览", "版本检索与问答", "版本与历史", "资料与来源")),
    ("变更审查", ("新建变更审查", "可能相关资料", "修改前后对照", "人工审核")),
    ("系统说明", ("检索评测", "已知限制", "系统说明")),
)
NAV_GROUP_LABELS = {
    "知识服务": "版本化知识服务",
    "变更审查": "变更影响审查",
    "系统说明": "系统说明",
}
NAV_PAGE_LABELS = {
    "总览": "总览",
    "版本检索与问答": "版本化知识检索",
    "版本与历史": "版本与历史",
    "资料与来源": "资料来源",
    "新建变更审查": "发起变更审查",
    "可能相关资料": "影响候选",
    "修改前后对照": "修改建议对照",
    "人工审核": "人工审核",
    "检索评测": "检索评测",
    "已知限制": "已知限制",
    "系统说明": "系统说明",
}
SECTION_LABELS = {
    "知识检索": "版本化知识服务",
    "知识服务": "版本化知识服务",
    "变更审查": "变更影响审查",
    "系统说明": "系统说明",
}
NAV_PAGES = frozenset(page for _, pages in NAV_GROUPS for page in pages)
PAGE_PARENTS = {
    "可能相关资料": "新建变更审查",
    "修改前后对照": "新建变更审查",
    "人工审核": "新建变更审查",
}
SECTION_ROUTES = {
    "知识检索": "版本检索与问答",
    "知识服务": "版本检索与问答",
    "变更审查": "新建变更审查",
    "系统说明": "系统说明",
}


def _navigate(destination: str) -> None:
    destination = destination if destination in NAV_PAGES else "总览"
    current = st.session_state.get("official_nav", "总览")
    if current not in NAV_PAGES:
        current = "总览"
    if current != destination:
        history = [page for page in st.session_state.get("official_nav_history", []) if page in NAV_PAGES]
        history.append(current)
        st.session_state["official_nav_history"] = history[-24:]
    st.session_state["official_nav"] = destination


def _navigate_back() -> None:
    current = st.session_state.get("official_nav", "总览")
    history = [page for page in st.session_state.get("official_nav_history", []) if page in NAV_PAGES]
    while history:
        previous = history.pop()
        if previous != current:
            st.session_state["official_nav_history"] = history
            st.session_state["official_nav"] = previous
            return
    st.session_state["official_nav_history"] = history
    st.session_state["official_nav"] = PAGE_PARENTS.get(current, "总览")


def _page_header(section: str, title: str, *, page_key: str, parent: str | None = None) -> None:
    with st.container(key=f"page_header_{page_key}"):
        back, trail = st.columns([.22, 4.2], gap="small")
        with back:
            st.button("←", key=f"nav_back_{page_key}", help="返回上一个操作模块",
                      on_click=_navigate_back)
        with trail:
            home, first_separator, group, second_separator, current = st.columns(
                [.65, .12, 1.05, .12, 1.75], gap="small"
            )
            with home:
                st.button("首页", key=f"breadcrumb_home_{page_key}", help="跳转到工作台总览",
                          on_click=_navigate, args=("总览",))
            with first_separator:
                st.markdown('<div class="breadcrumb-separator">›</div>', unsafe_allow_html=True)
            with group:
                st.button(SECTION_LABELS.get(section, section), key=f"breadcrumb_section_{page_key}",
                          on_click=_navigate, args=(SECTION_ROUTES.get(section, parent or "总览"),))
            if section != title:
                with second_separator:
                    st.markdown('<div class="breadcrumb-separator">›</div>', unsafe_allow_html=True)
                with current:
                    st.markdown(
                        f'<div class="breadcrumbs-current" aria-current="page">{escape(title)}</div>',
                        unsafe_allow_html=True,
                    )
    st.title(title)


def _home(ready: bool, workspace: dict | None) -> None:
    current = _confirmed_current_version(workspace)
    version_range = _version_option_label(current, workspace) if current else "服务未连接，无法确认"
    name = _workspace_name(workspace)
    locales = (workspace or {}).get("languages") or []
    language_names = {"zh": "中文", "zh-CN": "中文"}
    language_label = " / ".join(language_names.get(value, value) for value in locales) if locales else "尚无已审核语料"
    st.markdown(
        f'<div class="masthead"><span class="kicker">{escape(name)} · 版本化研发知识</span></div>',
        unsafe_allow_html=True,
    )
    ocr = _is_paddleocr(workspace)
    st.title("PaddleOCR 文档处理应用研发工作台" if ocr else "研发知识版本服务与变更影响审查")
    st.write(
        "维护扫描文档上传、OCR 识别与结果归一化应用：按固定依赖版本查阅官方中文资料，审查 v2.9.1 升级到 v3.0.0 对应用调用和结果消费的影响。"
        if ocr else f"基于 {name} 的官方中文研发资料进行版本检索；变更审查整理有来源支持的影响候选和证据缺口，最终由工程师确认。"
    )
    state = "已连接" if ready else (
        "许可待核实" if workspace and workspace.get("source_status") == "pending_redistribution_license"
        else "等待连接或语料激活"
    )
    status = [
        ("知识空间", name),
        ("资料来源", "PaddleOCR 官方中文资料与接口契约" if ocr else "PaddleDetection 官方开源资料"),
        ("默认检索范围", "当前应用依赖 v2.9.1 · 最新资料 v3.0.0" if ocr else version_range),
        ("语言", language_label),
        ("服务状态", state),
    ]
    cells = "".join(
        f'<div class="status-cell"><span class="status-label">{escape(label)}</span>'
        f'<span class="status-value">{escape(value)}</span></div>'
        for label, value in status
    )
    st.markdown(f'<div class="status-grid">{cells}</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, key="public_rag_module"):
            _module_heading("版本化研发知识服务 · RAG")
            st.markdown("### 版本化知识检索与问答")
            st.write("按依赖版本查询 OCR、版面解析和结果结构，保留固定提交的出处。" if ocr else "按正式版本、功能模块和配置检索 PP-Human 研发资料，并保留原文出处。")
            st.button("进入知识检索", type="primary", use_container_width=True,
                      on_click=_navigate, args=("版本检索与问答",))
    with right:
        with st.container(border=True, key="public_agent_module"):
            _module_heading("Agent · 应用兼容性审查" if ocr else "Agent · 研发资料变更审查")
            st.markdown("### 应用依赖升级兼容性审查" if ocr else "### 研发资料变更影响审查")
            st.write("提交原创应用样例或粘贴文本，核对调用、配置和结果消费方式，输出有出处的影响项与验证清单。" if ocr else "根据变更描述查找相关资料并整理修改建议，由人工确认。")
            st.button("发起兼容性审查" if ocr else "发起变更审查", type="primary", use_container_width=True,
                      on_click=_navigate, args=("新建变更审查",))
    st.markdown('<div class="section-rule">业务流程</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="flow-track"><span>应用文本</span><span>固定依赖版本</span><span>静态检查</span>'
        '<span>官方证据</span><span>影响清单</span><span>人工审核</span><span>推理回归验证</span></div>'
        if ocr else '<div class="flow-track"><span>研发资料</span><span>版本化检索</span>'
        '<span>引用溯源</span><span>资料变更</span><span>影响候选</span>'
        '<span>修改建议</span><span>人工审核</span></div>',
        unsafe_allow_html=True,
    )
    from components.paddleocr_review import coverage_panel
    coverage_panel(workspace or {})
    privacy_note = ((workspace or {}).get("domain_profile") or {}).get("privacy_boundary")
    if privacy_note:
        st.caption(str(privacy_note))
    if workspace:
        st.markdown(
            f'<div class="home-snapshot">{escape(workspace_snapshot(workspace))}</div>',
            unsafe_allow_html=True,
        )


def _use_example() -> None:
    st.session_state["official_question"] = st.session_state["official_example"]
    versions = set(re.findall(r'(?<![\w.])v?(\d+\.\d+\.\d+)(?![\w.])', st.session_state['official_example']))
    if versions == {'2.9.1', '3.0.0'}:
        st.session_state['internal_query_mode'] = 'manual'
        st.session_state['official_version'] = 'all'
    elif versions in ({'2.9.1'}, {'3.0.0'}):
        st.session_state['internal_query_mode'] = 'current' if '2.9.1' in versions else 'target'
        st.session_state['official_version'] = 'v' + next(iter(versions))


def _answer_gaps_panel(payload: dict) -> None:
    gaps = payload.get("evidence_gaps") or []
    if not isinstance(gaps, list):
        return
    gaps = [gap.strip() for gap in gaps[:5] if isinstance(gap, str) and gap.strip()]
    if not gaps:
        return
    st.markdown("**待补充的资料与信息**")
    for gap in gaps:
        st.markdown(f"- {gap}")
    st.caption("这些是模型指出的证据缺口，需要人工核对；不代表已确认不存在相关资料。")


def _knowledge(client: PublicKnowledgeClient, ready: bool, workspace: dict | None) -> None:
    _page_header("知识服务", "版本化知识检索与问答", page_key="knowledge")
    _knowledge_body(client, ready, workspace)


@st.fragment()
def _knowledge_body(client: PublicKnowledgeClient, ready: bool, workspace: dict | None) -> None:
    versions = _published_versions(workspace)
    current = _confirmed_current_version(workspace)
    internal_ocr = bool(workspace and workspace.get('workspace_id') == 'paddleocr')
    if internal_ocr:
        from services.internal_application_scope import dependency_scope
        labels = {'current': '当前应用依赖 · v2.9.1', 'target': '升级目标依赖 · v3.0.0', 'manual': '手动选择依赖资料范围'}
        mode = st.radio('查询任务', list(labels), format_func=labels.get, horizontal=True, key='internal_query_mode')
        try:
            scoped_version = dependency_scope(versions, mode, st.session_state.get('official_version'))
        except ValueError as exc:
            st.warning(str(exc))
            if mode != 'manual':
                return
            scoped_version = versions[0] if versions else 'all'
        if mode != 'manual':
            st.session_state['official_version'] = scoped_version
        current = scoped_version
    name = _workspace_name(workspace)
    rag_release=(workspace or {}).get('rag_quality_evaluation')
    default_policy = ('产线范围检索 + 模型重排' if isinstance(rag_release,dict) and
                      rag_release.get('selected_strategy') in ('contextual_llm_rerank','contextual_rerank') else
                      '产线范围检索 + 上下文补齐' if isinstance(rag_release,dict) else
                      str(workspace.get("retrieval_policy", "由服务配置") if workspace else "由服务配置").upper())
    default_version_label = _version_option_label(current, workspace) if current else "无法确认最新已收录版本"
    st.caption(f'{default_version_label} · 中文资料 · 选择示例后可编辑问题')
    with st.expander('资料范围与来源', expanded=internal_ocr and mode == 'manual'):
        if not current:
            st.caption('暂未获取当前版本信息；请以可选范围为准。')
        latest_source_time = _format_snapshot_timestamp(
            (workspace or {}).get('latest_source_retrieval_timestamp'))
        if latest_source_time:
            st.caption(f'最近收录：{latest_source_time}')
        if internal_ocr:
            st.caption('这里选择 PaddleOCR 依赖版本，内部应用版本尚未登记。官方资料作为内部研发知识的 POC 代理，保留真实来源。')
        st.caption(f'知识空间：{name}；默认检索：{default_policy}')
        a, b, c = st.columns([1, 1, .9], gap="medium")
        with a:
            version = st.selectbox(
                "版本范围", [*versions, "all"],
                format_func=lambda x: _version_option_label(x, workspace),
                key="official_version",
                disabled=internal_ocr and st.session_state.get('internal_query_mode') != 'manual',
            )
        with b:
            language_options = _published_language_options(workspace)
            language_values = [value for value, _label in language_options]
            language = st.selectbox(
                "资料语言", language_values,
                format_func=lambda x: dict(language_options)[x],
                key="official_language",
            )
        with c:
            st.markdown("**资料类型**")
            families = sorted({
                _document_family_label(row.get("document_family"))
                for row in (workspace or {}).get("source_breakdown", []) if isinstance(row, dict)
            })
            st.caption(" / ".join(families[:4]) if families else "当前公开工程资料")
    selected_scope = _device_scope_controls(workspace, prefix="official")
    filters = {key: value for key, value in selected_scope.items() if value is not None}
    examples = interview_queries(workspace)
    examples = [str(item) for item in examples if isinstance(item, str) and item.strip()]
    if not examples:
        examples = [f"{name} 中某个功能或参数是如何设计的？"]
    example_key = "official_example"
    if st.session_state.get(example_key) not in examples:
        st.session_state[example_key] = examples[0]
    st.selectbox(
        "示例问题（选择后可编辑）", examples, key="official_example",
        on_change=_use_example,
    )
    question = st.text_area("问题", value="", placeholder=examples[0], height=100, key="official_question")
    submitted_question = question.strip() or examples[0]
    with st.container(key='knowledge_actions'):
        generate_col, search_col, settings_col = st.columns([1.6, 1, 1], gap="small")
        with generate_col:
            ask_now = st.button("生成带引用回答", type="primary", key="knowledge_generate",
                                disabled=not ready, use_container_width=True)
        with search_col:
            search_now = st.button("仅查看检索原文", key="knowledge_search",
                                   disabled=not ready, use_container_width=True)
        with settings_col:
            with st.popover('检索设置', use_container_width=True):
                retrieval_policy = _retrieval_policy_selector(
                    workspace, key="official_retrieval_policy", label="检索策略",
                )
                top_k = st.slider(
                    "证据条数（Top-K）", min_value=1, max_value=20, value=int((rag_release or {}).get('top_k',5)),
                    key="official_top_k",
                    help="控制本次证据条数，版本和语言仍由服务端过滤。",
                )
                st.caption('生成可能调用已配置的模型服务，费用与限流以服务商规则为准。')
    st.markdown('<div id="knowledge-request-anchor"></div>', unsafe_allow_html=True)
    request_status = st.empty()
    scroll_to_results = ask_now or search_now
    if scroll_to_results:
        _scroll_to_anchor('knowledge-request-anchor')
        with request_status.container(), st.spinner('正在核对知识库状态……', show_time=True):
            submission_ready = _validate_submission_workspace(client, workspace)
        if not submission_ready:
            return
    if ask_now:
        with request_status.container(), st.spinner("正在检索资料、生成回答并核对引用……", show_time=True):
            payload = _request(
                lambda: client.query_official(
                    submitted_question, version=version, language=language, top_k=top_k,
                    retrieval_policy=retrieval_policy, **filters,
                ),
                fallback="知识问答暂不可用。",
            )
        if payload:
            st.session_state["official_result"] = (
                "query", submitted_question, version, language, selected_scope,
                retrieval_policy, payload,
            )
            st.session_state["official_result_top_k"] = top_k
    if search_now:
        with request_status.container(), st.spinner("正在检索资料……", show_time=True):
            payload = _request(
                lambda: client.search(
                    submitted_question, version=version, language=language, top_k=top_k,
                    retrieval_policy=retrieval_policy, **filters,
                ),
                fallback="资料检索暂不可用。",
            )
        if payload:
            st.session_state["official_result"] = (
                "search", submitted_question, version, language, selected_scope,
                retrieval_policy, payload,
            )
            st.session_state["official_result_top_k"] = top_k
    request_status.empty()
    st.markdown('<div id="knowledge-results-anchor"></div>', unsafe_allow_html=True)
    result = st.session_state.get("official_result")
    if (
        result
        and result[1:6] == (submitted_question, version, language, selected_scope, retrieval_policy)
        and st.session_state.get("official_result_top_k", 5) == top_k
    ):
        if ready and "official_document_titles" not in st.session_state:
            docs = _request(client.documents, fallback="来源目录暂不可用，仍可查看原文链接。")
            if docs is not None:
                _remember_document_titles(docs)
        mode, _, _, _, _filters, _policy, payload = result
        st.markdown('<div class="section-rule">结果与核验</div>', unsafe_allow_html=True)
        if mode == "query":
            st.subheader("回答")
            sources = payload.get("sources") or []
            if payload.get("status") == "OK" and sources:
                with st.container(border=True, key="generated_answer"):
                    from components.answer_reading import render_answer
                    render_answer(payload)
            else:
                if payload.get("status") == "OUT_OF_SCOPE":
                    st.warning("**缺少应用验证资料，无法确认兼容。** 当前知识库不能证明未提交的内部应用事实，已停止生成。")
                    st.write('若要确认应用升级兼容，需要应用调用与消费代码、输出契约和目标环境的回归记录。官方资料只能提供 SDK 行为依据。')
                elif payload.get("status") == "NO_EVIDENCE":
                    st.warning("**未找到支持答案的证据。** 当前版本与语言范围内未找到匹配资料；请核对资料范围。")
                elif payload.get("status") == "ABSTAINED":
                    diagnostic = payload.get("generation") or {}
                    reason = diagnostic.get("failure_reason")
                    if reason == 'MODEL_NO_SUPPORTED_ANSWER':
                        st.warning('**证据不足，未生成有依据的回答。** 具体缺口如下。')
                    else:
                        st.error('**回答未展示：证据核验未通过。** 请查看具体原因与检索原文。')
                    if reason == "MODEL_NO_SUPPORTED_ANSWER":
                        candidate_count = diagnostic.get("candidate_count", len(payload.get("evidence") or []))
                        coverage = diagnostic.get("evidence_coverage") or {}
                        missing_terms = coverage.get("missing_terms") or []
                        if payload.get("evidence_gaps"):
                            st.caption("模型服务正常，但未形成有证据支持的回答；请核对下方具体资料缺口。")
                        elif missing_terms:
                            missing_label = "、".join(str(term) for term in missing_terms[:6])
                            st.caption(
                                f"模型服务正常，但召回的 {candidate_count} 条资料未覆盖关键词：{missing_label}。"
                                "系统已拒答；请改用原文术语或缩小问题范围。"
                            )
                        else:
                            st.caption(
                                f"模型服务正常，已召回 {candidate_count} 条资料，但没有找到可支持答案的原文；"
                                "这是证据不足导致的拒答，不是网络或 API Key 故障。"
                            )
                    elif reason == "NO_VALID_EVIDENCE_CITATIONS":
                        claimed = diagnostic.get("claimed_citation_count", 0)
                        valid = diagnostic.get("valid_citation_count", 0)
                        candidates = diagnostic.get("candidate_count", 0)
                        st.caption(
                            f"模型输出未通过证据引用校验：检索到 {candidates} 条候选资料，"
                            f"{valid}/{claimed} 个引用能匹配本次结果。回答已隐藏以避免展示无法追溯内容；"
                            "请核对下方原文，或缩小问题范围后重试。"
                        )
                    else:
                        if reason=='CLAIM_SCOPE_MISMATCH':
                            st.caption('生成主张引用的模块或版本与问题不匹配，答案已隐藏；这不是模型网络故障，请核对下方资料范围。')
                        elif reason=='MECHANICAL_FACT_CONTRADICTION':
                            st.caption('生成的参数默认值与引用原文不一致，答案已隐藏；请以对应版本的参数说明为准。')
                        elif reason in {'CLAIM_SUPPORT_REJECTED','CLAIM_SUPPORT_CHECK_FAILED'}:
                            st.caption('生成主张未获得原文支持，已隐藏。' if reason=='CLAIM_SUPPORT_REJECTED'
                                       else '原文支持度复核未完成，未展示未经复核的答案；检索证据仍可查看。')
                        else:
                            st.caption(
                            f"回答因证据校验未通过而被隐藏（原因码：{reason or '未返回'}）；"
                            "请核对下方检索原文及本次检索技术详情。"
                            )
                elif payload.get("status") == "GENERATION_NOT_CONFIGURED":
                    st.error("**回答未生成：模型生成尚未启用。** 检索证据仍可查看。")
                elif payload.get("status") == "GENERATION_PROVIDER_UNAVAILABLE":
                    st.error("**回答未生成：后端无法连接模型服务。** 检索证据已保留，请稍后重试。")
                elif payload.get("status") == "GENERATION_PROVIDER_TIMEOUT":
                    st.error("**回答未生成：模型服务响应超时。** 检索证据已保留，请稍后重试。")
                elif payload.get("status") == "GENERATION_RATE_LIMITED":
                    st.error("**回答未生成：模型服务限流。** 检索证据已保留，请稍后重试。")
                elif payload.get("status") == "GENERATION_BILLING_REQUIRED":
                    st.error("**回答未生成：模型账户计费受限。** 检索证据已保留。")
                elif payload.get("status") == "GENERATION_AUTH_FAILED":
                    st.error("**回答未生成：模型服务鉴权失败。** 检索证据已保留，请联系维护者。")
                elif payload.get("status") == "GENERATION_PROVIDER_REJECTED":
                    st.caption("模型服务拒绝了请求；检索证据已保留，请稍后重试。")
                elif payload.get("status") == "GENERATION_RESPONSE_INVALID":
                    st.error(
                        "**回答未生成：模型输出未通过结构校验。** 检索证据已保留；这不是仍在加载，也不能算回答成功。"
                    )
                elif payload.get("status") == "GENERATION_RESPONSE_TRUNCATED":
                    st.error("**回答未生成：模型回复被截断。** 检索证据已保留，请核对原文。")
                else:
                    diagnostic = payload.get("generation") or {}
                    request_id = diagnostic.get("request_id") or "未返回"
                    st.caption(
                        f"生成未完成（{payload.get('status', 'UNKNOWN')}）；检索证据已保留。"
                        f"联系维护者时请提供请求编号：{request_id}。"
                    )
            primary = sources[0] if sources else None
            _consistency(payload.get("consistency_notes", []), primary=primary)
            _answer_gaps_panel(payload)
            correction=payload.get('correction') or {}
            if correction.get('attempts'):
                st.caption('已补查一次并重新核验。' + (
                    '下方保留通过核验的内容。' if correction.get('status')=='RECOVERED'
                    else '现有证据仍有缺口，请查看原文及具体核验原因。'))
                with st.expander('查看补查与核验过程'):
                    st.json(correction)
            verification=payload.get('claim_verification')
            if isinstance(verification,dict):
                with st.expander('查看原文支持度复核'):
                    st.json(verification)
            if payload.get("status") == "OK" and sources:
                _evidence(sources, heading="引用依据", query=submitted_question)
                cited_ids = {row.get("chunk_id") for row in sources}
                remaining = [
                    row for row in payload.get("evidence", [])
                    if row.get("chunk_id") not in cited_ids
                ]
                if remaining:
                    with st.expander(f"查看其余检索结果（{len(remaining)}）"):
                        _evidence(remaining, heading="其他相关资料", query=submitted_question)
            elif payload.get('status') != 'OUT_OF_SCOPE':
                _evidence(payload.get("evidence", []), heading="检索到的资料", query=submitted_question)
        else:
            _consistency(payload.get("consistency_notes", []))
            if payload.get("status") == "OUT_OF_SCOPE":
                st.caption("问题涉及当前公开语料无法提供的企业内部信息；系统已停止检索。")
            else:
                _evidence(payload.get("results", []), heading="检索到的资料", query=submitted_question)
        with st.expander("本次检索技术详情"):
            st.write(
                f"检索范围：{version} · 语言：{language} · Top-K：{top_k} · "
                f"本次策略：{payload.get('retrieval_policy', 'BM25')}"
            )
            for line in _retrieval_diagnostic_lines(payload):
                st.caption(line)
            selected_filters = [(label, selected_scope.get(key)) for key, _source, label in _DEVICE_SCOPE_FIELDS if selected_scope.get(key)]
            if selected_filters:
                st.caption("设备/软件范围：" + " · ".join(f"{label}：{value}" for label, value in selected_filters))
            st.caption("候选排序分数只用于同一检索策略内的排序，不代表事实正确性。")
            generation = payload.get("generation") if mode == "query" else None
            if isinstance(generation, dict):
                st.caption(
                    f"生成请求：{generation.get('request_id') or '未返回'} · "
                    f"服务商：{generation.get('provider') or '未确认'} · "
                    f"请求模型：{generation.get('requested_model') or '未确认'} · "
                    f"实际模型：{generation.get('returned_model') or '未返回'}"
                )
                st.caption(
                    f"结束原因：{generation.get('finish_reason') or '未返回'} · "
                    f"生成耗时：{generation.get('latency_ms') if generation.get('latency_ms') is not None else '未记录'} ms · "
                    f"Token 用量：{generation.get('usage') or '未返回'}"
                )
    if not ready:
        notice = workspace_page_readiness_notice(
            workspace, "知识服务可能正在冷启动；连接恢复后可继续检索。",
        )
        if notice:
            st.info(notice)
    if scroll_to_results:
        _scroll_to_results()


def _analyze_hypothetical(
    client: PublicKnowledgeClient, selected: dict, proposed: str,
    *, device_scope: dict[str, str | None] | None = None,
    retrieval_policy: str = "bm25",
) -> dict:
    agent_root = Path(__file__).resolve().parents[1] / "change-review-agent"
    if str(agent_root) not in sys.path:
        sys.path.insert(0, str(agent_root))
    from app.public_review import PublicReviewAgent
    return PublicReviewAgent(client).analyze(
        selected, proposed, retrieval_policy=retrieval_policy,
        **(device_scope or {}),
    )


def _analyze_change_request(
    client: PublicKnowledgeClient,
    change_summary: str,
    change_type: str | None = None,
    impact_scope: str | None = None,
    *,
    target_version: str | None = None,
    objective: str | None = None,
    constraints: str | None = None,
    validation_plan: str | None = None,
    language_mode: str = "zh",
    device_scope: dict[str, str | None] | None = None,
    retrieval_policy: str = "bm25",
) -> dict:
    agent_root = Path(__file__).resolve().parents[1] / "change-review-agent"
    if str(agent_root) not in sys.path:
        sys.path.insert(0, str(agent_root))
    from app.public_review import PublicReviewAgent
    return PublicReviewAgent(client).analyze_request(
        change_summary, change_type=change_type, impact_scope=impact_scope,
        target_version=target_version, objective=objective,
        constraints=constraints, validation_plan=validation_plan,
        language_mode=language_mode, retrieval_policy=retrieval_policy,
        **(device_scope or {}),
    )


def _review_steps(stage: int) -> None:
    labels = ("变更已提交", "证据已检索", "人工审核已记录" if stage >= 3 else "等待人工审核")
    cells = "".join(
        f'<span class="{"done" if i < stage else "current" if i == stage else ""}">'
        f'{i + 1}. {escape(label)}</span>'
        for i, label in enumerate(labels)
    )
    st.markdown(f'<div class="review-steps">{cells}</div>', unsafe_allow_html=True)


def _impact_panel(result: dict) -> None:
    st.subheader("影响候选")
    st.caption("以下为待核对线索，最终影响由人工确认。")
    references = result.get("confirmed_relations", [])
    if references:
        st.subheader("已确认文档关联")
        st.caption("该引用只确认文档关联，不代表所选段落受影响。")
        for reference in references:
            with st.container(border=True):
                relation_type = str(reference.get("relation_type") or "文档关联")
                st.markdown(f"**{escape(relation_type.replace('_', ' '))}**")
                st.caption(f"明确引用所在章节：{reference['source_heading']}")
                excerpt = _rewrite_relative_source_links(
                    _replace_markdown_images(reference["source_excerpt"]), reference.get("source_url", "")
                )
                st.write(excerpt[:300] + ("…" if len(excerpt) > 300 else ""))
                st.markdown(f"[阅读原始页面]({reference['source_url']})")
                if len(excerpt) > 300:
                    with st.expander("查看完整引用原文"):
                        st.write(excerpt)
    impacts = result.get("impacts", [])
    if not impacts:
        st.info("未检索到足够相关的其他资料；仍可人工审核当前段落。")
        return
    for index, item in enumerate(impacts):
        evidence = item["evidence"]
        with st.container(border=True, key=f"impact_row_{index}"):
            st.markdown(f"**待核对资料 · {evidence.get('heading') or evidence.get('document_key')}**")
            st.caption(f"{evidence.get('version', '')}　｜　{evidence.get('locale', '')}")
            st.write(item.get("reason", "请核对官方原文与显式引用。"))
            content = _rewrite_relative_source_links(
                _replace_markdown_images(evidence.get("content", "")), evidence.get("source_url", "")
            )
            st.write(content[:300] + ("…" if len(content) > 300 else ""))
            st.markdown(f"[阅读原始页面]({evidence['source_url']})")
            if len(content) > 300:
                with st.expander("查看完整相关片段"):
                    st.write(content)


def _review_evidence_panel(result: dict) -> None:
    st.subheader("引用依据")
    _source_card(result["selected_source"], index=1, key_prefix="selected_source")


def _review_advice_panel(result: dict, *, context: str = "review") -> None:
    advice = result.get("review_advice", {})
    if advice.get("status") == "OK" and advice.get("sources"):
        review = advice.get("review") or {}
        interpretation = review.get("change_interpretation") or advice.get("answer", "N/A")
        coverage = result.get("coverage") or {}
        coverage_complete = coverage.get("complete", True)
        if context == "变更分析" and not coverage_complete:
            interpretation = "本次只整理已命中的证据；检索覆盖不完整，不能据此判定无影响。请先补齐语言和检查项缺口。"
        candidates = review.get("impact_candidates", [])
        source_numbers = {
            row.get("chunk_id"): number
            for number, row in enumerate(advice["sources"], start=1)
        }
        if context == "变更分析":
            st.subheader("模型辅助核对建议")
            if not coverage_complete:
                st.warning(
                    "检索覆盖不完整：当前结果只能作为已命中资料的核对线索，不能据此得出无影响结论。"
                )
            st.markdown(
                '<div class="agent-review-summary">'
                '<div class="agent-review-summary-heading"><strong>本次分析结论</strong>'
                '<span>等待人工审核</span></div>'
                f'<p>{escape(str(interpretation))}</p></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"依据 {len(advice['sources'])} 条检索证据整理 · {len(candidates)} 项待核对")
            st.subheader("优先核对的影响候选")
            if candidates:
                sources_by_id = {
                    row.get("chunk_id"): row for row in advice["sources"] if row.get("chunk_id")
                }
                for number, candidate in enumerate(candidates, start=1):
                    evidence_id = candidate.get("evidence_chunk_id")
                    source = sources_by_id.get(evidence_id)
                    heading = (source or {}).get("heading") or (source or {}).get("document_key") or "待核对资料"
                    with st.container(key=f"review_candidate_{number}"):
                        st.markdown(
                            '<div class="agent-candidate-heading">'
                            f'<span>优先级 {number}</span><strong>{escape(str(heading))}</strong>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<div class="agent-field-label">影响判断</div>', unsafe_allow_html=True)
                        st.markdown(escape(str(candidate.get("reason") or "需要人工核对该资料。")))
                        st.markdown('<div class="agent-field-label">建议核对动作</div>', unsafe_allow_html=True)
                        st.markdown(escape(str(candidate.get("suggested_action") or "核对引用资料与变更范围后，由审核人决定后续动作。")))
                        if source:
                            evidence_number = source_numbers.get(evidence_id)
                            st.markdown(
                                '<div class="agent-evidence-heading">'
                                f'引用证据 [{evidence_number}] · {escape(str(heading))}'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"{source.get('version', '版本未标注')}　｜　"
                                f"{source.get('locale', '语言未标注')}　｜　公开来源资料"
                            )
                            content = _rewrite_relative_source_links(
                                _replace_markdown_images(source.get("content", "")),
                                source.get("source_url", ""),
                            )
                            if content:
                                excerpt = content.strip()
                                st.markdown(escape(excerpt[:360] + ("…" if len(excerpt) > 360 else "")))
                            source_url = source.get("source_url")
                            if source_url:
                                st.markdown(f"[打开官方原文]({escape(str(source_url), quote=True)})")
                            if content:
                                with st.expander(f"展开完整引用原文 [{evidence_number}]"):
                                    st.write(content)
                        else:
                            st.caption("此候选未匹配到有效引用，须先人工查证，不能视为证据支持的影响结论。")
            else:
                st.info("模型没有形成可引用的影响候选；下方检索命中仅供人工筛查。")
        else:
            st.subheader("模型辅助核对建议")
            st.markdown("**变更理解**")
            st.write(interpretation)
            st.caption("具体候选、原文和修改前后对照见下方；所有建议仍需人工确认。")
        structured_gap_text = {
            str(row.get("message") or row.get("description") or "").strip()
            for row in result.get("evidence_gap_details") or []
        }
        follow_up = []
        follow_up.extend(
            ("证据缺口", value) for value in (review.get("evidence_gaps") or [])
            if str(value).strip() not in structured_gap_text
        )
        follow_up.extend(
            ("版本或语言歧义", value) for value in (review.get("version_ambiguities") or [])
            if str(value).strip() not in structured_gap_text
        )
        follow_up.extend(("人工检查", value) for value in (review.get("reviewer_actions") or []))
        if follow_up:
            with st.expander(f"未解决事项与人工检查（{len(follow_up)}）"):
                for label, item in follow_up:
                    st.markdown(f"**{label}**")
                    st.write(item)
        st.caption("模型建议仅使用本次检索证据；人工确认前不会修改公共资料。")
    elif advice.get("status") == "GENERATION_NOT_CONFIGURED":
        st.subheader("模型辅助核对建议")
        st.caption("当前环境未启用模型建议；影响候选与原文仍可继续人工核对。")
    elif advice.get("status") == "GENERATION_PROVIDER_UNAVAILABLE":
        st.subheader("模型辅助核对建议")
        st.caption("模型服务暂不可用；本次检索候选与原文仍保留，建议人工核对。")
    elif advice.get("status") == "GENERATION_PROVIDER_TIMEOUT":
        st.subheader("模型辅助核对建议")
        st.caption("模型建议请求超时；检索候选与原文已保留，可稍后重试或人工核对。")
    elif advice.get("status") == "GENERATION_RATE_LIMITED":
        st.subheader("模型辅助核对建议")
        st.caption("模型服务当前限流；检索候选与原文已保留，可稍后重试。")
    elif advice.get("status") == "GENERATION_BILLING_REQUIRED":
        st.subheader("模型辅助核对建议")
        st.caption("模型服务返回计费或余额限制；请检查模型账户，当前候选仍可人工核对。")
    elif advice.get("status") == "GENERATION_AUTH_FAILED":
        st.subheader("模型辅助核对建议")
        st.caption("模型服务鉴权失败；请检查后端密钥与权限，当前候选和原文仍可人工核对。")
    elif advice.get("status") == "GENERATION_RESPONSE_TRUNCATED":
        st.subheader("模型辅助核对建议")
        st.caption("模型建议回复被截断，未作为有效建议展示；请依据已保留的候选原文人工核对。")
    elif advice.get("status") == "GENERATION_PROVIDER_REJECTED":
        st.subheader("模型辅助核对建议")
        st.caption("模型服务未接受本次建议请求；请检查后端模型配置，当前候选和原文仍可人工核对。")
    elif advice.get("status") == "ABSTAINED":
        st.subheader("模型辅助核对建议")
        st.info("模型提示待核对：当前证据不足以形成带有效引用的影响候选；以下缺口和动作尚未确认。")
        review = advice.get("review") if isinstance(advice.get("review"), dict) else {}
        if review.get("evidence_gaps"):
            st.caption("待补证据：" + "；".join(review["evidence_gaps"]))
        if review.get("version_ambiguities"):
            st.caption("版本或语言歧义：" + "；".join(review["version_ambiguities"]))
        if review.get("reviewer_actions"):
            st.caption("建议人工核对：" + "；".join(review["reviewer_actions"]))
    elif advice.get("status") == "NO_EVIDENCE":
        st.subheader("模型辅助核对建议")
        st.caption("没有找到可供模型引用的其他资料；请人工检查原文和变更草案。")
    elif advice.get("status") == "OUT_OF_SCOPE":
        st.subheader("当前资料范围不支持此请求")
        st.warning(advice.get(
            "message",
            "当前公开知识空间中没有适用于此请求的资料；未执行检索或模型生成。",
        ))
    else:
        st.subheader("模型辅助核对建议")
        st.caption("当前证据不足以形成带有效引用的模型建议；请按原文和候选资料人工核对。")

def _patch_panel(result: dict) -> None:
    st.subheader("修改建议对照")
    st.caption("草案仅用于本次审查，不会写回源资料。")
    before, after = result["patch_candidate"]["before"], result["patch_candidate"]["proposed_after"]
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, key="before_panel"):
            st.markdown("**当前官方原文**")
            st.write(before[:450] + ("…" if len(before) > 450 else ""))
    with right:
        with st.container(border=True, key="after_panel"):
            st.markdown("**会话内假设草案**")
            st.write(after[:450] + ("…" if len(after) > 450 else ""))
    if len(before) > 450 or len(after) > 450:
        with st.expander("查看完整修改前后内容"):
            st.markdown("**当前官方原文**")
            st.write(before)
            st.markdown("**会话内假设草案**")
            st.write(after)


def _review_target_id(result: dict) -> str:
    task_id = result.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _review_decision_for(result: dict) -> str | None:
    if st.session_state.get("official_review_decision_target") != _review_target_id(result):
        return None
    return st.session_state.get("official_review_decision")


def _set_review_decision(
    decision: str, target_id: str, result: dict, session_id: str,
) -> None:
    st.session_state["official_review_decision"] = decision
    st.session_state["official_review_decision_target"] = target_id
    decided_at = datetime.now(timezone.utc).isoformat()
    st.session_state["official_review_decision_at"] = decided_at
    st.session_state.pop("official_review_audit_event_id", None)
    st.session_state.pop("official_review_audit_persisted_at", None)
    st.session_state.pop("official_review_audit_error", None)
    try:
        report = _review_report(result, decision, decided_at)
        event = _audit_repository().record(report, session_id=session_id)
        st.session_state["official_review_audit_event_id"] = event["event_id"]
        st.session_state["official_review_audit_persisted_at"] = event["persisted_at_utc"]
    except Exception:
        # Keep the exportable decision visible, but never claim it was persisted.
        st.session_state["official_review_audit_error"] = "审核结果未能写入本地审查记录。"


def _review_report(result: dict, decision: str, decided_at: str) -> dict:
    """Export a session decision with evidence IDs and draft hashes for review."""
    source_rows = [result.get("selected_source") or {}]
    source_rows.extend(result.get("retrieved_results") or [])
    source_rows.extend(result.get("review_advice", {}).get("sources") or [])
    sources = list({
        row["chunk_id"]: {"chunk_id": row["chunk_id"], "source_url": row.get("source_url")}
        for row in source_rows if row.get("chunk_id")
    }.values())
    patch = result.get("patch_candidate") or {}
    def content_hash(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else None

    report = {
        "schema_version": 3,
        "record_scope": "review_decision_export",
        "task_id": _review_target_id(result),
        "request_fingerprint": result.get("request_fingerprint"),
        "request_mode": result.get("request_mode"),
        "request_summary": result.get("request_summary"),
        "request_plan": result.get("request_plan"),
        "selected_source_id": (result.get("selected_source") or {}).get("chunk_id"),
        "before_sha256": content_hash(patch.get("before")),
        "proposed_after_sha256": content_hash(patch.get("proposed_after")),
        "retrieval_policy": result.get("retrieval_policy"),
        "retrieval_policy_requested": result.get("retrieval_policy_requested"),
        "retrieval_policy_status": result.get("retrieval_policy_status"),
        "retrieval_latency_ms": result.get("retrieval_latency_ms"),
        "retrieval_trace": result.get("retrieval_trace"),
        "configuration_trace": result.get("configuration_trace"),
        "evidence_gaps": result.get("evidence_gaps", []),
        "evidence_gap_details": result.get("evidence_gap_details", []),
        "evidence_sources": sources,
        "model_status": result.get("review_advice", {}).get("status"),
        "model_review": result.get("review_advice", {}).get("review"),
        "human_decision": decision,
        "decided_at_utc": decided_at,
        "public_baseline_written": result.get("public_baseline_written"),
    }
    if result.get("request_mode") == "paddleocr_compatibility":
        report.update({
            "workspace_id": result.get("workspace_id"), "project_id": result.get("project_id"),
            "source_version": result.get("source_version"), "target_version": result.get("target_version"),
            "application_files": (result.get("compatibility_report") or {}).get("files", []),
            "compatibility_report": result.get("compatibility_report"),
            "report_fingerprint": result.get("report_fingerprint"), "mode": "static",
            "runtime_verified": False, "model_suggestions": result.get("model_suggestions", []),
            "model_status": result.get("model_status"), "model_calls": result.get("model_calls", 0),
            "model_advice_request_count": result.get("model_advice_request_count", 0),
            "model_generation": result.get("model_generation", {}),
            "investigation":result.get("investigation"),
            "evidence_sources": [citation for finding in (result.get("compatibility_report") or {}).get("findings", [])
                                 for citation in finding.get("evidence", [])],
            "human_review_scope": "已审阅静态报告；不代表实际推理兼容性已验证",
            "application_context": result.get('application_context'),
        })
    return report


def _clear_stale_review() -> None:
    st.session_state.pop("official_review", None)
    st.session_state.pop("official_review_decision", None)
    st.session_state.pop("official_review_decision_target", None)
    st.session_state.pop("official_review_decision_at", None)
    st.session_state.pop("official_review_audit_event_id", None)
    st.session_state.pop("official_review_audit_persisted_at", None)
    st.session_state.pop("official_review_audit_error", None)


def _save_review_draft() -> None:
    st.session_state["official_review_draft"] = st.session_state.get(
        "official_proposed_text", ""
    )
    _clear_stale_review()


def _review_panel(result: dict) -> None:
    st.subheader("人工审核")
    st.info(
        "此处为匿名演示审核记录，可能随实例重启清空，不能替代正式审批；请勿提交敏感信息。"
    )
    compatibility = result.get("request_mode") == "paddleocr_compatibility"
    request_only = result.get("request_mode") == "natural_language"
    review_target = "本次静态兼容性报告" if compatibility else "本次影响分析" if request_only else "会话草案"
    if compatibility:
        st.caption("确认仅记录已审阅报告。实际 OCR 推理、安装依赖和下游输出回归仍由开发者验证。")
    target_id = _review_target_id(result)
    session_id = st.session_state.setdefault("official_session_id", uuid.uuid4().hex)
    if compatibility:
        from components.paddleocr_disposition import render_disposition
        render_disposition(result, _audit_repository(), session_id)
    approve, reject = st.columns(2)
    approve.button(f"确认已审阅{review_target}", use_container_width=True,
                   on_click=_set_review_decision, args=("reviewed", target_id, result, session_id))
    reject.button(f"退回{review_target}", use_container_width=True,
                  on_click=_set_review_decision, args=("rejected", target_id, result, session_id))
    decision = _review_decision_for(result)
    if decision == "reviewed":
        if st.session_state.get("official_review_audit_event_id"):
            st.success(f"审核结果已写入本地审查记录。未创建候选版本，公共基线未修改。")
        else:
            st.warning(st.session_state.get("official_review_audit_error", "审核结果尚未确认持久化。"))
    elif decision == "rejected":
        if st.session_state.get("official_review_audit_event_id"):
            st.info(f"{review_target}已退回，决定已写入本地审查记录。公共基线未修改。")
        else:
            st.warning(st.session_state.get("official_review_audit_error", "退回决定尚未确认持久化。"))
    if decision:
        report = _review_report(result, decision, st.session_state.get("official_review_decision_at", ""))
        audit_event_id = st.session_state.get("official_review_audit_event_id")
        report["persisted_to_demo_sqlite"] = bool(audit_event_id)
        if audit_event_id:
            report["audit_event_id"] = audit_event_id
            report["persisted_at_utc"] = st.session_state.get("official_review_audit_persisted_at")
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", target_id)[:24] or "session"
        st.download_button(
            "下载本次审查记录（JSON）",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"review-{safe_id}.json", mime="application/json",
            key=f"review_export_{safe_id}",
        )

    with st.expander("最近审核记录"):
        try:
            history = _audit_repository().list_recent(session_id=session_id, limit=10)
            if compatibility:
                history = [event for event in history if event.get("workspace_id") == "paddleocr"
                           and event.get("request_mode") == "paddleocr_compatibility"]
            if history:
                st.table([
                    {
                        "时间（UTC）": event.get("decided_at_utc", ""),
                        "决定": ("继续人工升级流程" if event.get('upgrade_decision') == 'continue_manual_process' else
                                 "暂缓升级" if event.get('upgrade_decision') == 'hold' else
                                 "已审阅" if event.get("human_decision") == "reviewed" else "已退回"),
                        "审查内容": (event.get("request_summary") or "").replace("\n", " ")[:100],
                        "证据数": len(event.get("evidence_sources") or []),
                    }
                    for event in history
                ])
            else:
                st.caption("该会话还没有持久化的审核决定。")
        except Exception:
            st.warning("暂时无法读取本地审核记录；当前页面结果仍可下载为 JSON。")


def _configuration_trace_panel(result: dict) -> None:
    report = result.get("configuration_trace")
    if not isinstance(report, dict) or report.get("status") == "NOT_APPLICABLE":
        return
    st.subheader("显式资料关联")
    if report.get("status") in {"UNAVAILABLE", "NOT_AVAILABLE", "INVALID_RESPONSE"}:
        st.warning("本次未能核验配置引用关系；请人工核对原文中的文件引用。")
        return
    relations = report.get("relations") or []
    gaps = report.get("gaps") or []
    if gaps or (report.get("bounds") or {}).get("truncated"):
        st.warning(f"存在 {len(gaps)} 项引用待核对；未收录或未解析的目标已记入资料缺口。")
    if not relations:
        if not gaps and not (report.get("bounds") or {}).get("truncated"):
            st.caption("所选资料中未发现可解析的配置或文档引用。")
        return
    indexed = sum(row.get("target_status") == "indexed" for row in relations)
    st.caption(f"发现 {len(relations)} 条直接关联，{indexed} 条目标已收录，{len(gaps)} 项待补资料；关联存在仍需核对实际影响。")
    types = {"config_inherits": "配置继承", "config_references": "配置引用", "document_references": "文档引用"}
    states = {"indexed": "已收录", "not_indexed": "未收录", "different_version": "其他版本", "external_repository": "外部来源", "invalid_path": "路径无效"}
    with st.expander("查看关联路径与引用原文"):
        st.dataframe([{
            "关系": types.get(row.get("relation_type"), "待核对"),
            "引用文件": row.get("path"), "目标文件": row.get("target_path"),
            "目标状态": states.get(row.get("target_status"), "待核对"),
            "版本": row.get("version"),
        } for row in relations], width="stretch", hide_index=True)
        for row in relations[:5]:
            evidence = row.get("evidence") or {}
            st.code(str(evidence.get("text") or ""), language="text")
            st.markdown(f"[查看引用所在原文]({row.get('source_url', '')}#L{evidence.get('line_start', 1)})")
        if len(relations) > 5:
            st.caption("完整引用链随人工审核报告导出。")
    if (report.get("bounds") or {}).get("truncated"):
        st.warning("关联数量超过查询范围，报告已标记剩余待核对项。")


def _request_candidates_panel(result: dict, *, standalone: bool = False) -> None:
    _configuration_trace_panel(result)
    advice_sources = result.get("review_advice", {}).get("sources", [])
    retrieved = result.get("retrieved_results", [])
    advice_source_ids = {row.get("chunk_id") for row in advice_sources if row.get("chunk_id")}
    if standalone:
        st.subheader("可能相关资料")
        st.caption("检索与模型引用均为待核对线索。")
        if advice_sources:
            _evidence(advice_sources, heading="模型引用的官方片段")
        elif retrieved:
            _evidence(retrieved, heading="RAG 检索命中（供人工筛查）")
        else:
            st.info("当前没有可供人工筛查的检索资料。")
        remaining = [row for row in retrieved if row.get("chunk_id") not in advice_source_ids]
        if remaining:
            with st.expander(f"其他 RAG 检索命中（{len(remaining)}）"):
                _evidence(remaining, heading="尚未被模型引用的检索资料")
        return

    if not advice_sources:
        if retrieved:
            st.subheader("RAG 检索命中（供人工筛查）")
            st.caption("模型没有给出带有效引用的影响判断；以下结果只是检索线索。")
            _evidence(retrieved, heading="知识库检索结果")
        else:
            st.info("没有检索到可供核对的资料。")
        return

    review = result.get("review_advice", {}).get("review") or {}
    paired_ids = {
        candidate.get("evidence_chunk_id")
        for candidate in review.get("impact_candidates", [])
        if candidate.get("evidence_chunk_id")
    }
    unpaired_sources = [row for row in advice_sources if row.get("chunk_id") not in paired_ids]
    remaining_retrieval = [row for row in retrieved if row.get("chunk_id") not in advice_source_ids]
    if unpaired_sources:
        with st.expander(f"其他模型引用证据（{len(unpaired_sources)}）"):
            _evidence(unpaired_sources, heading="尚未配对到优先候选的引用")
    if remaining_retrieval:
        with st.expander(f"其他未被模型引用的检索命中（{len(remaining_retrieval)}）"):
            _evidence(remaining_retrieval, heading="RAG 检索补充结果")

def _retrieval_trace_panel(result: dict) -> None:
    trace = result.get("retrieval_trace")
    if not isinstance(trace, dict):
        return
    stage_status = result.get("stage_status") or {}
    coverage = result.get("coverage") or {}
    for line in _retrieval_diagnostic_lines(result):
        st.caption(line)
    if coverage:
        language_names = {"zh": "中文", "en": "英文"}
        attempted = "、".join(language_names.get(value, value) for value in coverage.get("attempted_languages", []))
        covered = "、".join(language_names.get(value, value) for value in coverage.get("covered_languages", [])) or "无"
        st.caption(
            f"检查项覆盖 {coverage.get('covered_check_count', 0)}/{coverage.get('required_check_count', 0)}"
            f" · 语言分支：{attempted or '未执行'} · 有命中：{covered} · "
            f"纳入证据 {coverage.get('selected_evidence_count', 0)}/{coverage.get('evidence_budget', 8)}"
        )
    if stage_status:
        stage_labels = {"planning": "问题拆解", "retrieval": "资料检索", "generation": "建议整理"}
        status_labels = {
            "OK": "完成", "EMPTY": "无结果", "FAILED": "失败",
            "SKIPPED": "未执行", "OUT_OF_SCOPE": "超出范围",
        }
        rendered = "　·　".join(
            f"{stage_labels.get(stage, stage)}：{status_labels.get(status, status)}"
            for stage, status in stage_status.items()
        )
        st.caption(f"流程状态　{rendered}")
    gap_details = result.get("evidence_gap_details") or []
    if gap_details:
        st.subheader("待补充核查")
        gap_labels = {
            "NO_REQUIRED_SOURCE": "缺少必要资料",
            "VERSION_AMBIGUITY": "版本适用性待确认",
            "IMAGE_NOT_REVIEWED": "图片证据待核验",
            "OUT_OF_SCOPE": "超出当前资料范围",
            "OUT_OF_SCOPE_PUBLIC_CORPUS": "超出当前资料范围",
            "RETRIEVAL_FAILED": "检索未完成",
            "INVALID_CITATION": "无效引用已移除",
            "MODEL_REPORTED": "模型提示待核验",
        }
        for gap in gap_details:
            gap_type = str(gap.get("gap_type") or "REVIEW_REQUIRED")
            description = str(gap.get("description") or gap.get("message") or "需要人工核查。")
            st.markdown(f"**{gap_labels.get(gap_type, '待核查事项')}** · {escape(description)}")
            missing_source = gap.get("missing_source_type") or gap.get("expected_materials")
            expected_version = gap.get("expected_version")
            metadata = []
            if missing_source:
                metadata.append(f"待补资料：{escape(str(missing_source))}")
            if expected_version:
                metadata.append(f"适用版本：{escape(str(expected_version))}")
            if metadata:
                st.caption("　｜　".join(metadata))
            suggested_query = gap.get("suggested_query")
            if suggested_query:
                st.caption(f"建议核查问题：{escape(str(suggested_query))}")
            action = gap.get("suggested_action")
            if action:
                st.markdown(escape(str(action)))
    with st.expander("检索过程与覆盖范围"):
        st.caption(f"任务编号：{result.get('task_id', '当前会话')} · 模型建议状态：{trace.get('model_status', '未调用')}")
        labels = {
            "candidate_found": "已纳入证据",
            "no_retrieval_match": "未检索到匹配",
            "candidate_outside_evidence_budget": "候选超出证据上限",
            "search_unavailable": "检索服务未完成",
        }
        for index, row in enumerate(trace.get("queries", []), 1):
            language_name = {"zh": "中文", "en": "英文"}.get(row.get("language"), row.get("language", ""))
            language_suffix = f" [{language_name}]" if language_name else ""
            st.markdown(f"{index}. **{labels.get(row.get('status'), '待核对')}**{language_suffix} · {row.get('query', '')}")
            st.caption(f"命中 {len(row.get('top_chunk_ids', []))} 条；纳入模型证据 {len(row.get('selected_chunk_ids', []))} 条。")


def _clear_change_request_results() -> None:
    st.session_state.pop("official_request_review", None)
    st.session_state.pop("official_review", None)
    st.session_state.pop("official_review_decision", None)
    st.session_state.pop("official_review_decision_target", None)
    st.session_state.pop("official_review_decision_at", None)
    st.session_state.pop("official_review_audit_event_id", None)
    st.session_state.pop("official_review_audit_persisted_at", None)
    st.session_state.pop("official_review_audit_error", None)


def _change_request_context_changed() -> None:
    _clear_change_request_results()


def _save_change_request() -> None:
    st.session_state["official_change_request_draft"] = st.session_state.get(
        "official_change_request", ""
    )
    _clear_change_request_results()


def _agent(
    client: PublicKnowledgeClient, ready: bool, docs: list[dict], workspace: dict | None = None,
) -> None:
    _page_header("变更审查", "研发资料变更影响审查", page_key="agent")
    current_version = st.session_state.get("official_current_version")
    version_scope = (
        f"知识库{_version_option_label(current_version, workspace)}"
        if current_version else "当前已收录资料"
    )
    if workspace:
        versions, default_version_index = _review_version_selector(workspace)
    else:
        versions, default_version_index = _published_versions(workspace), 0
    name = _workspace_name(workspace)
    profile = (workspace or {}).get("domain_profile") or {}
    example_queries = [
        str(value) for value in profile.get("example_queries", [])
        if isinstance(value, str) and value.strip()
    ]
    st.caption(f"用自然语言描述研发变更；Agent 调用 RAG 检索{version_scope}，再整理待核对的影响候选和建议。")
    st.info(f"本次分析仅保留在当前会话，不自动修改 {name} 上游项目或公共资料。")
    _remember_document_titles(docs)
    if "official_change_request" not in st.session_state:
        st.session_state["official_change_request"] = st.session_state.get(
            "official_change_request_draft", ""
        )
    summary = st.text_area(
        "描述研发变更", height=120,
        placeholder=(
            f"例如：{example_queries[0]}；请补充目标版本、功能模块和变更前后差异。"
            if example_queries else "说明变更目标版本、功能模块、变更前后行为和希望核查的资料。"
        ),
        key="official_change_request", on_change=_save_change_request,
    )
    type_options = {"自动识别": None}
    for change_type in profile.get("change_types", []):
        if not isinstance(change_type, dict):
            continue
        type_id = str(change_type.get("id") or "").strip()
        label = str(change_type.get("label") or type_id).strip()
        if type_id and label and type_id != "general":
            type_options[label] = type_id
    type_options["其他 / 待识别"] = "general"
    if st.session_state.get("official_change_type") not in type_options:
        st.session_state["official_change_type"] = "自动识别"
    with st.expander("可选：补充结构化变更信息"):
        selected_type = st.selectbox(
            "变更类型", list(type_options), key="official_change_type",
            on_change=_change_request_context_changed,
        )
        impact_scope = st.text_input(
            "影响范围（模块、项目或对象）", max_chars=160,
            placeholder="例如：行人跟踪器、行为识别流水线、模型配置或推理部署",
            key="official_impact_scope", on_change=_change_request_context_changed,
        )
        target_version = st.selectbox(
            "审查目标版本", versions, index=default_version_index,
            format_func=lambda value: _version_option_label(value, workspace),
            key="agent_target_version", on_change=_change_request_context_changed,
        )
        objective = st.text_input(
            "变更目标（选填）", max_chars=800,
            placeholder="这次变更希望解决什么问题？", key="official_change_objective",
            on_change=_change_request_context_changed,
        )
        constraints = st.text_input(
            "约束条件（选填）", max_chars=800,
            placeholder="例如：保持现有接口兼容", key="official_change_constraints",
            on_change=_change_request_context_changed,
        )
        validation_plan = st.text_input(
            "验证计划（选填）", max_chars=800,
            placeholder="例如：运行规划验证器回归测试", key="official_change_validation_plan",
            on_change=_change_request_context_changed,
        )
        st.caption("未填写的目标、约束或验证计划会明确标为待补充；Agent 不会代替你推断事实。")
    device_scope = _device_scope_controls(workspace, prefix="agent")
    retrieval_policy = _retrieval_policy_selector(
        workspace, key="agent_retrieval_policy", label="Agent 检索策略",
    )
    language_mode = "zh"
    st.caption(
        "Agent 保留原始描述，最多拆分 4 项检查；实验策略会批量召回并统一重排，"
        f"最多选取 8 条 {target_version} 中文证据供模型分析。"
    )
    if st.button("检索资料并分析影响", type="primary", disabled=not ready or not summary.strip()):
        with st.spinner(f"正在检索 {name} {target_version} 版本资料并整理影响建议……"):
            result = _request(
                lambda: _analyze_change_request(
                    client, summary, type_options[selected_type], impact_scope,
                    target_version=target_version, objective=objective,
                    constraints=constraints, validation_plan=validation_plan,
                    language_mode=language_mode, device_scope=device_scope,
                    retrieval_policy=retrieval_policy,
                ),
                fallback="变更影响分析暂未完成。",
            )
        if result:
            st.session_state["official_request_review"] = result
            st.session_state.pop("official_review", None)
            st.session_state["official_review_decision"] = None
            st.rerun()

    request_result = st.session_state.get("official_request_review")
    selected_type_code = type_options.get(st.session_state.get("official_change_type", "自动识别"))
    active_request = request_result if _request_context_matches(
        request_result, summary=summary, target_version=target_version,
        objective=objective, constraints=constraints, validation_plan=validation_plan,
        selected_type_code=selected_type_code,
        impact_scope=st.session_state.get("official_impact_scope", ""),
        device_scope=device_scope,
        retrieval_policy=retrieval_policy,
    ) else None
    if active_request:
        stage = 3 if _review_decision_for(active_request) else 2
        _review_steps(stage)
        plan = active_request.get("request_plan") or {}
        st.caption(
            f"变更类型：{plan.get('change_type_label', '待识别')} · "
            f"识别方式：{'人工选择' if plan.get('classification_source') == 'user_selected' else '规则识别'} · "
            f"检索关注点：{plan.get('retrieval_focus', '按原始描述检索')}"
        )
        for warning in plan.get("scope_warnings") or []:
            st.warning(warning)
        context = active_request.get("request_context") or {}
        context_labels = {
            "objective": "变更目标", "constraints": "约束条件",
            "validation_plan": "验证计划",
        }
        st.markdown("**提案上下文**")
        context_cells = [
            ("审查目标版本", context.get("target_version") or plan.get("target_version") or target_version),
            *[
                (label, context.get(key) or value.strip() or "待补充")
                for key, label, value in (
                    ("objective", context_labels["objective"], objective),
                    ("constraints", context_labels["constraints"], constraints),
                    ("validation_plan", context_labels["validation_plan"], validation_plan),
                )
            ],
        ]
        st.dataframe(
            [{"字段": label, "当前内容": value} for label, value in context_cells],
            hide_index=True, use_container_width=True,
        )
        if context.get("missing_fields"):
            missing_labels = [context_labels.get(key, key) for key in context["missing_fields"]]
            st.caption("尚待补充：" + "、".join(missing_labels) + "。这些内容不会由 Agent 推断。")
        st.markdown('<div class="section-rule">本次变更分析</div>', unsafe_allow_html=True)
        _review_advice_panel(active_request, context="变更分析")
        _request_candidates_panel(active_request)
        _retrieval_trace_panel(active_request)
        _review_panel(active_request)

        candidates = active_request.get("retrieved_results", [])
        if candidates:
            with st.expander("可选：针对候选资料制作修改草案"):
                st.caption("需要形成段落建议时，可选择一份候选资料。")
                documents_by_id = {row["document_id"]: row for row in docs if row.get("document_id")}
                candidate_document_ids = list(dict.fromkeys(row["document_id"] for row in candidates))
                for row in candidates:
                    documents_by_id.setdefault(row["document_id"], row)
                if st.session_state.get("official_change_document") not in candidate_document_ids:
                    st.session_state["official_change_document"] = candidate_document_ids[0]
                document_id = st.selectbox(
                    "聚焦一份检索候选资料", candidate_document_ids,
                    key="official_change_document",
                    format_func=lambda key: (
                        f"{documents_by_id[key].get('title') or documents_by_id[key].get('document_key')} · "
                        f"{documents_by_id[key].get('version', '')} · {documents_by_id[key].get('locale', '')}"
                    ),
                )
                chunks = _request(lambda: client.document(document_id), fallback="所选候选资料暂未加载。") if ready else None
                if chunks:
                    by_id = {row["chunk_id"]: row for row in chunks}
                    if st.session_state.get("official_change_chunk") not in by_id:
                        preferred = next((row["chunk_id"] for row in candidates if row["document_id"] == document_id), None)
                        st.session_state["official_change_chunk"] = preferred if preferred in by_id else next(iter(by_id))
                    chunk_id = st.selectbox(
                        "聚焦段落（可选）", list(by_id), key="official_change_chunk",
                        format_func=lambda key: f"{by_id[key].get('heading') or '正文'} · 第 {list(by_id).index(key)+1} 段",
                    )
                    selected = by_id[chunk_id]
                    if st.session_state.get("official_draft_source_id") != chunk_id:
                        st.session_state["official_draft_source_id"] = chunk_id
                        st.session_state["official_review_draft"] = selected["content"]
                        st.session_state["official_proposed_text"] = selected["content"]
                        _clear_stale_review()
                    elif "official_proposed_text" not in st.session_state:
                        st.session_state["official_proposed_text"] = st.session_state.get(
                            "official_review_draft", selected["content"]
                        )
                    with st.expander("查看所选官方原文"):
                        st.write(_rewrite_relative_source_links(
                            _replace_markdown_images(selected["content"]), selected.get("source_url", "")
                        ))
                        st.markdown(f"[阅读原始页面]({selected['source_url']})")
                    proposed = st.text_area(
                        "目标段落草案", height=170, key="official_proposed_text",
                        on_change=_save_review_draft,
                    )
                    exact_retrieval_policy = _retrieval_policy_selector(
                        workspace, key="official_exact_retrieval_policy",
                        label="草案影响检索策略",
                    )
                    if st.button(
                        "生成修改前后对照", type="secondary", key="official_create_patch",
                        disabled=not ready or proposed.strip() == selected["content"].strip(),
                    ):
                        with st.spinner("正在核对段落差异、关联资料与引用依据……"):
                            result = _request(
                                lambda: _analyze_hypothetical(
                                    client, selected, proposed, device_scope=device_scope,
                                    retrieval_policy=exact_retrieval_policy,
                                ),
                                fallback="具体草案核对暂未完成。",
                            )
                        if result:
                            st.session_state["official_review"] = result
                            st.session_state["official_exact_scope"] = device_scope.copy()
                            st.session_state["official_exact_policy"] = exact_retrieval_policy
                            st.session_state["official_review_decision"] = None
                            st.rerun()
        exact_result = st.session_state.get("official_review")
        active_exact = (
            exact_result
            if exact_result
            and exact_result.get("selected_source", {}).get("chunk_id") == st.session_state.get("official_change_chunk")
            and st.session_state.get("official_exact_scope") == device_scope
            and exact_result.get(
                "retrieval_policy_requested", exact_result.get("retrieval_policy", "bm25")
            ) == st.session_state.get("official_exact_retrieval_policy", "bm25")
            else None
        )
        if active_exact:
            st.markdown('<div class="section-rule">具体段落草案核对</div>', unsafe_allow_html=True)
            _review_advice_panel(active_exact, context="草案核对")
            _impact_panel(active_exact)
            _review_evidence_panel(active_exact)
            _patch_panel(active_exact)
            _retrieval_trace_panel(active_exact)
            _review_panel(active_exact)
    else:
        _review_steps(0)
        if not ready:
            notice = workspace_page_readiness_notice(
                workspace, "知识服务正在启动或暂不可用；连接恢复后可直接提交自然语言变更描述。",
            )
            if notice:
                st.info(notice)


def _review_subpage(choice: str) -> None:
    _page_header("变更审查", NAV_PAGE_LABELS[choice], page_key="review", parent=PAGE_PARENTS[choice])
    result = st.session_state.get("official_review") or st.session_state.get("official_request_review")
    if not result:
        st.info("当前会话还没有变更分析结果。请先用自然语言描述变更并检索相关资料。")
        return
    if choice == "可能相关资料":
        if result.get("request_mode") == "natural_language":
            _request_candidates_panel(result, standalone=True)
        else:
            _impact_panel(result)
    elif choice == "修改前后对照":
        if result.get("patch_candidate"):
            _patch_panel(result)
        elif result.get("request_mode") == "natural_language":
            _review_advice_panel(result)
            st.info("尚未为具体段落制作修改草案。返回“新建变更审查”，可从检索候选中选择资料继续。")
    else:
        _review_panel(result)


_OCR_EXAMPLES = {
    "旧版结构分析接口": ("removed_structure_api.py", "from paddleocr import PPStructure\n\ndef extract_layout(image_path):\n    engine = PPStructure()\n    return engine(image_path)\n"),
    "旧版结果消费方式": ("legacy_result_consumer.py", "from paddleocr import PaddleOCR\n\ndef extract_text(image_path):\n    engine = PaddleOCR(lang='ch')\n    result = engine.ocr(image_path)\n    return [line[1][0] for line in result[0]]\n"),
    "基础 OCR 调用": ("compatible_basic.py", "from paddleocr import PaddleOCR\n\ndef recognize(image_path):\n    engine = PaddleOCR(lang='ch')\n    return engine.ocr(image_path)\n"),
}


def _ocr_example(label: str) -> tuple[str, str]:
    filename, fallback = _OCR_EXAMPLES[label]
    sample = Path(__file__).resolve().parents[1] / "examples" / "paddleocr_document_app" / filename
    try:
        return f"app/{filename}", sample.read_text(encoding="utf-8")
    except OSError:
        return f"app/{filename}", fallback


def _clear_ocr_review() -> None:
    job=st.session_state.pop('ocr_job',None)
    if job:job['cancel'].set()
    st.session_state.pop("ocr_compatibility_review", None)
    st.session_state.pop("ocr_review_input_fingerprint", None)
    st.session_state.pop('ocr_disposition_state', None)
    for key in ("official_review_decision", "official_review_decision_target", "official_review_decision_at",
                "official_review_audit_event_id", "official_review_audit_persisted_at", "official_review_audit_error"):
        st.session_state.pop(key, None)


def _select_ocr_example() -> None:
    _clear_ocr_review()
    path, content = _ocr_example(st.session_state.get("ocr_example", next(iter(_OCR_EXAMPLES))))
    st.session_state["ocr_application_path"] = path
    st.session_state["ocr_application_content"] = content


def _pick_ocr_bundle_file():
    rows=st.session_state.get('ocr_bundle_files',[])
    name=st.session_state.get('ocr_bundle_file',rows[0]['path'] if rows else '')
    st.session_state['ocr_bundle_content']=next((r['content'] for r in rows if r['path']==name),'')


def _select_ocr_demo():
    from services.interview_examples import demo_application
    _clear_ocr_review()
    st.session_state['ocr_bundle_files'] = demo_application(st.session_state['ocr_demo_case'])
    st.session_state.pop('ocr_bundle_file', None)


def _edit_ocr_bundle_file():
    for row in st.session_state.get('ocr_bundle_files',[]):
        if row['path']==st.session_state.get('ocr_bundle_file'):
            row['content']=st.session_state.get('ocr_bundle_content','')
    _clear_ocr_review()


def _analyze_ocr_application(client: PublicKnowledgeClient, files: list[dict], *,
                             source_version: str, target_version: str, generate_advice: bool = False,
                             on_progress=None, resume=None,cancelled=None,investigation_progress=None) -> dict:
    agent_root = Path(__file__).resolve().parents[1] / "change-review-agent"
    # The Streamlit entry point is also named app.py. Prefer the Agent package
    # even when PYTHONPATH already includes it after demo-ui.
    if str(agent_root) in sys.path:
        sys.path.remove(str(agent_root))
    sys.path.insert(0, str(agent_root))
    from app.paddleocr_review import PaddleOCRReviewAgent
    return PaddleOCRReviewAgent(client).analyze(
        files, source_version=source_version, target_version=target_version, generate_advice=generate_advice,
        on_progress=on_progress,
        resume=resume,cancelled=cancelled,investigation_progress=investigation_progress,
    )


@st.fragment(run_every=1)
def _ocr_job_progress(input_fingerprint):
    from services.paddleocr_jobs import snapshot
    job=st.session_state.get('ocr_job')
    if not job:return
    if job['identity']!=input_fingerprint:
        job['cancel'].set();st.session_state.pop('ocr_job',None);return
    state=snapshot(job)
    stages={'static_review':'检查应用调用与配置','evidence_validation':'核验固定版本证据','official_lookup':'逐项查询官方资料','model_advice':'核对模型建议'}
    if not state['done']:
        with st.status(stages.get(state.get('stage'),'正在审查'),expanded=True):
            st.write(f"已用 {state['elapsed_seconds']:.1f} 秒；已处理查证项 {state.get('processed',0)}，最多 12 项。")
            if st.button('取消后续查证',key='ocr_cancel_job'):
                job['cancel'].set()
                st.info('已请求取消；当前网络请求结束后停止，保留已完成的静态报告与证据。')
        return
    st.session_state.pop('ocr_job',None)
    try:result=job['future'].result()
    except (ValueError,ServiceError) as exc:st.session_state['ocr_job_error']=str(exc)
    except Exception:st.session_state['ocr_job_error']='任务失败，未生成可核验报告。请重试。'
    else:
        st.session_state['ocr_compatibility_review']=result
        st.session_state['ocr_review_input_fingerprint']=input_fingerprint
    st.rerun()


def _ocr_findings(result: dict, *, show_related: bool = False) -> None:
    from components.reading_layout import paragraph_blocks, unique_citations
    report = result["compatibility_report"]
    priority = {"supported_risk": 0, "needs_verification": 1, "unaffected": 2}
    findings = sorted(report.get("findings") or [], key=lambda row: priority.get(row.get("status"), 1))
    labels = {"supported_risk": "有证据支持的静态风险", "needs_verification": "待验证",
              "unaffected": "基础调用形式未发现变更"}
    for index, finding in enumerate(findings):
        location = finding["application"]
        with (st.expander(f"调用形式未发现变更，仍需回归 · {finding['title']}")
              if finding['status'] == 'unaffected' else st.container(border=True, key=f"ocr_finding_{index}")):
            status_label = labels.get(finding['status'], '待验证')
            if finding['status'] == 'supported_risk':
                st.error(status_label, icon='🔴')
            elif finding['status'] == 'needs_verification':
                st.warning(status_label)
            else:
                st.caption(status_label + ' · 仍需运行回归')
            st.markdown(f"**{finding['title']}**")
            st.caption(f"应用位置：{location['path']} · 第 {location['line']}–{location['end_line']} 行")
            st.markdown('**下一步核查**')
            for paragraph in paragraph_blocks(finding['desired_check']):
                st.markdown(paragraph)
            with st.expander('原因与应用代码'):
                for paragraph in paragraph_blocks(finding['explanation']):
                    st.markdown(paragraph)
                st.code(location["snippet"], language="python" if location["path"].endswith(".py") else "yaml", wrap_lines=True)
            citations = unique_citations(finding['evidence'])
            with st.expander(f'双版本依据 · {len(citations)} 条来源'):
                before, after = st.columns(2, gap="medium")
                for column, version, label in ((before, report["source_version"], "当前依赖证据"),
                                               (after, report["target_version"], "升级目标证据")):
                    with column:
                        st.markdown(f"**{label} · {version}**")
                        version_citations = [row for row in citations if row['version'] == version]
                        if not version_citations:
                            st.caption("本项缺少该版本证据，需人工补查。")
                        for citation in version_citations:
                            source_line = citation['line_start']
                            line_url = f"{citation['url']}#L{source_line}-L{citation['line_end']}"
                            st.markdown(f"[{citation.get('title') or citation['path']}]({line_url})")
                            st.caption(f"{citation['path']} · 第 {source_line}–{citation['line_end']} 行")
                            st.caption("官方接口 / 配置契约" if citation['path'].endswith(('.py', '.yml', '.yaml')) else "官方中文文档")
                st.caption('来源提交与文件哈希（按上述来源顺序）')
                for citation in citations:
                    st.code(f"{citation['version']} · {citation['path']}:{citation['line_start']}–{citation['line_end']}\ncommit={citation['commit']}\nsha256={citation['sha256']}", language='text', wrap_lines=True)
    if not findings:
        st.info("当前文本没有形成可绑定双版本证据的影响项；请查看待验证项。")
    if show_related:
        related = result.get("related_materials") or []
        if related:
            st.subheader("相关官方中文资料")
            st.caption("补查资料供人工阅读；检索候选不增加或改变静态影响结论。")
            _evidence(related, heading="同一知识空间中的补查资料")
        for gap in result.get("lookup_gaps") or []:
            st.warning(gap)


def _ocr_report(result: dict) -> None:
    report = result["compatibility_report"]
    status_labels = {"supported_risk": "存在有证据支持的静态升级风险", "needs_verification": "当前静态证据不足，需要补充验证",
                     "unaffected": "基础调用形式未发现变更，仍需实际推理验证"}
    st.subheader("应用兼容性静态报告")
    st.write(f"{report['source_version']} → {report['target_version']}：{status_labels[report['status']]}")
    st.caption("实际 OCR 推理未验证。静态报告不证明运行环境、模型下载、文档质量或下游输出已兼容。")
    risks = [f for f in report.get('findings', []) if f['status'] == 'supported_risk']
    unresolved = (result.get('investigation') or {}).get('unresolved', [])
    st.caption(f"静态风险 {len(risks)} 处 · 待验证记录 {len(report.get('gaps') or [])} 项 · 未完成查证 {len(unresolved)} 项")
    if risks:
        st.warning(f'先处理 {len(risks)} 处已定位的静态升级风险，再执行目标环境回归。')
    else:
        st.info('当前支持范围内未定位已证实的升级风险；不能据此确认兼容。')
    st.markdown('**风险位置与版本依据**')
    _ocr_findings(result)
    if unresolved:
        reasons = {'required_source_not_verified': '缺少核验通过的固定版本来源',
                   'content_requirements_not_covered': '返回片段没有覆盖必要契约事实',
                   'content_criteria_not_defined': '尚无可自动关闭的核查标准',
                   'application_validation_required': '需补充应用资料或在内部环境回归，官方资料不能代替验证'}
        st.markdown('**尚未完成的查证**')
        grouped_reasons = {}
        for gap in unresolved:
            versions = grouped_reasons.setdefault(gap['reason'], set())
            versions.update(gap['versions'])
        for reason, versions in grouped_reasons.items():
            st.markdown(f"- **{reasons.get(reason, reason)}**（{', '.join(sorted(versions))}）")
        st.caption('按缺口原因合并展示；逐项义务与编号保留在查证轨迹中，义务数不等于业务风险数。')
    from components.paddleocr_review import impact_panel
    impact_panel(result)
    st.markdown("**待验证项与证据缺口**")
    groups={}
    for gap in report.get('gaps') or []:
        groups.setdefault((gap.get('application') or {}).get('path','任务范围'),[]).append(gap)
    if groups:st.caption(f"{len(report['gaps'])} 项待验证记录，涉及 {len(groups)} 个文件或范围。")
    for filename,items in groups.items():
        with st.expander(f'{filename} · {len(items)} 项待验证'):
            for gap in items:
                location=gap.get('application') or {}
                prefix=f"第 {location['line']} 行：" if location else ''
                st.write(prefix+gap['detail'])
                if location.get('snippet'):st.code(location['snippet'],language='python')
    if not report.get("gaps"):
        st.caption("本次受限静态范围未记录额外证据缺口；实际推理仍待验证。")
    with st.expander('开发者完整验证清单'):
        for step in dict.fromkeys(report.get("verification_steps") or []):
            st.markdown(f"- {step}")
    suggestions = result.get("model_suggestions") or []
    if suggestions:
        with st.expander(f'模型核查建议（待人工确认）· {len(suggestions)} 项'):
            st.caption("模型建议未验证，不改变上述静态发现或实际推理状态。")
            for row in suggestions:
                source = row["evidence"]
                st.write(row["reason"])
                st.markdown(f"建议核查：{row['suggested_action']} · [目标版出处]({source['source_url']})")
    elif result.get("model_status") not in {None, "NOT_REQUESTED"}:
        explanations = {
            "GENERATION_RESPONSE_INVALID": "模型已返回结果，但建议结构或引用未通过校验，已隐藏建议。",
            "GENERATION_RESPONSE_TRUNCATED": "模型回复达到输出长度上限，建议未完整生成。",
            "GENERATION_NOT_CONFIGURED": "建议生成服务尚未配置。",
            "GENERATION_AUTH_FAILED": "模型服务拒绝认证，请由管理员核对后端密钥。",
            "GENERATION_BILLING_REQUIRED": "模型服务返回余额或计费问题。",
            "GENERATION_RATE_LIMITED": "模型服务返回限流，请稍后重试。",
            "GENERATION_PROVIDER_TIMEOUT": "等待模型服务回复超时。",
            "GENERATION_PROVIDER_UNAVAILABLE": "本次建议生成请求未完成。",
            "ABSTAINED": "本次证据不足以生成可绑定出处的核查建议。",
            "NO_VERIFIED_TARGET_EVIDENCE": "本次没有已核验的目标版证据，不调用模型生成建议。",
        }
        st.info(explanations.get(result.get("model_status"), "本次未获得可绑定目标版证据的模型核查建议。")
                + "静态报告仍可审阅。")
    generation = result.get("model_generation") or {}
    if generation.get("requested_model") or generation.get("returned_model") or generation.get("usage"):
        with st.expander("本次建议生成记录"):
            st.write(f"请求模型：{generation.get('requested_model') or '未确认'}")
            st.write(f"实际模型：{generation.get('returned_model') or '未返回'}")
            st.write(f"Token 用量：{generation.get('usage') or '未返回'}")
    st.download_button("下载静态报告（JSON）", data=json.dumps(result, ensure_ascii=False, indent=2),
                       file_name=f"paddleocr-static-{result.get('task_id', 'review')}.json", mime="application/json",
                       key="ocr_static_report_export")


def _ocr_agent(client: PublicKnowledgeClient, ready: bool, workspace: dict | None) -> None:
    _page_header("变更审查", "应用依赖升级兼容性审查", page_key="ocr_agent")
    st.write("审查文档处理应用从 PaddleOCR v2.9.1 升级到 v3.0.0 后的接口、配置和结果消费方式。")
    st.info('可直接运行默认案例：系统已装载原创应用文件，无需安装 OCR。也可选择修改后复查或缺资料案例。')
    with st.expander('需要人工完成什么？'):
        st.write('核对提交文件和风险证据 → 确认或退回报告 → 在授权开发环境修改并执行回归 → 负责人决定是否升级。')
        st.caption('已审阅不等于批准升级。系统不执行上传代码、不自动修改应用；默认案例的静态结论不证明用户应用运行成功。')
    app_name = st.text_input('内部应用名称（演示标签）', value='文档处理应用', max_chars=80,
                             key='ocr_internal_app_name', on_change=_clear_ocr_review)
    change_reason = st.text_input('升级目的与验收说明', value='保留中文 OCR 归一化与下游 JSON 契约',
                                  max_chars=1000, key='ocr_change_reason', on_change=_clear_ocr_review)
    application_context = {'application_name': app_name, 'application_version': None,
                           'change_reason': change_reason, 'source_identity': 'student_poc_application',
                           'business_scope': 'internal_document_processing'}
    st.caption('内部应用版本未登记。依赖版本独立管理；默认原创应用使用 OCR 主链，PPStructure 仅是相关接口检查分支。')
    before, after = st.columns(2, gap="medium")
    with before:
        source_version = st.selectbox("应用当前依赖", ["v2.9.1"], key="ocr_source_version")
    with after:
        target_version = st.selectbox("升级目标依赖", ["v3.0.0"], key="ocr_target_version")
    from services.paddleocr_application_inputs import load_original_application, parse_application_inputs, application_fingerprint
    mode=st.selectbox('应用输入',['完整文档处理应用','规则片段'],key='ocr_input_mode',on_change=_clear_ocr_review)
    if mode=='完整文档处理应用':
        from services.interview_examples import DEMO_CASES, demo_application
        label = st.selectbox('应用演示案例', list(DEMO_CASES), key='ocr_demo_case', on_change=_select_ocr_demo)
        st.caption(DEMO_CASES[label])
        if 'ocr_bundle_files' not in st.session_state:
            st.session_state['ocr_bundle_files']=demo_application(label)
        rows=st.session_state['ocr_bundle_files']
        names=[r['path'] for r in rows]
        if st.session_state.get('ocr_bundle_file') not in names:
            st.session_state['ocr_bundle_file']=names[0]
            _pick_ocr_bundle_file()
        st.selectbox('应用文件',names,key='ocr_bundle_file',on_change=_pick_ocr_bundle_file)
        st.text_area('文件文本',height=260,max_chars=50000,key='ocr_bundle_content',on_change=_edit_ocr_bundle_file)
        st.caption(f'当前 {len(rows)} 个文件；路径仅作相对标签，不读取或执行你的应用。')
        uploaded=st.file_uploader('导入 UTF-8 应用文件',type=['py','json','md','txt','yaml','yml'],accept_multiple_files=True,key='ocr_uploads')
        if uploaded and st.button('使用上传文件',key='ocr_import_files'):
            try:
                new_rows=parse_application_inputs([{'path':u.name,'content':u.getvalue().decode('utf-8-sig')} for u in uploaded])
                st.session_state['ocr_bundle_files']=new_rows
                st.session_state.pop('ocr_bundle_file',None)
                _clear_ocr_review();st.rerun()
            except (ValueError,UnicodeDecodeError) as exc:st.error(str(exc))
        remove,reset=st.columns(2)
        if remove.button('移除当前文件',disabled=len(rows)<=1,key='ocr_remove_file'):
            st.session_state['ocr_bundle_files']=[r for r in rows if r['path']!=st.session_state['ocr_bundle_file']]
            st.session_state.pop('ocr_bundle_file',None);_clear_ocr_review();st.rerun()
        if reset.button('恢复当前演示案例',key='ocr_reset_bundle'):
            st.session_state['ocr_bundle_files']=demo_application(label)
            st.session_state.pop('ocr_bundle_file',None);_clear_ocr_review();st.rerun()
        files=rows
        content=''.join(r['content'] for r in rows)
    else:
        st.selectbox('原创应用示例',list(_OCR_EXAMPLES),key='ocr_example',on_change=_select_ocr_example)
        if 'ocr_application_content' not in st.session_state:_select_ocr_example()
        path=st.text_input('应用文件相对路径',max_chars=160,key='ocr_application_path',on_change=_clear_ocr_review)
        content=st.text_area('应用文件文本',height=240,max_chars=50000,key='ocr_application_content',on_change=_clear_ocr_review)
        files=[{'path':path,'content':content}]
    available = bool((workspace or {}).get("generation_available"))
    generate = st.checkbox("生成核查建议", value=available, disabled=not available,
                           key="ocr_generate_advice", on_change=_clear_ocr_review,
                           help="可选模型建议仅使用已核验的目标版官方证据，不改变静态报告。")
    try:
        files=parse_application_inputs(files)
        input_fingerprint=application_fingerprint(files,(source_version,target_version),
            {'generate_advice':generate, 'application_context':application_context},(workspace or {}).get('corpus_fingerprint'))
    except ValueError as exc:
        st.error(str(exc));_clear_ocr_review();return
    previous=st.session_state.get('ocr_compatibility_review')
    active_job=st.session_state.get('ocr_job')
    stale_report=previous and st.session_state.get('ocr_review_input_fingerprint')!=input_fingerprint
    stale_job=active_job and active_job['identity']!=input_fingerprint
    if stale_report or stale_job:
        _clear_ocr_review()
        previous=None
    resume_clicked=False
    if previous and previous.get('investigation',{}).get('unresolved'):
        resume_clicked=st.button('继续未完成查证',key='ocr_resume_job',
            disabled=previous['investigation']['elapsed_seconds']>=240 or bool(st.session_state.get('ocr_job')))
    run_clicked=st.button('审查应用兼容性',type='primary',disabled=not ready or not content.strip() or bool(st.session_state.get('ocr_job')),key='ocr_run_review')
    if run_clicked or resume_clicked:
        if not _validate_submission_workspace(client, workspace):
            return
        from services.paddleocr_jobs import start_job
        from concurrent.futures import TimeoutError as FutureTimeout
        import copy
        restored=previous.get('resume_state') if resume_clicked else None
        _clear_ocr_review()
        st.session_state.pop('ocr_job_error',None)
        submitted=copy.deepcopy(files)
        def work(cancelled,update):
            result = _analyze_ocr_application(client,submitted,source_version=source_version,target_version=target_version,
                generate_advice=generate,on_progress=update,investigation_progress=update,cancelled=cancelled,resume=restored)
            result['application_context'] = application_context
            return result
        job=start_job(work,input_fingerprint);st.session_state['ocr_job']=job
        # Fast local checks finish in-place; longer requests are polled by the fragment.
        try:result=job['future'].result(timeout=1)
        except FutureTimeout:pass
        except (ValueError,ServiceError) as exc:
            st.session_state['ocr_job_error']=str(exc);st.session_state.pop('ocr_job',None)
        else:
            st.session_state.pop('ocr_job',None)
            st.session_state['ocr_compatibility_review']=result
            st.session_state['ocr_review_input_fingerprint']=input_fingerprint
    if st.session_state.get('ocr_job'):_ocr_job_progress(input_fingerprint)
    if st.session_state.get('ocr_job_error'):st.error(st.session_state['ocr_job_error'])
    result = st.session_state.get("ocr_compatibility_review")
    if result and st.session_state.get("ocr_review_input_fingerprint") == input_fingerprint:
        _review_steps(3 if _review_decision_for(result) else 2)
        _ocr_report(result)
        _review_panel(result)
    elif not ready:
        notice = workspace_page_readiness_notice(workspace, "知识服务暂不可用，连接恢复后可审查应用文本。")
        if notice:
            st.info(notice)


def _ocr_review_subpage(choice: str, workspace: dict) -> None:
    _page_header("变更审查", _page_label(choice, workspace), page_key="ocr_review", parent="新建变更审查")
    result = st.session_state.get("ocr_compatibility_review")
    if not result or result.get("workspace_id") != "paddleocr":
        st.info("先提交应用文本，生成带官方证据的静态兼容性报告。")
        st.button("发起兼容性审查", on_click=_navigate, args=("新建变更审查",))
        return
    _review_steps(3 if _review_decision_for(result) else 2)
    if choice == "可能相关资料":
        _ocr_findings(result, show_related=True)
    elif choice == "修改前后对照":
        _ocr_report(result)
    else:
        _ocr_report(result)
        _review_panel(result)


def _versions(client: PublicKnowledgeClient, ready: bool, workspace: dict | None) -> None:
    _page_header("知识服务", "版本与历史", page_key="versions")
    if not ready:
        notice = workspace_page_readiness_notice(
            workspace, "知识服务暂不可用，连接恢复后可查看各版本的真实资料。",
        )
        if notice:
            st.info(notice)
        return
    docs = _request(client.documents, fallback="版本资料目录暂不可用。") or []
    if (workspace or {}).get("workspace_id") == "pphuman":
        st.write("每组资料固定到 PaddleDetection 对应的正式 release tag；可按版本回看 PP-Human 教程和配置变更。")
        snapshots = (workspace or {}).get("snapshots") or []
        columns = st.columns(3, gap="medium")
        for index, snapshot in enumerate(snapshots):
            version = str(snapshot.get("version") or "未标记版本")
            members = [row for row in docs if row.get("version") == version]
            with columns[index % len(columns)]:
                with st.container(border=True, key=f"pphuman_snapshot_{version}"):
                    st.markdown(f"### {version}　{snapshot.get('label', '')}")
                    st.caption(
                        f"{len(members)} 份版本资料 · 固定提交 `{snapshot.get('commit') or '未声明'}`"
                    )
                    for row in members[:4]:
                        st.markdown(f"- {row.get('title') or row['document_key']} · [查看官方原文]({row['source_url']})")
                    if len(members) > 4:
                        with st.expander(f"查看其余 {len(members) - 4} 份资料"):
                            for row in members[4:]:
                                st.markdown(f"- {row.get('title') or row['document_key']} · [查看官方原文]({row['source_url']})")
        st.caption("这些是各 release 的文档快照；项目未随此工作台下载模型权重、样例视频或第三方数据集。")
        return
    if (workspace or {}).get("workspace_id") == "edge_ai_device":
        st.write("资料快照记录语料的固定来源版本；JetPack/L4T 与设备型号是检索范围，不代表历史语料快照。")
        snapshots = (workspace or {}).get("snapshots") or []
        for index, snapshot in enumerate(snapshots):
            version = str(snapshot.get("version") or _confirmed_current_version(workspace) or "当前快照")
            with st.container(border=True, key=f"edge_snapshot_{index}"):
                st.markdown(f"### 当前固定中文资料快照 · `{version}`")
                details = [
                    f"资料数：{snapshot.get('source_count', len(docs))}",
                    f"仓库：{snapshot.get('repository') or (workspace or {}).get('repository', '未声明')}",
                    f"来源提交：`{snapshot.get('commit') or '未声明'}`",
                ]
                if snapshot.get("captured_at_date"):
                    details.append(f"固定日期：{snapshot['captured_at_date']}")
                st.caption("　·　".join(details))
        st.markdown("**资料中标注的设备与软件范围**")
        for key, label in (
            ("hardware_models", "设备型号"), ("module_skus", "模组 SKU"),
            ("carrier_boards", "载板"), ("software_baselines", "JetPack / L4T 基线"),
        ):
            values = (workspace or {}).get(key) or []
            if values:
                st.write(f"**{label}：** " + "、".join(str(value) for value in values))
        st.caption("这些是来源资料标注的适用范围；出现某个版本或型号不等于已验证兼容。")
        if docs:
            with st.expander(f"查看固定快照中的资料（{len(docs)}）"):
                for row in docs:
                    title = row.get("title") or row.get("document_key") or "公开资料"
                    st.markdown(f"- {title} · [阅读原始页面]({row['source_url']})")
        return
    st.write("仅展示已收录的公开资料快照；资料快照与软件发行版本分别管理。")
    baseline = workspace.get("baseline_version") if workspace else None
    current = _confirmed_current_version(workspace)
    latest_source_time = _format_snapshot_timestamp(
        workspace.get("latest_source_retrieval_timestamp") if workspace else None
    )
    if latest_source_time:
        st.caption(f"最近收录：{latest_source_time} · 新资料入库后即可检索。")
    first, second = st.columns(2, gap="medium")
    for column, version, label in (
        (first, baseline, _version_option_label(baseline, workspace) if baseline else "历史基线"),
        (second, current, _version_option_label(current, workspace) if current else "最新已收录"),
    ):
        if not version:
            continue
        with column:
            with st.container(border=True):
                st.markdown(f"### {version}　{label}")
                members = _scope_members(version, workspace)
                matching = [row for row in docs if members is None or row.get("version") in members]
                st.write(f"知识库收录 {len(matching)} 份该版本资料。")
                for row in matching[:5]:
                    st.markdown(
                        f"- {row.get('title') or row['document_key']} · "
                        f"[阅读原始页面]({row['source_url']})"
                    )
                    if row.get("rendered_url"):
                        st.markdown(f"  [阅读排版页面]({row['rendered_url']})")
                if len(matching) > 5:
                    with st.expander(f"查看其余 {len(matching)-5} 份资料"):
                        for row in matching[5:]:
                            st.markdown(
                                f"- {row.get('title') or row['document_key']} · "
                                f"[阅读原始页面]({row['source_url']})"
                            )
                            if row.get("rendered_url"):
                                st.markdown(f"  [阅读排版页面]({row['rendered_url']})")
    st.info("版本对照请在知识检索中选择“全部已收录版本”。")
    st.button("进入知识检索", on_click=_navigate, args=("版本检索与问答",))


def _sources(client: PublicKnowledgeClient, ready: bool, workspace: dict | None) -> None:
    _page_header("知识服务", "资料来源", page_key="sources")
    name = _workspace_name(workspace)
    profile = (workspace or {}).get("domain_profile") or {}
    source_scope = profile.get("source_scope") or (workspace or {}).get("data_origin")
    st.write(source_scope or f"资料来自 {name} 的公开来源固定快照。")
    coverage = _source_coverage_text(workspace)
    if coverage:
        with st.expander("查看收录范围与语言对应情况"):
            st.write(coverage)
    if not ready:
        notice = workspace_page_readiness_notice(
            workspace, "知识服务暂不可用，资料目录将在连接恢复后显示。",
        )
        if notice:
            st.info(notice)
        return
    docs = _request(client.documents, fallback="官方资料目录暂不可用。") or []
    version = st.selectbox("资料版本", [*_published_versions(workspace), "all"], format_func=lambda x: _version_option_label(x, workspace), key="source_version")
    term = st.text_input("按资料名称或工程标识筛选", key="source_filter")
    filtered = [
        row for row in docs
        if (version == "all" or row.get("version") in (_scope_members(version, workspace) or set()))
        and term.casefold() in (row.get("title", "") + " " + row.get("document_key", "")).casefold()
    ]
    st.caption(f"当前条件下有 {len(filtered)} 份固定来源资料。")
    kind = {
        "official_documentation": "公开工程文档",
        "community_translation": "中文社区译本",
        "github_release": "官方 Release",
        "github_issue": "官方 Issue",
        "github_pull_request": "官方 PR",
    }
    for index, row in enumerate(filtered[:12]):
        with st.container(border=True, key=f"source_row_{index}"):
            st.markdown(f"**{row.get('title') or row['document_key']}**")
            st.caption(f"{_version_option_label(row.get('version', ''), workspace)}　｜　{row.get('locale', '')}　｜　{kind.get(row.get('source_type'), '公开资料')}")
            st.markdown(f"[阅读原始页面]({row['source_url']})")
            if row.get("rendered_url"):
                st.markdown(f"[阅读排版页面]({row['rendered_url']})")
            if row.get("source_type") == "community_translation":
                st.caption("社区维护的中文译本；此链接固定到收录提交。")
    if len(filtered) > 12:
        with st.expander(f"查看其余 {len(filtered)-12} 份资料"):
            for row in filtered[12:]:
                st.markdown(
                    f"- {row.get('title') or row['document_key']} · "
                    f"[阅读原始页面]({row['source_url']})"
                )


def _pphuman_development_evaluation(workspace: dict) -> None:
    report = workspace.get("development_evaluation")
    if (not isinstance(report, dict) or report.get("assessment_type") != "dev_diagnostic_only"
            or report.get("default_promotion_eligible") is not False):
        st.warning("当前服务未返回匹配语料与代码指纹的开发集诊断；不展示旧构建的分数。")
        return
    counts = report.get("denominators") or {}
    st.info("以下是当前构建的开发集诊断，尚非独立测试结果，不用于宣称回答准确率或自动切换默认策略。")
    st.caption(f"RAG：{counts.get('rag_answerable', 0)} 个可回答问题，另有 {counts.get('rag_unanswerable', 0)} 个语料外问题；只评来源覆盖。")
    rows = [{
        "策略": "BM25" if name.startswith("bm25@") else "术语扩展 RRF",
        "Top-K": values.get("top_k"), "必需来源召回": values.get("mean_required_source_recall"),
        "完整来源集率": values.get("complete_required_source_set_rate"),
        "来源 nDCG@K": values.get("mean_ndcg_at_k"),
    } for name, values in (report.get("rag") or {}).items() if values.get("top_k") in {5, 10}]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"Agent：{counts.get('agent_answerable', 0)} 个有标签任务，每项查询 Top-5；以下为最终证据整理前的检索覆盖。")
    agent_rows = [{
        "策略": "BM25" if name == "bm25" else "术语扩展 RRF",
        "必需来源召回": values.get("required_evidence_source_recall"),
        "完整来源集率": values.get("complete_required_evidence_set_rate"),
    } for name, values in (report.get("agent") or {}).items()]
    if agent_rows:
        st.dataframe(agent_rows, width="stretch", hide_index=True)
    st.warning("模型重排、回答事实性、引用支持和实际影响判断尚未完成独立盲评；开发集分数不能证明新问题上的泛化。")
    st.caption("语料外问题仍可能有检索候选；是否正确拒答需单独评估。当前保留 BM25 默认，可手动对比 RRF 和模型重排。")


def _benchmark(workspace: dict | None) -> None:
    _page_header("系统说明", "检索评测", page_key="benchmark")
    st.subheader("当前语料评测状态")
    if not workspace:
        st.info("知识服务尚未连接，当前无法确认语料与检索策略。")
        return

    status = str(workspace.get("retrieval_evaluation_status") or "pending")
    policy = str(workspace.get("retrieval_policy") or "由服务配置").upper()
    st.markdown(f"**当前策略：{policy}**")
    if _is_paddleocr(workspace):
        current=workspace.get('impact_evaluation')
        rag_quality=workspace.get('rag_quality_evaluation')
        if isinstance(rag_quality,dict):
            st.caption(f"问答默认策略：{rag_quality['selected_strategy']}；上方服务策略为保留的基础检索策略。")
            st.subheader('当前 RAG 验证结果')
            rows=[]
            for name,splits in rag_quality['retrieval'].items():
                for split,value in splits.items():
                    if value.get('evaluation_status','complete')!='complete':continue
                    m=value['metrics']
                    rows.append({'策略':name,'题集':{'dev':'开发','regression':'已知回归','validation':'验证','sealed':'首次封存检验','holdout':'分组回归留出'}.get(split,split),
                                 '完整证据题':f"{m['complete_count']} / {m['count']}",
                                 '非目标模块片段':m['wrong_module_count'],'错版片段':m['wrong_version_count'],
                                 '执行情况':value.get('execution','—')})
            st.dataframe(rows,hide_index=True,width='stretch')
            st.caption(rag_quality['limitations'])
        if isinstance(current,dict):
            st.subheader('当前语料与实现的冻结评测')
            rows=[]
            for name,splits in current['retrieval'].items():
                for split,values in splits.items():
                    m=values['metrics']
                    rows.append({'策略':name,'题集':{'dev':'开发','validation':'验证','sealed':'首次封存'}.get(split,split),
                        '完整证据题':f"{m['complete_count']} / {m['count']}",'必需事实':f"{m['fact_hit_count']} / {m['fact_count']}",
                        '错版片段':m['wrong_version_count'],'非目标模块片段':m['wrong_module_count']})
            st.dataframe(rows,hide_index=True,width='stretch')
            st.write('默认选型：'+current['selected_strategy']+'；'+current['promotion_reason'])
            review=current.get('review',[])
            st.write(f"应用组织变体 {len(review)} 个；按规则类别计，漏报 {sum(r['C_rag']['false_negative_count'] for r in review)} 项，误报 {sum(r['C_rag']['false_positive_count'] for r in review)} 项。")
            st.caption('开发过程中 AI 辅助原文标注，按章节分组，尚无人类业务专家独立复核。证据覆盖不等于答案正确率；模型辅助消融尚未执行，上传应用的实际运行也未执行。')
            with st.expander('当前评测的限制与失败项'):st.json(current.get('limitations'))
        st.write("当前 PaddleOCR 范围采用来源派生问题与固定静态案例验收。")
        st.info("来源命中只衡量检索覆盖；静态规则验收不代表实际 OCR 运行准确率，也不代表新应用上的影响判断准确率。")
        st.caption("固定静态案例：已移除结构分析 API、旧结果消费方式、基础 OCR 调用。用户应用运行验证和人工盲审需单独完成。")
        quality=workspace.get('quality_comparison')
        if isinstance(quality,dict):
            st.subheader('同语料检索策略对比')
            labels={'bm25':'BM25','semantic':'中文语义召回','bm25_rerank':'BM25 + 重排',
                    'hybrid':'混合 RRF','hybrid_rerank':'混合 RRF + 重排'}
            rows=[]
            for name,values in quality.get('strategies',{}).items():
                for split,title in (('dev','开发集'),('holdout','内部留出集')):
                    metrics=values.get(split,{})
                    rows.append({'策略':labels.get(name,name),'题集':title,'问题数':metrics.get('count'),
                                 '完整事实召回':f"{metrics.get('complete_rate',0):.1%}",
                                 '必需事实召回':f"{metrics.get('fact_recall',0):.1%}",
                                 '错版片段数':metrics.get('wrong_version_count')})
            if rows:st.dataframe(rows,width='stretch',hide_index=True)
            st.caption(f"开发集选型：{labels.get(quality.get('selected_on_dev'),quality.get('selected_on_dev'))}；统一 Top-{quality.get('top_k',5)}，先按完整召回、再按事实召回选型。留出集不用于选型。")
            st.caption('题目由开发者根据固定来源标注；不是独立专家盲测。事实片段命中不等于答案准确率，重排分数也不是正确概率。')
        probes=workspace.get('independent_probes')
        if isinstance(probes,dict):
            st.markdown('**选型后的一次性补充探针**')
            st.write(f"检索完整命中 {probes.get('retrieval_complete')} / {probes.get('retrieval_count')}；受限静态审查符合预期 {probes.get('static_passed')} / {probes.get('static_count')}。")
            st.caption('独立代码审查者按原文另行编写；未据此修改策略或标签。审查者了解实现，这不是行业专家盲测，也不代表任意应用兼容。')
        runtime=workspace.get('upgrade_demo')
        if isinstance(runtime,dict):
            st.subheader('原创应用真实升级回归')
            rows=[{'依赖版本':r.get('paddleocr_version'),
                   '旧结果消费方式':'可运行' if r.get('legacy_consumer_success') else '失败',
                   '版本适配器三页回归':'通过' if r.get('sample_regression_passed') else '未通过'}
                  for r in runtime.get('reports',[])]
            if rows:st.dataframe(rows,width='stretch',hide_index=True)
            st.caption('两页原创中文文档 + 一页空白页。真实推理只验证此样例，不代表用户上传应用已运行，也不是识别准确率测试。旧、新版使用不同 OCR 模型，不作质量或延迟对比。')
            with st.expander('升级回归环境与记录'):st.json(runtime)
        report = workspace.get("retrieval_evaluation")
        if (isinstance(report, dict) and report.get("project_id") == "paddleocr"
                and report.get("assessment_type") == "source_derived_acceptance"):
            strategy_labels = {"bm25": "BM25 基线", "bm25_fields": "字段 BM25", "rrf": "RRF"}
            metrics = []
            for strategy, values in (report.get("retrieval") or {}).items():
                if not isinstance(values, dict):
                    continue
                count, hits, wrong = values.get("count"), values.get("evidence_span_hit_count"), values.get("wrong_version_count")
                if (all(type(value) is int for value in (count, hits, wrong))
                        and 0 <= hits <= count and count > 0 and wrong >= 0):
                    metrics.append({"策略": strategy_labels.get(strategy, strategy),
                                    f"命中事实片段（Top-{report.get('top_k', 5)}）": f"{hits} / {count}", "错版数": wrong})
            if metrics and not isinstance(quality,dict):
                st.dataframe(metrics, width="stretch", hide_index=True)
                st.caption("策略使用同一冻结题目范围；命中事实片段不能替代回答事实性、引用支持或业务准确率评测。")
            case_labels = {"removed-api": "旧版结构分析接口", "result-consumer": "旧版结果消费方式", "basic-surface": "基础 OCR 调用"}
            status_labels = {"supported_risk": "静态风险有证据支持", "needs_verification": "待验证", "unaffected": "基础调用形式未发现变更"}
            cases = [{"原创应用案例": case_labels.get(row.get("id"), row.get("id", "")),
                      "静态结果": status_labels.get(row.get("status"), "待验证"),
                      "验收": "通过" if row.get("passed") is True else "未通过",
                      "实际推理": "未验证"}
                     for row in report.get("compatibility") or [] if isinstance(row, dict) and row.get("runtime_verified") is False]
            if cases:
                st.dataframe(cases, width="stretch", hide_index=True)
            if not isinstance(quality,dict):
                st.caption(str(report.get("selection_decision") or "当前保留 BM25 基线，不自动切换策略。"))
            with st.expander("当前构建验收记录"):
                st.json(report)
        elif not isinstance(current,dict) and not isinstance(quality,dict):
            st.caption("当前服务尚未返回与 PaddleOCR 活动语料绑定的评测记录。")
        st.caption(f"{workspace.get('source_count', 0)} 份登记来源 · {workspace.get('chunk_count', 0)} 个检索片段")
        return
    if workspace.get("workspace_id") == "pphuman":
        _pphuman_development_evaluation(workspace)
        st.caption(f"{workspace.get('source_count', 0)} 份中文来源 · {workspace.get('chunk_count', 0)} 个检索片段")
        return
    if status == "pending_project_evaluation" and workspace.get("source_status") == "pending_redistribution_license":
        st.warning("目标项目尚未获得可核实的内容再分发许可，目前没有可索引的项目正文，因此尚不能评测本项目 RAG 或 Agent 效果。历史其他语料的成绩不作为当前项目成绩。")
        return
    if status != "edge_ai_retrieval_v2_validated":
        st.warning(
            "当前项目尚未完成与活动语料指纹绑定的冻结评测；"
            "此前其他领域语料上的分数不适用于本知识空间，因此这里不展示为当前成绩。"
        )
    else:
        report = workspace.get("retrieval_evaluation")
        if not isinstance(report, dict) or report.get("name") != "edge_ai_retrieval_v2":
            st.warning("后端标记为已评测，但没有返回匹配的边缘设备评测报告；当前不展示未经核验的数字。")
        else:
            st.info(
                f"评测报告已绑定当前语料与策略指纹：固定题集 {report.get('case_count', 0)} 题，"
                f"DEV/HOLDOUT 各 {report.get('case_split_counts', {}).get('dev', 0)} / "
                f"{report.get('case_split_counts', {}).get('holdout', 0)} 题。"
            )
            st.caption(str(report.get("selection_reason") or "当前策略按冻结评测结果选择。"))
            rows = []
            candidates = report.get("candidates") or {}
            for split in ("dev", "holdout"):
                for strategy, values in (candidates.get(split) or {}).items():
                    rows.append({
                        "数据切分": "开发集" if split == "dev" else "留出集",
                        "检索策略": strategy,
                        "必需来源召回": values.get("mean_required_source_recall"),
                        "完整来源集率": values.get("complete_required_source_set_rate"),
                        "范围错误命中": values.get("wrong_scope_result_count"),
                        "无答案题返回候选比例": values.get("unanswerable_candidate_rate"),
                        "检索 P95 (ms)": (values.get("latency_ms") or {}).get("p95"),
                    })
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
                st.caption(
                    "无答案题返回检索候选不等同于最终回答错误；它提示候选里有噪声，"
                    "需结合生成拒答与人工核验评估。当前 DEV/HOLDOUT 指标未显示 RRF 优于 BM25。"
                )
            agent_report = workspace.get("change_review_evaluation")
            if isinstance(agent_report, dict):
                st.markdown("**变更审查流程评测**")
                st.caption(
                    f"固定场景 {agent_report.get('case_count', 0)} 个；仅评估规则分类、检索覆盖、"
                    "范围缺口和人工审核边界，不代表大模型影响建议准确率。"
                )
                agent_rows = []
                for split, values in (agent_report.get("splits") or {}).items():
                    agent_rows.append({
                        "数据切分": "开发集" if split == "dev" else "留出集",
                        "变更类型识别率": values.get("expected_change_type_accuracy"),
                        "必需来源召回": values.get("required_source_recall_across_planned_queries"),
                        "完整来源集": f"{values.get('complete_required_source_set_count', 0)} / {values.get('answerable_case_count', 0)}",
                        "范围缺口识别率": values.get("scope_gap_detection_accuracy"),
                        "人工审核边界": values.get("manual_review_boundary_accuracy"),
                    })
                if agent_rows:
                    st.dataframe(agent_rows, width="stretch", hide_index=True)
                st.warning(
                    "影响候选精确率、建议正确性和最终回答事实性尚未完成人工盲评，不能据此宣称 Agent 准确率。"
                )

    corpus = workspace.get("corpus_fingerprint") or {}
    st.markdown("**当前运行范围**")
    st.write(
        f"{workspace.get('source_count', 0)} 份中文来源 · "
        f"{workspace.get('chunk_count', 0)} 个检索片段 · "
        f"语料指纹 `{corpus.get('fingerprint_sha256', '未返回')}`"
    )
    if status != "edge_ai_retrieval_v2_validated":
        st.caption("完成与当前语料、配置和冻结题集指纹匹配的评测后，才会展示可复现的策略比较结果。")
    st.caption("当前新语料默认使用可解释的 BM25 基线；重排策略需在同一冻结题集上验证后再考虑启用。")


def _limits(workspace: dict | None = None) -> None:
    _page_header("系统说明", "已知限制", page_key="limits")
    for index, (title, detail) in enumerate((
        ("资料范围", "仅覆盖已收录的公开资料。"),
        ("回答", "请对照引用原文核验；证据不足时系统会拒答。"),
        ("影响分析", "静态检查只覆盖固定版本和可识别的应用调用；动态调用、证据不足和未实际推理保持待验证。" if _is_paddleocr(workspace) else "没有显式关联时，Agent 提供的是待核对候选。"),
        ("人工审核", "审核记录按匿名会话保存，公网实例重启后可能清空；不会写回源资料。"),
        ("检索策略", "默认方案依据当前语料的冻结实验选择；模型不可用时明确回退。检索覆盖不等于回答准确率，静态结论仍需实际文档回归。" if _is_paddleocr(workspace) else "BM25 为默认；PP-Human 任务自适应混合检索与重排是实验选项。检索召回提升不等于答案或影响判断更准确，端到端效果待独立盲审。"),
    )):
        with st.container(border=True, key=f"limit_row_{index}"):
            st.markdown(f"**{title}**")
            st.write(detail)


def _about(workspace: dict | None) -> None:
    _page_header("系统说明", "系统说明", page_key="about")
    st.write("面向内部研发团队，RAG 查询版本化研发知识，Agent 审查同一文档处理应用升级。当前 POC 使用 PaddleOCR 真实资料代理组件知识；应用输出契约和回归状态以提交的应用文件与记录为准。" if _is_paddleocr(workspace) else "RAG 按版本检索公开资料并保留来源；Agent 根据证据整理影响候选和修改建议，交由人工确认。")
    name = _workspace_name(workspace)
    st.caption('学生 POC · 非 PaddlePaddle 官方产品 · 不会写回源资料。' if _is_paddleocr(workspace) else f"独立工程演示 · 非 {name} 官方产品 · 不会写回源资料。")
    if _is_paddleocr(workspace):
        from components.retrieval_comparison import render_comparison
        render_comparison((workspace or {}).get('retrieval_comparison'))
    repositories = (workspace or {}).get("repositories") or [(workspace or {}).get("repository")]
    repository_links = [
        f"[{repository}](https://github.com/{repository})"
        for repository in repositories
        if isinstance(repository, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
    ]
    if repository_links:
        st.markdown("公开来源仓库：" + " · ".join(repository_links))
    with st.expander("运行版本与资料指纹"):
        st.write(f"前端构建：`{ui_build_revision()}`")
        st.write(f"RAG 后端构建：`{(workspace or {}).get('build_revision') or 'unknown'}`")
        corpus_fingerprint = (workspace or {}).get("corpus_fingerprint") or {}
        corpus_hash = corpus_fingerprint.get("fingerprint_sha256", "unknown")
        st.write(f"语料指纹：`{corpus_hash}`")
        st.write(f"检索配置指纹：`{(workspace or {}).get('retrieval_config_fingerprint') or 'unknown'}`")
        st.write(f"评测集指纹：`{(workspace or {}).get('evaluation_fingerprint') or 'unknown'}`")
        st.caption("SHA 仅在 Git 提交可验证且工作区干净时显示；公网 Streamlit 与 RAG 服务分别部署，需核对两端版本。")


def render() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    choice = st.session_state.setdefault("official_nav", "总览")
    st.session_state.setdefault("official_nav_history", [])
    if choice not in NAV_PAGES:
        choice = "总览"
        st.session_state["official_nav"] = choice
        st.session_state["official_nav_history"] = [
            page for page in st.session_state["official_nav_history"] if page in NAV_PAGES
        ]
    client = _client()
    workspace = st.session_state.get('official_workspace')
    if workspace is None:
        workspace = _request_workspace(client)
    with st.sidebar:
        st.markdown('<div class="sidebar-mark">工作台导航</div>', unsafe_allow_html=True)
        for group, pages in NAV_GROUPS:
            st.markdown(f'<div class="nav-heading">{escape(NAV_GROUP_LABELS.get(group, group))}</div>', unsafe_allow_html=True)
            for page in pages:
                st.button(_page_label(page, workspace), key="nav_" + page,
                          type="primary" if choice == page else "secondary",
                          on_click=_navigate, args=(page,), use_container_width=True)
        st.button('刷新知识库状态', key='refresh_workspace',
                  on_click=lambda: st.session_state.pop('navigation_workspace_cache', None),
                  use_container_width=True)
    workspace = _request_workspace(client)
    ready = bool(workspace) and workspace.get("rag_ready", True) is not False
    _sync_workspace_version(workspace)
    mismatch = public_workspace_mismatch(
        workspace, public_demo=_setting("APP_ENV", "local").strip().casefold() == "public_demo",
    )
    if mismatch:
        st.warning(mismatch)
        st.info("为避免误用不匹配的资料，知识检索和变更审查暂不可用。")
        return
    readiness_message = workspace_readiness_message(workspace)
    if readiness_message:
        st.warning(readiness_message)
    if choice == "总览":
        _home(ready, workspace)
    elif choice == "版本检索与问答":
        _knowledge(client, ready, workspace)
    elif choice == "版本与历史":
        _versions(client, ready, workspace)
    elif choice == "资料与来源":
        _sources(client, ready, workspace)
    elif choice == "新建变更审查":
        if _is_paddleocr(workspace):
            _ocr_agent(client, ready, workspace)
        else:
            docs = _request(client.documents, fallback="官方资料目录暂不可用。") if ready else []
            _agent(client, ready, docs or [], workspace)
    elif choice in ("可能相关资料", "修改前后对照", "人工审核"):
        if _is_paddleocr(workspace):
            _ocr_review_subpage(choice, workspace)
        else:
            _review_subpage(choice)
    elif choice == "检索评测":
        _benchmark(workspace)
    elif choice == "已知限制":
        _limits(workspace)
    else:
        _about(workspace)
