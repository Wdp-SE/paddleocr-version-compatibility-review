"""UI-neutral retrieval policy options and diagnostic copy."""

from __future__ import annotations


def retrieval_policy_options(workspace: dict | None) -> list[dict[str, str | bool]]:
    options: list[dict[str, str | bool]] = [
        {"id": "bm25", "label": "BM25 基线（默认）", "experimental": False},
    ]
    quality=(workspace or {}).get('quality_retrieval',{})
    impact=(workspace or {}).get('impact_evaluation')
    rag=(workspace or {}).get('rag_quality_evaluation')
    if (workspace or {}).get('workspace_id')=='paddleocr' and isinstance(rag,dict):
        options[0]['label']='BM25 基线（对照）'
        options.insert(0,{'id':'paddleocr_evidence','label':'按产线范围检索与证据核验（默认）','experimental':False})
    if (workspace or {}).get('workspace_id')=='paddleocr' and isinstance(impact,dict) and not isinstance(rag,dict):
        selected=impact.get('selected_strategy','bm25')
        if selected!='bm25':
            option={'id':'paddleocr_evidence','label':'当前验证通过的证据检索（默认）','experimental':False}
            options[0]['label']='BM25 基线（对照）'
            options.insert(0,option)
    if ((workspace or {}).get('workspace_id')=='paddleocr' and quality.get('configured')
            and quality.get('strategy') in {'semantic','bm25_rerank','hybrid','hybrid_rerank'}):
        option={'id':'paddleocr_quality','label':'质量优先检索（语义召回 / 重排）','experimental':True}
        if isinstance(rag,dict):options.append(option)
        else:options.insert(0,option)
    if (workspace or {}).get("workspace_id") == "pphuman":
        options.append({
            "id": "bm25_pphuman_term_expansion_rrf",
            "label": "术语扩展 + RRF（可对比）",
            "experimental": True,
        })
        options.append({
            "id": "task_adaptive_rerank",
            "label": "术语融合 + 模型重排（实验）",
            "experimental": True,
        })
    return options


def _format_ms(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return f"{value:g} ms"


def retrieval_diagnostic_lines(result: dict) -> list[str]:
    """Summarize known strategy, rerank outcome, evidence counts, and timings."""
    lines: list[str] = []
    correction = result.get('correction') or {}
    if correction.get('attempts'):
        outcome = '已获得核验通过的回答' if correction.get('status') == 'RECOVERED' else '仍有缺口，需核对原文'
        lines.append(f"证据补查 {correction['attempts']} 次：{outcome}")
    usage = result.get('workflow_usage') or {}
    if usage.get('total_tokens') is not None:
        suffix = '' if usage.get('complete') else '（仅统计已返回用量的调用）'
        lines.append(f"生成、核验与重排合计 Token：{usage['total_tokens']}{suffix}")
    trace = result.get("retrieval_trace") if isinstance(result.get("retrieval_trace"), dict) else {}
    actual = result.get("actual_policy") or result.get("retrieval_policy")
    requested = result.get("requested_policy") or result.get("retrieval_policy_requested")
    if actual or requested:
        if requested and actual and requested != actual:
            lines.append(f"检索策略：请求策略 {requested}；实际策略 {actual}")
        else:
            lines.append(f"检索策略：{actual or requested}")
    policy_status = result.get("retrieval_policy_status")
    if policy_status in {"MISMATCH", "PARTIAL", "UNCONFIRMED", "NOT_AVAILABLE_FALLBACK"}:
        status_labels = {
            "MISMATCH": "后端策略与请求不一致",
            "PARTIAL": "部分调用未确认",
            "UNCONFIRMED": "后端未确认策略",
            "NOT_AVAILABLE_FALLBACK": "自适应策略不可用，已回退",
        }
        lines.append(f"策略状态：{status_labels[policy_status]}")

    route = result.get("route")
    if route:
        route_labels = {
            "DIRECT_LOOKUP": "精确标识符查询",
            "CONCEPTUAL_LOOKUP": "概念查询",
            "CHANGE_REVIEW": "变更审查",
            "GENERAL_LOOKUP": "通用查询",
        }
        lines.append(f"查询类型：{route_labels.get(str(route), route)}")

    candidate_count = result.get("candidate_count")
    final_count = result.get("final_evidence_count")
    if candidate_count is None:
        candidate_count = trace.get("candidate_count")
    if final_count is None:
        final_count = trace.get("final_evidence_count")
    if candidate_count is None and isinstance(result.get("candidates"), list):
        candidate_count = len(result["candidates"])
    if final_count is None:
        evidence = result.get("evidence")
        results = result.get("results")
        if isinstance(evidence, list):
            final_count = len(evidence)
        elif isinstance(results, list):
            final_count = len(results)
    if candidate_count is not None or final_count is not None:
        lines.append(
            f"候选资料 {candidate_count if candidate_count is not None else '未返回'} 条；"
            f"最终证据 {final_count if final_count is not None else '未返回'} 条"
        )

    rerank_status = result.get("rerank_status") or trace.get("rerank_status")
    diagnostics = result.get("rerank_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = trace.get("rerank_diagnostics") if isinstance(trace.get("rerank_diagnostics"), dict) else {}
    rerank_calls = diagnostics.get("rerank_calls", result.get("rerank_calls", trace.get("rerank_calls")))
    if rerank_status == 'QUALITY_FALLBACK':
        rerank_line='语义或重排模型不可用，已回退 BM25；原因：'+str(diagnostics.get('failure_reason') or '未返回')
    elif rerank_status == "RERANK_FALLBACK":
        rerank_line = "重排失败，按 BM25/RRF 候选顺序回退；未执行重排排序"
    elif rerank_status == "NOT_AVAILABLE_FALLBACK":
        rerank_line = "批量接口不可用，未执行重排，已回退 BM25"
    elif rerank_status == "OK":
        rerank_line = f"重排已完成（调用 {rerank_calls or 1} 次）"
    elif rerank_status == "SKIPPED_DIRECT_LOOKUP":
        rerank_line = "精确标识符查询按 BM25 返回，跳过重排"
    elif rerank_status == "SKIPPED_NO_CANDIDATES":
        rerank_line = "没有可重排候选，未调用重排模型"
    elif rerank_status == "NOT_RUN_SCOPE_GUARD":
        rerank_line = "请求超出当前公开资料范围，未检索或重排"
    elif rerank_status == "NOT_REQUESTED":
        rerank_line = "未请求模型重排"
    elif rerank_status == "FAILED":
        rerank_line = "重排流程失败；请查看检索状态和服务请求编号"
    elif rerank_status:
        rerank_line = f"重排状态：{rerank_status}"
    else:
        rerank_line = None
    if rerank_line:
        lines.append(rerank_line)
        if rerank_status in {"OK", "RERANK_FALLBACK"}:
            usage = diagnostics.get("usage")
            if isinstance(usage, dict) and usage:
                known_usage = [
                    f"{key} {usage[key]}" for key in (
                        "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens",
                    ) if usage.get(key) is not None
                ]
                usage_text = "、".join(known_usage) if known_usage else "已返回"
            else:
                usage_text = "未返回"
            lines.append(f"重排 Token 用量：{usage_text}")

    if trace.get("http_request_count") is not None or trace.get("subquery_count") is not None:
        lines.append(
            f"RAG 请求 {trace.get('http_request_count', '未返回')} 次；"
            f"检查项/子查询 {trace.get('subquery_count', '未返回')} 项"
        )

    stage_timings = result.get("stage_latency_ms")
    if not isinstance(stage_timings, dict):
        stage_timings = {}
    retrieval = _format_ms(stage_timings.get("retrieval", result.get("retrieval_latency_ms")))
    rerank = _format_ms(stage_timings.get("rerank", diagnostics.get("latency_ms")))
    generation = _format_ms(stage_timings.get("generation"))
    total = _format_ms(stage_timings.get("total"))
    timed = []
    if retrieval:
        timed.append(f"检索 {retrieval}")
    if rerank and rerank_status not in {None, "NOT_REQUESTED", "SKIPPED_DIRECT_LOOKUP", "NOT_RUN_SCOPE_GUARD"}:
        timed.append(f"重排 {rerank}")
    if generation:
        timed.append(f"生成 {generation}")
    if total:
        timed.append(f"总耗时 {total}")
    if timed:
        lines.append("处理耗时：" + "；".join(timed))
    return lines
