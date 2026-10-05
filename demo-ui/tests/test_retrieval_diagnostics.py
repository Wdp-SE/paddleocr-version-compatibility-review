from __future__ import annotations

from services.retrieval_diagnostics import retrieval_diagnostic_lines, retrieval_policy_options


def test_verified_rag_release_stays_default_with_optional_experiments():
    options=retrieval_policy_options({'workspace_id':'paddleocr','rag_quality_evaluation':{},
        'impact_evaluation':{'selected_strategy':'window_rerank_120'},
        'quality_retrieval':{'configured':True,'strategy':'hybrid_rerank'}})
    assert options[0]['id']=='paddleocr_evidence'
    assert sum(o['id']=='paddleocr_evidence' for o in options)==1
    assert options[-1]['experimental']


def test_diagnostics_disclose_corrective_pass_and_all_stage_token_usage():
    lines = retrieval_diagnostic_lines({'correction': {'attempts': 1, 'status': 'UNRESOLVED'},
        'workflow_usage': {'total_tokens': 1234}, 'final_evidence_count': 8})
    assert any('补查' in line and '1' in line and '仍有缺口' in line for line in lines)
    assert any('1234' in line for line in lines)


def test_policy_options_limit_experiment_to_pphuman():
    options = retrieval_policy_options({"workspace_id": "pphuman"})
    assert [row["id"] for row in options] == [
        "bm25", "bm25_pphuman_term_expansion_rrf", "task_adaptive_rerank",
    ]
    assert [row["id"] for row in retrieval_policy_options({"workspace_id": "edge_ai_device"})] == ["bm25"]


def test_diagnostics_report_actual_fallback_counts_and_stage_timings():
    lines = retrieval_diagnostic_lines({
        "retrieval_policy": "bm25_rrf_fallback",
        "retrieval_policy_requested": "task_adaptive_rerank",
        "route": "CHANGE_REVIEW",
        "rerank_status": "RERANK_FALLBACK",
        "candidate_count": 18,
        "final_evidence_count": 6,
        "stage_latency_ms": {"retrieval": 120.5, "rerank": 900.0, "total": 1100.0},
    })

    assert any("请求策略" in line and "实际策略" in line for line in lines)
    assert any("重排失败" in line and "BM25/RRF" in line for line in lines)
    assert any("18" in line and "6" in line for line in lines)
    assert any("检索 120.5 ms" in line and "重排 900 ms" in line for line in lines)


def test_diagnostics_do_not_invent_missing_model_latency_or_rerank_success():
    lines = retrieval_diagnostic_lines({
        "retrieval_policy": "bm25",
        "retrieval_policy_requested": "task_adaptive_rerank",
        "rerank_status": "NOT_AVAILABLE_FALLBACK",
        "retrieval_trace": {"http_request_count": 5, "subquery_count": 4},
    })
    assert any("未执行重排" in line and "回退 BM25" in line for line in lines)
    assert any("RAG 请求 5 次" in line and "检查项/子查询 4 项" in line for line in lines)
    assert not any("ms" in line for line in lines)


def test_paddleocr_optional_quality_and_unavailable_diagnostics():
    options=retrieval_policy_options({'workspace_id':'paddleocr','quality_retrieval':{'configured':True,'strategy':'hybrid_rerank'}})
    assert 'paddleocr_quality' in [row['id'] for row in options]
    lines=retrieval_diagnostic_lines({'requested_policy':'paddleocr_quality','actual_policy':'bm25',
                                     'rerank_status':'QUALITY_FALLBACK',
                                     'rerank_diagnostics':{'failure_reason':'QUALITY_MODELS_NOT_CONFIGURED'}})
    assert any('模型不可用' in line and 'BM25' in line for line in lines)


def test_baseline_selection_is_not_advertised_as_neural_ranking():
    options=retrieval_policy_options({'workspace_id':'paddleocr','quality_retrieval':{'configured':True,'strategy':'bm25'}})
    assert [row['id'] for row in options]==['bm25']
