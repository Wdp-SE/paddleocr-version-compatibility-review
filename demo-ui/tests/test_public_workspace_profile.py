from services.public_workspace_profile import (
    clear_workspace_bound_results,
    public_workspace_mismatch,
    workspace_identity_changed,
    workspace_page_readiness_notice,
    workspace_readiness_message,
    workspace_snapshot,
)


def _paddleocr_workspace(**overrides):
    return {
        "workspace_id": "paddleocr",
        "domain_profile": {"id": "paddleocr"},
        "workspace": "PaddleOCR 文档处理应用研发知识",
        "repository": "PaddlePaddle/PaddleOCR",
        "repositories": ["PaddlePaddle/PaddleOCR"],
        "languages": ["zh"],
        **overrides,
    }


def test_public_demo_accepts_the_active_paddleocr_workspace():
    assert public_workspace_mismatch(_paddleocr_workspace(), public_demo=True) is None


def test_public_demo_rejects_old_or_unrelated_workspace_ids():
    warning = public_workspace_mismatch(
        _paddleocr_workspace(workspace_id="unrelated_workspace"), public_demo=True,
    )

    assert warning is not None
    assert "PaddleOCR" in warning
    assert "RAG_API_BASE_URL" not in warning


def test_public_demo_rejects_wrong_repository_domain_profile_or_non_chinese_data():
    for patch in (
        {"repository": "other/project"},
        {"domain_profile": {"id": "other"}},
        {"repositories": ["PaddlePaddle/PaddleOCR", "other/project"]},
        {"languages": ["zh", "en"]},
        {"languages": []},
    ):
        assert public_workspace_mismatch(_paddleocr_workspace(**patch), public_demo=True)


def test_workspace_snapshot_reports_the_current_corpus_size():
    snapshot = workspace_snapshot({
        "unique_document_count": 14, "source_count": 83, "chunk_count": 761,
    })

    assert "14 个主题" in snapshot
    assert "83 条版本/语言来源" in snapshot
    assert "761 个检索片段" in snapshot


def test_pending_domain_evaluation_is_explicit_without_reusing_old_scores():
    workspace = _paddleocr_workspace(
        retrieval_evaluation_status="new_corpus_pending_rebenchmark",
        activation_block_reason="pending_project_evaluation",
        rag_ready=False,
        source_count=83,
        chunk_count=761,
    )

    assert public_workspace_mismatch(workspace, public_demo=True) is None
    assert "PaddleOCR 语料尚未完成评测" in workspace_readiness_message(workspace)
    assert "83 份来源" in workspace_snapshot(workspace)


def test_page_readiness_notice_does_not_repeat_global_blocker():
    workspace = _paddleocr_workspace(
        activation_block_reason="pending_project_evaluation", rag_ready=False,
    )

    assert workspace_page_readiness_notice(
        workspace, "知识服务可能正在冷启动；连接恢复后可继续检索。",
    ) is None


def test_page_readiness_notice_keeps_generic_fallback_without_workspace():
    fallback = "知识服务暂不可用，连接恢复后可查看资料。"

    assert workspace_page_readiness_notice(None, fallback) == fallback


def test_workspace_identity_change_invalidates_cached_results_across_corpus_versions():
    previous = _paddleocr_workspace(current_version="v2.9.1")
    current = _paddleocr_workspace(current_version="v3.0.0")

    assert workspace_identity_changed(previous, current)
    assert not workspace_identity_changed(current, dict(current))


def test_switching_to_paddleocr_clears_old_corpus_evidence_decisions_and_question():
    previous = {
        "workspace_id": "edge_ai_device",
        "repository": "Seeed-Studio/wiki-documents",
        "current_version": "old-snapshot",
    }
    state = {
        "official_workspace": previous,
        "official_result": ("query", "old question", "old-snapshot", "zh", {}, {}),
        "official_review": {"answer": "old corpus answer"},
        "official_review_decision": "approved",
        "official_question": "keep user draft",
    }

    changed = clear_workspace_bound_results(state, _paddleocr_workspace())

    assert changed is True
    assert "official_result" not in state
    assert "official_review" not in state
    assert "official_review_decision" not in state
    assert "official_question" not in state
    assert state["official_workspace"]["workspace_id"] == "paddleocr"
