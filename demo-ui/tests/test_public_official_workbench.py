from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).parents[1] / "app.py"
CHUNK = {
    "chunk_id": "wiki-1eadc6584f96:zh:seeed-jetson-flash-firmware:1",
    "document_id": "wiki-1eadc6584f96:zh:seeed-jetson-flash-firmware",
    "document_key": "seeed-jetson-flash-firmware",
    "source_id": "seeed-jetson-flash-firmware",
    "version": "wiki-1eadc6584f96", "source_snapshot": "wiki-1eadc6584f96",
    "locale": "zh-CN", "language": "zh",
    "heading": "Jetson 刷写与软件基线", "content": "刷写前核对设备型号、JetPack 与 L4T 版本。",
    "source_type": "official_documentation",
    "source_url": "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/",
    "repository": "Seeed-Studio/wiki-documents",
    "commit": "1eadc6584f962b6efdbdb3e49b2b4ce30c85be08",
    "sha256": "a" * 64,
    "retrieval_score": 12.0, "retrieval_policy": "bm25",
}


def _mock_client(monkeypatch):
    from services.public_knowledge_client import PublicKnowledgeClient
    import public_workbench

    monkeypatch.setenv("DEMO_LEGACY_FIXTURES", "false")
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())
    monkeypatch.setattr(PublicKnowledgeClient, "documents", lambda self: [{
        "document_id": CHUNK["document_id"], "document_key": CHUNK["document_key"],
        "title": "Jetson 刷写与软件基线", "version": "wiki-1eadc6584f96", "locale": "zh-CN",
        "source_type": "official_documentation", "source_url": CHUNK["source_url"],
    }])
    monkeypatch.setattr(PublicKnowledgeClient, "document", lambda self, document_id: [dict(CHUNK)])
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "query": question, "results": [dict(CHUNK)], "retrieval_policy": "bm25",
        "consistency_notes": [],
    })


    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "上游参数优先于启动参数。", "sources": [dict(CHUNK)],
        "claims": [{"text": "上游参数优先于启动参数。", "source_indexes": [1]}],
        "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
        "evidence_support": {
            "level": "partial", "label": "一般",
            "summary": "引用覆盖了部分问题关键词；该提示不代表答案正确率。",
        },
    })
    monkeypatch.setattr(PublicKnowledgeClient, "review_advice", lambda self, change_summary, evidence_chunk_ids: {
        "status": "OK", "answer": "依据引用片段，建议核对相关资料中的参数顺序。",
        "sources": [dict(CHUNK)] if CHUNK["chunk_id"] in evidence_chunk_ids else [],
        "evidence": [dict(CHUNK)] if CHUNK["chunk_id"] in evidence_chunk_ids else [],
        "review": {
            "change_interpretation": "依据引用片段，建议核对相关资料中的参数顺序。",
            "impact_candidates": [{
                "evidence_chunk_id": CHUNK["chunk_id"],
                "reason": "该章节说明参数优先级。",
                "suggested_action": "核对示例与运维说明是否同步。",
            }],
            "evidence_gaps": ["尚未提供目标设备的实测记录。"],
            "version_ambiguities": [],
            "reviewer_actions": ["逐版本确认变更影响。"],
            "review_status": "REQUIRES_HUMAN_REVIEW",
        },
    })
    monkeypatch.setattr(
        PublicKnowledgeClient, "review_advice_for_version",
        lambda self, change_summary, evidence_chunk_ids, *, version, **scope:
            self.review_advice(change_summary, evidence_chunk_ids),
    )
    monkeypatch.setattr(public_workbench, "_analyze_hypothetical", lambda client, selected, proposed_text, **kwargs: {
        "change": {"change_type": "MODIFIED"}, "selected_source": dict(CHUNK),
        "impacts": [{"relation": "suggested", "status": "SUGGESTED", "reason": "主题相关，需人工核验。", "evidence": dict(CHUNK)}],
        "confirmed_relations": [], "patch_candidate": {
            "before": CHUNK["content"], "proposed_after": proposed_text,
            "status": "REQUIRES_HUMAN_REVIEW",
        }, "sandbox_only": True, "public_baseline_written": False,
        "review_advice": {
            "status": "OK", "answer": "依据本次官方片段，建议核对相关资料中的参数顺序。",
            "sources": [dict(CHUNK)],
            "review": {
                "change_interpretation": "依据本次官方片段，建议核对相关资料中的参数顺序。",
                "impact_candidates": [{
                    "evidence_chunk_id": CHUNK["chunk_id"],
                    "reason": "该章节说明参数优先级。",
                    "suggested_action": "核对示例与运维说明是否同步。",
                }],
                "evidence_gaps": ["尚未提供目标设备的实测记录。"],
                "version_ambiguities": [],
                "reviewer_actions": ["逐版本确认变更影响。"],
                "review_status": "REQUIRES_HUMAN_REVIEW",
            },
        },
    })


def _edge_workspace():
    return {
        "workspace_id": "edge_ai_device",
        "domain_profile": {
            "id": "edge_ai_device",
            "name": "边缘 AI 设备研发变更审查",
            "example_queries": [
                "J4012 在 JetPack 7.2 下部署工业视觉需要检查哪些环境？",
                "reComputer Industrial J4011 升级 JetPack 6.2 前应核对哪些刷写和热设计注意事项？",
                "J30/J40 设备如何获取系统日志并用于故障排查？",
            ],
            "change_types": [
                {"id": "software_baseline", "label": "软件基线变更"},
                {"id": "device_configuration", "label": "设备与接口配置变更"},
                {"id": "deployment_operations", "label": "AI 部署与运维变更"},
            ],
        },
        "workspace": "reComputer Industrial / Jetson 边缘 AI 工程知识",
        "repository": "Seeed-Studio/wiki-documents",
        "repositories": ["Seeed-Studio/wiki-documents"],
        "current_version": "wiki-1eadc6584f96",
        "available_versions": ["wiki-1eadc6584f96"],
        "version_labels": {"wiki-1eadc6584f96": "当前固定资料快照"},
        "version_scopes": {"latest": {"versions": ["wiki-1eadc6584f96"]}},
        "languages": ["zh"],
        "source_count": 18, "chunk_count": 394, "unique_document_count": 18,
        "hardware_models": ["reComputer Industrial J4012", "reComputer Industrial J3011"],
        "module_skus": ["P3767-0000"], "carrier_boards": ["J401"],
        "software_baselines": ["JetPack 6.2", "JetPack 7.2 (L4T 39.2.0)"],
        "source_registry": [{
            "source_id": "seeed-jetson-flash-firmware",
            "source_url": "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/",
            "repository": "Seeed-Studio/wiki-documents", "source_snapshot": "wiki-1eadc6584f96",
            "commit": "1eadc6584f962b6efdbdb3e49b2b4ce30c85be08", "sha256": "a" * 64,
        }],
        "snapshots": [{"version": "wiki-1eadc6584f96", "commit": "1eadc6584f962b6efdbdb3e49b2b4ce30c85be08"}],
        "retrieval_evaluation_status": "new_corpus_pending_rebenchmark",
    }


def _pphuman_workspace():
    return {
        "workspace_id": "pphuman",
        "domain_profile": {
            "id": "pphuman",
            "name": "PP-Human 行人分析系统研发",
            "example_queries": [
                "PP-Human v2.9.0 如何启用行为识别并选择对应的模型配置？",
                "调整行人跟踪器配置后，需要核对哪些部署步骤和验证项？",
            ],
            "change_types": [
                {"id": "model_config", "label": "检测或属性模型配置变更"},
                {"id": "behavior_pipeline", "label": "行为识别或属性分析流程变更"},
                {"id": "tracking", "label": "行人跟踪或跨镜跟踪变更"},
                {"id": "deployment", "label": "推理部署和运行配置变更"},
            ],
        },
        "workspace": "PP-Human 行人分析工程知识",
        "repository": "PaddlePaddle/PaddleDetection",
        "repositories": ["PaddlePaddle/PaddleDetection"],
        "current_version": "v2.9.0",
        "baseline_version": "v2.5.0",
        "available_versions": ["v2.9.0", "v2.8.1", "v2.8.0", "v2.7.0", "v2.6.0", "v2.5.0"],
        "version_labels": {"v2.9.0": "当前最新版"},
        "version_scopes": {"latest": {"versions": ["v2.9.0"]}},
        "languages": ["zh"],
        "source_count": 83,
        "chunk_count": 761,
        "unique_document_count": 14,
        "source_breakdown": [{
            "version": "v2.9.0", "locale": "zh-CN", "document_family": "tracking",
            "source_format": "markdown", "count": 14,
        }],
        "retrieval_evaluation_status": "new_corpus_pending_rebenchmark",
        "rag_ready": True,
    }


def test_edge_knowledge_filters_are_forwarded_to_rag(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())
    seen = []
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: (
        seen.append((question, scope)) or {
            "answer": "仅展示带来源的资料回答。", "sources": [dict(CHUNK)],
            "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
        }
    ))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    app.selectbox(key="official_device_model").set_value("reComputer Industrial J4012")
    app.selectbox(key="official_software_baseline").set_value("JetPack 7.2 (L4T 39.2.0)").run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    assert seen[0][1]["language"] == "zh"
    assert seen[0][1]["device_model"] == "reComputer Industrial J4012"
    assert seen[0][1]["software_baseline"] == "JetPack 7.2 (L4T 39.2.0)"
    assert "JetPack 7.2 下部署工业视觉" in app.selectbox(key="official_example").options[0]


def test_edge_agent_shows_domain_change_types_and_passes_confirmed_scope(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    import public_workbench

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())
    calls = []
    monkeypatch.setattr(public_workbench, "_analyze_change_request", lambda *args, **kwargs: (
        calls.append(kwargs) or {"request_summary": args[1], "request_plan": {}, "retrieved_results": []}
    ))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()

    labels = {item for item in app.selectbox(key="official_change_type").options}
    assert "软件基线变更" in labels
    assert "设备与接口配置变更" in labels
    assert not any("规划 / 轨迹" in item or "工作流 / 行为" in item for item in labels)
    app.text_area(key="official_change_request").set_value("升级 J4012 的 JetPack 软件基线")
    app.selectbox(key="agent_device_model").set_value("reComputer Industrial J4012")
    app.selectbox(key="agent_module_sku").set_value("P3767-0000")
    app.selectbox(key="agent_carrier_board").set_value("J401")
    app.selectbox(key="agent_software_baseline").set_value("JetPack 6.2").run()
    next(button for button in app.button if button.label == "检索资料并分析影响").click().run()

    assert not app.exception
    assert calls[0]["device_scope"]["device_model"] == "reComputer Industrial J4012"
    assert calls[0]["device_scope"]["module_sku"] == "P3767-0000"
    assert calls[0]["device_scope"]["carrier_board"] == "J401"
    assert calls[0]["device_scope"]["software_baseline"] == "JetPack 6.2"


def test_workspace_scoped_selectors_follow_the_latest_manifest_version():
    import public_workbench

    workspace = {
        "workspace": "其他领域", "repository": "Seeed-Studio/wiki-documents",
        "baseline_version": "0.51.0", "current_version": "0.52.0",
        "available_versions": ["0.51.0", "0.52.0"], "languages": ["en-US"],
    }

    assert public_workbench._published_versions(workspace) == ["0.52.0", "0.51.0"]
    assert public_workbench._review_version_selector(workspace) == (["0.52.0", "0.51.0"], 0)
    assert public_workbench._published_language_options(workspace) == [("en", "English")]
    assert public_workbench._published_language_options({"languages": []}) == [("all", "语言元数据未声明")]


def test_chinese_only_language_options_do_not_offer_english_or_all_languages():
    import public_workbench

    options = public_workbench._published_language_options({"languages": ["en-US", "zh-CN"]})

    assert options == [("zh", "中文")]


def test_legacy_project_workspace_is_rejected_and_query_controls_stay_disabled(monkeypatch):
    from services.public_knowledge_client import PublicKnowledgeClient

    workspace = {
        "workspace_id": "industrial-inspection",
        "domain_profile": {"id": "industrial_inspection", "name": "工业视觉安全监控应用研发"},
        "workspace": "工业视觉安全监控应用研发",
        "repository": "xbs0325/industrial-inspection",
        "repositories": ["xbs0325/industrial-inspection"],
        "current_version": "6d0df954f26b1810910db9f50727ca8bd19afa9f",
        "available_versions": ["6d0df954f26b1810910db9f50727ca8bd19afa9f"],
        "languages": [],
        "source_status": "pending_redistribution_license",
        "activation_block_reason": "pending_redistribution_license",
        "rag_ready": False,
        "source_count": 0,
        "chunk_count": 0,
        "retrieval_evaluation_status": "pending_project_evaluation",
    }
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: workspace)
    app = AppTest.from_file(APP, default_timeout=40).run()

    assert not app.exception
    assert any("PaddleOCR 官方中文资料不匹配" in item.value for item in app.warning)
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    assert not any(button.label == "生成带引用回答" for button in app.button)
    assert any("为避免误用不匹配的资料" in item.value for item in app.info)


def test_relationship_labels_keep_path_candidates_distinct_from_confirmed_translation():
    import public_workbench

    candidate = public_workbench._document_relationship_caption({
        "document_relationships": [{
            "relation_type": "translation_of", "verification_status": "candidate",
        }],
    })
    verified = public_workbench._document_relationship_caption({
        "document_relationships": [{
            "relation_type": "translation_of", "verification_status": "verified",
        }],
    })
    localized = public_workbench._document_relationship_caption({
        "document_relationships": [{
            "relation_type": "localized_variant_of", "verification_status": "verified",
        }],
    })

    assert "待核验" in candidate
    assert "不据此判断同步差异" in candidate
    assert "已核验" in verified
    assert "译文" not in localized


def test_source_coverage_summary_reports_edge_device_snapshot_and_families():
    import public_workbench

    summary = public_workbench._source_coverage_text({
        "current_version": "wiki-1eadc6584f96", "source_count": 18,
        "source_breakdown": [
            {"version": "wiki-1eadc6584f96", "locale": "zh-CN", "document_family": "ai_deployment", "count": 4},
            {"version": "wiki-1eadc6584f96", "locale": "zh-CN", "document_family": "hardware_interface", "count": 3},
        ],
    })

    assert summary is not None
    assert "18 份资料" in summary
    assert "ai deployment" in summary and "hardware interface" in summary


def test_workbench_warns_when_public_rag_workspace_is_not_paddleocr(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setenv("APP_ENV", "public_demo")

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: {
        "workspace": "reComputer Industrial / Jetson",
        "repository": "Seeed-Studio/wiki-documents",
        "baseline_version": "0.51.0", "current_version": "0.52.0",
        "languages": ["zh-CN", "en-US"],
    })

    app = AppTest.from_file(APP, default_timeout=40).run()

    assert not app.exception
    warning = "\n".join(item.value for item in app.warning)
    assert "PaddleOCR 官方中文资料不匹配" in warning
    assert "RAG_API_BASE_URL" not in warning
    assert "public_corpus_other" not in warning
    assert not any("资料规模" in item.value for item in list(app.markdown) + list(app.caption) + list(app.error))


def test_home_hides_operational_and_corpus_detail_copy(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    original_workspace = PublicKnowledgeClient.workspace
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: {
        **original_workspace(self),
        "corpus_scope": "Detailed corpus paths, repositories and version matching notes",
    })
    app = AppTest.from_file(APP, default_timeout=40).run()

    assert not app.exception
    visible = "\n".join(
        item.value
        for group in (app.markdown, app.caption, app.info, app.warning)
        for item in group
    )
    assert "选择要查看的功能页面" not in visible
    assert "精选资料范围，不代表上游项目全量" not in visible
    assert "独立工程演示，并非上游官方产品" not in visible
    assert "community Chinese translation snapshot" not in visible
    assert "Detailed corpus paths" not in visible
    assert "进入知识检索" in "\n".join(item.label for item in app.button)


def test_agent_context_filter_keeps_scope_guard_and_rejects_stale_results():
    import public_workbench

    args = {
        "summary": "change the planner behavior",
        "target_version": "0.52.0",
        "objective": "reduce false pull-out",
        "constraints": "keep interface stable",
        "validation_plan": "run planner tests",
        "selected_type_code": "workflow_behavior",
        "impact_scope": "behavior path planner",
    }
    assert public_workbench._request_context_matches({
        "request_summary": args["summary"], "scope_status": "OUT_OF_SCOPE",
    }, **args)

    valid = {
        "request_summary": args["summary"],
        "request_plan": {
            "target_version": "0.52.0", "impact_scope": "behavior path planner",
            "change_type": "workflow_behavior", "classification_source": "user_selected",
        },
        "request_context": {
            "target_version": "0.52.0", "objective": args["objective"],
            "constraints": args["constraints"], "validation_plan": args["validation_plan"],
        },
    }
    assert public_workbench._request_context_matches(valid, **args)
    assert not public_workbench._request_context_matches(
        valid, **{**args, "target_version": "0.51.0"}
    )


def test_retrieval_policy_options_only_offer_pphuman_experiment_in_its_workspace():
    import public_workbench

    baseline = {"id": "bm25", "label": "BM25 基线（默认）", "experimental": False}
    experiment = {
        "id": "task_adaptive_rerank",
        "label": "术语融合 + 模型重排（实验）",
        "experimental": True,
    }
    rrf = {"id": "bm25_pphuman_term_expansion_rrf", "label": "术语扩展 + RRF（可对比）", "experimental": True}
    assert public_workbench._retrieval_policy_options({"workspace_id": "pphuman"}) == [baseline, rrf, experiment]
    assert public_workbench._retrieval_policy_options({"workspace_id": "edge_ai_device"}) == [baseline]


def test_retrieval_diagnostics_explain_actual_policy_rerank_and_timings():
    import public_workbench

    lines = public_workbench._retrieval_diagnostic_lines({
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


def test_agent_context_filter_invalidates_results_after_retrieval_policy_change():
    import public_workbench

    result = {
        "request_summary": "调整行人跟踪阈值",
        "retrieval_policy": "bm25",
        "request_plan": {"target_version": "v2.9.0", "impact_scope": "跟踪"},
        "request_context": {"target_version": "v2.9.0"},
    }
    common = {
        "summary": "调整行人跟踪阈值", "target_version": "v2.9.0", "objective": "",
        "constraints": "", "validation_plan": "", "selected_type_code": None,
        "impact_scope": "跟踪", "retrieval_policy": "bm25",
    }
    assert public_workbench._request_context_matches(result, **common)
    assert not public_workbench._request_context_matches(
        result, **{**common, "retrieval_policy": "bm25_pphuman_term_expansion_rrf"}
    )


def _state_get(session_state, key, default=None):
    try:
        return session_state[key]
    except KeyError:
        return default


def _start_agent_request(app, summary="假设调整全局参数优先级，并找出需要核对的资料。"):
    if _state_get(app.session_state, "official_nav") != "新建变更审查":
        next(button for button in app.button if button.label == "发起变更审查").click().run()
    app.text_area(key="official_change_request").set_value(summary).run()
    next(button for button in app.button if button.label == "检索资料并分析影响").click().run()
    return app


def test_public_home_has_pphuman_modules_and_change_review_flow(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _pphuman_workspace())
    app = AppTest.from_file(APP, default_timeout=40).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["研发知识版本服务与变更影响审查"]
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "PP-Human 行人分析工程知识" in text
    assert "版本化研发知识服务 · RAG" in text
    assert "Agent · 研发资料变更审查" in text
    assert "研发资料变更影响审查" in text
    assert "PP-Human 行人分析系统研发" not in text
    assert "v2.9.0" in text
    assert "官方英文资料" not in text
    assert "中英文资料" not in text
    assert "进入知识检索" in {button.label for button in app.button}
    assert "发起变更审查" in {button.label for button in app.button}
    assert "Case A" not in text and "演示案例 A" not in text
    assert "Case B" not in text and "演示案例 B" not in text
    assert app.session_state["official_nav"] == "总览"
    assert "总览" in {button.label for button in app.button}


def test_public_rag_keeps_answer_before_real_cited_source(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    assert not app.exception
    assert app.selectbox(key="official_language").value == "zh"
    assert app.selectbox(key="official_version").value == "wiki-1eadc6584f96"
    next(button for button in app.button if button.label == "生成带引用回答").click().run()
    assert not app.exception
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert {item.value for item in app.subheader} >= {"回答", "引用依据"}
    assert "阅读原始页面" in text
    assert "[1] Jetson 刷写与软件基线" in text
    assert "引用编号对应本次检索片段" in text
    assert "证据支撑度：一般" not in text
    assert "不代表事实正确性" in text
    assert "证据可信度" not in text


def test_rag_renders_approved_image_ocr_as_derived_evidence(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())

    image = {
        **CHUNK,
        "chunk_id": "wiki-1eadc6584f96:zh:guide/parameter/context:figure:1",
        "document_key": "guide/parameter/context",
        "document_id": "wiki-1eadc6584f96:zh:guide/parameter/context",
        "heading": "参数上下文 / 查看运行结果",
        "content": "Node_A 日志截图显示输出 100 和 66。",
        "modality": "image_ocr", "figure_id": "64bd324feb4e55a2",
        "review_status": "approved", "sha256": "a" * 64,
        "repository": "Seeed-Studio/wiki-documents",
        "commit": "1eadc6584f962b6efdbdb3e49b2b4ce30c85be08",
        "raw_url": "https://raw.githubusercontent.com/Seeed-Studio/wiki-documents/1eadc6584f962b6efdbdb3e49b2b4ce30c85be08/docs/img/example.png",
        "source_url": "https://github.com/Seeed-Studio/wiki-documents/blob/1eadc6584f962b6efdbdb3e49b2b4ce30c85be08/docs/cn/jetson_developtool_flash_firmware.md",
    }
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "query": question, "results": [dict(image)], "retrieval_policy": "bm25_figure_ocr",
    })

    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "仅查看检索原文").click().run()

    assert not app.exception
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "截图 OCR 证据" in text
    assert "人工校对的 OCR 派生证据，请对照原图" in text
    assert "[查看原图]" in text
    assert "阅读原始页面" in text


def test_rag_hides_approved_image_ocr_from_a_repository_outside_edge_workspace(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())

    image = {
        **CHUNK,
        "modality": "image_ocr", "figure_id": "foreign-figure",
        "review_status": "approved", "sha256": "a" * 64,
        "repository": "Seeed-Studio/wiki-documents",
        "commit": "a190201acffa03d199d4ca216288734a6513de3d",
        "raw_url": "https://raw.githubusercontent.com/Seeed-Studio/wiki-documents/a190201acffa03d199d4ca216288734a6513de3d/docs/img/example.png",
        "source_url": "https://wiki.seeedstudio.com/cn/a190201acffa03d199d4ca216288734a6513de3d/docs/docs/zh/guide/example.md",
        "content": "foreign project screenshot OCR",
    }
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "query": question, "results": [dict(image)], "retrieval_policy": "bm25_figure_ocr",
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "仅查看检索原文").click().run()

    assert not app.exception
    visible = "\n".join(item.value for group in (app.markdown, app.caption, app.info, app.warning) for item in group)
    assert "截图证据未通过来源校验，已隐藏识别文本" in visible
    assert "foreign project screenshot OCR" not in visible


def test_rag_hides_image_ocr_when_approval_or_pinned_image_url_is_invalid(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    image = {
        **CHUNK,
        "modality": "image_ocr", "figure_id": "untrusted-figure",
        "review_status": "pending", "sha256": "bad-hash",
        "raw_url": "https://example.com/image.png",
        "content": "untrusted screenshot text",
    }
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "query": question, "results": [dict(image)], "retrieval_policy": "bm25_figure_ocr",
    })

    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "仅查看检索原文").click().run()

    assert not app.exception
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info) + list(app.warning))
    assert "截图证据未通过来源校验，已隐藏" in text
    assert "untrusted screenshot text" not in text


def test_version_selector_uses_latest_published_workspace_version(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: {
        "workspace": "reComputer Industrial / Jetson", "baseline_version": "wiki-1eadc6584f96",
        "current_version": "wiki-next-snapshot", "available_versions": ["wiki-next-snapshot", "wiki-1eadc6584f96"],
        "source_count": 70, "chunk_count": 800,
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    assert not app.exception
    assert app.selectbox(key="official_version").value == "wiki-next-snapshot"
    labels = app.selectbox(key="official_version").options
    assert labels[0].startswith("wiki-next-snapshot") and labels[1].startswith("wiki-1eadc6584f96")
    assert labels[2] == "全部已收录版本"


def test_version_selector_tracks_new_latest_release_after_manual_old_selection(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    workspace = {
        "workspace": "reComputer Industrial / Jetson", "baseline_version": "wiki-previous-snapshot",
        "current_version": "wiki-1eadc6584f96", "available_versions": ["wiki-1eadc6584f96", "wiki-previous-snapshot"],
        "source_count": 52, "chunk_count": 659,
        "latest_source_retrieval_timestamp": "2026-09-27T16:48:07.391969+00:00",
    }
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: dict(workspace))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    assert app.selectbox(key="official_version").value == "wiki-1eadc6584f96"

    workspace.update(
        baseline_version="wiki-1eadc6584f96", current_version="wiki-next-snapshot",
        available_versions=["wiki-next-snapshot", "wiki-1eadc6584f96"],
        latest_source_retrieval_timestamp="2026-09-28T09:15:00+00:00",
    )
    app.button(key='refresh_workspace').click().run()

    assert not app.exception
    assert app.selectbox(key="official_version").value == "wiki-next-snapshot"
    app.selectbox(key="official_version").set_value("wiki-1eadc6584f96").run()
    assert app.selectbox(key="official_version").value == "wiki-1eadc6584f96"
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "最新已收录" in app.selectbox(key="official_version").options[0]
    assert "最近收录：" in visible and "2026-09-28 09:15 UTC" in visible


def test_offline_version_fallback_is_not_presented_as_latest(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: None)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    assert not app.exception
    assert app.selectbox(key="official_version").value == "等待连接"
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "无法确认最新已收录版本" in visible
    assert "暂未获取当前版本信息" in visible


def test_agent_result_prioritizes_analysis_and_pairs_actions_with_evidence(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)

    assert not app.exception
    visible = "\n".join(
        item.value for collection in (app.markdown, app.caption, app.subheader, app.info)
        for item in collection
    )
    assert "本次分析结论" in visible
    assert "优先核对的影响候选" in visible
    assert "建议核对动作" in visible
    assert "引用证据" in visible
    assert "等待人工审核" in visible
    source_links = [item.value for item in app.markdown if "[打开官方原文]" in item.value]
    assert len(source_links) == 1
    assert "可能相关资料" not in {item.value for item in app.subheader}


def test_review_decision_cannot_approve_a_different_analysis(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)
    next(button for button in app.button if button.label == "确认已审阅本次影响分析").click().run()
    request_target = app.session_state["official_review_decision_target"]

    app.session_state["official_review"] = {
        "request_mode": "selected_source",
        "selected_source": {"chunk_id": "another-source"},
        "patch_candidate": {"proposed_after": "另一个会话草案"},
    }
    next(button for button in app.button if button.label == "人工审核").click().run()

    assert not app.exception
    assert not any("已记录本次会话对会话草案" in item.value for item in app.success)
    next(button for button in app.button if button.label == "确认已审阅会话草案").click().run()
    assert app.session_state["official_review_decision_target"] != request_target


def test_agent_shows_retrieval_trace_and_uncovered_change_clause(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    def search(self, question, **scope):
        rows = [] if question.startswith("待排查") else [dict(CHUNK)]
        return {"query": question, "results": rows, "retrieval_policy": "bm25"}

    monkeypatch.setattr(PublicKnowledgeClient, "search", search)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app, "调整参数优先级；待排查调度失败恢复说明。")

    assert not app.exception
    assert app.session_state["official_request_review"]["retrieval_trace"]["uncovered_queries"]
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "检索过程与覆盖范围" in {item.label for item in app.expander}
    assert "待排查调度失败恢复说明" in visible


def test_agent_surfaces_stage_status_and_unverified_translation_check(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    related = {
        "gap_type": "UNVERIFIED_TRANSLATION",
        "message": "中英文对应关系待核验；不能据此判定内容不一致。",
        "description": "中英文对应关系待核验；不能据此判定内容不一致。",
        "missing_source_type": "经人工确认的中英文对应关系",
        "expected_version": "0.52.0",
        "suggested_query": "核验对应语种资料：docs-main:en:planning/example",
        "suggested_action": "先确认两份资料确为同一范围，再检查更新时间。",
    }
    original_search = PublicKnowledgeClient.search

    def search(self, question, **scope):
        result = original_search(self, question, **scope)
        result["results"] = [{
            **result["results"][0],
            "document_relationships": [{
                "relation_type": "translation_of", "verification_status": "candidate",
            }],
        }]
        return result

    monkeypatch.setattr(PublicKnowledgeClient, "search", search)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)

    assert not app.exception
    result = app.session_state["official_request_review"]
    result["stage_status"] = {"planning": "OK", "retrieval": "OK", "generation": "OK"}
    result["evidence_gap_details"] = [related]
    app.session_state["official_request_review"] = result
    app.run()

    assert not app.exception
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "流程状态" in visible
    assert "对应关系待核验" in visible
    assert "0.52.0" in visible
    assert "核验对应语种资料" in visible
    assert "待补充核查" in {item.value for item in app.subheader}


def test_agent_shows_model_abstention_gaps_as_unconfirmed_prompts(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "review_advice", lambda self, change_summary, evidence_chunk_ids: {
        "status": "ABSTAINED", "answer": "N/A", "sources": [],
        "evidence": [dict(CHUNK)],
        "review": {
            "change_interpretation": "需要进一步核对。",
            "impact_candidates": [],
            "evidence_gaps": ["缺少下游节点恢复行为说明。"],
            "version_ambiguities": ["尚未核对历史版本。"],
            "reviewer_actions": ["补充恢复策略来源后重新审查。"],
            "review_status": "REQUIRES_HUMAN_REVIEW",
        },
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app, "调整故障恢复策略")

    assert not app.exception
    result = app.session_state["official_request_review"]
    assert result["impacts"] == []
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "模型提示待核对" in visible
    assert "缺少下游节点恢复行为说明" in visible
    assert "尚未核对历史版本" in visible
    assert "补充恢复策略来源后重新审查" in visible


def test_review_export_records_task_evidence_and_human_decision_without_raw_draft():
    from public_workbench import _review_report

    result = {
        "task_id": "task-123", "request_fingerprint": "request-sha", "request_mode": "selected_source",
        "selected_source": {"chunk_id": "wiki-1eadc6584f96:zh:guide/test:1", "source_url": "https://github.com/Seeed-Studio/wiki-documents/example"},
        "patch_candidate": {"before": "official text", "proposed_after": "private proposed text"},
        "retrieved_results": [], "review_advice": {"status": "OK", "review": {"impact_candidates": []}},
        "retrieval_trace": {"queries": []}, "evidence_gaps": [], "public_baseline_written": False,
    }

    report = _review_report(result, "reviewed", "2026-09-28T00:00:00+00:00")

    assert report["task_id"] == "task-123"
    assert report["human_decision"] == "reviewed"
    assert report["selected_source_id"] == "wiki-1eadc6584f96:zh:guide/test:1"
    assert report["public_baseline_written"] is False
    assert report["proposed_after_sha256"]
    assert report["schema_version"] == 3
    assert "private proposed text" not in json.dumps(report, ensure_ascii=False)


def test_generation_rate_limit_keeps_evidence_and_explains_retry(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "GENERATION_RATE_LIMITED", "consistency_notes": [],
        "generation": {
            "request_id": "test-request-1", "provider": "deepseek",
            "requested_model": "deepseek-chat", "returned_model": None,
            "finish_reason": None, "usage": None, "latency_ms": 1480,
        },
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert app.error
    assert "限流" in visible and "稍后重试" in visible
    assert "[1] Jetson 刷写与软件基线" in visible
    assert "test-request-1" in visible


def test_long_hit_fragment_is_complete_once_and_links_to_original_page(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    long_hit = {**CHUNK, "content": "命中证据。" * 200}
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "检索证据支持的回答。", "sources": [long_hit], "evidence": [long_hit],
        "status": "OK", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    visible = "\n".join(item.value for item in app.markdown)
    assert sum(item.value == long_hit['content'] for item in app.markdown) == 1
    detail = next(item for item in app.expander if item.label == '查看原文片段 · 1 段')
    assert not detail.proto.expanded
    assert all("完整命中片段" not in item.label for item in app.expander)
    assert "阅读原始页面" in visible


def test_generated_answer_shows_cited_evidence_first_and_collapses_other_hits(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    other = {**CHUNK, "chunk_id": "wiki-1eadc6584f96:zh:guide/parameter/local:2", "heading": "本地参数"}
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "启动参数优先于本地参数。", "sources": [dict(CHUNK)],
        "evidence": [dict(CHUNK), other], "status": "OK", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    assert "引用依据" in {item.value for item in app.subheader}
    assert any(item.label == "查看其余检索结果（1）" for item in app.expander)
    assert app.button(key='knowledge_search')


def test_generated_answer_renders_claim_level_references_without_confidence_grade(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "当前片段记录上游参数优先。", "sources": [dict(CHUNK)],
        "claims": [{"text": "当前片段记录上游参数优先。", "source_indexes": [1]}],
        "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "当前片段记录上游参数优先" in visible
    assert "[1]" in visible
    assert "引用编号对应本次检索片段" in visible
    assert "证据支撑度：" not in visible


def test_agent_starts_with_natural_language_and_uses_rag_to_find_candidates(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    calls = []
    def search(self, question, *, version, language, top_k=5, **scope):
        calls.append((question, version, language, top_k))
        return {"query": question, "results": [dict(CHUNK)], "retrieval_policy": "bm25"}

    monkeypatch.setattr(PublicKnowledgeClient, "search", search)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()

    assert app.text_area(key="official_change_request").label == "描述研发变更"
    assert not any(box.label == "选择真实研发资料" for box in app.selectbox)
    change_request = "计划将 J4012 的软件基线升级至 JetPack 7.2，请核对升级和验证资料。"
    app.text_area(key="official_change_request").set_value(change_request).run()
    next(button for button in app.button if button.label == "检索资料并分析影响").click().run()

    assert not app.exception
    assert calls
    assert all(call[2] == "zh" for call in calls)
    assert all(call[1] == "wiki-1eadc6584f96" for call in calls)
    assert app.session_state["official_request_review"]["request_summary"] == change_request
    assert app.session_state["official_request_review"]["impacts"][0]["evidence"]["chunk_id"] == CHUNK["chunk_id"]
    headings = [item.value for item in app.subheader]
    assert headings.index("模型辅助核对建议") < headings.index("优先核对的影响候选")
    assert "建议引用的官方片段（变更分析）" not in headings
    evidence_ids = [item["chunk_id"] for item in app.session_state["official_request_review"]["retrieved_results"]]
    assert len(evidence_ids) == len(set(evidence_ids))


def test_rag_suggested_question_is_muted_placeholder_and_used_when_submitted_blank(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())

    submitted = []
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: (
        submitted.append(question) or {
            "answer": "依据官方资料生成的回答。", "sources": [dict(CHUNK)],
            "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
        }
    ))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    question = app.text_area(key="official_question")
    assert question.value == ""
    example = _edge_workspace()["domain_profile"]["example_queries"][0]
    assert question.placeholder == example
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    assert submitted == [example]
    assert "依据官方资料生成的回答。" in "\n".join(item.value for item in app.markdown)


def test_rag_generation_uses_user_edited_question(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    submitted = []
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: (
        submitted.append(question) or {
            "answer": "依据官方资料生成的回答。", "sources": [dict(CHUNK)],
            "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
        }
    ))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    app.text_area(key="official_question").set_value("规划验证器如何检查轨迹？").run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    assert submitted == ["规划验证器如何检查轨迹？"]


def test_suggested_questions_match_current_edge_ai_profile(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    workspace = _edge_workspace()
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: workspace)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    example_picker = app.selectbox(key="official_example")
    options = example_picker.options
    assert options == workspace["domain_profile"]["example_queries"]
    assert not any(question.isascii() for question in options)
    assert not any("Jetson" in question or "API server" in question for question in options)
    assert not any(item.label == "从已收录资料选择示例问题" for item in app.expander)
    assert example_picker.label == "示例问题（选择后可编辑）"

    selected_question = options[1]
    example_picker.set_value(selected_question).run()
    assert app.text_area(key="official_question").value == selected_question


def test_evidence_image_markdown_uses_text_placeholder_instead_of_missing_asset(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    image_chunk = {
        **CHUNK,
        "content": "Dependencies support execution tracking.\n\n"
                   "![reComputer Industrial / Jetson](../../../img/introduction_ui.png)",
    }
    monkeypatch.setattr(PublicKnowledgeClient, "search", lambda self, question, **scope: {
        "query": question, "results": [dict(image_chunk)], "retrieval_policy": "bm25",
        "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "仅查看检索原文").click().run()

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "原文配图「reComputer Industrial / Jetson」" in rendered
    assert "![reComputer Industrial / Jetson]" not in rendered


def test_agent_original_source_uses_placeholder_for_missing_markdown_images(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    image_chunk = {
        **CHUNK,
        "content": "reComputer Industrial / Jetson guide.\n\n"
                   "![reComputer Industrial / Jetson](../../../img/introduction_ui.png)",
    }
    monkeypatch.setattr(PublicKnowledgeClient, "document", lambda self, document_id: [dict(image_chunk)])

    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()
    _start_agent_request(app)

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "原文配图「reComputer Industrial / Jetson」" in rendered
    assert "![reComputer Industrial / Jetson]" not in rendered


def test_public_rag_without_generation_shows_compact_evidence_fallback(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "GENERATION_NOT_CONFIGURED", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert not app.exception
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert app.error
    assert "检索证据仍可查看" in visible
    assert "模型生成尚未启用" in visible
    assert "API_KEY" not in visible
    assert "start_prototype.ps1" not in visible
    assert "[1] Jetson 刷写与软件基线" in visible
    assert "检索候选" not in visible
    assert app.button(key='knowledge_search')
    assert "仅查看检索原文" in {button.label for button in app.button}


def test_public_rag_generation_failure_explains_backend_fallback(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "FAIL_CLOSED", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "生成未完成（FAIL_CLOSED）" in visible
    assert "联系维护者时请提供请求编号" in visible
    assert "[1] Jetson 刷写与软件基线" in visible


def test_abstained_answer_explains_missing_retrieval_terms(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "ABSTAINED", "consistency_notes": [],
        "generation": {
            "failure_reason": "MODEL_NO_SUPPORTED_ANSWER",
            "candidate_count": 5,
            "evidence_coverage": {"matched_terms": ["api", "server"], "missing_terms": ["health", "check", "endpoint"]},
            "request_id": "request-abstained-test",
        },
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "模型服务正常" in visible
    assert "health、check、endpoint" in visible
    assert "未覆盖关键词" in visible
    assert "不是网络或 API Key 故障" not in visible
    assert "[1] Jetson 刷写与软件基线" in visible


def test_abstained_answer_distinguishes_keyword_match_from_supported_answer(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "ABSTAINED", "consistency_notes": [],
        "generation": {
            "failure_reason": "MODEL_NO_SUPPORTED_ANSWER",
            "candidate_count": 5,
            "evidence_coverage": {
                "matched_terms": ["api", "server", "health", "check", "endpoint"],
                "missing_terms": [],
            },
        },
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.info))
    assert "没有找到可支持答案的原文" in visible
    assert "这是证据不足导致的拒答，不是网络或 API Key 故障" in visible
    assert "当前证据覆盖了部分问题关键词" not in visible


def test_public_rag_has_no_fixed_generation_count_limit(monkeypatch):
    _mock_client(monkeypatch)
    monkeypatch.setenv("MAX_LLM_CALLS_PER_SESSION", "10")
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    app.session_state["official_generation_calls"] = 1000
    app.run()

    assert app.button(key="knowledge_generate").disabled is False
    captions = [item.value for item in app.caption]
    assert not any("剩余生成次数" in caption or "生成额度已用完" in caption for caption in captions)
    assert any("费用与限流以服务商规则为准" in caption for caption in captions)


def test_provider_connection_failure_explains_proxy_or_network(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": "GENERATION_PROVIDER_UNAVAILABLE", "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert app.error
    assert "后端无法连接模型服务" in visible
    assert "检索证据已保留" in visible
    assert "[1] Jetson 刷写与软件基线" in visible


def test_provider_rejection_and_invalid_response_explain_safe_fallback(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient

    statuses = iter(("GENERATION_PROVIDER_REJECTED", "GENERATION_RESPONSE_INVALID"))
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "N/A", "sources": [], "evidence": [dict(CHUNK)],
        "status": next(statuses), "consistency_notes": [],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "模型服务拒绝了请求" in visible
    assert "检索证据已保留" in visible

    next(button for button in app.button if button.label == "生成带引用回答").click().run()
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "模型输出未通过结构校验" in visible
    assert "检索证据已保留" in visible


def test_generated_answer_typography_is_large_and_readable():
    from components.public_theme import PUBLIC_CSS

    assert '.st-key-generated_answer [data-testid="stMarkdownContainer"] p {' in PUBLIC_CSS
    assert "font-size:1.3rem!important;line-height:1.75!important;" in PUBLIC_CSS
    assert "font-size:1.3rem!important;line-height:1.75!important;" in PUBLIC_CSS


def test_breadcrumbs_are_clickable_and_back_returns_to_previous_module(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()

    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    labels = {button.label for button in app.button}
    assert {"←", "首页", "版本化知识服务"} <= labels
    assert "返回首页" not in labels
    assert any(
        'class="breadcrumbs-current"' in item.value
        and 'aria-current="page"' in item.value
        and "版本化知识检索与问答" in item.value
        for item in app.markdown
    )

    next(button for button in app.button if button.label == "发起变更审查").click().run()
    next(button for button in app.button if button.label == "←").click().run()
    assert app.session_state["official_nav"] == "版本检索与问答"

    next(button for button in app.button if button.label == "首页").click().run()
    assert app.session_state["official_nav"] == "总览"

    next(button for button in app.button if button.label == "发起变更审查").click().run()
    next(button for button in app.button if button.label == "变更影响审查").click().run()
    assert app.session_state["official_nav"] == "新建变更审查"
    next(button for button in app.button if button.label == "首页").click().run()
    assert app.session_state["official_nav"] == "总览"


def test_review_subpage_breadcrumb_and_back_history(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)

    next(button for button in app.button if button.label == "人工审核").click().run()
    assert app.session_state["official_nav"] == "人工审核"
    assert {"←", "首页", "变更影响审查"} <= {button.label for button in app.button}
    assert "返回审查" not in {button.label for button in app.button}

    next(button for button in app.button if button.label == "变更影响审查").click().run()
    assert app.session_state["official_nav"] == "新建变更审查"
    assert app.session_state["official_request_review"]["sandbox_only"] is True

    next(button for button in app.button if button.label == "←").click().run()
    assert app.session_state["official_nav"] == "人工审核"


def test_rag_actions_alerts_and_typography_use_neutral_accessible_styles(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    assert app.button(key="knowledge_generate")
    assert app.button(key="knowledge_search")

    from components.public_theme import PUBLIC_CSS

    assert ".st-key-knowledge_generate button" in PUBLIC_CSS
    assert ".st-key-knowledge_search button" in PUBLIC_CSS
    assert "[data-testid=\"stAlert\"]" in PUBLIC_CSS
    assert "background:var(--paper)!important" in PUBLIC_CSS
    assert "[data-testid=\"stMarkdownContainer\"] p" in PUBLIC_CSS
    assert "font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem)!important;" in PUBLIC_CSS


def test_knowledge_top_k_applies_to_generation_and_raw_search_and_scrolls_to_results(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    import public_workbench

    calls = []
    scrolled = []

    def query_official(self, question, *, version, language, top_k=5, retrieval_policy="bm25"):
        calls.append(("query", question, version, language, top_k, retrieval_policy))
        return {
            "answer": "上游参数优先于启动参数。", "sources": [dict(CHUNK)],
            "claims": [{"text": "上游参数优先于启动参数。", "source_indexes": [1]}],
            "evidence": [dict(CHUNK)], "status": "OK", "consistency_notes": [],
        }

    def search(self, question, *, version, language, top_k=5, retrieval_policy="bm25"):
        calls.append(("search", question, version, language, top_k, retrieval_policy))
        return {
            "query": question, "results": [dict(CHUNK)], "retrieval_policy": "bm25",
            "consistency_notes": [],
        }

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", query_official)
    monkeypatch.setattr(PublicKnowledgeClient, "search", search)
    monkeypatch.setattr(public_workbench, "_scroll_to_results", lambda: scrolled.append(True))
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()

    app.slider(key="official_top_k").set_value(8).run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()
    assert calls[-1][0] == "query" and calls[-1][4] == 8 and calls[-1][5] == "bm25"
    assert scrolled == [True]

    next(button for button in app.button if button.label == "仅查看检索原文").click().run()
    assert calls[-1][0] == "search" and calls[-1][4] == 8 and calls[-1][5] == "bm25"
    assert scrolled == [True, True]


def test_public_knowledge_client_sends_selected_policy_to_search_and_generation(monkeypatch):
    from services.public_knowledge_client import PublicKnowledgeClient

    calls = []
    client = object.__new__(PublicKnowledgeClient)
    client.timeout = 30.0
    monkeypatch.setattr(client, "_request", lambda method, path, **kwargs: calls.append((method, path, kwargs)) or {})
    client.search(
        "问题", version="v2.9.0", language="zh",
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )
    client.query_official(
        "问题", version="v2.9.0", language="zh",
        retrieval_policy="bm25_pphuman_term_expansion_rrf",
    )

    assert calls[0][2]["json"]["retrieval_policy"] == "bm25_pphuman_term_expansion_rrf"
    assert calls[1][2]["json"]["retrieval_policy"] == "bm25_pphuman_term_expansion_rrf"


def test_pphuman_rag_page_exposes_experimental_policy_and_passes_it_to_answer(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    import public_workbench

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _pphuman_workspace())
    calls = []

    def query_official(self, question, *, version, language, top_k=5, retrieval_policy="bm25", **scope):
        calls.append(retrieval_policy)
        return {
            "answer": "N/A", "sources": [], "claims": [], "evidence": [],
            "status": "NO_EVIDENCE", "consistency_notes": [],
            "retrieval_policy": retrieval_policy,
        }

    monkeypatch.setattr(PublicKnowledgeClient, "query_official", query_official)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    policy_widget = app.selectbox(key="official_retrieval_policy")
    assert "术语融合 + 模型重排（实验）" in policy_widget.options
    policy_widget.set_value("术语融合 + 模型重排（实验）").run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()

    assert calls == ["task_adaptive_rerank"]


def test_pphuman_agent_page_exposes_experimental_policy_and_passes_it_to_review(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    import public_workbench

    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _pphuman_workspace())
    calls = []
    monkeypatch.setattr(
        public_workbench, "_analyze_change_request",
        lambda client, summary, *args, **kwargs: calls.append(kwargs["retrieval_policy"]),
    )
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()
    app.text_area(key="official_change_request").set_value(
        "调整行人跟踪器配置，核对相关部署和回归验证。"
    ).run()
    policy_widget = app.selectbox(key="agent_retrieval_policy")
    assert "术语融合 + 模型重排（实验）" in policy_widget.options
    policy_widget.set_value("术语融合 + 模型重排（实验）").run()
    next(button for button in app.button if button.label == "检索资料并分析影响").click().run()

    assert calls == ["task_adaptive_rerank"]


def test_scroll_to_results_uses_same_document_html_script(monkeypatch):
    import public_workbench

    rendered = []
    monkeypatch.setattr(
        public_workbench.st, "html",
        lambda body, *, unsafe_allow_javascript=False: rendered.append((body, unsafe_allow_javascript)),
    )

    public_workbench._scroll_to_results()

    assert rendered and rendered[0][1] is True
    assert 'getElementById("knowledge-results-anchor")' in rendered[0][0]
    assert "scrollIntoView" in rendered[0][0]


def test_streamlit_alert_inner_layers_cannot_restore_blue_backgrounds():
    from components.public_theme import PUBLIC_CSS

    assert '[data-testid="stAlert"] *' in PUBLIC_CSS
    assert '[role="alert"] *' in PUBLIC_CSS
    assert "background-image:none!important;" in PUBLIC_CSS
    assert "background-color:transparent!important;" in PUBLIC_CSS


def test_legacy_ui_theme_has_no_blue_tinted_surfaces():
    app_source = APP.read_text(encoding="utf-8")

    for color in ("#f4f7fb", "#f7f9fc", "#eaf0f7", "#f5f9ff", "#edf4fb", "#eef4fa"):
        assert color not in app_source


def test_workbench_theme_keeps_neutral_base_and_equal_home_cards():
    from components.public_theme import PUBLIC_CSS, _frame_data_uri

    assert "--canvas:#f5f5f5" in PUBLIC_CSS
    assert "--ink:#171717" in PUBLIC_CSS
    assert "--quiet:#f7f7f7" in PUBLIC_CSS
    assert "--blue:" not in PUBLIC_CSS
    assert "background:var(--ink)!important" not in PUBLIC_CSS
    assert ".st-key-public_rag_module,.st-key-public_agent_module {" in PUBLIC_CSS
    assert "background:var(--paper)!important;color:var(--ink);" in PUBLIC_CSS
    assert '[data-testid="stSelectbox"] .react-aria-ComboBox [role="group"] {' in PUBLIC_CSS
    assert "min-height:25rem;padding:4.2rem clamp(3.5rem,4.6vw,4.5rem) 3.7rem!important;" in PUBLIC_CSS
    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in PUBLIC_CSS
    assert '[data-testid="stHorizontalBlock"]:has(.st-key-public_rag_module) {flex-direction:column!important;' in PUBLIC_CSS
    assert '.st-key-public_rag_module::before {background-image:url("' + _frame_data_uri("rag-book-frame.svg") + '");}' in PUBLIC_CSS
    assert '.st-key-public_agent_module::before {background-image:url("' + _frame_data_uri("agent-robot-frame.svg") + '");}' in PUBLIC_CSS
    assert ".flow-track span:not(:last-child)::after" in PUBLIC_CSS
    assert "[class*=\"st-key-source_card_\"]" in PUBLIC_CSS
    assert ".st-key-generated_answer {" in PUBLIC_CSS
    assert "border-left:3px solid var(--ink)!important;border-radius:0!important;" in PUBLIC_CSS


def test_public_workbench_theme_sets_black_primary_accent():
    import tomllib

    config_path = APP.parent / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["theme"]["primaryColor"] == "#171717"


def test_home_module_action_clearance_and_readable_text_scale_are_preserved():
    """Keep the book action above its inner page seam and ordinary UI copy legible."""
    from components.public_theme import PUBLIC_CSS

    book_frame = (APP.parent / "assets" / "rag-book-frame.svg").read_text(encoding="utf-8")

    assert "v368" in book_frame
    assert "M320 381v32" in book_frame
    assert "min-height:25rem;padding:4.2rem clamp(3.5rem,4.6vw,4.5rem) 3.7rem!important;" in PUBLIC_CSS
    assert "p,li {font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem);line-height:1.65;color:var(--body);}" in PUBLIC_CSS
    assert "[data-testid=\"stButton\"] button {min-height:3rem;border-radius:4px;font-size:clamp(1.12rem,1.05rem + .15vw,1.24rem);" in PUBLIC_CSS


def test_workbench_typography_overrides_streamlit_default_small_text_nodes():
    from components.public_theme import PUBLIC_CSS

    assert '[data-testid="stMarkdownContainer"] p,' in PUBLIC_CSS
    assert '[data-testid="stMarkdownContainer"] li {font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem)!important;' in PUBLIC_CSS
    assert '[data-testid="stButton"] button [data-testid="stMarkdownContainer"] p {color:inherit!important;font-size:clamp(1.12rem,1.05rem + .15vw,1.24rem)!important;line-height:1.25!important;}' in PUBLIC_CSS
    assert '[data-testid="stWidgetLabel"] p {font-size:1.12rem!important;line-height:1.45!important;}' in PUBLIC_CSS
    assert '[data-testid="stExpander"] summary p {font-size:1.14rem!important;' in PUBLIC_CSS
    assert '.home-snapshot {color:var(--muted);font-size:1.18rem;line-height:1.55;}' in PUBLIC_CSS


def test_corpus_image_references_have_readable_local_descriptions():
    from public_workbench import _replace_markdown_images

    content = '前文 ![流程图](../flow.png) <p align="center"><img src="../step.png" alt="执行步骤"/></p>'
    rendered = _replace_markdown_images(content)
    assert "原文配图「流程图」" in rendered
    assert "原文配图「执行步骤」" in rendered
    assert "图中文字未纳入检索" in rendered
    assert "<img" not in rendered and "<p align" not in rendered


def test_relative_corpus_links_point_to_the_pinned_official_source():
    from public_workbench import _rewrite_relative_source_links

    source_url = (
        "https://wiki.seeedstudio.com/cn/"
        "a190201acffa03d199d4ca216288734a6513de3d/"
        "docs/docs/zh/guide/parameter/priority.md"
    )
    content = (
        "[内置参数](built-in.md) [本章](#priority) "
        "[官方发布](https://wiki.seeedstudio.com/cn/)"
    )
    rendered = _rewrite_relative_source_links(content, source_url)

    assert (
        "[内置参数](https://wiki.seeedstudio.com/cn/"
        "a190201acffa03d199d4ca216288734a6513de3d/"
        "docs/docs/zh/guide/parameter/built-in.md)"
    ) in rendered
    assert f"[本章]({source_url}#priority)" in rendered
    assert "[官方发布](https://wiki.seeedstudio.com/cn/)" in rendered


def test_relative_wiki_links_resolve_to_seeed_origin_without_rewriting_external_links():
    from public_workbench import _rewrite_relative_source_links

    source_url = "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/"
    content = "[刷写指南](/cn/flash/jetpack_to_selected_product/) [同目录](../faq/) [外部](https://example.com/a)"
    rendered = _rewrite_relative_source_links(content, source_url)

    assert "[刷写指南](https://wiki.seeedstudio.com/cn/flash/jetpack_to_selected_product/)" in rendered
    assert "[同目录](https://wiki.seeedstudio.com/cn/faq/)" in rendered
    assert "[外部](https://example.com/a)" in rendered


def test_wide_layout_uses_full_main_column_and_unframed_back_arrow():
    from components.public_theme import PUBLIC_CSS

    assert "max-width:none!important;width:100%!important;margin:0!important;" in PUBLIC_CSS
    assert ".block-container {background:var(--paper);" in PUBLIC_CSS
    assert '[data-testid="stHeader"] {background:var(--paper);}' in PUBLIC_CSS
    assert '[data-testid="stSidebar"][aria-expanded="true"]' in PUBLIC_CSS
    assert "width:clamp(260px,17vw,330px)!important;" in PUBLIC_CSS
    assert '[data-testid="stSidebar"][aria-expanded="false"]' in PUBLIC_CSS
    assert "flex:0 0 0!important;" in PUBLIC_CSS
    assert '[class*="st-key-page_header_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {' in PUBLIC_CSS
    assert "min-height:2.35rem;display:flex;align-items:center;justify-content:center;" in PUBLIC_CSS
    assert '[class*="st-key-page_header_"] [data-testid="stMarkdownContainer"] {overflow:visible;display:flex;align-items:center;min-height:2.35rem;margin:0!important;' in PUBLIC_CSS
    assert '[class*="st-key-page_header_"] [data-testid="stMarkdownContainer"] p {margin:0!important;' in PUBLIC_CSS
    assert ".breadcrumb-separator {height:2.35rem;box-sizing:border-box;display:flex;align-items:center;justify-content:center;" in PUBLIC_CSS
    assert "font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem);line-height:1.65" in PUBLIC_CSS
    assert "font-size:clamp(1.12rem,1.05rem + .15vw,1.24rem);font-weight:640" in PUBLIC_CSS
    assert "font-size:1.3rem;font-weight:560" in PUBLIC_CSS
    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in PUBLIC_CSS
    assert ".status-cell {display:flex;flex-direction:column;justify-content:center;min-height:6.1rem;" in PUBLIC_CSS
    assert "background:transparent!important;color:var(--ink)!important;border:0!important;box-shadow:none!important;" in PUBLIC_CSS


def test_workbench_navigation_is_larger_and_content_has_no_forced_empty_viewport_height():
    from components.public_theme import PUBLIC_CSS

    assert ".breadcrumbs {font-size:1.2rem;" in PUBLIC_CSS
    assert ".breadcrumbs-current {height:2.35rem;" in PUBLIC_CSS
    assert "color:var(--ink);font-size:1.2rem;font-weight:740" in PUBLIC_CSS
    assert "[data-testid=\"stSidebar\"] [data-testid=\"stButton\"] button" in PUBLIC_CSS
    assert "font-size:1.3rem;font-weight:560" in PUBLIC_CSS
    assert '[data-testid="stWidgetLabel"] p {font-size:1.12rem!important;line-height:1.45!important;}' in PUBLIC_CSS
    assert 'width:100%!important;max-width:100%!important;flex-wrap:wrap!important;' in PUBLIC_CSS
    assert '.breadcrumbs-current {height:auto;min-height:2.35rem;}' in PUBLIC_CSS
    assert "margin:1.7rem 0 .85rem" in PUBLIC_CSS
    assert "[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] {min-height:calc(100vh" not in PUBLIC_CSS
    assert ".block-container {background:var(--paper);max-width:none!important;width:100%!important;margin:0!important;box-sizing:border-box;\n  padding:3rem 1.25rem 1.5rem;overflow:visible;}" in PUBLIC_CSS
    assert "textarea::placeholder" in PUBLIC_CSS


def test_home_snapshot_follows_content_with_consistent_spacing(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()

    assert not app.exception
    assert any('class="home-snapshot"' in item.value for item in app.markdown)

    from components.public_theme import PUBLIC_CSS

    assert '[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {min-height:calc(100vh - 4.5rem);' not in PUBLIC_CSS
    assert '[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.home-snapshot) {' in PUBLIC_CSS
    assert "margin-top:1.15rem!important;border-top:1px solid var(--line);" in PUBLIC_CSS
    assert '[data-testid="stElementContainer"]:has(.home-snapshot) {\n  margin-top:auto!important' not in PUBLIC_CSS
    assert "@media (max-width:600px) {.status-grid {grid-template-columns:minmax(0,1fr);}" in PUBLIC_CSS
    assert ('[data-testid="stElementContainer"]:has(.home-snapshot) [data-testid="stMarkdownContainer"] {\n'
            '  margin:0!important;') in PUBLIC_CSS


def test_stale_navigation_state_recovers_to_home(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    app.session_state["official_nav"] = "removed page"
    app.run()

    assert not app.exception
    assert app.session_state["official_nav"] == "总览"
    assert [item.value for item in app.title] == ["研发知识版本服务与变更影响审查"]


def test_public_agent_change_and_review_are_session_local(monkeypatch):
    _mock_client(monkeypatch)
    first = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(first, "计划将全局参数优先级调整为最高，并找出要同步的资料。")
    assert not first.exception
    assert first.session_state["official_request_review"]["sandbox_only"] is True
    assert first.session_state["official_request_review"]["public_baseline_written"] is False
    assert "模型辅助核对建议" in {item.value for item in first.subheader}
    assert "依据引用片段，建议核对相关资料中的参数顺序。" in "\n".join(
        item.value for item in first.markdown
    )
    visible = "\n".join(item.value for item in list(first.markdown) + list(first.caption))
    assert "该章节说明参数优先级。" in visible
    assert "核对示例与运维说明是否同步。" in visible
    assert "尚未提供目标设备的实测记录。" in visible
    assert "等待人工审核" in visible
    headings = [item.value for item in first.subheader]
    assert headings.index("模型辅助核对建议") < headings.index("优先核对的影响候选")
    next(button for button in first.button if button.label == "确认已审阅本次影响分析").click().run()
    assert first.session_state["official_review_decision"] == "reviewed"

    second = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in second.button if button.label == "发起变更审查").click().run()
    assert "official_request_review" not in second.session_state
    assert "official_review_decision" not in second.session_state


def test_agent_draft_survives_navigation_away_and_back(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()
    request = "把全局参数调整为最高优先级，并检查需要同步的资料。"
    app.text_area(key="official_change_request").set_value(request).run()
    next(button for button in app.button if button.label == "总览").click().run()
    next(button for button in app.button if button.label == "发起变更审查").click().run()

    assert not app.exception
    assert app.text_area(key="official_change_request").value == request

def test_public_agent_shows_explicit_document_reference_without_confirming_paragraph_impact(monkeypatch):
    _mock_client(monkeypatch)
    import public_workbench

    analyze = public_workbench._analyze_hypothetical

    def with_document_reference(client, selected, proposed, **kwargs):
        result = analyze(client, selected, proposed, **kwargs)
        result["confirmed_relations"] = [{
            "relation_type": "DOCUMENT_REFERENCE",
            "source_document_id": "wiki-1eadc6584f96:zh:jetson-flash-guide",
            "target_document_id": "wiki-1eadc6584f96:zh:jetson-carrier-board-guide",
            "source_chunk_id": "wiki-1eadc6584f96:zh:jetson-flash-guide:2",
            "source_heading": "固件刷写与载板选择",
            "source_url": "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/",
            "source_excerpt": "刷写前需要确认设备型号、目标载板与系统镜像版本。",
        }]
        return result

    monkeypatch.setattr(public_workbench, "_analyze_hypothetical", with_document_reference)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)
    app.text_area(key="official_proposed_text").set_value("假设将启动参数提高到第一优先级。").run()
    next(button for button in app.button if button.label == "生成修改前后对照").click().run()
    assert not app.exception
    assert "已确认文档关联" in {item.value for item in app.subheader}
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "刷写前需要确认设备型号、目标载板与系统镜像版本" in visible
    assert "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/" in visible
    assert "官方实现 PR" not in visible and "DSIP" not in visible
    assert "不代表所选段落受影响" in visible
    assert "待核对资料" in visible
    assert "已确认关系" not in visible


def test_public_navigation_exposes_single_domain_and_validated_edge_evaluation(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    workspace = {
        **_edge_workspace(),
        "retrieval_evaluation_status": "edge_ai_retrieval_v2_validated",
        "retrieval_evaluation": {
            "name": "edge_ai_retrieval_v2", "policy": "bm25", "case_count": 22,
            "case_split_counts": {"dev": 11, "holdout": 11},
            "selection_reason": "DEV/HOLDOUT 上 BM25 Faceted RRF 未提高来源召回，保留 BM25。",
            "candidates": {
                "dev": {
                    "bm25": {
                        "mean_required_source_recall": 0.95,
                        "complete_required_source_set_rate": 0.90,
                        "wrong_scope_result_count": 0,
                        "unanswerable_candidate_rate": 1.0,
                        "latency_ms": {"p95": 4.085},
                    },
                    "bm25_faceted_rrf": {
                        "mean_required_source_recall": 0.95,
                        "complete_required_source_set_rate": 0.90,
                        "wrong_scope_result_count": 0,
                        "unanswerable_candidate_rate": 1.0,
                        "latency_ms": {"p95": 4.91},
                    },
                },
                "holdout": {
                    "bm25": {
                        "mean_required_source_recall": 0.90,
                        "complete_required_source_set_rate": 0.90,
                        "wrong_scope_result_count": 0,
                        "unanswerable_candidate_rate": 1.0,
                        "latency_ms": {"p95": 2.514},
                    },
                    "bm25_faceted_rrf": {
                        "mean_required_source_recall": 0.90,
                        "complete_required_source_set_rate": 0.90,
                        "wrong_scope_result_count": 0,
                        "unanswerable_candidate_rate": 1.0,
                        "latency_ms": {"p95": 2.764},
                    },
                },
            },
        },
    }
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: workspace)
    app = AppTest.from_file(APP, default_timeout=40).run()
    labels = {button.label for button in app.button}
    assert {"总览", "版本化知识检索", "版本与历史", "资料来源", "发起变更审查", "影响候选", "修改建议对照", "检索评测", "已知限制"} <= labels
    next(button for button in app.button if button.label == "检索评测").click().run()
    assert not app.exception
    visible = "\n".join(
        item.value for group in (app.markdown, app.caption, app.subheader, app.warning, app.info)
        for item in group
    )
    assert "当前语料评测状态" in visible
    assert "固定题集 22 题" in visible
    assert "BM25" in visible
    assert "Autoware" not in visible and "DolphinScheduler" not in visible
    assert "43 条可回答" not in visible


def test_benchmark_ignores_other_corpus_numbers_for_new_edge_workspace(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    workspace = {
        **_edge_workspace(),
        "retrieval_evaluation": {
            "name": "other_corpus_v2", "case_count": 90,
            "holdout": {"required_source_recall_at_5": 0.99},
        },
    }
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: workspace)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "检索评测").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.subheader) + list(app.warning))
    assert not app.exception
    assert "尚未完成与活动语料指纹绑定的冻结评测" in visible
    assert "0.99" not in visible and "other_corpus_v2" not in visible
    assert "其他领域语料上的分数不适用于本知识空间" in visible


def test_benchmark_displays_only_fingerprinted_edge_metrics_and_agent_limits(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    workspace = {
        **_edge_workspace(),
        "retrieval_evaluation_status": "edge_ai_retrieval_v2_validated",
        "retrieval_evaluation": {
            "name": "edge_ai_retrieval_v2", "policy": "bm25", "case_count": 22,
            "case_split_counts": {"dev": 11, "holdout": 11},
            "selection_reason": "当前冻结 DEV/HOLDOUT 测试下保留 BM25。",
            "candidates": {
                split: {
                    strategy: {
                        "mean_required_source_recall": 0.90,
                        "complete_required_source_set_rate": 0.90,
                        "wrong_scope_result_count": 0,
                        "unanswerable_candidate_rate": 1.0,
                        "latency_ms": {"p95": 2.0},
                    }
                    for strategy in ("bm25", "bm25_faceted_rrf")
                }
                for split in ("dev", "holdout")
            },
        },
        "change_review_evaluation": {
            "case_count": 12,
            "splits": {
                split: {
                    "expected_change_type_accuracy": 0.83,
                    "required_source_recall_across_planned_queries": 0.75,
                    "complete_required_source_set_count": 3,
                    "answerable_case_count": 4,
                    "scope_gap_detection_accuracy": 1.0,
                    "manual_review_boundary_accuracy": 1.0,
                    "human_quality_scoring": {"status": "not_scored"},
                }
                for split in ("dev", "holdout")
            },
        },
    }
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: workspace)
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "检索评测").click().run()

    assert not app.exception
    assert len(app.dataframe) == 2
    assert set(app.dataframe[0].value.iloc[:, 1]) == {"bm25", "bm25_faceted_rrf"}
    assert app.dataframe[0].value.iloc[0, 5] == 1.0
    assert app.dataframe[1].value.iloc[0, 5] == 1.0


def test_edge_version_page_distinguishes_snapshot_from_software_baselines(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: _edge_workspace())
    monkeypatch.setattr(PublicKnowledgeClient, "documents", lambda self: [])
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本与历史").click().run()

    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert not app.exception
    assert "当前固定中文资料快照" in visible
    assert "JetPack / L4T 基线" in visible
    assert "资料快照与软件发行版本分别管理" not in visible
    assert "不等于已验证兼容" in visible
def test_relative_official_link_stays_on_commit_or_becomes_plain_text():
    from public_workbench import _rewrite_relative_source_links

    source = "https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/"
    raw = "[安全章节](./jetson_developtool_supported_devices/) [中文目录](../) [越界章节](../../another-repo/README.md) [编码越界](%2e%2e/%2e%2e/en/README.md) [外部](https://example.com/x)"
    visible = _rewrite_relative_source_links(raw, source)

    assert "[安全章节](https://wiki.seeedstudio.com/cn/jetson_developtool_flash_firmware/jetson_developtool_supported_devices/)" in visible
    assert "[中文目录](https://wiki.seeedstudio.com/cn/)" in visible
    assert "越界章节" in visible
    assert "[越界章节](" not in visible
    assert "编码越界" in visible
    assert "[编码越界](" not in visible
    assert "[外部](https://example.com/x)" in visible


def test_verified_consistency_notice_names_primary_basis_and_both_versions(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    old = {**CHUNK, "version": "wiki-previous-snapshot", "source_url": "https://wiki.seeedstudio.com/cn//tag/wiki-previous-snapshot"}
    note = {
        "kind": "verified_literal_value_difference", "document_key": CHUNK["document_key"],
        "heading": CHUNK["heading"], "parameter": "worker.threads", "values": ["1000", "500"],
        "message": "同一章节中有不同的明确值。", "sources": [
            {"version": "wiki-1eadc6584f96", "locale": "zh-CN", "source_url": CHUNK["source_url"]},
            {"version": "wiki-previous-snapshot", "locale": "zh-CN", "source_url": old["source_url"]},
        ],
    }
    monkeypatch.setattr(PublicKnowledgeClient, "query_official", lambda self, question, **scope: {
        "answer": "当前片段记录 1000。", "sources": [dict(CHUNK)],
        "evidence": [dict(CHUNK), old], "status": "OK", "consistency_notes": [note],
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "版本化知识检索").click().run()
    next(button for button in app.button if button.label == "生成带引用回答").click().run()
    assert not app.exception
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.warning))
    assert "版本差异提醒" in visible
    assert "引用依据之一" in visible
    assert "wiki-previous-snapshot" in visible and "wiki-1eadc6584f96" in visible
    assert "概率" not in visible


def test_agent_review_sections_remain_session_bound_after_navigation(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)
    assert not app.exception
    next(button for button in app.button if button.label == "影响候选").click().run()
    visible = "\n".join(item.value for item in list(app.markdown) + list(app.caption) + list(app.error))
    assert "可能相关资料" in visible or "阅读原始页面" in visible
    next(button for button in app.button if button.label == "人工审核").click().run()
    assert app.session_state["official_request_review"]["sandbox_only"] is True
    next(button for button in app.button if button.label == "确认已审阅本次影响分析").click().run()
    assert app.session_state["official_review_decision"] == "reviewed"


def test_agent_stepper_marks_human_review_after_analysis(monkeypatch):
    _mock_client(monkeypatch)
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)
    tracks = [item.value for item in app.markdown if 'review-steps' in item.value]
    assert any('class="current">3. 等待人工审核' in track for track in tracks)
    next(button for button in app.button if button.label == "确认已审阅本次影响分析").click().run()
    tracks = [item.value for item in app.markdown if 'review-steps' in item.value]
    assert any('class="done">3. 人工审核已记录' in track for track in tracks)


def test_switching_source_resets_previous_unsent_draft(monkeypatch):
    _mock_client(monkeypatch)
    from services.public_knowledge_client import PublicKnowledgeClient
    second = {**CHUNK, "chunk_id": "wiki-1eadc6584f96:zh:guide/parameter/priority:2",
              "heading": "启动参数", "content": "启动参数是第二优先级。"}
    monkeypatch.setattr(PublicKnowledgeClient, "document",
                        lambda self, document_id: [dict(CHUNK), second])
    app = AppTest.from_file(APP, default_timeout=40).run()
    _start_agent_request(app)
    app.text_area(key="official_proposed_text").set_value("上一个段落的未提交草案").run()
    app.selectbox(key="official_change_chunk").set_value(second["chunk_id"]).run()
    assert not app.exception
    assert app.text_area(key="official_proposed_text").value == second["content"]
    assert _state_get(app.session_state, "official_review_decision") is None


def test_about_page_exposes_separate_frontend_backend_and_asset_fingerprints(monkeypatch):
    _mock_client(monkeypatch)
    import public_workbench

    monkeypatch.setattr(public_workbench, "ui_build_revision", lambda: "a" * 40)
    from services.public_knowledge_client import PublicKnowledgeClient
    monkeypatch.setattr(PublicKnowledgeClient, "workspace", lambda self: {
        "workspace": "其他领域", "repository": "Seeed-Studio/wiki-documents",
        "repositories": ["Seeed-Studio/wiki-documents"],
        "build_revision": "b" * 40,
        "corpus_fingerprint": {"fingerprint_sha256": "c" * 64},
        "retrieval_config_fingerprint": "d" * 64,
        "evaluation_fingerprint": "e" * 64,
    })
    app = AppTest.from_file(APP, default_timeout=40).run()
    next(button for button in app.button if button.label == "系统说明").click().run()
    assert not app.exception
    assert any(item.label == "运行版本与资料指纹" for item in app.expander)
