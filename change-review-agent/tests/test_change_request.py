from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.change_request import (
    CHANGE_TYPES,
    build_request_plan,
    classify_change_type,
    is_out_of_scope_public_request,
)
from app.domain_profile import load_change_profile


PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "pphuman_change_profile.json"


def test_active_profile_is_chinese_pphuman_engineering_only():
    profile = load_change_profile(PROFILE_PATH)

    assert profile["id"] == "pphuman"
    assert profile["languages"] == ["zh"]
    assert set(CHANGE_TYPES) == {
        "model_config", "behavior_pipeline", "tracking", "deployment", "general",
    }
    assert all(row["checklist"] for row in profile["change_types"])


@pytest.mark.parametrize(("summary", "expected"), [
    ("调整 PP-Human 的行人检测模型配置和推理阈值。", "model_config"),
    ("变更行人跟踪器参数并核对跨镜跟踪流程。", "tracking"),
    ("变更 PP-Human 推理流水线部署配置。", "deployment"),
    ("在 PP-Human v2.8.1 部署时调整行人跟踪模型的命令行参数。", "deployment"),
    ("启用行为识别并核对行为分析配置。", "behavior_pipeline"),
    ("变更视频行为识别任务开关和模型目录，核对行为教程与推理配置。", "behavior_pipeline"),
    ("检查当前资料里是否有对应的工程说明。", "general"),
])
def test_change_type_classification_uses_only_current_pphuman_profile(summary, expected):
    assert classify_change_type(summary) == expected


def test_request_plan_uses_the_same_specific_type_as_user_facing_classifier():
    summary = "变更视频行为识别任务开关和模型目录，核对行为教程与推理配置。"

    plan = build_request_plan(summary)

    assert plan["change_type"] == classify_change_type(summary) == "behavior_pipeline"


def test_compound_request_keeps_full_text_and_stops_at_profile_query_limit():
    summary = "调整行人跟踪参数并核对跨镜流程；更新行为识别配置；检查推理部署；补充回归测试。"

    plan = build_request_plan(summary)

    assert plan["queries"][0]["query"] == summary
    assert len(plan["queries"]) <= 4
    assert plan["query_limit"] == 4
    assert plan["languages"] == ["zh"]
    assert plan["manual_review_required"] is True


def test_profile_rejects_device_scope_when_the_domain_does_not_define_it():
    profile = load_change_profile(PROFILE_PATH)
    assert profile["scope_fields"] == []

    with pytest.raises(ValueError, match="不支持所填的设备范围字段"):
        build_request_plan("核对跟踪器配置变更。", profile=profile, device_model="Unknown Board X")


def test_private_company_material_is_stopped_but_public_project_questions_are_allowed():
    assert is_out_of_scope_public_request("查询本公司的内部 API 和私有工单。")
    assert not is_out_of_scope_public_request("核对 PP-Human 跟踪流程的公开配置说明。")
