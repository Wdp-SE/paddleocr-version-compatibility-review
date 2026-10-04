from pathlib import Path

from streamlit.testing.v1 import AppTest

FIXTURE = Path(__file__).parent / "fixtures" / "release_candidate_app.py"

def test_configuration_tool_failure_stays_visible_without_a_false_no_impact_result():
    app = AppTest.from_file(FIXTURE)
    app.session_state["test_panel"] = "configuration"
    app.run()
    assert not app.exception
    assert app.warning
    assert "无影响" not in app.warning[0].value


def test_current_dev_metrics_are_separate_from_answer_accuracy_and_defaults():
    app = AppTest.from_file(FIXTURE).run()
    assert not app.exception
    assert len(app.dataframe) == 2
    assert "11 个可回答" in "\n".join(row.value for row in app.caption)
    visible = "\n".join(row.value for row in list(app.info) + list(app.warning) + list(app.caption))
    assert "独立" in visible and "回答准确率" in visible


def test_unresolved_reference_is_not_presented_as_an_absence_of_references():
    app = AppTest.from_file(FIXTURE)
    app.session_state["test_panel"] = "unresolved"
    app.run()
    assert not app.exception
    assert app.warning
    assert "待核对" in "\n".join(row.value for row in app.warning)
    assert not any("未发现" in row.value for row in app.caption)


def test_answer_shows_the_concrete_missing_material_instead_of_an_api_key_error():
    app = AppTest.from_file(FIXTURE)
    app.session_state["test_panel"] = "answer_gaps"
    app.run()
    assert not app.exception
    assert any("跟踪回归记录" in row.value for row in app.markdown)
    assert not any("API" in row.value for row in app.caption)


def test_export_preserves_configuration_trace_for_human_review():
    from public_workbench import _review_report
    report = _review_report({
        "task_id": "config-review", "configuration_trace": {
            "status": "partial", "relations": [{"target_path": "deploy/pipeline/config/tracker_config.yml"}],
            "gaps": [{"kind": "not_indexed"}],
        }, "public_baseline_written": False,
    }, "reviewed", "2026-10-04T01:00:00Z")
    assert report["configuration_trace"]["relations"][0]["target_path"] == "deploy/pipeline/config/tracker_config.yml"
    assert report["public_baseline_written"] is False
