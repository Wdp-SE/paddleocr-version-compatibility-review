import copy
from services.review_action_summary import summarize_review


def test_same_limitations_group_without_losing_locations_or_hiding_blockers():
    gaps = [{'code': 'uncertain_control_flow', 'detail': '路径未证明', 'blocking': False,
             'application': {'path': 'adapter.py', 'line': i}} for i in range(1, 30)]
    gaps += [{'code': 'missing_call', 'detail': '缺调用层', 'application': {'path': 'client.py', 'line': 1}}]
    result = {'compatibility_report': {'findings': [], 'gaps': gaps},
              'investigation': {'unresolved': [{'check_id': 'c1', 'reason': 'application_validation_required',
                                               'versions': ['v3.0.0']}]}}
    original = copy.deepcopy(result)
    summary = summarize_review(result)
    assert len(summary['limitations']) == 1
    assert summary['limitations'][0]['count'] == 29
    assert len(summary['limitations'][0]['locations']) == 29
    assert len(summary['input_gaps']) == 1
    assert len(summary['runtime_actions']) == 1
    assert summary['record_count'] == 31
    assert result == original


def test_missing_blocking_flag_is_conservative_and_other_paths_stay_separate():
    result = {'compatibility_report': {'gaps': [
        {'code': 'x', 'detail': '缺资料', 'application': {'path': p, 'line': 1}} for p in ['a.py', 'b.py']]}}
    summary = summarize_review(result)
    assert len(summary['input_gaps']) == 2
    assert not summary['limitations']


def test_report_groups_limitations_in_collapsed_panel_with_original_locations():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
from components.review_action_summary import render_gap_summary
render_gap_summary({'compatibility_report':{'gaps':[
 {'code':'uncertain_control_flow','detail':'路径未证明','blocking':False,
  'application':{'path':'adapter.py','line':i}} for i in range(1,30)]}})
''').run()
    assert not app.exception
    panel = next(p for p in app.expander if '一般分析限制' in p.label)
    assert '1 组' in panel.label and '29 条原始记录' in panel.label
    assert not panel.proto.expanded
    assert any('adapter.py:29' in r.value for r in app.caption)
    assert any('分支路径尚未核实' in r.value for r in app.markdown)
