from streamlit.testing.v1 import AppTest


def test_graph_details_do_not_claim_runtime_verification():
    app = AppTest.from_string('''
from components.graph_execution import graph_execution_panel
graph_execution_panel({'graph_execution': {'engine': 'langgraph', 'workflow': 'upgrade_review',
    'completed_nodes': ['validate_input', 'static_review'], 'status': 'completed', 'graph_version': '1'}})
''').run()
    assert not app.exception
    assert len(app.expander) == 1
    text = '\n'.join(str(x.value) for x in list(app.caption) + list(app.markdown))
    assert '静态检查' in text
    assert '兼容验证通过' not in text


def test_old_report_has_no_graph_panel():
    app = AppTest.from_string('''
from components.graph_execution import graph_execution_panel
graph_execution_panel({})
''').run()
    assert not app.exception
    assert not app.expander
