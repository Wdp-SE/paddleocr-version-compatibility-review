"""Read-only execution detail. Completed nodes never imply OCR compatibility."""
import streamlit as st

_LABELS = {
    'scope': '范围检查', 'validate_input': '输入核对',
    'static_review': '静态检查', 'evidence_validation': '证据核对',
    'official_lookup': '版本查证', 'model_advice': '建议生成',
    'retrieve': '检索资料', 'prepare_context': '组织证据',
    'generate': '生成回答', 'verify_claims': '核验主张',
    'render_answer': '组织回答', 'report': '整理报告',
    'first_pass': '首次查询', 'correction_gate': '检查补查条件',
    'repair_retrieval': '补充检索', 'repaired_pass': '核对补查回答',
    'select_answer': '选择有依据的回答', 'finish': '完成查询',
}


def graph_execution_panel(payload):
    execution = payload.get('graph_execution')
    if not isinstance(execution, dict) or execution.get('engine') != 'langgraph':
        return
    with st.expander('执行步骤', expanded=False):
        nodes = execution.get('completed_nodes', [])
        if isinstance(nodes, list):
            for index, node in enumerate(nodes[:32], 1):
                if isinstance(node, str):
                    st.caption(f'{index}. {_LABELS.get(node, node)}')
        passes = execution.get('passes', [])
        if isinstance(passes, list):
            for index, record in enumerate(passes[:2], 1):
                if not isinstance(record, dict):
                    continue
                steps = record.get('completed_nodes', [])
                if isinstance(steps, list):
                    st.caption(f'第 {index} 轮：' + ' → '.join(
                        _LABELS.get(n, n) for n in steps[:16] if isinstance(n, str)))
        st.caption('步骤完成仅表示分析执行结束，实际运行与人工处置见对应记录。')
