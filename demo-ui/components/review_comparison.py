"""Small, explicit before/after comparison within one session."""
import streamlit as st
from services.review_comparison import capture_baseline, compare_reviews


def render_review_comparison(result):
    corpus = result.get('corpus_fingerprint')
    with st.expander('修改后复查对照'):
        baseline = st.session_state.get('ocr_comparison_baseline')
        if baseline:
            comparison = compare_reviews(baseline, result, corpus_fingerprint=corpus)
            st.markdown('**与基准报告对照**')
            if comparison['status'] != 'COMPARED':
                st.warning(comparison['reason'])
            else:
                st.caption(comparison['notice'])
                labels = {'PERSISTS': '原规则风险仍存在', 'NEW': '新出现的规则风险',
                          'UNRESOLVED': '资料或分析不足，不能判断原风险已消除',
                          'NOT_OBSERVED_NOW': '本次未观察到原规则风险，仍需实际回归'}
                for row in comparison['findings']:
                    st.write(f"{labels[row['change']]}：{row['path']} · {row['rule_id']} "
                             f"（原 {row['previous_count']} 处，现 {row['current_count']} 处）")
                with st.expander('文件变化与待验证项'):
                    names = {'ADDED': '新增', 'REMOVED': '移除', 'MODIFIED': '修改', 'UNCHANGED': '未变'}
                    for row in comparison['files']:
                        st.caption(f"{names[row['change']]}：{row['path']}")
                    st.caption(f"当前仍有 {comparison['remaining_gaps']} 项缺口；按规则与文件比较，不声称逐语句修复认证。")
        else:
            st.caption('先保存当前报告为基准，再修改或补充文件并重新审查。版本、目的和验收要求沿用原任务。')
        save, clear = st.columns(2)
        if save.button('将当前报告设为对照基准', key='ocr_save_baseline', disabled=not bool(corpus)):
            st.session_state['ocr_comparison_baseline'] = capture_baseline(result, corpus_fingerprint=corpus)
            st.rerun()
        if clear.button('清除对照基准', key='ocr_clear_baseline', disabled=not bool(baseline)):
            st.session_state.pop('ocr_comparison_baseline', None)
            st.rerun()
