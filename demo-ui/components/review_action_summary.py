"""Progressive disclosure of exact report records, grouped by cause and file."""
import streamlit as st
from services.review_action_summary import summarize_review

_LABELS = {'ocr_result_flow_unresolved': '结果来源尚未核实',
           'unreviewed_configuration': '依赖与运行环境待验证',
           'uncertain_control_flow': '分支路径尚未核实',
           'missing_call': '缺少调用层资料', 'unparsed_python': '文件无法完成静态解析',
           'application_evidence_unavailable': '应用契约与测试资料待补充'}


def render_gap_summary(result):
    summary = summarize_review(result)
    if not any(summary[key] for key in ('input_gaps','lookup_actions','runtime_actions','limitations')):
        return
    st.markdown('**待核查事项**')
    groups = summary['input_gaps']
    if groups:st.caption(f'{len(groups)} 项资料核查')
    for group in groups:
        paths = list(dict.fromkeys(l.get('path', '任务范围') for l in group['locations']))
        title = ' / '.join(paths) or '任务范围'
        label = _LABELS.get(group['code'], '需补充或核对资料')
        with st.expander(f"{title} · {label} · {group['count']} 条记录"):
            for detail in dict.fromkeys(r.get('detail', '') for r in group['records']):
                st.write(detail)
            if group['next_action']:
                st.caption('下一步：' + group['next_action'])
            for loc in group['locations']:
                st.caption(f"{loc.get('path')}:{loc.get('line', '')}")
                if loc.get('snippet'):
                    st.code(loc['snippet'], language='python', wrap_lines=True)
    if summary['lookup_actions']:
        with st.expander('版本资料待补查'):
            reasons={'required_source_not_verified':'需要核验固定版本来源',
                     'content_requirements_not_covered':'现有片段不足以支持结论',
                     'content_criteria_not_defined':'需要人工明确核查标准'}
            for group in summary['lookup_actions']:
                st.write(reasons.get(group['code'],'组件资料尚待查证'))
                st.caption('适用版本：'+', '.join(group['versions']))
    if summary['runtime_actions']:
        st.info('待完成应用验证：补齐应用资料或关联当前提交的回归记录。')
    limitations = summary['limitations']
    if limitations:
        with st.expander(f"一般分析限制 · {len(limitations)} 组（{sum(g['count'] for g in limitations)} 条原始记录）"):
            st.caption('这些是静态分析能力边界，不是额外业务风险，也不要求逐条重复检索。')
            for group in limitations:
                st.markdown('**' + _LABELS.get(group['code'], '静态分析范围限制') + '**')
                st.write(group['detail'])
                locs = group['locations']
                st.caption('位置：' + '、'.join(f"{loc.get('path')}:{loc.get('line', '')}" for loc in locs))
                if group['next_action']:
                    st.caption(group['next_action'])
