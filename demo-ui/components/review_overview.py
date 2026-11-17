"""One concise task summary; static completion never implies runtime approval."""
from html import escape
import streamlit as st
from services.review_action_summary import review_overview


def render_overview(result, *, records=None):
    view=review_overview(result,records=records)
    snapshot=result.get('application_snapshot') or {}
    commit=snapshot.get('head_commit')
    scope=f"依赖 {result.get('source_version','')} → {result.get('target_version','')}"
    if commit:scope+=' · 应用提交 '+commit[:12]
    else:scope+=' · 临时诊断'
    st.caption(scope)
    if view['risk_count']:
        st.error('有证据支持的静态风险，需要处理')
    elif view['change_count']:
        st.info('配置行为发生变化，需要确认并回归')
    elif result.get('workflow_type')=='configuration_change':
        st.info('尚无可确认的配置变化，请检查输入与缺口')
    else:
        st.info('未确认升级风险，仍需核查资料与运行结果')
    # Validate the stored record against this report before displaying its state.
    saved=st.session_state.get('ocr_disposition_state',{})
    label=view['runtime_label']
    if label=='未关联回归记录' and saved.get('regression'):
        from services.paddleocr_disposition import binding,validate_regression
        if saved.get('report_binding')==binding(result):
            try:
                checked=validate_regression(result,saved['regression'])
                label='已关联夹具记录' if checked['record_status']=='FIXTURE_ONLY' else '已关联人工记录，待核验'
            except ValueError:label='回归记录已失效'
    configuration=result.get('workflow_type')=='configuration_change'
    primary=(f"配置变化 {view['change_count']} 项" if configuration else f"风险问题 {view['risk_count']} 项")
    locations=(f"{view['risk_location_count']} 个位置" if view['risk_count'] else
               '同版本参数对照' if configuration else '当前静态检查范围')
    summary_html=f"""<style>
    .review-summary {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;
      padding:14px 0 18px;border-bottom:1px solid #d8d8d8;margin-bottom:14px;}}
    .review-summary strong {{display:block;font-size:15px;line-height:1.5;color:#252525;}}
    .review-summary span {{display:block;font-size:14px;line-height:1.5;color:#606060;margin-top:4px;overflow-wrap:anywhere;}}
    @media(max-width:420px) {{.review-summary {{grid-template-columns:1fr;gap:10px;}}
      .review-summary>div {{display:flex;justify-content:space-between;gap:12px;}}
      .review-summary span {{margin:0;}}}}
    </style><div class="review-summary" role="group" aria-label="审查摘要">
    <div><strong>{escape(primary)}</strong><span>{escape(locations)}</span></div>
    <div><strong>待核查 {view['gap_count']} 项</strong><span>资料与版本依据</span></div>
    <div><strong>运行验证</strong><span>{escape(label)}</span></div></div>"""
    # Markdown treats four-space indentation after a block as code.
    st.markdown('\n'.join(line.lstrip() for line in summary_html.splitlines()),unsafe_allow_html=True)
    return view
