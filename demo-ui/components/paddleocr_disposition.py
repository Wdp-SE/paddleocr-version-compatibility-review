"""Human-only continuation panel; it never executes code or rewrites sources."""
import json
from datetime import datetime, timezone
import streamlit as st
from services.paddleocr_disposition import binding, gap_id, regression_template, validate_regression, assess_disposition


def render_disposition(result, repository, session_id):
    key = binding(result)
    prefix = 'ocr_disposition_' + key[:16]
    state = st.session_state.setdefault('ocr_disposition_state', {})
    if state.get('report_binding') != key:
        state.clear()
        state.update(report_binding=key, regression=None)
    st.subheader('风险处置与回归记录')
    st.caption('填写处置理由、外部测试结果及残余风险。记录属于匿名演示，不能替代企业审批或自动证明升级成功。')
    risks = {}
    choices = {'pending': '待处理', 'needs_change': '需修改', 'mitigated': '已采取措施，待核对', 'not_applicable': '判定不适用，须说明理由'}
    report = result['compatibility_report']
    for finding in report.get('findings', []):
        if finding.get('status') != 'supported_risk':
            continue
        fid = finding['finding_id']
        with st.expander(f"{fid} · {finding.get('rule_id', '升级风险')}"):
            decision = st.selectbox('处置状态', list(choices), format_func=choices.get, key=prefix + fid + '_decision')
            reason = st.text_area('处置理由 / 修改位置', max_chars=2000, key=prefix + fid + '_reason')
            risks[fid] = {'decision': decision, 'reason': reason}
    gaps = {}
    for gap in [*report.get('gaps', []), *(result.get('investigation') or {}).get('unresolved', [])]:
        gid = gap_id(gap)
        with st.expander('残余风险 · ' + gid):
            st.write(gap.get('detail') or gap.get('reason') or gap)
            accepted = st.checkbox('已人工评估并记录接受该残余风险的理由', key=prefix + gid + '_accepted')
            reason = st.text_area('理由与后续核查责任', max_chars=2000, key=prefix + gid + '_reason')
            gaps[gid] = {'decision': 'accepted_residual' if accepted else 'pending', 'reason': reason}
    st.download_button('下载外部回归记录模板（JSON）',
        json.dumps(regression_template(result), ensure_ascii=False, indent=2),
        file_name='ocr-regression-template.json', mime='application/json', key=prefix + '_template')
    uploaded = st.file_uploader('导入外部回归记录（最多 200 KB）', type=['json'], key=prefix + '_upload')
    if uploaded is not None and st.button('校验并关联回归记录', key=prefix + '_validate'):
        try:
            if uploaded.size > 200_000:
                raise ValueError('回归记录超过 200 KB。')
            state['regression'] = validate_regression(result, json.loads(uploaded.getvalue()))
            st.success('已关联人工提交的回归记录；系统未执行或认证该测试。')
        except (ValueError, UnicodeDecodeError) as exc:
            state['regression'] = None
            st.error(str(exc))
    assessment = assess_disposition(result, risks, gaps, state.get('regression'))
    if assessment['blockers']:
        st.info(f"还有 {len(assessment['blockers'])} 项未完成。可暂缓升级，或补齐处置与回归记录。")
        with st.expander('查看未完成项'):
            for message in assessment['blockers']:
                st.write(message)
    if state.get('regression'):
        st.caption('已关联外部回归记录：人工提交，真实性未认证。')
    continue_col, hold_col = st.columns(2)
    continued = continue_col.button('人工记录：继续升级流程', disabled=not assessment['eligible_for_human_continuation'], key=prefix + '_continue')
    held = hold_col.button('人工记录：暂缓升级', key=prefix + '_hold')
    if continued or held:
        event = {'task_id': result['task_id'], 'workspace_id': 'paddleocr',
                 'request_mode': 'paddleocr_compatibility', 'request_fingerprint': result.get('request_fingerprint'),
                 'human_decision': 'reviewed' if continued else 'rejected',
                 'decided_at_utc': datetime.now(timezone.utc).isoformat(),
                 'event_type': 'upgrade_disposition', 'application_context': result.get('application_context'),
                 'upgrade_decision': 'continue_manual_process' if continued else 'hold', **assessment}
        try:
            state['saved_event'] = repository.record(event, session_id=session_id)
        except Exception:
            state.pop('saved_event', None)
            st.error('处置决定未能写入本地记录，请重试。')
    saved = state.get('saved_event')
    if saved:
        st.caption('已保存的决定仅代表当时的处置快照；继续编辑后须重新记录决定。')
        st.download_button('下载已记录的处置快照', json.dumps(saved, ensure_ascii=False, indent=2),
            file_name='ocr-disposition.json', mime='application/json', key=prefix + '_export')
