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
    st.subheader('处理事项')
    st.caption('演示处理记录，不构成发布审批。')
    risks = {}
    configuration = result.get('workflow_type') == 'configuration_change'
    choices = ({'pending': '待确认', 'accepted_change': '行为变化符合预期（须说明）'} if configuration else
               {'pending': '待处理', 'needs_change': '需修改', 'mitigated': '已采取措施，待核对', 'not_applicable': '判定不适用，须说明理由'})
    report = result['compatibility_report']
    from services.review_action_summary import finding_groups
    for group in finding_groups(result,statuses=('supported_risk','behavior_change')):
        finding=group['finding']
        with st.expander(f"{finding['title']} · {len(group['locations'])} 个位置"):
            # Display grouping does not batch-accept individual findings.
            for member in group['findings']:
                fid=member['finding_id'];location=member.get('application') or {}
                st.caption(f"{location.get('path','')}:{location.get('line','')} · {fid}")
                decision = st.selectbox('处置状态', list(choices), format_func=choices.get, key=prefix + fid + '_decision')
                reason = st.text_area('处置理由 / 修改位置', max_chars=2000, key=prefix + fid + '_reason')
                risks[fid] = {'decision': decision, 'reason': reason}
    gaps = {}
    residual = [g for g in [*report.get('gaps', []), *(result.get('investigation') or {}).get('unresolved', [])]
                if g.get('blocking') is not False]
    if residual:
        with st.expander('待确认事项'):
            st.caption('逐项记录资料缺口和未验证事项的处理理由。')
            for gap in residual:
                gid = gap_id(gap)
                location = gap.get('application') or {}
                label = (f"{location['path']}:{location.get('line', '')}" if location.get('path')
                         else gap.get('check_id') or '任务范围')
                with st.container(border=True):
                    st.markdown('**' + label + '**')
                    st.write(gap.get('detail') or gap.get('reason') or gap)
                    accepted = st.checkbox('已评估此未验证事项，并明确接受其不确定性', key=prefix + gid + '_accepted')
                    reason = st.text_area('理由与后续核查责任', max_chars=2000, key=prefix + gid + '_reason')
                    gaps[gid] = {'decision': 'accepted_residual' if accepted else 'pending', 'reason': reason}
    st.download_button('下载外部回归记录模板（JSON）',
        json.dumps(regression_template(result), ensure_ascii=False, indent=2),
        file_name='ocr-regression-template.json', mime='application/json', key=prefix + '_template')
    with st.container(key='review_regression_upload'):
        st.markdown('''<style>
        .st-key-review_regression_upload [data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
        </style>''',unsafe_allow_html=True)
        uploaded = st.file_uploader('导入回归记录', type=['json'], key=prefix + '_upload',max_upload_size=1)
        st.caption('JSON · 最大 200 KB')
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
        st.info('尚有未完成的处理事项，暂不能继续评审。')
        with st.expander('查看处理要求'):
            for message in assessment['blockers']:
                st.write(message)
    if state.get('regression'):
        st.caption('已关联外部回归记录：人工提交，真实性未认证。')
    continue_col, hold_col = st.columns(2)
    continued = continue_col.button('记录继续评审', disabled=not assessment['eligible_for_human_continuation'], key=prefix + '_continue')
    held = hold_col.button('记录暂缓变更', key=prefix + '_hold')
    if continued or held:
        event = {'task_id': result['task_id'], 'workspace_id': 'paddleocr',
                 'request_mode': result.get('request_mode', 'paddleocr_compatibility'), 'request_fingerprint': result.get('request_fingerprint'),
                 'human_decision': 'reviewed' if continued else 'rejected',
                 'decided_at_utc': datetime.now(timezone.utc).isoformat(),
                 'event_type': 'configuration_disposition' if configuration else 'upgrade_disposition',
                 'workflow_type': result.get('workflow_type', 'dependency_upgrade'), 'application_context': result.get('application_context'),
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
