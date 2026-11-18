"""Authenticated frontend for the independent internal review control plane."""
import hashlib
import json
import os
import streamlit as st

from services.internal_client import InternalClient
from services.rag_client import ServiceError


def identity_header():
    # The deployment proxy must authenticate and strip client-supplied identity headers.
    # The API verifies the forwarded token independently. Never accept a URL token.
    context=getattr(st,'context',None)
    headers=getattr(context,'headers',{})
    value=headers.get('Authorization','')
    return value if value.startswith('Bearer ') else None


def render():
    st.title('内部研发知识与变更审查')
    authorization=identity_header()
    if not authorization:
        st.error('企业身份尚未建立。请通过已配置的企业登录入口访问；不会以匿名身份提交内部任务。')
        return
    subject_session=hashlib.sha256(authorization.encode()).hexdigest()
    if st.session_state.get('internal_identity')!=subject_session:
        for key in list(st.session_state):
            if str(key).startswith('internal_'):del st.session_state[key]
        st.session_state['internal_identity']=subject_session
    base=os.getenv('RAG_API_BASE_URL','http://127.0.0.1:8765')
    client=InternalClient(base,authorization)
    try:
        applications=client._request('GET','/internal/applications')['applications']
        if not applications:
            st.warning('当前身份没有可访问的登记应用。请联系应用维护者。');return
        names={r['id']:r['name'] for r in applications}
        application=st.selectbox('内部应用',list(names),format_func=names.get,key='internal_application')
        client.application=application
        mode=st.radio('工作模式',['研发知识查询','变更审查'],horizontal=True)
        if mode=='研发知识查询':
            include_application = st.checkbox('关联应用提交', value=False)
            head = st.text_input('应用提交 / 分支', 'HEAD').strip() if include_application else None
            entries = next(row.get('entrypoints', []) for row in applications if row['id'] == application)
            entrypoint = None
            versions = ['v2.9.1', 'v3.0.0']
            if include_application:
                versions.append('all')
                if entries:
                    entrypoint = st.selectbox('应用入口', entries, index=None, placeholder='选择本次使用的入口')
                    versions.insert(0, 'auto')
            version = st.selectbox('组件资料版本', versions, format_func=lambda item: {
                'auto': '按应用入口选择', 'all': '两版对照',
            }.get(item, item))
            question=st.text_area('问题',key='internal_question')
            left,right=st.columns(2)
            search=left.button('检索原文',use_container_width=True)
            generate=right.button('生成引用回答',use_container_width=True)
            if (search or generate) and question.strip():
                if include_application and not head:
                    st.warning('请提供应用提交或分支。')
                    return
                with st.spinner('正在查阅对应版本资料…'):
                    if include_application:
                        payload = {
                            'application_id': application, 'head': head, 'version': version,
                            'query': question.strip(), 'generate': generate}
                        if entrypoint:
                            payload['entrypoint'] = entrypoint
                        value = client._request('POST', '/internal/application-knowledge', json=payload,
                                                request_timeout=90 if generate else 20)
                    else:
                        value=(client.query_official(question,version=version,retrieval_policy='paddleocr_evidence') if generate
                               else client.search(question,version=version,retrieval_policy='paddleocr_evidence'))
                    st.session_state['internal_answer']={'application':application,'version':version,
                        'question':question.strip(),'head':head,'entrypoint':entrypoint,'value':value}
            saved=st.session_state.get('internal_answer')
            if (saved and saved['application']==application and saved.get('version')==version
                    and saved.get('question')==question.strip() and saved.get('head')==head
                    and saved.get('entrypoint')==entrypoint):
                value=saved['value']
                if value.get('answer') and value['answer']!='N/A':st.markdown(value['answer'])
                elif value.get('status') not in (None,'OK','EVIDENCE_ONLY','NEEDS_ENTRYPOINT_CONTEXT'):
                    st.warning('本次未生成完整回答。请核对原文或查看服务策略。')
                if value.get('application_commit'):
                    st.caption('应用提交 '+value['application_commit'])
                context = value.get('entrypoint_context', {})
                if context.get('declared_version'):
                    dependency = context['dependency_source']
                    st.caption(f"入口声明依赖 {context['declared_version']} · "
                               f"{dependency['path']}:{dependency['line']}")
                if context.get('gap'):
                    st.warning(context['gap'])
                for gap in value.get('evidence_gaps', []):
                    st.warning(gap)
                for gap in value.get('snapshot_gaps', []):
                    if isinstance(gap, dict) and gap.get('detail'):
                        st.caption(gap['detail'])
                for row in (value.get('results') or value.get('evidence') or [])[:8]:
                    with st.expander(row.get('document_title') or row.get('document_path','证据')):
                        st.caption(f"{row.get('version','')} · {row.get('document_path','')} · {row.get('line_start','')}–{row.get('line_end','')}")
                        st.code(row.get('content',''),language=None)
            return
        purpose=st.selectbox('审查目的',['依赖升级','同版本配置变更'])
        configuration=purpose=='同版本配置变更'
        if configuration:
            st.caption('组件 v2.9.1 · drop_score 配置对照')
        else:
            st.caption('组件依赖 v2.9.1 → v3.0.0')
        with st.form('internal_submission'):
            head=st.text_input('待审应用提交 / 分支','HEAD')
            baseline=st.text_input('基准应用提交（配置审查必填）' if configuration else '基准应用提交（可选）')
            advice=False if configuration else st.checkbox('生成模型核查建议（受项目数据出口策略约束）',value=False)
            submitted=st.form_submit_button('提交配置审查' if configuration else '提交升级审查',type='primary')
        if submitted:
            if configuration and not baseline.strip():
                st.warning('请填写基准应用提交，配置审查需要比较修改前后两份快照。')
            else:
                value=client._request('POST','/internal/tasks',json={'application_id':application,'head':head.strip(),
                    'base':baseline.strip() or None,'generate_advice':advice,
                    'workflow_type':'configuration_change' if configuration else 'upgrade_review'})
                st.session_state['internal_task_id']=value['id']
                st.session_state['internal_task_input']=value['id']
                st.success('任务已进入后端队列，可以关闭页面后通过任务编号继续查看。')
        task_id=st.text_input('任务编号',key='internal_task_input')
        if not task_id:return
        if len(task_id)!=32 or any(c not in 'abcdef0123456789' for c in task_id):
            st.warning('请输入完整任务编号。');return
        if st.session_state.get('internal_observed_task')!=task_id:
            st.session_state['internal_observed_task']=task_id
            st.session_state['internal_polling']=task_id
        # Stop timed polling at a terminal state. Reads never execute the review.
        @st.fragment(run_every='3s' if st.session_state.get('internal_polling')==task_id else None)
        def task_view():
            try:
                value=client._request('GET','/internal/tasks/'+task_id)
                if value['state'] in ('DONE','FAILED','CANCELLED') and st.session_state.get('internal_polling')==task_id:
                    st.session_state['internal_polling']=None
                    st.rerun()
                states={'QUEUED':'等待执行','RUNNING':'正在审查','DONE':'审查完成','FAILED':'执行失败','CANCELLED':'已取消'}
                stages={'queued':'排队','static_review':'静态检查','evidence_validation':'核验证据',
                    'official_lookup':'查阅版本资料','model_advice':'生成核查建议','completed':'结束'}
                st.markdown('**'+states.get(value['state'],value['state'])+'**')
                if value['state'] in ('QUEUED','RUNNING'):
                    st.caption(stages.get(value['stage'],value['stage']))
                    preview=value.get('static_preview')
                    if preview:
                        st.info(f"静态证据核验已完成：发现 {preview['risk_count']} 个风险位置。后续查证仍在进行。")
                        for item in preview['findings'][:3]:st.write(f"{item['path']}:{item['line']} · {item['title']}")
                with st.expander('任务执行记录'):
                    st.caption(f"任务 {task_id} · 执行尝试 {value['attempts']} · {stages.get(value['stage'],value['stage'])}")
                if value['state'] in ('QUEUED','RUNNING'):
                    if st.button('取消任务',key='internal_cancel'):
                        client._request('POST','/internal/tasks/'+task_id+'/cancel');st.rerun(scope='fragment')
                elif value['state']=='DONE':
                    response=client._request('GET','/internal/tasks/'+task_id+'/report')
                    if response['report'].get('workflow_type')=='configuration_change':
                        from components.configuration_workbench import render_configuration_findings
                        render_configuration_findings(response['report'],records=response['records'])
                    else:
                        from public_workbench import _ocr_report
                        _ocr_report(response['report'],records=response['records'])
                    coverage = response.get('regression_coverage', [])
                    if coverage:
                        labels = {'NOT_RUN': '未运行', 'PASSED': '实际 OCR 通过', 'FAILED': '失败',
                                  'SKIPPED': '已跳过', 'FIXTURE_PASSED': '夹具通过'}
                        st.subheader('逐项回归结果')
                        st.table([{'回归项': item['label'], '状态': labels[item['status']],
                                   '运行编号': item.get('run_id', '—')} for item in coverage])
                    with st.expander('运行与人工审核记录'):
                        st.json(response['records'])
                    st.download_button('下载任务记录',json.dumps(response,ensure_ascii=False,indent=2),
                        file_name='internal-review-'+task_id+'.json',mime='application/json')
                    with st.form('internal_decision'):
                        decision=st.selectbox('审核动作',['ACKNOWLEDGED','DEFERRED'])
                        reason=st.text_area('审核理由')
                        confirm=st.form_submit_button('记录人工审核')
                    if confirm:
                        client._request('POST','/internal/tasks/'+task_id+'/decisions',json={
                            'decision':decision,'reason':reason,'report_fingerprint':response['report']['report_fingerprint']})
                        st.success('已记录当前身份的审核意见；不代表自动批准升级。')
                elif value['state'] in ('FAILED','CANCELLED'):
                    st.warning('任务状态：'+value['state']+' · '+(value.get('error') or ''))
                    if st.button('显式重试任务',key='internal_retry'):
                        client._request('POST','/internal/tasks/'+task_id+'/retry')
                        st.session_state['internal_polling']=task_id;st.rerun()
            except ServiceError as exc:st.error(str(exc))
        task_view()
    except ServiceError as exc:
        st.error(str(exc))
        st.caption('请核对企业登录、应用权限、项目数据出口策略及后端状态。')
