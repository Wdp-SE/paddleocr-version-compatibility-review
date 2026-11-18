"""Same-app configuration change input and readable report."""
import json
import streamlit as st
from services.configuration_workflow import configuration_demo_inputs, run_configuration_review, configuration_identity
from services.review_tasks import ReviewTaskStore, CachedReviewGateway
from components.paddleocr_disposition import render_disposition


def render_configuration_workbench(client, ready, workspace, repository, session_id):
    st.caption('OCR 组件 v2.9.1 · drop_score 参数对照')
    mode = st.selectbox('配置审查输入', ['原创应用配置对照', '已登记应用（两个 Git 提交）'], key='config_input_mode')
    snapshot = base_snapshot = None
    if mode.startswith('已登记'):
        from components.registered_application import render_registered_input
        snapshot, base_snapshot = render_registered_input(client, ready, on_change=lambda: st.session_state.pop('config_result', None),
                                                          workflow_type='configuration_change')
        if not snapshot or not base_snapshot:
            st.info('配置审查请选择候选提交审查，读取基准与候选两个提交。这里只读 Git 已提交文件。')
            return
        before, after = base_snapshot['files'], snapshot['files']
    else:
        try:
            before, after = configuration_demo_inputs()
        except ValueError as exc:
            st.error(str(exc)); return
        st.caption('原创示例应用 · 配置前后对照')
        scenario = st.selectbox('配置案例', ['提高阈值 0.5 → 0.8', 'JSON 配置传递 0.5 → 0.8', '不变 0.5 → 0.5', '动态阈值，证据不足'], key='config_scenario')
        if scenario.startswith('JSON'):before,after=configuration_demo_inputs(style='json')
        default_after = '0.8' if scenario.startswith('提高') else '0.5' if scenario.startswith('不变') else 'runtime_threshold'
        old = next(f for f in before if f['path'] == 'application/ocr_client.py')
        new = next(f for f in after if f['path'] == 'application/ocr_client.py')
        with st.expander('查看或编辑配置代码'):
            left, right = st.columns(2)
            old['content'] = left.text_area('修改前调用层', value=old['content'], height=220, max_chars=50000, key='config_before_code_'+scenario)
            new['content'] = right.text_area('候选调用层', value=new['content'].replace('drop_score=0.8', 'drop_score='+default_after),
                                              height=220, max_chars=50000, key='config_after_code_'+scenario)
        with st.expander('同时纳入的应用契约与消费者'):
            st.write([f['path'] for f in after if f['path'] != 'application/ocr_client.py'])
    identity = configuration_identity(before, after, workspace)
    identity += ':' + json.dumps([snapshot and snapshot['head_commit'], base_snapshot and base_snapshot['head_commit']])
    store = ReviewTaskStore(repository.database_path.with_name('application_tasks.sqlite3'))
    with st.expander('恢复本次输入的配置审查任务'):
        recovery = st.text_area('本地恢复凭据 JSON', height=80, max_chars=2000, key='config_recovery')
        if st.button('读取配置任务', key='config_restore'):
            try:
                ticket = json.loads(recovery)
                saved = store.load(ticket, identity=identity)
                if saved.get('result'):
                    st.session_state.update(config_result=saved['result'], config_identity=identity, config_ticket=ticket)
                    st.success('已恢复与本次输入、版本及语料绑定的报告。')
                else:
                    st.session_state['config_ticket'] = ticket
                    st.info('任务未完成，再次审查可复用已完成的查证。')
            except (ValueError, TypeError):
                st.error('恢复凭据或输入身份不一致，未读取任务。')
    if st.button('审查配置变化', type='primary', disabled=not ready, key='config_review_run'):
        st.session_state.pop('config_result', None)
        ticket = st.session_state.get('config_ticket')
        try:
            store.load(ticket, identity=identity)
        except ValueError:
            ticket = store.create(identity)
        st.session_state['config_ticket'] = ticket
        try:
            store.start(ticket)
            with st.status('正在核对配置与同版本证据', expanded=True) as progress:
                st.write('解析两份应用 → 核对阈值定义 → 读取应用契约 → 列出比较回归要求')
                gateway = CachedReviewGateway(client, store, ticket, expected_workspace=workspace)
                result = run_configuration_review(gateway, before, after, snapshot=snapshot, base_snapshot=base_snapshot)
                store.finish(ticket, result)
                st.session_state['config_result'] = result
                st.session_state['config_identity'] = identity
                progress.update(label='配置静态审查完成', state='complete', expanded=False)
        except Exception as exc:
            store.fail(ticket, type(exc).__name__)
            st.error('配置审查未完成，请检查服务连接或补齐应用输入。')
    result = st.session_state.get('config_result')
    if not result or st.session_state.get('config_identity') != identity:
        return
    render_configuration_findings(result,repository=repository,session_id=session_id)
    st.download_button('下载配置审查报告', json.dumps(result, ensure_ascii=False, indent=2),
                       file_name='ocr-configuration-review.json', mime='application/json', key='config_download')
    st.download_button('下载本地任务恢复凭据', json.dumps(st.session_state.get('config_ticket'), indent=2),
                       file_name='configuration-task-ticket.json', mime='application/json', key='config_ticket_download')


def render_configuration_findings(result, *, repository=None, session_id=None, records=None):
    from components.graph_execution import graph_execution_panel
    graph_execution_panel(result)
    """Shared result view; public and authenticated records remain separate."""
    from components.review_overview import render_overview
    from components.review_action_summary import render_gap_summary
    from services.review_action_summary import finding_groups
    st.subheader('配置审查结果')
    render_overview(result,records=records)
    report=result['compatibility_report']
    labels=['审查结果','证据与代码','回归要求']+(['处理记录'] if repository is not None else [])
    tabs=st.tabs(labels)
    groups=finding_groups(result)
    with tabs[0]:
        if not groups:st.info('未形成可确认的配置变化，请检查输入与资料缺口。')
        for group in groups:
            finding=group['finding']
            with st.container(border=True):
                st.markdown('**'+finding['title']+'**')
                st.write(finding['explanation'])
                st.caption(f"{finding['application']['path']} · 第 {finding['application']['line']} 行")
                st.markdown('**需要处理：** '+finding['desired_check'])
        render_gap_summary(result)
    with tabs[1]:
        from components.registered_application import render_snapshot_report
        render_snapshot_report(result)
        for group in groups:
            finding=group['finding']
            st.markdown('**'+finding['title']+'**')
            old,new=st.columns(2)
            with old:
                st.caption('基准配置');st.code(finding['before_application']['snippet'],language='python')
            with new:
                st.caption('候选配置');st.code(finding['application']['snippet'],language='python')
            for phase,key in [('基准','before_application'),('候选','application')]:
                for source in finding[key].get('value_sources',[]):
                    with st.expander(f"{phase}配置来源 · {source['path']}"):
                        st.caption('字段路径：'+' / '.join(source.get('key_path',[])) if source.get('key_path') else f"原文第 {source['line']} 行")
                        st.code(source['snippet'],language='json' if source['path'].endswith('.json') else
                                'yaml' if source['path'].endswith(('.yaml','.yml')) else 'python')
            for row in finding['evidence']:
                st.markdown(f"[{row['title']}]({row['url']}#L{row['line_start']}-L{row['line_end']})")
                with st.expander('原文片段'):st.code(row['definition'],language='text',wrap_lines=True)
    with tabs[2]:
        for req in report['regression_requirements']:
            st.write('输入：'+' / '.join(req['input_conditions']))
            for name in req['checks']:st.markdown('- '+name)
        if not report['regression_requirements']:st.info('请先补齐配置输入，生成回归要求。')
        st.caption('回归记录需绑定本次提交；夹具与实际 OCR 运行分别记录。')
    if repository is not None:
        with tabs[3]:render_disposition(result,repository,session_id)
