"""Compact formal Git input plus separate snapshot knowledge lookup."""
import json
import streamlit as st
from services.registered_application import load_registry, collect_snapshot
from services.review_tasks import ReviewTaskStore


def render_registered_input(client, ready, *, on_change, public_demo=True):
    registry = load_registry()
    if public_demo:
        registry = {key: row for key, row in registry.items() if row['source_identity'] == 'student_poc_application'}
    if not registry:
        st.warning('尚无可用的已登记应用。公网演示只开放学生原创资料。')
        return None, None
    app_id = st.selectbox('已登记内部应用', list(registry), format_func=lambda key: registry[key]['name'], key='registered_app')
    app = registry[app_id]
    st.caption(f"维护者：{app['owner']} · 学生原创应用 · 文件范围：{', '.join(app['include'])}")
    with st.form('git_snapshot_form'):
        mode = st.radio('审查阶段', ['升级前预审', '候选提交审查'], horizontal=True, key='git_review_mode')
        left, right = st.columns(2)
        head = left.text_input('待审应用提交 / 本地分支', value='HEAD', max_chars=160, key='git_head')
        base = right.text_input('基准应用提交（候选审查时必填）', value='', max_chars=160, key='git_base')
        submitted = st.form_submit_button('读取提交与检查范围')
    if submitted:
        on_change()
        st.session_state.pop('registered_snapshot', None)
        st.session_state.pop('registered_base_snapshot', None)
        try:
            if mode == '候选提交审查' and not base.strip(): raise ValueError('请填写候选提交对应的基准提交。')
            snapshot = collect_snapshot(app, head.strip(), base_commit=base.strip() if mode == '候选提交审查' else None)
            baseline = collect_snapshot(app, snapshot['base_commit']) if snapshot['base_commit'] else None
            st.session_state['registered_snapshot'] = snapshot
            st.session_state['registered_base_snapshot'] = baseline
        except ValueError as exc: st.error(str(exc))
    snapshot = st.session_state.get('registered_snapshot')
    if not snapshot or snapshot['application_id'] != app_id:
        st.info('先读取应用提交。系统从 Git 对象收集登记范围，包含未修改的消费者；不读取工作区未提交改动。')
        return None, None
    st.success(f"已装载应用提交 {snapshot['head_commit'][:12]} · {len(snapshot['files'])} 个文件")
    st.caption('依赖升级 v2.9.1 → v3.0.0；上述提交号属于应用，两者分别管理。')
    if snapshot['gaps']: st.warning(f"收集不完整：{len(snapshot['gaps'])} 项缺口。报告将保留这些缺口。")
    with st.expander('已检查范围与缺失资料'):
        st.write(snapshot['coverage']['scope'])
        st.table([{'文件': row['path'], '内容哈希': row['sha256'][:12],
                   '候选变化': '已修改' if row['path'] in snapshot['changed_paths'] else '纳入上下文'} for row in snapshot['inventory']])
        for gap in snapshot['gaps']: st.warning(gap['detail'])
        if snapshot['base_commit']:
            st.caption(f"基准提交：{snapshot['base_commit']}；变更文件：{', '.join(snapshot['changed_paths']) or '登记目录无变更'}")
        st.download_button('下载收集清单', json.dumps({k: v for k, v in snapshot.items() if k != 'files'}, ensure_ascii=False, indent=2),
                           file_name='application-snapshot.json', mime='application/json')
    with st.expander('查询这次提交的应用契约与测试资料'):
        question = st.text_input('应用资料查询', value='text confidence page_index 输出契约', key='application_knowledge_query')
        if st.button('检索应用原文', disabled=not ready, key='application_knowledge_search'):
            try:
                from services.registered_review import validate_application_evidence
                value = validate_application_evidence(client.application_search(snapshot['files'], question,
                    application_id=app_id, application_commit=snapshot['head_commit']), snapshot)
                for row in value.get('results', []):
                    st.markdown(f"**{row['path']} · 第 {row['line_start']}–{row['line_end']} 行**")
                    st.code(row['content'], language=None)
                if not value.get('results'): st.info('当前应用提交未检索到匹配资料。')
                st.caption(value.get('notice', '仅当前提交的应用资料；不是运行测试证明。'))
            except Exception: st.warning('应用资料检索暂不可用，请查看收集清单中的文件。')
    return snapshot, st.session_state.get('registered_base_snapshot')


def render_snapshot_report(result):
    snapshot = result.get('application_snapshot')
    if not snapshot: return
    st.markdown('**审查对象与检查范围**')
    st.caption(f"{snapshot['application_id']} · 应用提交 {snapshot['head_commit'][:12]} · "
               f"{snapshot.get('review_mode', 'preflight')} · 依赖 {result['source_version']} → {result['target_version']}")
    if snapshot['coverage']['status'] != 'REGISTERED_SCOPE_COMPLETE':
        st.warning('登记范围内资料未收集完整；不能将未发现风险解释为应用兼容。')
    else: st.caption('已收集登记角色所需文件；静态关系和实际回归仍须分别核查。')
    with st.expander('应用提交中的契约与测试依据'):
        for row in (result.get('application_evidence') or {}).get('results', []):
            st.markdown(f"**{row['path']} · 第 {row['line_start']}–{row['line_end']} 行**")
            st.code(row['content'], language=None)
        st.caption('学生原创应用资料，与官方 SDK 资料分别管理；测试定义不能证明已执行。')
    comparison = result.get('commit_comparison')
    if comparison:
        with st.expander('基准提交与候选提交对照'):
            if comparison['status'] != 'COMPARED': st.warning(comparison['reason'])
            else:
                st.caption(f"{comparison['previous_application_commit'][:12]} → {comparison['current_application_commit'][:12]}")
                st.write(comparison['notice'])
                labels = {'PERSISTS': '原风险仍存在', 'NEW': '新出现的规则风险',
                          'UNRESOLVED': '资料不足，不能确认已消除', 'NOT_OBSERVED_NOW': '本次未观察到，仍需回归'}
                for row in comparison['findings']: st.write(f"{row['path']}：{labels[row['change']]}")
                if not comparison['findings']: st.caption('本次未形成可对照的静态风险项；仍需核对缺口与实际测试。')


def render_task_recovery(store: ReviewTaskStore, identity):
    with st.expander('恢复本地审查任务'):
        st.caption('凭据只用于取回本地任务，不代表企业身份。实例磁盘被重建后记录可能丢失，请下载报告留存。')
        task = st.text_input('任务编号', max_chars=64, key='restore_task_id')
        token = st.text_input('任务恢复凭据', type='password', max_chars=200, key='restore_task_token')
        if st.button('读取任务', key='restore_task'):
            try:
                ticket = {'task_id': task, 'token': token}
                value = store.load(ticket, identity=identity)
                st.session_state['ocr_task_ticket'] = ticket
                if value['result']:
                    st.session_state['ocr_compatibility_review'] = value['result']
                    st.session_state['ocr_review_input_fingerprint'] = identity
                    st.success('已恢复本次输入绑定的静态报告。')
                else: st.info(f"任务状态：{value['state']}。再次提交时可复用已保存且身份一致的查证响应；不会自动执行代码。")
            except ValueError as exc: st.error(str(exc))
