"""Impact levels and investigation traces; UI never promotes model conclusions."""
import streamlit as st


def impact_panel(result):
    report=result.get('compatibility_report',{})
    if report.get('impact_schema_version')!=1:
        st.caption('当前服务提供基础静态报告；应用关系扩展未返回。')
        return
    st.markdown('**待核查影响路径**')
    paths=report.get('impact_paths',[])
    if not paths:st.caption('当前提交文本没有可证明的跨文件影响路径。')
    for i,path in enumerate(paths):
        label='直接结果流' if path['flow_proven'] else '引用关系'
        with st.expander(f"{label}候选 · {path['origin_node']} → {path['target_node']}"):
            st.caption('关系已定位；下游业务是否破坏仍需核查和运行回归。')
            for edge in path['edges']:
                loc=edge['application']
                st.write(f"{edge['source']} → {edge['target']} · {edge['kind']}")
                st.caption(f"{loc['path']}:{loc['line_start']}–{loc['line_end']}")
                st.code(loc['excerpt'],language='python')
    st.markdown('**回归要求（尚未执行）**')
    for req in report.get('regression_requirements',[]):
        st.write(req['expected_contract'])
        st.caption('输入条件：'+' / '.join(req['input_conditions']))
    investigation=result.get('investigation')
    if investigation:
        with st.expander('逐项查证轨迹'):
            st.caption(f"检索调用 {investigation['search_calls']} 次 · 查证阶段 {investigation['elapsed_seconds']:.1f} 秒 · 停止原因 {investigation['stop_reason']}")
            if result.get('task_elapsed_seconds') is not None:
                st.caption(f"本次任务总用时（含模型建议）：{result['task_elapsed_seconds']:.1f} 秒")
            st.dataframe([{k:v for k,v in t.items() if k not in ('content',)} for t in investigation['trace']],use_container_width=True)
            for gap in investigation['unresolved']:
                st.warning(f"{gap['check_id']}：{gap['reason']} · {', '.join(gap['versions'])}")
            st.caption(f"规划请求 {result.get('model_planning_request_count',0)} · 建议请求 {result.get('model_advice_request_count',0)} · 支持复核请求 {result.get('model_support_request_count',0) if result.get('model_support_request_count') is not None else '服务未返回计数'}")


def coverage_panel(workspace):
    coverage=workspace.get('task_coverage')
    if not isinstance(coverage,dict):return
    with st.expander('任务资料覆盖与缺口'):
        st.write('已核验来源格式：'+' / '.join(f'{k} {v}' for k,v in coverage.get('formats',{}).items()))
        st.caption('官方资料与作者原创应用分开标记；不代表全企业资料覆盖。')
        for gap in coverage.get('gaps',[]):
            st.warning('图片信息未提取：'+str(gap.get('count',0))+' 个引用' if gap['code']=='image_information_not_extracted'
                       else f"{gap.get('version','')} / {gap.get('family','')}：资料缺口")
