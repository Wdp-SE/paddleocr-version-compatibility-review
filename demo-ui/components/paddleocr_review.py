"""Impact levels and investigation traces; UI never promotes model conclusions."""
import streamlit as st
from contextlib import nullcontext


def impact_panel(result, *, show_paths=True, show_regression=True, show_trace=True):
    if show_trace:
        from components.graph_execution import graph_execution_panel
        graph_execution_panel(result)
    report=result.get('compatibility_report',{})
    if report.get('impact_schema_version')!=1:
        st.caption('当前服务提供基础静态报告；应用关系扩展未返回。')
        return
    paths=report.get('impact_paths',[])
    with (st.expander(f'可能受影响的位置 · {len(paths)} 条路径') if show_paths else nullcontext()):
        if not show_paths:paths=[]
        if show_paths and not paths:st.caption('尚未定位跨文件影响路径。')
        if show_paths:st.caption('候选关系，实际影响需回归确认。')
        for i,path in enumerate(paths):
            label='直接结果流' if path['flow_proven'] else '引用关系'
            st.markdown(f"**{i + 1}. {label}候选**")
            st.write(f"{path['origin_node']} → {path['target_node']}")
            for edge in path['edges']:
                loc=edge['application']
                st.write(f"{edge['source']} → {edge['target']} · {edge['kind']}")
                st.caption(f"{loc['path']}:{loc['line_start']}–{loc['line_end']}")
                st.code(loc['excerpt'],language='python',wrap_lines=True)
    if show_regression:st.markdown('**待执行回归**')
    seen=set()
    for req in report.get('regression_requirements',[]):
        if not show_regression:break
        key=(req['expected_contract'],tuple(req['input_conditions']))
        if key in seen:continue
        seen.add(key)
        st.markdown(f"**{len(seen)}.** {req['expected_contract']}")
        st.caption('输入条件：'+' / '.join(req['input_conditions']))
        if req.get('kind')=='application_contract':
            st.caption('依据：本次应用提交的 '+req['source']['path']+'；应用要求与官方 SDK 事实分别核对。')
            for check in req['checks']:st.markdown('- '+check)
    investigation=result.get('investigation')
    if investigation and show_trace:
        with st.expander('逐项查证轨迹'):
            st.caption(f"检索调用 {investigation['search_calls']} 次 · 查证阶段 {investigation['elapsed_seconds']:.1f} 秒 · 停止原因 {investigation['stop_reason']}")
            if result.get('task_elapsed_seconds') is not None:
                st.caption(f"本次任务总用时（含模型建议）：{result['task_elapsed_seconds']:.1f} 秒")
            st.dataframe([{k:v for k,v in t.items() if k not in ('content',)} for t in investigation['trace']],use_container_width=True)
            for gap in investigation['unresolved']:
                if gap['reason']=='application_validation_required':
                    st.caption(f"{gap['check_id']}：待补充应用资料或运行回归 · {', '.join(gap['versions'])}")
                else:
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
