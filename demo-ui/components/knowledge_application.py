"""Optional immutable app context for the same versioned knowledge service."""
import streamlit as st


def application_context():
    from services.registered_application import load_registry, collect_snapshot
    with st.expander('关联应用提交（可选）'):
        registry = {k:v for k,v in load_registry().items() if v.get('source_identity')=='student_poc_application'}
        if not registry:
            st.caption('暂无可用于公开演示的已登记应用。')
            return None
        selected=st.selectbox('应用',list(registry),format_func=lambda k:registry[k]['name'],key='knowledge_application')
        ref=st.text_input('应用提交 / 本地分支',value='HEAD',max_chars=160,key='knowledge_application_ref')
        left,right=st.columns(2)
        if left.button('加载提交',key='knowledge_load_application'):
            st.session_state.pop('knowledge_snapshot',None)
            st.session_state.pop('official_result',None)
            try:
                snapshot=collect_snapshot(registry[selected],ref.strip())
                st.session_state['knowledge_snapshot']=snapshot
            except ValueError as exc:
                st.error(str(exc))
        if right.button('移除关联',key='knowledge_clear_application'):
            st.session_state.pop('knowledge_snapshot',None)
            st.session_state.pop('official_result',None)
        snapshot=st.session_state.get('knowledge_snapshot')
        if not snapshot or snapshot['application_id']!=selected:
            return None
        st.caption('已关联 '+snapshot['head_commit'][:12]+' · 未提交修改不纳入查询')
        if snapshot.get('gaps'):
            st.warning('应用收集存在缺口，查询不能据此确认完整兼容。')
        return snapshot
