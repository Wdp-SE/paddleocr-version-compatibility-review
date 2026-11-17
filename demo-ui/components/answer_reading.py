"""Readable verified statements; formatting never changes their factual text."""
import re
import streamlit as st
from components.reading_layout import paragraph_blocks


def render_claim(text):
    # Long inline code is hard to read. Preserve its bytes in a code block.
    for paragraph in paragraph_blocks(text):
        parts = re.split(r'(```[\s\S]*?```|`[^`\n]{65,}`)', paragraph)
        for part in parts:
            if part.startswith('```'):
                st.markdown(part)
            elif part.startswith('`') and part.endswith('`') and len(part) >= 67:
                st.code(part[1:-1], language='python' if '(' in part else 'text', wrap_lines=True)
            elif part.strip():
                st.markdown(part)


def render_answer(payload, *, show_heading=True):
    from components.graph_execution import graph_execution_panel
    graph_execution_panel(payload)
    claims = payload.get('claims') or []
    coverage = payload.get('requirement_coverage') or {}
    partial = payload.get('answer_completeness') == 'PARTIAL_SUPPORTED' or bool(payload.get('evidence_gaps'))
    if coverage.get('status') == 'CONFLICTING_ASSESSMENTS':
        st.warning('完整性核验意见不一致：以下保留有证据支持的内容；未解决的要点仍需核对，不能视为已经答全。')
    elif partial:
        st.warning('部分要点有证据支持；其余要点仍需补充资料。')
    if show_heading:st.markdown('**直接回答**')
    if not claims:
        render_claim(payload['answer'])
    for i, claim in enumerate(claims):
        with st.container(key=f'answer_claim_{i}'):
            if len(claims) > 1:
                st.markdown(f'**要点 {i + 1}**')
            render_claim(claim.get('text', ''))
            st.caption('依据：' + '、'.join(f'[{n}]' for n in claim.get('source_indexes', [])))
    with st.expander('回答核验说明'):
        st.caption('引用编号对应本次检索片段；请对照原文确认语义。模型核验不替代人工确认，也不是正确率认证。')
        retired = [r for r in (payload.get('generator_gap_review') or {}).get('items', []) if r.get('retired')]
        for row in retired:
            st.caption('已复核的生成器疑问：' + row['original_gap'])
            st.caption(row['reason'])
        if coverage.get('items'):
            st.markdown('**逐要点覆盖（模型复核）**')
            labels = {'COVERED': '有支持的回答', 'PARTIAL': '仅部分覆盖', 'MISSING': '仍缺依据或回答',
                      'CONFLICTING': '核验意见不一致，需人工确认'}
            for row in coverage['items']:
                st.write(f"{labels.get(row['status'], '尚未核对')}：{row['query']}")
                st.caption(row['reason'])
        else:
            st.caption('逐要点完整性尚未评估；不能仅凭引用通过判断问题已答全。')
