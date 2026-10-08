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


def render_answer(payload):
    claims = payload.get('claims') or []
    partial = payload.get('answer_completeness') == 'PARTIAL_SUPPORTED' or bool(payload.get('evidence_gaps'))
    if partial:
        st.warning('部分要点有证据支持；其余要点仍需补充资料。')
    st.markdown('**直接回答**')
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
