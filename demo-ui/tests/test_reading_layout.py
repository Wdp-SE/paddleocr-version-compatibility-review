from streamlit.testing.v1 import AppTest

from components.reading_layout import paragraph_blocks, unique_citations, evidence_excerpt


def test_paragraphs_keep_code_and_every_fact_without_splitting_code_punctuation():
    text = '旧接口读取位置需要调整。' * 12 + '`consume("a。b")` 返回字段需核查。\n```python\nx = "句。号"\n```\n仍需运行回归。'
    blocks = paragraph_blocks(text)
    assert len(blocks) > 1
    assert ''.join(blocks) == text
    assert any('`consume("a。b")`' in block for block in blocks)
    assert any('```python\nx = "句。号"\n```' in block for block in blocks)


def test_duplicate_sources_merge_only_when_version_location_and_provenance_match():
    citation = dict(version='v2.9.1', path='ocr.py', url='https://example.com/ocr.py',
                    commit='a' * 40, sha256='b' * 64, line_start=1, line_end=3)
    distinct = [{**citation, 'version': 'v3.0.0'}, {**citation, 'line_end': 4},
                {**citation, 'commit': 'c' * 40}, {**citation, 'sha256': 'd' * 64}]
    assert unique_citations([citation, dict(citation), *distinct]) == [citation, *distinct]
    # Missing provenance is insufficient to collapse records.
    assert len(unique_citations([{'path': 'ocr.py'}, {'path': 'ocr.py'}])) == 2


def test_answer_keeps_partial_warning_and_citations_next_to_numbered_points():
    app = AppTest.from_string('''
from components.answer_reading import render_answer
render_answer({'answer': 'unused', 'answer_completeness': 'PARTIAL_SUPPORTED',
 'claims': [{'text': '旧结果按位置读取。新结果按字段读取。', 'source_indexes': [1, 2]},
            {'text': '实际输出仍需回归。', 'source_indexes': [3]}]})
''').run()
    assert not app.exception
    assert '部分要点有证据支持' in app.warning[0].value
    assert [item.value for item in app.caption][:2] == ['依据：[1]、[2]', '依据：[3]']
    assert any('要点 2' in item.value for item in app.markdown)


def test_long_original_evidence_is_collapsed_and_preview_escapes_html():
    content = '<script>bad()</script>' + '原始字段说明。' * 80
    row = dict(content=content, version='v3.0.0', document_key='ocr-results',
               heading='输出字段', source_url='https://example.com/ocr', line_start=1, line_end=1)
    app = AppTest.from_string('from public_workbench import _source_card\n'
                             f'_source_card({row!r}, index=1, key_prefix="test")').run()
    assert not app.exception
    detail = next(item for item in app.expander if item.label == '查看原文片段 · 1 段')
    assert not detail.proto.expanded
    preview = next(item.value for item in app.markdown if 'class="evidence-preview"' in item.value)
    assert '&lt;script&gt;' in preview and '<script>' not in preview
    assert len(preview) < 260
    assert any(item.value == content for item in app.markdown)


def test_claim_fenced_code_is_not_misread_as_long_inline_code():
    code = 'payload = ' + repr('示例。' * 50)
    text = '核对结果字段。\n```python\n' + code + '\n```\n仍需回归。'
    app = AppTest.from_string('from components.answer_reading import render_claim\n'
                             f'render_claim({text!r})').run()
    assert not app.exception
    assert any(item.value == '```python\n' + code + '\n```' for item in app.markdown)


def test_excerpt_removes_table_markup_and_focuses_actual_query_fields():
    texts = ['无关参数。' * 80, '<td><code>rec_scores</code></td><td>识别置信度列表</td>']
    excerpt = evidence_excerpt(texts, 'rec_scores 表示什么？')
    assert 'rec_scores' in excerpt and '识别置信度列表' in excerpt
    assert '<td>' not in excerpt and '<code>' not in excerpt
    assert len(excerpt) <= 182
    assert evidence_excerpt(['类型 <Unknown> 必须保留']) == '类型 <Unknown> 必须保留'
