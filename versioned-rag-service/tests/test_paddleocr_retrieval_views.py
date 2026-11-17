import copy
import pytest

PARENT = {'chunk_id':'c1','source_id':'s1','version':'v3.0.0', 'document_path':'docs/version3.x/pipeline_usage/OCR.md',
          'source_sha256':'abc','line_start':10,'line_end':13,
          'content':'# 参数\n| 参数 | 值 |\n| a | 1 |\n| b | 2 |', 'heading':'参数'}


def test_two_required_rows_in_one_parent_survive_evidence_dedup():
    from src.paddleocr_retrieval_views import materialize_evidence, evidence_text
    rows=materialize_evidence([PARENT],[{'parent_chunk_id':'c1','line_start':12,'line_end':12},
                                     {'parent_chunk_id':'c1','line_start':13,'line_end':13}])
    assert len(rows)==1 and rows[0]['content']==PARENT['content']
    assert {s['line_start'] for s in rows[0]['evidence_spans']}=={12,13}
    assert '| a | 1 |' in evidence_text(rows[0]) and '| b | 2 |' in evidence_text(rows[0])


def test_long_table_last_row_is_indexed_and_view_never_changes_parent():
    from src.paddleocr_retrieval_views import build_views, validate_views
    text='| 参数 | 值 |\n| --- | --- |\n'+'\n'.join(f'| param{i} | {i} |' for i in range(100))
    parent={**PARENT,'content':text,'line_end':111}
    views=build_views([parent],token_count=len,max_tokens=100)
    assert len(views)>10
    assert any('param99' in v['retrieval_text'] for v in views)
    assert all(len(v['retrieval_text'])<=100 for v in views)
    validate_views([parent],views)
    forged=copy.deepcopy(views); forged[-1]['source_id']='another'
    with pytest.raises(ValueError,match='identity'): validate_views([parent],forged)


def test_forged_span_content_cannot_enter_generation():
    from src.paddleocr_retrieval_views import evidence_text
    with pytest.raises(ValueError,match='span'):
        evidence_text({**PARENT,'evidence_spans':[{'line_start':12,'line_end':12,'content':'b is safe'}]})


def test_single_long_line_preserves_exact_character_offsets():
    from src.paddleocr_retrieval_views import build_views, materialize_evidence, evidence_text
    parent={**PARENT,'content':'<table>'+('field|42;'*100)+'</table>','line_end':10,'heading':''}
    views=build_views([parent],token_count=len,max_tokens=80)
    assert all(len(v['retrieval_text'])<=80 for v in views)
    row=materialize_evidence([parent],[views[-1]])[0]
    assert evidence_text(row).endswith('</table>')
    assert row['content']==parent['content']


def test_padded_markdown_parameter_row_keeps_name_description_and_default_together():
    from src.paddleocr_retrieval_views import build_structure_views, validate_views
    row='| drop_score'+(' '*180)+'| 低于此识别分数的文字不返回'+(' '*180)+'| 0.5 |'
    text='| 参数 | 说明 | 默认值 |\n| --- | --- | --- |\n'+row+'\n| other | 其他参数 | False |'
    parent={**PARENT,'content':text,'line_end':13}
    views=build_structure_views([parent],max_chars=384)
    selected=[v for v in views if 'drop_score' in v['content']]
    assert len(selected)==1 and selected[0]['content']==row
    assert '0.5' in selected[0]['content']
    validate_views([parent],views)


def test_markdown_like_rows_inside_fence_stay_one_code_example():
    from src.paddleocr_retrieval_views import build_structure_views
    text='```text\n| field | example |\n| value | 42 |\n```'
    views=build_structure_views([{**PARENT,'content':text,'line_end':13}],max_chars=384)
    assert len(views)==1 and views[0]['content']==text
def test_generation_table_context_keeps_header_without_copying_unrelated_parameter_rows():
    from src.paddleocr_retrieval_views import generation_evidence_text
    header='| 参数 | 默认值 | 说明 |\n|---|---|---|\n'
    middle=''.join(f'| unrelated_{i} | 1 | 其他阶段参数 |\n' for i in range(35))
    selected='| drop_score | 0.5 | 过滤识别得分 |\n'
    text=header+middle+selected
    start=len(header+middle)
    row={'content':text,'line_start':1,'evidence_spans':[{'char_start':start,'char_end':len(text),
         'line_start':38,'line_end':38,'content':selected}]}
    result=generation_evidence_text(row)
    assert '| 参数 | 默认值 | 说明 |' in result and selected in result
    assert 'unrelated_0 ' not in result
    assert len(result)<len(text)/3


def test_python_generation_context_closes_the_actual_return_without_rewriting_retrieval_span():
    import ast
    from src.paddleocr_retrieval_views import generation_evidence_text, evidence_text
    text='def run(self, value):\n    return list(\n        self.predict_iter(\n            value,\n        )\n    )\n'
    start=text.index('    return');end=text.index('            value,')
    span={'char_start':start,'char_end':end,'line_start':2,'line_end':3,'content':text[start:end]}
    hit={'content':text,'line_start':1,'document_path':'pipeline.py','evidence_spans':[span]}
    assembled=generation_evidence_text(hit)
    ast.parse(assembled)
    assert assembled==text
    assert evidence_text(hit)==text[start:end]


def test_oversized_python_function_does_not_bypass_generation_context_budget():
    from src.paddleocr_retrieval_views import generation_evidence_text
    text='def large():\n    selected = 1\n'+('    # filler\n'*600)+"    return 'unselected_tail'\n"
    start=text.index('    selected');end=text.index('    # filler')
    span={'char_start':start,'char_end':end,'line_start':2,'line_end':2,'content':text[start:end]}
    hit={'content':text,'line_start':1,'document_path':'pipeline.py','evidence_spans':[span]}
    value=generation_evidence_text(hit)
    assert len(value)<=6000
    assert 'unselected_tail' not in value
