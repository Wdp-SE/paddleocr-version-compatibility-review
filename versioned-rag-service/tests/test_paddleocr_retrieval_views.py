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
