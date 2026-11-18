from types import SimpleNamespace

from src.paddleocr_evidence_search import EvidenceSearch
from src.paddleocr_query_plan import plan_query
from src.paddleocr_retrieval_views import build_structure_views


def test_identifier_parts_preserve_whole_symbol():
    from src.paddleocr_evidence_units import retrieval_tokens
    parts=retrieval_tokens('text_det_box_thresh numpy.ndarray')
    assert all(t in parts for t in ('text_det_box_thresh','text','det','box','thresh','numpy.ndarray','numpy','ndarray'))


def test_layout_exclusion_does_not_delete_code_comments_or_runtime_yaml():
    from src.paddleocr_evidence_units import is_layout_only
    assert is_layout_only('---\ncomments: true\n---','guide.md')
    assert is_layout_only('# 使用教程','guide.md')
    assert is_layout_only('</table>','guide.md')
    assert not is_layout_only('# use_angle_cls controls orientation','client.py')
    assert not is_layout_only('---\nlang: ch\n---','runtime.yaml')


def test_only_declared_parameter_column_has_definition_priority():
    from src.paddleocr_evidence_units import parameter_definition_priority
    assert parameter_definition_priority('<tr><td><code>drop_score</code></td><td>过滤阈值</td></tr>',{'drop_score'})==1
    assert parameter_definition_priority('| `drop_score` | float | 0.5 |',{'drop_score'})==1
    assert parameter_definition_priority('<tr><td>other_key</td><td>参见drop_score</td></tr>',{'drop_score'})==0
    assert parameter_definition_priority('drop_score=drop_score',{'drop_score'})==0


def test_parameter_meaning_survives_many_repeated_call_references():
    base={'source_id':'sdk','version':'v3.0.0','language':'zh','namespace':'project_primary',
          'source_sha256':'a'*64,'line_start':1,'heading':'OCR'}
    code=[{**base,'chunk_id':f'code-{i}','document_path':'paddleocr/_pipelines/ocr.py',
           'content':'text_det_box_thresh=text_det_box_thresh '*12,'line_end':1} for i in range(100)]
    definition={**base,'chunk_id':'definition','document_path':'docs/version3.x/pipeline_usage/OCR.md',
                'content':'<tr><td>text_det_box_thresh</td><td>平均得分大于阈值才保留文字区域</td></tr>','line_end':1}
    index=SimpleNamespace(chunks=code+[definition])
    result=EvidenceSearch(index,build_structure_views(index.chunks)).search(
        plan_query('OCR text_det_box_thresh 如何决定保留检测框？',versions=('v3.0.0',)),top_k=1)
    assert result['results'][0]['chunk_id']=='definition'
    assert index.chunks[-1]==definition


def test_natural_batch_lookup_keeps_definition_alongside_conflicting_package_guide():
    from pathlib import Path
    from src.public_knowledge import PublicKnowledgeIndex
    from src.paddleocr_evidence_search import configured_evidence_search
    from src.paddleocr_retrieval_views import generation_evidence_text
    index=PublicKnowledgeIndex(Path(__file__).parents[1]/'public_corpus_paddleocr')
    plan=plan_query('v2.9.1 旧版 OCR 一批识别多少张裁剪图，默认是多少？',versions=('v2.9.1',))
    r=configured_evidence_search(index,structure_windows=True).search(plan,top_k=5,candidate_budget=80)
    definitions=[h for h in r['results'] if h['document_path']=='doc/doc_ch/inference_args.md'
                 and 'rec_batch_num' in generation_evidence_text(h)]
    assert definitions, 'batch definition must not be displaced by FAQ references'
    assert any(h['document_path']=='doc/doc_ch/whl.md' and 'rec_batch_num' in generation_evidence_text(h)
               for h in r['results']), 'keep package context instead of hiding the conflicting source'


def test_batch_definition_priority_does_not_promote_references_or_other_stage():
    from src.paddleocr_evidence_units import task_definition_priority
    intent={'kind':'parameter_lookup','aliases':['识别 batch size 批量']}
    assert task_definition_priority('| rec_batch_num | int | 6 | 识别的batch size |',intent)==1
    assert task_definition_priority('rec_batch_num = args.rec_batch_num # 识别 batch size',intent)==0
    assert task_definition_priority('| cls_batch_num | int | 6 | 分类的batch size |',intent)==0


def test_code_search_aliases_do_not_become_required_words_in_parameter_tables():
    from src.paddleocr_evidence_units import task_definition_priority
    intent = {'kind':'parameter_lookup', 'aliases':['识别 batch size 批量','img_list img_num batch_num'],
              'definition_aliases':['识别 batch size 批量']}
    assert task_definition_priority('| rec_batch_num | int | 6 | 识别的batch size |', intent) == 1
    assert task_definition_priority('| rec_batch_num | 进行识别时，同时前向的图片数 | 30 |', intent) == 1
    assert task_definition_priority('| cls_batch_num | 进行分类时，同时前向的图片数 | 6 |', intent) == 0
def test_literal_context_expansion_preserves_source_and_respects_bounds():
    import copy
    from src.paddleocr_context_selection import complete_parameter_context
    from src.paddleocr_retrieval_views import build_views, validate_views
    body = 'MAPPING = {\n' + ''.join(f'    "key_{i}": "value_{i}",\n' for i in range(20)) + '}\n'
    parent = {'chunk_id': 'p', 'source_id': 's', 'source_sha256': 'abc',
              'version': 'v3.0.0', 'document_path': 'example.py',
              'line_start': 1, 'line_end': len(body.splitlines()), 'content': body}
    views = build_views([parent], token_count=len, max_tokens=128)
    original = copy.deepcopy(views)
    expanded = complete_parameter_context(views[:1], {'p': parent})
    assert expanded[0]['content'] == body
    validate_views([parent], expanded)
    assert views == original
    assert complete_parameter_context(views[:1], {'p': parent}, max_chars=128) == views[:1]
    # A different part of the source may not fetch an unrelated dictionary.
    assert complete_parameter_context(views[-1:], {'p': parent}) == views[-1:]


def test_parameter_context_never_discards_code_already_in_a_retrieved_window():
    from src.paddleocr_context_selection import complete_parameter_context
    from src.paddleocr_retrieval_views import build_views, validate_views
    for body in ('base_params = {}\nraise ValueError("Unknown argument")\n',
                 'def small():\n    return {}\n\nraise ValueError("Unknown argument")\n'):
        parent = {'chunk_id': 'p', 'source_id': 's', 'source_sha256': 'abc',
                  'version': 'v3.0.0', 'document_path': 'example.py', 'line_start': 1, 'content': body}
        views = build_views([parent], token_count=len)
        expanded = complete_parameter_context(views, {'p': parent})
        assert body in expanded[0]['content']
        validate_views([parent], expanded)
