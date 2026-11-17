from src.paddleocr_query_plan import plan_query, retrieval_intent
import pytest


@pytest.mark.parametrize('query', ['不要检测，只识别文字', '不是 OCR，是 PPStructureV3 的字段。'])
def test_negative_wording_keeps_the_positive_business_request(query):
    plan = plan_query(query, versions=('v2.9.1',))
    assert plan['requirements']


def test_colloquial_multi_part_upgrade_reserves_each_dimension_for_each_version():
    plan = plan_query('OCR 2.9.1 升到 3.0.0，怎样初始化、怎么调用、结果怎么读？请提供两版出处。',
                      versions=('v2.9.1','v3.0.0'))
    obligations={(r['version'],r['fact_dimension']) for r in plan['requirements']}
    assert obligations=={(v,d) for v in ('v2.9.1','v3.0.0') for d in ('call','parameters','result')}
    assert len(plan['requirements'])==6


def test_output_operations_are_not_ranked_as_generic_product_introduction():
    value=retrieval_intent('OCR predict 拿到一批结果，逐项打印并导出图片怎么写？')
    assert value['kind']=='result_usage'
    assert 'print' in ' '.join(value['aliases'])
    assert 'save_to_img' in ' '.join(value['aliases'])


def test_text_score_question_maps_to_result_schema_without_inventing_values():
    value=retrieval_intent('OCR 识别出来的文字和置信度在哪些字段里？')
    assert value['kind']=='result_fields'
    assert {'rec_texts','rec_scores'}.issubset(set(' '.join(value['aliases']).split()))


def test_existing_crops_uses_recognition_only_route_without_changing_version():
    value=retrieval_intent('2.9.1 已经裁好小图，只做识别，不要再次找框，该怎么调？')
    assert value['kind']=='api_usage'
    assert 'ocr det rec False' in value['aliases']


def test_ocr_product_shorthand_keeps_method_relationship_lookup():
    value=retrieval_intent('OCR 3.0.0 的 ocr() 方法还保留吗，它与 predict() 是什么关系？')
    assert value['kind']=='api_usage'
    assert value['method_lookup'] is True


@pytest.mark.parametrize('query,symbol', [
    ('请提供 OCR 3.0.0 中 rec_texts 和 rec_scores 含义的依据。','rec_texts'),
    ('请给出 PaddleOCR drop_score 默认值的证据。','drop_score'),
])
def test_requesting_evidence_does_not_discard_the_fact_being_requested(query,symbol):
    plan=plan_query(query,versions=('v3.0.0',))
    assert plan['requirements']
    assert any(symbol in r['symbols'] for r in plan['requirements'])
