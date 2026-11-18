import pytest
from src.paddleocr_query_plan import query_module, plan_query


@pytest.mark.parametrize('query,expected',[
    ('只看 PaddleOCR OCR 产线的 rec_texts，别拿 PPStructureV3 的结果当证据。','ocr'),
    ('PPStructureV3 的 overall_ocr_res 是什么？这里不是独立 OCR 调用。','structure'),
    ('不是 OCR，是 PPStructureV3 的字段。','structure'),
    ('PaddleOCR OCR 的结果，而非 PPStructureV3 的输出。','ocr'),
])
def test_explicitly_excluded_pipeline_does_not_take_over_query_scope(query,expected):
    assert query_module(query)==expected
    assert {r['module'] for r in plan_query(query,versions=('v3.0.0',))['requirements']}=={expected}
