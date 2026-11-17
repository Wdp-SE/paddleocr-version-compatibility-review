from src.paddleocr_query_plan import claim_scope_module


def test_legacy_orientation_description_keeps_owning_ocr_scope():
    requirements=[{'module':'ocr','symbols':['use_angle_cls'],'version':'v2.9.1'}]
    assert claim_scope_module('方向分类器只有在 use_angle_cls 与 cls 同时为真时执行。',requirements)=='ocr'


def test_explicit_standalone_orientation_api_still_changes_scope():
    requirements=[{'module':'ocr','symbols':['use_angle_cls'],'version':'v2.9.1'}]
    assert claim_scope_module('方向分类模块 DocImgOrientationClassification 使用 predict。',requirements)=='doc_img_orientation_classification'
