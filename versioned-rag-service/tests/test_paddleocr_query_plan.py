def test_model_plan_cannot_silently_replace_explicit_version():
    from src.paddleocr_query_plan import plan_query
    proposed=lambda **kw: {'requirements':[{'query':'v3.0.0 参数','version':'v3.0.0'}]}
    plan=plan_query('v2.9.1 的 page_num 默认值？',versions=('v2.9.1',),planner=proposed)
    assert plan['versions']==['v2.9.1'] and plan['planner_status']=='REJECTED'


def test_explicit_version_outside_selected_scope_needs_clarification():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('v2.9.1 的调用',versions=('v3.0.0',))
    assert plan['status']=='NEEDS_CLARIFICATION'


def test_compound_query_preserves_original_and_bounded_requirements():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('OCR如何安装？以及结果如何保存？',versions=('v3.0.0',))
    assert plan['original_query']=='OCR如何安装？以及结果如何保存？'
    assert 1<=len(plan['requirements'])<=4

def test_compound_requirements_keep_their_own_modules():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('文本检测 thresh 默认值；文本识别 batch_size 默认值',versions=('v3.0.0',))
    assert [r['module'] for r in plan['requirements']]==['text_detection','text_recognition']

def test_compound_requirements_keep_their_own_versions():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('v2.9.1 OCR cpu_threads 默认值；v3.0.0 OCR cpu_threads 默认值',
                    versions=('v2.9.1','v3.0.0'))
    assert [r['version'] for r in plan['requirements']]==['v2.9.1','v3.0.0']
