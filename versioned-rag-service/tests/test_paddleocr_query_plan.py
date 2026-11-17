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


def test_direction_classifier_scope_distinguishes_text_lines_from_documents():
    from src.paddleocr_query_plan import query_module
    assert query_module('文本行方向分类的批量大小如何设置？')=='text_line_orientation_classification'
    assert query_module('文档图像方向分类如何配置？')=='doc_img_orientation_classification'
    assert query_module('OCR 如何启用文本行方向分类？')=='ocr'


def test_colloquial_upgrade_index_read_routes_to_result_contracts():
    from src.paddleocr_query_plan import plan_query
    query='从 PaddleOCR 2.9.1 升级到 3.0.0，我们原来按 line[1][0] 取文字，还能这么取吗？要核对哪些返回字段？'
    plan=plan_query(query,versions=('v2.9.1','v3.0.0'))
    assert plan['retrieval_intent']['kind']=='result_migration'
    assert any('rec_texts' in q for q in plan['retrieval_intent']['aliases'])
    assert {r['module'] for r in plan['requirements']} == {'ocr'}
    assert plan['original_query']==query


def test_named_structure_upgrade_keeps_explicit_pipeline_instead_of_ocr_default():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('PPStructure 从 2.9.1 升级到 3.0.0，旧结果结构怎么变化？',
                    versions=('v2.9.1','v3.0.0'))
    assert {r['module'] for r in plan['requirements']}=={'structure'}


def test_plain_text_extraction_does_not_imply_upgrade_comparison():
    from src.paddleocr_query_plan import plan_query
    plan=plan_query('OCR 怎样取文字？',versions=('v3.0.0',))
    assert plan['retrieval_intent']['kind']!='result_migration'
def test_old_ocr_direction_switch_does_not_select_standalone_document_orientation_module():
    from src.paddleocr_query_plan import plan_query
    out = plan_query('扫描件方向不固定。PaddleOCR 2.9.1 初始化启用了 use_angle_cls，但调用 ocr 时 cls=False，还会做方向分类吗？', versions=('v2.9.1',))
    assert all(r['module'] == 'ocr' for r in out['requirements'])


def test_constructor_parameter_transfer_between_versions_is_migration_without_upgrade_keyword():
    from src.paddleocr_query_plan import plan_query
    out = plan_query('PaddleOCR 2.9.1 的 show_log 传入 3.0.0 构造器后会怎样？请查参数转发和校验代码，不能只看文档有没有列出。',
                     versions=('v2.9.1', 'v3.0.0'))
    assert out['retrieval_intent']['kind'] == 'parameter_migration'


def test_single_version_constructor_usage_and_version_comparison_without_parameter_are_not_migration():
    from src.paddleocr_query_plan import retrieval_intent
    for query in ('PaddleOCR 3.0.0 构造器的 cpu_threads 参数怎样设置？',
                  'PaddleOCR 2.9.1 和 3.0.0 的安装要求有什么不同？',
                  'PaddleOCR 2.9.1 与 3.0.0 的 rec_batch_num 参数默认值分别是多少？'):
        assert retrieval_intent(query)['kind'] != 'parameter_migration'


def test_named_paddleocr_method_question_keeps_ocr_scope_without_call_parentheses():
    from src.paddleocr_query_plan import plan_query
    out = plan_query('目标依赖是 PaddleOCR 3.0.0，应用应如何用 predict 处理一张图？还能找到 ocr 方法吗？',
                     versions=('v3.0.0',))
    assert {r['module'] for r in out['requirements']} == {'ocr'}
    assert out['retrieval_intent']['kind'] == 'api_usage'


def test_named_standalone_module_does_not_become_ocr_when_predict_is_mentioned():
    from src.paddleocr_query_plan import query_module, retrieval_intent
    assert query_module('PaddleOCR 独立文本识别模块使用 predict 方法吗？') == 'text_recognition'
    for name, module in [('版面检测', 'layout_detection'), ('文本检测', 'text_detection')]:
        question = f'PaddleOCR 3.0.0 {name}的 predict 方法如何调用？'
        assert query_module(question) == module
        assert not retrieval_intent(question).get('method_lookup')
