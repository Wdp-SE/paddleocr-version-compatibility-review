from pathlib import Path
import pytest
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_query_plan import plan_query
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_retrieval_views import generation_evidence_text


@pytest.fixture(scope='module')
def index():
    return PublicKnowledgeIndex(Path(__file__).resolve().parents[1]/'public_corpus_paddleocr')


def test_followup_clause_keeps_explicit_parent_pipeline_scope():
    plan = plan_query('PaddleOCR 3.0.0 通用 OCR 的 use_textline_orientation 为 None 时怎样处理？如何关闭文本行方向分类？', versions=('v3.0.0',))
    assert {r['module'] for r in plan['requirements']} == {'ocr'}


def test_each_version_has_a_fact_obligation_not_the_entire_migration_in_isolation():
    plan = plan_query('PaddleOCR 2.9.1 升级到 3.0.0 结果读取为什么要调整？', versions=('v2.9.1', 'v3.0.0'))
    assert [r['version_role'] for r in plan['requirements']] == ['baseline', 'target']
    assert all(r['evidence_obligation'] for r in plan['requirements'])


def test_parameter_combination_question_retains_python_usage_witness(index):
    plan = plan_query('扫描页有180度旋转文字时，PaddleOCR 2.9.1 的 use_angle_cls 与 ocr(cls=True) 如何配合？', versions=('v2.9.1',))
    result = configured_evidence_search(index, structure_windows=True).search(plan, top_k=5, candidate_budget=80)
    assert any('PaddleOCR(use_angle_cls=True' in generation_evidence_text(r)
               and '.ocr(' in generation_evidence_text(r) and 'cls=True' in generation_evidence_text(r)
               for r in result['results'])


def test_multiple_result_fields_keep_each_definition_not_only_serialized_example(index):
    plan = plan_query('PaddleOCR 3.0.0 通用 OCR 的 rec_texts 和 rec_scores 分别表示什么？', versions=('v3.0.0',))
    result = configured_evidence_search(index, structure_windows=True).search(plan, top_k=5, candidate_budget=80)
    text = '\n'.join(generation_evidence_text(r) for r in result['results'])
    assert '`rec_texts`: `(List[str])`' in text
    assert '`rec_scores`: `(List[float])`' in text


def test_threshold_boundary_keeps_comparison_and_retained_output_together(index):
    plan=plan_query('OCR 2.9.1 结果分数刚好等于 drop_score 时，主链过滤代码保留还是丢弃？',versions=('v2.9.1',))
    result=configured_evidence_search(index,structure_windows=True).search(plan,top_k=5,candidate_budget=80)
    witnesses=[generation_evidence_text(r) for r in result['results']
               if r['document_path']=='tools/infer/predict_system.py']
    assert any('if score >= self.drop_score:' in t and 'filter_rec_res.append(rec_result)' in t for t in witnesses)


def test_two_classification_switches_retain_forwarding_and_boolean_guard(index):
    plan=plan_query('PaddleOCR 2.9.1 初始化 use_angle_cls=True，调用 ocr 时 cls=False，还会执行方向分类吗？',versions=('v2.9.1',))
    result=configured_evidence_search(index,structure_windows=True).search(plan,top_k=5,candidate_budget=80)
    text='\n'.join(generation_evidence_text(r) for r in result['results'])
    assert 'self.__call__(img, cls, slice)' in text
    assert 'if self.use_angle_cls and cls:' in text


def test_complex_upgrade_budget_keeps_initializers_not_only_predict_arguments(index):
    plan=plan_query('PaddleOCR 2.9.1 升级到 3.0.0，调用入口、初始化参数和结果读取应分别核查什么？',versions=('v2.9.1','v3.0.0'))
    result=configured_evidence_search(index,structure_windows=True).search(plan,top_k=8,candidate_budget=120)
    for version in ('v2.9.1','v3.0.0'):
        assert any(r['version']==version and r['document_path'].endswith('.py') and
                   'def __init__(' in generation_evidence_text(r) for r in result['results'])


def test_parameter_migration_retains_all_keyword_routing_branches(index):
    plan=plan_query('PaddleOCR 2.9.1 升级到 3.0.0 后，show_log 和 det_db_thresh 这些旧参数应如何处理？',
                    versions=('v2.9.1','v3.0.0'))
    result=configured_evidence_search(index,structure_windows=True).search(plan,top_k=5,candidate_budget=80)
    text='\n'.join(generation_evidence_text(r) for r in result['results'])
    for code in ('for name, val in kwargs.items():','if name in _DEPRECATED_PARAM_NAME_MAPPING:',
                 'params[new_name] = val','else:\n                base_params[name] = val','unknown_names'):
        assert code in text,code


def test_explicit_detection_component_keeps_its_own_scope_after_ocr_clause(index):
    plan=plan_query('PaddleOCR 3.0.0 通用 OCR 的 use_textline_orientation 怎样设置？'
                    '文本检测 TextDetection() 的 thresh 默认值是多少？',versions=('v3.0.0',))
    assert [r['module'] for r in plan['requirements']]==['ocr','text_detection']
    result=configured_evidence_search(index,structure_windows=True).search(plan,top_k=5,candidate_budget=80)
    assert any(r['module']=='text_detection' for r in result['results'])


def test_all_collected_component_apis_keep_explicit_subquestion_scope():
    import re
    root=Path(__file__).resolve().parents[1]/'public_corpus_paddleocr/sources/v3.0.0/docs/version3.x/module_usage'
    checked=0
    for path in root.glob('*.md'):
        match=re.search(r'^from paddleocr import ([A-Za-z]+)\s*$',path.read_text(encoding='utf8'),re.M)
        if not match:continue
        plan=plan_query('PaddleOCR 3.0.0 通用 OCR 的 use_textline_orientation 如何设置？'
                        f'{match[1]}() 如何调用？',versions=('v3.0.0',))
        assert [r['module'] for r in plan['requirements']]==['ocr',path.stem],path.name
        checked+=1
    assert checked==9
