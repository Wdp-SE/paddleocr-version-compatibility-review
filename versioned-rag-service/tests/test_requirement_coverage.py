import copy
import pytest

from src.claim_support import check_claim_support


CLAIMS = [{'text': '文字字段为 rec_texts', 'evidence_ids': ['a']}]
EVIDENCE = [{'chunk_id': 'a', 'version': 'v3.0.0', 'module': 'ocr', 'content': 'rec_texts 包含识别文字'}]
REQUIREMENTS = [
    {'id': 'r1', 'query': '文字怎么取', 'version': 'v3.0.0', 'module': 'ocr'},
    {'id': 'r2', 'query': '空页如何处理', 'version': 'v3.0.0', 'module': 'ocr'},
]


def response(rows=None):
    return {'verdicts': [{'claim_index': 0, 'supported': True, 'reason': '原文支持'}],
            'requirement_verdicts': rows or [
                {'requirement_id': 'r1', 'status': 'COVERED', 'claim_indexes': [0], 'reason': '文字字段已回答'},
                {'requirement_id': 'r2', 'status': 'MISSING', 'claim_indexes': [], 'reason': '没有空页行为依据'}]}


def run(value):
    return check_claim_support(copy.deepcopy(CLAIMS), EVIDENCE, lambda _: (value, {}),
                               question='文字怎么取以及空页如何处理', requirements=REQUIREMENTS)


def test_true_claim_does_not_mean_all_question_requirements_answered():
    seen = []
    def judge(payload):
        seen.extend(payload)
        return response(), {}
    out = check_claim_support(CLAIMS, EVIDENCE, judge, requirements=REQUIREMENTS)
    assert seen[0]['requirements'] == REQUIREMENTS
    assert out['status'] == 'SUPPORTED'
    assert out['coverage']['status'] == 'PARTIAL_SUPPORTED'
    assert out['coverage']['items'][0]['evidence_ids'] == ['a']
    assert out['coverage']['items'][1]['status'] == 'MISSING'


def test_legacy_or_failed_coverage_keeps_supported_claim_but_never_marks_complete():
    out = run({'verdicts': response()['verdicts']})
    assert out['claims'] == CLAIMS
    assert out['coverage']['status'] == 'NOT_ASSESSED'


@pytest.mark.parametrize('rows', [
    [{'requirement_id': 'r1', 'status': 'COVERED', 'claim_indexes': [0], 'reason': 'missing r2'}],
    [{'requirement_id': 'foreign', 'status': 'COVERED', 'claim_indexes': [0], 'reason': 'foreign'}]*2,
    [{'requirement_id': 'r1', 'status': 'COVERED', 'claim_indexes': [999], 'reason': 'foreign claim'},
     {'requirement_id': 'r2', 'status': 'MISSING', 'claim_indexes': [], 'reason': 'gap'}],
])
def test_forged_or_incomplete_requirement_verdicts_never_become_complete(rows):
    assert run(response(rows))['coverage']['status'] == 'NOT_ASSESSED'


def test_rejected_claim_cannot_close_a_requirement():
    value = response()
    value['verdicts'][0]['supported'] = False
    out = run(value)
    assert out['claims'] == []
    assert out['coverage']['items'][0]['status'] == 'MISSING'


def test_missing_verdict_with_supporting_claim_is_a_conflict_not_a_source_gap():
    value = response()
    value['requirement_verdicts'][0].update(status='MISSING', claim_indexes=[0])
    out = run(value)
    assert out['coverage']['status'] == 'CONFLICTING_ASSESSMENTS'
    assert out['coverage']['items'][0]['status'] == 'CONFLICTING'
    assert out['coverage']['items'][0]['evidence_ids'] == ['a']


def test_wrong_version_evidence_cannot_cover_requirement():
    evidence = [{**EVIDENCE[0], 'version': 'v2.9.1'}]
    out = check_claim_support(CLAIMS, evidence, lambda _: (response(), {}), requirements=REQUIREMENTS)
    assert out['coverage']['items'][0]['status'] == 'MISSING'


def test_complete_is_labeled_as_model_assessment_not_human_accuracy():
    rows = [{'requirement_id': r['id'], 'status': 'COVERED', 'claim_indexes': [0], 'reason': '覆盖'}
            for r in REQUIREMENTS]
    out = run(response(rows))
    assert out['coverage']['status'] == 'MODEL_ASSESSED_COMPLETE'
    assert out['coverage']['assessment_type'] == 'model_assessment_not_human_validation'


def test_coverage_checker_receives_generator_gaps_as_untrusted_context():
    seen=[]
    def judge(payload):
        seen.extend(payload)
        return response(),{}
    result=check_claim_support(CLAIMS,EVIDENCE,judge,requirements=REQUIREMENTS,
                               reported_gaps=['尚未明确空页行为'])
    assert seen[0]['reported_gaps']==['尚未明确空页行为']
    assert result['coverage']['status']=='PARTIAL_SUPPORTED'


def test_query_plan_exposes_truncated_requirements_instead_of_silently_omitting():
    from src.paddleocr_query_plan import plan_query
    plan = plan_query('安装；调用；结果；阈值；配置；参数；返回；推理；空页；多页', versions=('v3.0.0',))
    assert plan['omitted_requirements'] == ['空页', '多页']


def test_repair_expands_source_units_without_changing_requested_version_or_question():
    from pathlib import Path
    from src.public_api import SearchRequest, _execute_public_search
    from src.public_knowledge import PublicKnowledgeIndex
    index = PublicKnowledgeIndex(Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr')
    payload = SearchRequest(query='OCR 默认识别批次数量是什么', version='v2.9.1', language='zh',
                            retrieval_policy='paddleocr_evidence', candidate_budget=120, top_k=8)
    out = _execute_public_search(index, payload, None, top_k=8, correction=True)
    assert out['rerank_diagnostics']['context_window_limit_per_source'] == 8
    assert out['rerank_diagnostics']['window_unit'] == 'structured_characters'
    assert {r['version'] for r in out['results']} == {'v2.9.1'}
    assert all(r['query'] == payload.query for r in out['requirements'])


def test_public_query_surfaces_missing_requirement_even_when_generator_omits_gaps():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from src.public_knowledge import PublicKnowledgeIndex
    from src.public_server import create_app
    class Generator:
        provider, model = 'test', 'test'
        def generate(self, **kwargs):
            return {'claims': [{'text': '请核对 OCR 结果字段。', 'evidence_ids': ['E1']}],
                    'relevant_sources': [], 'evidence_gaps': []}
        def verify_claims_with_diagnostics(self, payload):
            reqs = payload[0].get('requirements', [])
            return {'verdicts': [{'claim_index': 0, 'supported': True, 'reason': '测试接线'}],
                    'requirement_verdicts': [
                        {'requirement_id': r['id'], 'status': 'MISSING', 'claim_indexes': [],
                         'reason': '该要点未被回答'} for r in reqs]}, {}
    index = PublicKnowledgeIndex(Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=Generator())) as client:
        result = client.post('/public/query', json={
            'query': 'OCR 结果有哪些字段；OCR 结果怎样读取', 'version': 'v3.0.0', 'language': 'zh'}).json()
    assert result['status'] == 'OK'
    assert result['requirement_coverage']['status'] == 'PARTIAL_SUPPORTED'
    assert result['answer_completeness'] == 'PARTIAL_SUPPORTED'
    assert any('未被回答' in gap for gap in result['evidence_gaps'])


def test_public_query_keeps_supported_fact_when_completeness_assessments_conflict():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from src.public_knowledge import PublicKnowledgeIndex
    from src.public_server import create_app

    class Generator:
        provider, model = 'test', 'test'

        def generate(self, **kwargs):
            return {'claims': [{'text': 'rec_batch_num 默认为 6。', 'evidence_ids': ['E1']}],
                    'relevant_sources': [], 'evidence_gaps': ['裁剪图输入条件仍需核对']}

        def verify_claims_with_diagnostics(self, payload):
            assert payload[0]['reported_gaps'] == ['裁剪图输入条件仍需核对']
            return {'verdicts': [{'claim_index': 0, 'supported': True, 'reason': '参数定义支持'}],
                    'requirement_verdicts': [
                        {'requirement_id': r['id'], 'status': 'COVERED', 'claim_indexes': [0],
                         'reason': '参数和默认值已回答'} for r in payload[0]['requirements']]}, {}

    index = PublicKnowledgeIndex(Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr')
    with TestClient(create_app(index=index, generator=Generator())) as client:
        result = client.post('/public/query', json={
            'query': 'rec_batch_num 默认值是多少，用于什么输入？',
            'version': 'v2.9.1', 'language': 'zh'}).json()
    assert result['status'] == 'OK'
    assert 'rec_batch_num' in result['answer']
    assert result['answer_completeness'] == 'PARTIAL_SUPPORTED'
    assert result['requirement_coverage']['status'] == 'CONFLICTING_ASSESSMENTS'
    assert '裁剪图输入条件仍需核对' in result['evidence_gaps']
