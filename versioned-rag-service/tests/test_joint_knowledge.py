import asyncio
import hashlib
import pytest

from src.joint_knowledge import answer_joint, application_hits


def app_evidence():
    text = '{"required_line_fields":["text","confidence","page_index"]}'
    return {'path': 'application/contract.json', 'content': text,
            'sha256': hashlib.sha256(text.encode()).hexdigest(), 'line_start': 1, 'line_end': 1,
            'application_commit': 'a'*40, 'application_id': 'ocr', 'source_identity': 'submitted_application',
            'evidence_id': 'app-contract'}


def test_application_hit_never_impersonates_official_version_or_publisher():
    hit = application_hits([app_evidence()])[0]
    assert hit['source_identity'] == 'submitted_application'
    assert hit['application_commit'] == 'a'*40
    assert hit['version'] not in ('v2.9.1','v3.0.0')
    assert hit['source_url'] is None


def test_joint_no_model_still_returns_both_evidence_scopes_without_runtime_verdict():
    official = {'chunk_id': 'off', 'content': 'rec_texts rec_scores', 'version': 'v3.0.0'}
    out = asyncio.run(answer_joint('如何满足应用字段？', [official], [app_evidence()], None))
    assert out['status'] == 'GENERATION_NOT_CONFIGURED'
    assert {row['source_identity'] for row in out['evidence']} == {'official_component', 'submitted_application'}
    assert out['runtime_verified'] is False


@pytest.mark.parametrize('application', [[], [app_evidence()]])
def test_supported_component_claim_survives_missing_application_support(application):
    class Generator:
        def generate(self, **kwargs):
            return {'claims': [
                {'text': 'rec_texts 是 OCR 识别文本。', 'evidence_ids': ['E1']},
                {'text': '本应用已经兼容。', 'evidence_ids': ['E1']},
                {'text': '虚构依据。', 'evidence_ids': ['invented']},
            ], 'evidence_gaps': []}

        def verify_claims_with_diagnostics(self, rows):
            assert [row['claim'] for row in rows] == ['rec_texts 是 OCR 识别文本。']
            return {'verdicts': [{'claim_index': 0, 'supported': True, 'reason': 'supported'}]}, {}

    out = asyncio.run(answer_joint('如何让 rec_texts 满足本应用契约？',
        [{'chunk_id': 'off', 'version': 'v3.0.0', 'module': 'ocr', 'content': 'rec_texts'}],
        application, Generator()))
    assert out['status'] == 'OK'
    assert out['answer_completeness'] == 'PARTIAL_SUPPORTED'
    assert out['answer'] == 'rec_texts 是 OCR 识别文本。'
    assert out['evidence_gaps']
    assert len(out['claim_verification']['deterministic_rejections']) == 2


def test_fabricated_citations_are_hidden():
    class Generator:
        def generate(self, **kwargs):
            return {'claims': [{'text': '已通过全部测试', 'evidence_ids': ['invented']}], 'evidence_gaps': []}
    out = asyncio.run(answer_joint('应用契约', [{'chunk_id':'off','content':'example','version':'v3.0.0'}], [app_evidence()], Generator()))
    assert out['status'] == 'ABSTAINED'
    assert out['answer'] == 'N/A'


def test_supported_joint_answer_retains_two_distinct_citation_identities():
    class Generator:
        def generate(self, **kwargs):
            assert 'identity=submitted_application' in kwargs['context']
            return {'claims':[{'text':'组件有 rec_texts；本应用声明 text 字段，应在适配层转换。',
                               'evidence_ids':['E1','E2']}], 'evidence_gaps':[]}
        def verify_claims_with_diagnostics(self, rows):
            return {'verdicts':[{'claim_index':0,'supported':True,'reason':'Two evidence scopes provided'}]}, {}
    out = asyncio.run(answer_joint('应用 text 契约', [{'chunk_id':'off','content':'rec_texts','version':'v3.0.0'}], [app_evidence()], Generator()))
    assert out['status']=='OK'
    assert {r['source_identity'] for r in out['sources']}=={'submitted_application','official_component'}
    assert out['runtime_verified'] is False


def test_official_evidence_cannot_alone_establish_application_requirements():
    class Generator:
        def generate(self, **kwargs):
            return {'claims':[{'text':'本应用要求 pages 字段','evidence_ids':['E1']}], 'evidence_gaps':[]}
    out = asyncio.run(answer_joint('应用契约', [{'chunk_id':'off','content':'pages','version':'v3.0.0'}], [app_evidence()], Generator()))
    assert out['status']=='ABSTAINED'


def test_joint_rejects_wrong_version_even_if_semantic_judge_would_accept():
    class Generator:
        def generate_with_diagnostics(self, **kwargs):
            return {'claims':[{'text':'PaddleOCR v3.0.0 的 ocr 方法返回嵌套列表。','evidence_ids':['E1']}], 'evidence_gaps':[]}, {'usage':{'total_tokens':17}}
        def verify_claims_with_diagnostics(self, rows):
            pytest.fail('deterministic version rejection must precede judge')
    out=asyncio.run(answer_joint('PaddleOCR 3.0.0 的 OCR 返回什么？',
        [{'chunk_id':'old','content':'PaddleOCR returns nested list','version':'v2.9.1','module':'ocr'}],
        [app_evidence()],Generator()))
    assert out['status']=='ABSTAINED'
    assert out['generation']['usage']['total_tokens']==17


@pytest.mark.parametrize('with_app_citation,expected', [(True,'MODEL_ASSESSED_COMPLETE'), (False,'PARTIAL_SUPPORTED')])
def test_joint_completeness_requires_supported_claims_from_both_source_scopes(with_app_citation, expected):
    class Generator:
        def generate(self, **kwargs):
            refs = ['E1', 'E2'] if with_app_citation else ['E1']
            return {'claims': [{'text': 'rec_texts 对应 text。', 'evidence_ids': refs}], 'evidence_gaps': []}

        def verify_claims_with_diagnostics(self, rows):
            return {'verdicts': [{'claim_index': 0, 'supported': True, 'reason': 'test'}],
                    'requirement_verdicts': [{'requirement_id': r['id'], 'status': 'COVERED',
                        'claim_indexes': [0], 'reason': 'test'} for r in rows[0].get('requirements', [])]}, {}

    result = asyncio.run(answer_joint('OCR 的 rec_texts 如何对应应用输出？',
        [{'chunk_id':'off','content':'rec_texts','version':'v3.0.0','module':'ocr'}],
        [app_evidence()], Generator()))
    assert result['status'] == 'OK'
    assert result['answer_completeness'] == expected
    assert result['runtime_verified'] is False
