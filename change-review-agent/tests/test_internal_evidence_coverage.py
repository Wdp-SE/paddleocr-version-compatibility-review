from app.paddleocr_investigation import investigate, build_checks


class Gateway:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def search(self, question, **scope):
        self.calls += 1
        text = self.text(question)
        return {'results': [{'chunk_id': text, 'content': text, 'version': scope['version']}]}


def check():
    return {'check_id': 'fields', 'query': 'OCR 返回字段', 'versions': ['v3.0.0'],
            'required_tokens': {'v3.0.0': ['rec_texts', 'rec_scores']}}


def test_real_but_irrelevant_evidence_does_not_close_check():
    result = investigate([check()], Gateway(lambda q: '安装 CPU 环境'), clock=lambda: 0,
                         validate_evidence=lambda hit, remaining: hit)
    assert result['unresolved'][0]['reason'] == 'content_requirements_not_covered'
    assert result['checked_evidence']  # genuine sources remain available for inspection
    assert result['check_assessments'][0]['status'] == 'INSUFFICIENT_CONTENT'


def test_all_required_fields_close_only_the_matching_version():
    result = investigate([check()], Gateway(lambda q: 'rec_texts rec_scores'), clock=lambda: 0,
                         validate_evidence=lambda hit, remaining: hit)
    assert result['unresolved'] == []
    assert result['check_assessments'][0]['status'] == 'REQUIREMENTS_COVERED'


def test_check_without_explicit_requirements_stays_candidate():
    row = {**check(), 'required_tokens': {}}
    result = investigate([row], Gateway(lambda q: 'rec_texts rec_scores'), clock=lambda: 0,
                         validate_evidence=lambda hit, remaining: hit)
    assert result['unresolved'][0]['reason'] == 'content_criteria_not_defined'


def test_successful_replan_closes_parent_but_keeps_requirements():
    planner = lambda **kwargs: [{'parent_check_id': 'fields', 'query': '新问题', 'versions': ['v3.0.0'],
                                'required_tokens': {}}]
    result = investigate([check()], Gateway(lambda q: 'rec_texts rec_scores' if q == '新问题' else '安装'),
                         clock=lambda: 0, planner=planner, validate_evidence=lambda hit, remaining: hit)
    assert result['unresolved'] == []
    assert result['search_calls'] == 2


def test_static_result_consumption_provides_source_specific_criteria():
    rows = build_checks({'findings': [{'finding_id': 'f1', 'rule_id': 'legacy_ocr_result',
                        'title': '旧结果读取', 'desired_check': '核对返回', 'application': {}}]})
    facts = rows[0]['required_facts']['v3.0.0']
    assert any('rec_texts' in f['tokens'] and 'rec_scores' in f['tokens'] for f in facts)
    assert any(f['path'] == 'paddleocr/_pipelines/ocr.py' for f in facts)


def test_partial_verified_evidence_accumulates_across_replans_and_resume():
    gateway = Gateway(lambda q: 'rec_scores' if q == '补查得分' else 'rec_texts')
    planner = lambda **kwargs: [{'parent_check_id': 'fields', 'query': '补查得分', 'versions': ['v3.0.0']}]
    result = investigate([check()], gateway, clock=lambda: 0, planner=planner,
                         validate_evidence=lambda hit, remaining: hit)
    assert result['unresolved'] == []
    assert set(result['check_assessments'][0]['evidence_ids']) == {'rec_texts', 'rec_scores'}
    resumed = investigate([check()], gateway, clock=lambda: 0, resume=result['resume'],
                          validate_evidence=lambda hit, remaining: hit)
    assert resumed['unresolved'] == []


def test_partial_sources_from_other_check_do_not_close_obligation():
    second = {**check(), 'check_id': 'other', 'query': '只查得分',
              'required_tokens': {'v3.0.0': ['rec_scores']}}
    result = investigate([check(), second], Gateway(lambda q: 'rec_scores' if q=='只查得分' else 'rec_texts'),
                         clock=lambda:0, validate_evidence=lambda hit, remaining:hit)
    assert [row['check_id'] for row in result['unresolved']] == ['fields']
