import copy
import pytest
from services.review_comparison import compare_reviews, capture_baseline


def report(status='supported_risk', *, path='application/normalizer.py', missing=False):
    return {'workspace_id': 'paddleocr', 'source_version': 'v2.9.1', 'target_version': 'v3.0.0',
            'request_fingerprint': 'request', 'report_fingerprint': 'report', 'runtime_verified': False,
            'compatibility_report': {'files': [{'path': path, 'sha256': 'a'*64}],
                'findings': [{'rule_id': 'legacy_ocr_result', 'finding_id': 'f1', 'status': status,
                              'application': {'path': path, 'line': 6}, 'title': '旧式结果读取'}],
                'gaps': [{'code': 'missing_application_module', 'detail': '缺调用层'}] if missing else []}}


def test_same_rule_in_changed_file_remains_and_changed_hash_invalidates_old_records():
    before = capture_baseline(report(), corpus_fingerprint='corpus')
    current = report()
    current['compatibility_report']['files'][0]['sha256'] = 'b'*64
    out = compare_reviews(before, current, corpus_fingerprint='corpus')
    assert out['findings'][0]['change'] == 'PERSISTS'
    assert out['files'][0]['change'] == 'MODIFIED'
    assert out['prior_records_reusable'] is False
    assert out['runtime_verified'] is False


def test_removed_risk_is_only_cleared_with_analysis_of_same_file_and_no_new_gap():
    before = capture_baseline(report(), corpus_fingerprint='corpus')
    current = report('unaffected')
    out = compare_reviews(before, current, corpus_fingerprint='corpus')
    assert out['findings'][0]['change'] == 'NOT_OBSERVED_NOW'
    assert '兼容' not in out['findings'][0]['change']


@pytest.mark.parametrize('mode', ['missing_file', 'unresolved', 'global_gap'])
def test_disappearing_risk_is_not_resolution_when_input_or_analysis_is_missing(mode):
    before = capture_baseline(report(), corpus_fingerprint='corpus')
    current = report('needs_verification' if mode == 'unresolved' else 'unaffected', missing=mode == 'global_gap')
    if mode == 'missing_file':
        current['compatibility_report']['files'] = []
        current['compatibility_report']['findings'] = []
    out = compare_reviews(before, current, corpus_fingerprint='corpus')
    assert out['findings'][0]['change'] == 'UNRESOLVED'


def test_version_corpus_or_workspace_changes_refuse_comparison():
    before = capture_baseline(report(), corpus_fingerprint='corpus')
    assert compare_reviews(before, report(), corpus_fingerprint='other')['status'] == 'INCOMPARABLE'
    current = report(); current['target_version'] = 'v3.1.0'
    assert compare_reviews(before, current, corpus_fingerprint='corpus')['status'] == 'INCOMPARABLE'


@pytest.mark.parametrize('field', ['analysis_policy_fingerprint', 'application_context'])
def test_changed_policy_or_business_context_cannot_look_like_a_fixed_application(field):
    previous = {**report(), field: 'old'}
    before = capture_baseline(previous, corpus_fingerprint='corpus')
    current = {**report('unaffected'), field: 'new'}
    assert compare_reviews(before, current, corpus_fingerprint='corpus')['status'] == 'INCOMPARABLE'


def test_baseline_is_immutable_and_tampering_is_rejected():
    source = report(); before = capture_baseline(source, corpus_fingerprint='corpus')
    source['compatibility_report']['findings'].clear()
    assert before['result']['compatibility_report']['findings']
    before['result']['compatibility_report']['findings'].clear()
    assert compare_reviews(before, report(), corpus_fingerprint='corpus')['status'] == 'INCOMPARABLE'


def test_new_risk_and_file_change_and_duplicates_are_not_hidden():
    before = capture_baseline(report(), corpus_fingerprint='corpus')
    current = report()
    current['compatibility_report']['findings'].append({
        'rule_id': 'removed_api', 'finding_id': 'f2', 'status': 'supported_risk',
        'application': {'path': 'application/normalizer.py', 'line': 10}, 'title': '接口变化'})
    out = compare_reviews(before, current, corpus_fingerprint='corpus')
    assert {r['change'] for r in out['findings']} == {'PERSISTS', 'NEW'}
