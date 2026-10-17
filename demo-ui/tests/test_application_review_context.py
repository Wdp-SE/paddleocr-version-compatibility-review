import copy
import pytest
from services.application_review_context import attach_application_context
from services.paddleocr_disposition import regression_template, validate_regression
from services.paddleocr_disposition import assess_disposition, gap_id
from services.review_comparison import capture_baseline, compare_reviews


def result():
    return {'workspace_id': 'paddleocr', 'runtime_verified': False, 'source_version': 'v2.9.1',
            'target_version': 'v3.0.0', 'request_fingerprint': 'r', 'report_fingerprint': 'f',
            'application_context': {'application_name': 'demo'},
            'compatibility_report': {'findings': [], 'files': [], 'gaps': [], 'regression_requirements': []}}


def snapshot(sha='a' * 40):
    return {'application_id': 'ocr', 'head_commit': sha, 'base_commit': None,
            'files': [], 'inventory': [], 'gaps': [{'code': 'missing_registered_role', 'detail': '缺调用层'}],
            'coverage': {'status': 'INCOMPLETE'}, 'source_identity': 'student_poc_application'}


def test_incomplete_snapshot_is_bound_and_cannot_be_treated_as_complete():
    original = result()
    original['status'] = original['compatibility_report']['status'] = 'unaffected'
    r = attach_application_context(original, snapshot())
    assert not original['compatibility_report']['gaps']
    assert r['compatibility_report']['gaps'][0]['detail'] == '缺调用层'
    assert r['application_snapshot']['head_commit'] == 'a' * 40
    assert r['request_fingerprint'] != original['request_fingerprint']
    assert r['runtime_verified'] is False
    assert r['status'] == r['compatibility_report']['status'] == 'needs_verification'
    assert r['compatibility_report']['summary']['gap_count'] == 1
    gap = r['compatibility_report']['gaps'][0]
    assessment = assess_disposition(r, {}, {gap_id(gap): {'decision': 'accepted_residual', 'reason': 'reviewed'}}, None)
    assert any('输入不完整' in text for text in assessment['blockers'])


def test_external_regression_requires_exact_application_commit():
    r = attach_application_context(result(), snapshot())
    record = regression_template(r)
    assert record['application_commit'] == 'a' * 40
    record.update(environment='local controlled CI', actual_dependency='v3.0.0')
    record['application_commit'] = 'b' * 40
    with pytest.raises(ValueError, match='提交'): validate_regression(r, record)


def test_change_of_commit_is_comparable_but_not_different_app():
    a = attach_application_context(result(), snapshot())
    b = attach_application_context(result(), snapshot('b' * 40))
    baseline = capture_baseline(a, corpus_fingerprint='corpus')
    assert compare_reviews(baseline, b, corpus_fingerprint='corpus')['status'] == 'COMPARED'
    b['application_snapshot']['application_id'] = 'another'
    assert compare_reviews(baseline, b, corpus_fingerprint='corpus')['status'] == 'INCOMPARABLE'
