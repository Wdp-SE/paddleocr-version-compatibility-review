import copy
import pytest

from services.paddleocr_disposition import binding, regression_template, assess_disposition, validate_regression


def report():
    return {'report_fingerprint': 'report-a', 'request_fingerprint': 'request-a',
            'source_version': 'v2.9.1', 'target_version': 'v3.0.0',
            'compatibility_report': {'files': [{'path': 'main.py', 'sha256': 'a' * 64}],
                'findings': [{'finding_id': 'risk-1', 'status': 'supported_risk'}],
                'gaps': [], 'regression_requirements': [{'finding_id': 'risk-1', 'input_conditions': ['中文页'], 'checks': ['JSON 契约']} ]}}


def record(r):
    data = regression_template(r)
    data.update(environment='Python 3.10 / PaddleOCR 3.0.0', actual_dependency='v3.0.0')
    data['results'] = [{'finding_id': 'risk-1', 'input_conditions': ['中文页'],
                        'checks': [{'name': 'JSON 契约', 'passed': True}], 'log_excerpt': 'manual run passed'}]
    return data


def test_changed_report_or_files_cannot_reuse_external_records():
    r = report(); data = record(r)
    r['compatibility_report']['files'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='绑定'):
        validate_regression(r, data)


def test_passed_record_does_not_authenticate_runtime_or_clear_static_risk():
    r = report(); original = copy.deepcopy(r)
    parsed = validate_regression(r, record(r))
    assert parsed['record_status'] == 'HUMAN_SUBMITTED_UNVERIFIED'
    result = assess_disposition(r, {'risk-1': {'decision': 'mitigated', 'reason': '适配器已修改'}}, {}, parsed)
    assert result['eligible_for_human_continuation'] is True
    assert result['runtime_verified'] is False and r == original


def test_missing_risk_test_or_pending_gap_blocks_continuation():
    r = report()
    assert not assess_disposition(r, {}, {}, None)['eligible_for_human_continuation']
    risks = {'risk-1': {'decision': 'mitigated', 'reason': '适配器已修改'}}
    assert not assess_disposition(r, risks, {}, None)['eligible_for_human_continuation']
    r['compatibility_report']['gaps'] = [{'gap_id': 'gap-a', 'detail': '动态依赖未解析'}]
    assert not assess_disposition(r, risks, {}, validate_regression(r, record(r)))['eligible_for_human_continuation']


def test_wrong_dependency_missing_check_and_forged_status_are_not_passes():
    r = report(); data = record(r)
    data['actual_dependency'] = 'v2.9.1'
    with pytest.raises(ValueError): validate_regression(r, data)
    data = record(r); data['results'][0]['checks'][0]['passed'] = 'true'
    with pytest.raises(ValueError): validate_regression(r, data)
    data = record(r); data['results'][0]['checks'] = []
    validated = validate_regression(r, data)
    risks = {'risk-1': {'decision': 'mitigated', 'reason': '适配器已修改'}}
    assert not assess_disposition(r, risks, {}, validated)['eligible_for_human_continuation']


def test_empty_report_never_becomes_an_upgrade_approval():
    r = report(); r['compatibility_report']['findings'] = []
    assert not assess_disposition(r, {}, {}, None)['eligible_for_human_continuation']
    assert binding(r) != binding({**r, 'request_fingerprint': 'other'})
