"""Shared, explicit hand-off state for upgrade and configuration workflows."""
from services.paddleocr_disposition import validate_regression


def lifecycle_summary(result, regression=None):
    report = result.get('compatibility_report', {})
    findings, gaps = report.get('findings', []), report.get('gaps', [])
    snapshot = result.get('application_snapshot') or {}
    runtime = 'NOT_EXECUTED'
    if regression:
        try:
            runtime = validate_regression(result, regression)['record_status']
        except ValueError:
            runtime = 'STALE_OR_INVALID'
    return {'workflow_type': result.get('workflow_type', 'dependency_upgrade'),
            'application_commit': snapshot.get('head_commit'),
            'baseline_commit': (result.get('baseline_application_snapshot') or {}).get('head_commit')
                               or (result.get('commit_comparison') or {}).get('previous_application_commit'),
            'static_supported': [f for f in findings if f.get('status') in ('supported_risk', 'behavior_change', 'unaffected')],
            'actions': [{'finding_id': f['finding_id'], 'path': f.get('application', {}).get('path'),
                         'action': f.get('desired_check', '补齐应用上下文后重新检查。')} for f in findings],
            'input_gaps': [g for g in gaps if g.get('blocking') is not False],
            'regression_requirements': report.get('regression_requirements', []),
            'runtime_record_status': runtime, 'runtime_verified': False,
            'human_decision': 'PENDING', 'automatic_approval': False}
