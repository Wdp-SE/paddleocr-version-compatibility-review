from services.review_lifecycle import lifecycle_summary


def test_three_upgrade_outcomes_have_explicit_actions_and_runtime_boundary():
    for status in ('supported_risk', 'unaffected', 'needs_verification'):
        result = {'application_snapshot': {'head_commit': 'b'*40}, 'compatibility_report': {
            'findings': [{'finding_id': 'f1', 'status': status, 'desired_check': '空页与 JSON',
                          'application': {'path': 'application/normalizer.py'}}],
            'gaps': [], 'regression_requirements': [{'finding_id': 'f1'}]}}
        out = lifecycle_summary(result)
        assert out['runtime_record_status'] == 'NOT_EXECUTED'
        assert out['actions'][0]['path'] == 'application/normalizer.py'
        assert not out['automatic_approval']
        assert bool(out['static_supported']) == (status != 'needs_verification')
