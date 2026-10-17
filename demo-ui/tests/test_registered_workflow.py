from services.registered_review import run_registered_review


def test_candidate_review_checks_both_snapshots_and_keeps_app_evidence_separate():
    calls = []
    def analyze(client, files, **kwargs):
        calls.append(files)
        return {'workspace_id': 'paddleocr', 'runtime_verified': False, 'source_version': 'v2.9.1',
                'target_version': 'v3.0.0', 'corpus_fingerprint': 'corpus',
                'request_fingerprint': 'r', 'report_fingerprint': 'f',
                'compatibility_report': {'files': [], 'findings': [], 'gaps': [], 'regression_requirements': []}}
    class Gateway:
        def application_search(self, files, question, **scope):
            import hashlib
            return {'results': [{'source_identity': 'submitted_application', 'application_commit': scope['application_commit'],
                'application_id': scope['application_id'], 'path': files[0]['path'], 'line_start': 1, 'line_end': 1,
                'sha256': hashlib.sha256(files[0]['content'].encode()).hexdigest(), 'content': files[0]['content']}]}
    base = {'application_id': 'ocr', 'head_commit': 'a'*40, 'base_commit': None, 'inventory': [],
            'files': [{'path': 'consumer.py', 'content': 'old'}], 'gaps': [], 'coverage': {'status': 'REGISTERED_SCOPE_COMPLETE'}}
    head = {**base, 'head_commit': 'b'*40, 'base_commit': 'a'*40, 'files': [{'path': 'consumer.py', 'content': 'new'}]}
    result = run_registered_review(Gateway(), head, base_snapshot=base, analyze=analyze,
        context={'application_name': 'demo'}, source_version='v2.9.1', target_version='v3.0.0')
    assert len(calls) == 2
    assert result['commit_comparison']['status'] == 'COMPARED'
    assert result['application_evidence']['results'][0]['application_commit'] == 'b'*40
    assert result['runtime_verified'] is False


def test_wrong_commit_application_evidence_is_rejected():
    import pytest
    from services.registered_review import validate_application_evidence
    with pytest.raises(ValueError):
        validate_application_evidence({'results': [{'path': 'consumer.py', 'application_commit': 'b'*40}]},
            {'application_id': 'ocr', 'head_commit': 'a'*40, 'files': [{'path': 'consumer.py', 'content': 'text'}]})
