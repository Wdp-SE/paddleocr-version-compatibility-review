import json
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.error import HTTPError


def load_runner():
    path = Path(__file__).resolve().parents[2] / 'evaluation/business_quality_v1/query_acceptance.py'
    spec = spec_from_file_location('business_query_acceptance', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_http_failures_keep_all_rows_and_fail_cli(tmp_path, monkeypatch):
    runner = load_runner()

    @contextmanager
    def open_request(request, timeout):
        from io import BytesIO
        if isinstance(request, str):
            yield BytesIO(b'{"workspace_id":"paddleocr"}')
        else:
            raise HTTPError(request.full_url, 503, 'fake unavailable', {}, None)

    monkeypatch.setattr(runner.urllib.request, 'urlopen', open_request)
    report = tmp_path / 'acceptance.json'
    assert runner.main(['--output', str(report)]) != 0
    result = json.loads(report.read_text(encoding='utf-8'))
    assert len(result['rows']) >= 6
    assert all(row['status'] == 'REQUEST_FAILED' for row in result['rows'])


def test_success_status_cannot_hide_missing_or_cross_version_evidence():
    runner = load_runner()
    row = {'version': 'v3.0.0', 'repository': 'PaddlePaddle/PaddleOCR',
           'source_id': 'official', 'chunk_id': 'test', 'line_start': 1, 'line_end': 2}
    assert runner.check_response({'results': [row]}, 'v3.0.0', mode='search') == []
    assert runner.check_response({'results': []}, 'v3.0.0', mode='search')
    assert runner.check_response({'results': [{**row, 'version': 'v2.9.1'}]}, 'v3.0.0', mode='search')
    assert runner.check_response({'status': 'NEEDS_CLARIFICATION', 'results': []}, 'v2.9.1', mode='conflict') == []
    assert runner.check_response({'status': 'NEEDS_CLARIFICATION', 'results': [row]}, 'v2.9.1', mode='conflict')


def test_joint_gate_requires_current_citations_and_complete_application_provenance():
    runner = load_runner()
    commit = 'a' * 40
    official = {'version': 'v3.0.0', 'repository': 'PaddlePaddle/PaddleOCR',
                'source_id': 'release', 'chunk_id': 'off', 'line_start': 1, 'line_end': 2}
    app = {'source_identity': 'submitted_application', 'chunk_id': 'app',
           'path': 'contract.json', 'line_start': 1, 'line_end': 1, 'sha256': 'c' * 64,
           'content': '{"text": "example"}', 'application_id': 'ocr', 'application_commit': commit}
    snapshot = {'application_id': 'ocr', 'head_commit': commit,
                'files': [{'path': app['path'], 'content': app['content']}],
                'inventory': [{'path': app['path'], 'sha256': app['sha256']}]}
    value = {'status': 'OK', 'application_id': 'ocr', 'application_commit': commit,
             'evidence': [official, app], 'sources': [official, app],
             'claims': [{'text': 'SDK field maps to the app field.', 'evidence_ids': ['off', 'app']}]}

    def check(payload):
        return runner.check_response(payload, 'v3.0.0', mode='joint', snapshot=snapshot)

    assert check(value) == []
    assert check({**value, 'claims': [{'text': 'invalid', 'evidence_ids': ['off', 'invented']}]})
    assert check({**value, 'sources': [official],
                  'claims': [{'text': 'component only', 'evidence_ids': ['off']}]})
    assert check({**value, 'evidence': [official, {**app, 'sha256': None}]})
    other_app = {**app, 'application_id': 'other'}
    assert check({**value, 'application_id': 'other', 'evidence': [official, other_app],
                  'sources': [official, other_app]}) == ['INVALID_APPLICATION_PROVENANCE']
    altered = {**app, 'content': 'altered'}
    assert check({**value, 'evidence': [official, altered],
                  'sources': [official, altered]}) == ['INVALID_APPLICATION_PROVENANCE']
