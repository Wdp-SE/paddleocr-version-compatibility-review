"""Counterexamples must remain visible even if the adapter probe passes."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/paddleocr_document_app'))
import run_upgrade_demo as runner


def load_acceptance():
    path = ROOT / 'evaluation/business_quality_v1/runtime_acceptance.py'
    assert path.exists(), 'controlled runtime acceptance runner is missing'
    spec = importlib.util.spec_from_file_location('business_runtime_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_runtime_always_writes_failure_record(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise AssertionError('controlled application entry point failed')
    monkeypatch.setattr(runner, '_run', failed, raising=False)
    output = tmp_path / 'failed.json'
    with pytest.raises(AssertionError):
        runner.run(tmp_path, output)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['runtime_status'] == 'FAILED'
    assert report['actual_entrypoint_contract_passed'] is False
    assert report['error_type'] == 'AssertionError'
    assert report['application_files_sha256']['application/consumer.py']


def test_run_is_invalidated_when_source_changes_during_inference(tmp_path, monkeypatch):
    identities = iter([{'application/consumer.py': 'before'}, {'application/consumer.py': 'after'}])
    monkeypatch.setattr(runner, 'application_identity', lambda: next(identities))
    monkeypatch.setattr(runner, '_run', lambda *args, **kwargs: {
        'runtime_status': 'PASSED', 'actual_entrypoint_contract_passed': True})
    report = runner.run(tmp_path, tmp_path / 'report.json')
    assert report['runtime_status'] == 'INVALIDATED'
    assert report['source_changed_during_run'] is True
    assert report['source_identity_at_start']['application_files_sha256']['application/consumer.py'] == 'before'


@pytest.mark.parametrize('major,raw', [
    (2, [[[None, ('first', 0.98)]], [], [[None, ('last', 0.71)]]]),
    (3, [{'rec_texts': ['first'], 'rec_scores': [0.98]},
         {'rec_texts': [], 'rec_scores': []}, {'rec_texts': ['last'], 'rec_scores': [0.71]}]),
])
def test_real_runner_verifies_confidence_correspondence_not_only_text(major, raw):
    checker = getattr(runner, 'normalization_contract', None)
    assert callable(checker), 'real runner does not check raw-to-normalized confidence mapping'
    rows = [{'page_index': 0, 'text': 'first', 'confidence': 0.98},
            {'page_index': 2, 'text': 'last', 'confidence': 0.71}]
    from application.consumer import to_json
    output = to_json(rows, page_count=3)
    assert checker(raw, rows, output, major=major) is True
    rows[0]['confidence'] = 0.71
    assert checker(raw, rows, output, major=major) is False


def test_acceptance_records_expected_failures_without_hiding_them(tmp_path):
    report = load_acceptance().run_fixture_acceptance(tmp_path / 'report.json')
    assert report['execution_kind'] == 'fixed_score_fixture'
    assert report['expectations_met'] is True
    rows = {row['case_id']: row for row in report['cases']}
    assert rows['json_old']['output']['pages'][0]['text'] == 'boundary\nbetween\nhigh'
    assert rows['json_new']['output']['pages'][0]['text'] == 'high'
    assert rows['file_changed_not_adopted']['runtime_status'] == 'FAILED'
    assert rows['file_changed_not_adopted']['configuration']['value_sources']['drop_score'] == 'explicit_override'
    assert rows['v3_default_entry_fails']['adapter_probe_passed'] is True
    assert rows['v3_default_entry_fails']['runtime_status'] == 'FAILED'
    assert rows['v3_candidate_absorbs_shape']['runtime_status'] == 'PASSED'
    assert rows['unchanged_call_runtime_bad_type']['runtime_status'] == 'FAILED'
    assert len({row['consumer_sha256'] for row in rows.values()}) == 1
    for row in report['cases']:
        assert row['input_sha256'] and row['application_files_sha256']


@pytest.mark.parametrize('mutation', ['confidence_mismatch', 'bool_score', 'wrong_page_order', 'wrong_text_type', 'dropped_empty_page'])
def test_contract_probe_detects_mapping_type_and_page_failures(mutation):
    module = load_acceptance()
    rows = [{'page_index': 0, 'text': 'first', 'confidence': 0.98},
            {'page_index': 2, 'text': 'last', 'confidence': 0.71}]
    output = {'schema_version': 1, 'pages': [
        {'page_index': 0, 'text': 'first'}, {'page_index': 1, 'text': ''},
        {'page_index': 2, 'text': 'last'}]}
    if mutation == 'confidence_mismatch': rows[0]['confidence'] = 0.71
    if mutation == 'bool_score': rows[0]['confidence'] = True
    if mutation == 'wrong_page_order': output['pages'].reverse()
    if mutation == 'wrong_text_type': output['pages'][0]['text'] = 123
    if mutation == 'dropped_empty_page': output['pages'].pop(1)
    assert module.check_contract(rows, output, [
        {'page_index': 0, 'text': 'first', 'confidence': 0.98},
        {'page_index': 2, 'text': 'last', 'confidence': 0.71}], page_count=3) is False


def test_native_process_failure_still_gets_an_identity_bound_record(tmp_path, monkeypatch):
    module = load_acceptance()
    supervisor = getattr(module, 'run_real_case', None)
    assert callable(supervisor), 'real OCR process needs an external failure-record supervisor'
    import subprocess
    original_run = module.subprocess.run
    def native_exit(command, *args, **kwargs):
        if any(str(part).endswith('run_upgrade_demo.py') for part in command):
            return subprocess.CompletedProcess(command, 1)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(module.subprocess, 'run', native_exit)
    output = tmp_path / 'native-failed.json'
    report = supervisor('v3', tmp_path, output)
    assert report['runtime_status'] == 'NOT_AVAILABLE'
    assert report['failure_reason'] == 'child_exited_without_report'
    assert report['child_exit_code'] == 1
    assert report['application_files_sha256']['application/consumer.py']
    assert json.loads(output.read_text(encoding='utf-8')) == report
