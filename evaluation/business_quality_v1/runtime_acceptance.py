"""Bounded developer acceptance for this repository's original OCR application.

This runner has no application/source/plugin argument. Fixed-score fixtures are
independent of real OCR and are never reported as model compatibility evidence.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / 'examples/paddleocr_document_app'
sys.path.insert(0, str(APP_ROOT))
from application import pipeline
from application.consumer import to_json
from result_adapter import normalize_v2, normalize_v3
from run_upgrade_demo import application_identity


V2_RAW = [
    [[None, ['below', 0.49]], [None, ['boundary', 0.5]],
     [None, ['between', 0.7]], [None, ['high', 0.9]]],
    [], [[None, ['last', 0.95]]],
]
V3_RAW = [
    {'rec_texts': ['first'], 'rec_scores': [0.98]},
    {'rec_texts': [], 'rec_scores': []},
    {'rec_texts': ['last'], 'rec_scores': [0.71]},
]
V3_EXPECTED = [
    {'page_index': 0, 'text': 'first', 'confidence': 0.98},
    {'page_index': 2, 'text': 'last', 'confidence': 0.71},
]


def check_contract(rows, output, expected_rows, *, page_count):
    """Check score-to-text correspondence as well as the consumer's JSON schema."""
    if not isinstance(rows, list) or rows != expected_rows:
        return False
    if any(not isinstance(row, dict) or set(row) != {'page_index', 'text', 'confidence'}
           or type(row['page_index']) is not int or not 0 <= row['page_index'] < page_count
           or not isinstance(row['text'], str) or type(row['confidence']) not in (int, float)
           or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1
           for row in rows):
        return False
    if not isinstance(output, dict) or set(output) != {'schema_version', 'pages'}:
        return False
    if type(output['schema_version']) is not int or output['schema_version'] != 1:
        return False
    pages = output['pages']
    if not isinstance(pages, list) or len(pages) != page_count:
        return False
    for index, page in enumerate(pages):
        if (not isinstance(page, dict) or set(page) != {'page_index', 'text'}
                or type(page['page_index']) is not int or page['page_index'] != index
                or not isinstance(page['text'], str)):
            return False
        if page['text'] != '\n'.join(row['text'] for row in expected_rows if row['page_index'] == index):
            return False
    return True


class FixedScoreEngine:
    """Only deterministic vendor-shaped fixture data, never submitted source."""
    def __init__(self, *, lang, drop_score=0.5, raw=None, shape='v2'):
        if lang != 'ch':
            raise ValueError('fixed fixture is Chinese OCR only')
        self.threshold = drop_score
        self.raw = V2_RAW if raw is None else raw
        self.shape = shape

    def ocr(self, image):
        if self.shape == 'v3':
            return self.raw
        return [[line for line in page if line[1][1] >= self.threshold] for page in self.raw]


@contextmanager
def candidate_adapter(enabled):
    """A fixed test candidate; does not edit source or claim a released app."""
    previous = pipeline.normalize_legacy
    if enabled:
        pipeline.normalize_legacy = normalize_v3
    try:
        yield
    finally:
        pipeline.normalize_legacy = previous


def _identity():
    identity = {'application_files_sha256': application_identity(),
                'adapter_sha256': hashlib.sha256((APP_ROOT / 'result_adapter.py').read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        identity['git_base_commit'] = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        identity['application_worktree_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'examples/paddleocr_document_app'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        identity.update(git_base_commit=None, application_worktree_dirty=True)
    return identity


def _case(case_id, *, identity, config_path=None, override=None, shape='v2',
          candidate=False, expected_rows, expected_status='PASSED', bad_type=False):
    raw = V3_RAW if shape == 'v3' else V2_RAW
    if bad_type:
        raw = [[[None, [123, 0.9]]], [], []]
    row = {'case_id': case_id, **identity, 'expected_runtime_status': expected_status,
           'consumer_sha256': identity['application_files_sha256']['application/consumer.py'],
           'input_sha256': hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
           'dependency_contract': '3.0.0' if shape == 'v3' else '2.9.1',
           'execution_kind': 'fixed_score_fixture', 'configuration': {},
           'candidate_adapter_injected': candidate, 'runtime_status': 'FAILED'}
    # Adapter probe runs independently and can succeed when the entry point fails.
    try:
        probe = normalize_v3(raw) if shape == 'v3' else normalize_v2(raw)
        row['adapter_probe_passed'] = check_contract(probe, to_json(probe, page_count=3),
                                                     expected_rows, page_count=3)
    except (TypeError, ValueError, KeyError, IndexError):
        row['adapter_probe_passed'] = False
    def factory(**parameters):
        return FixedScoreEngine(**parameters, raw=raw, shape=shape)
    try:
        with candidate_adapter(candidate):
            output = pipeline.process_configured('fixed-original-three-page-fixture', config_path=config_path,
                                      drop_score=override, engine_factory=factory,
                                      provenance=row['configuration'])
            # Separate normalization inspection validates score mapping omitted by JSON.
            effective_raw = FixedScoreEngine(**row['configuration']['effective_values'], raw=raw, shape=shape).ocr(None)
            normalized = normalize_v3(effective_raw) if candidate else normalize_v2(effective_raw)
        row.update(output=output, normalized_results=normalized)
        row['contract_passed'] = check_contract(normalized, output, expected_rows, page_count=3)
        if row['contract_passed']:
            row['runtime_status'] = 'PASSED'
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc), contract_passed=False)
    row['expectation_met'] = row['runtime_status'] == expected_status
    return row


def run_fixture_acceptance(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_dir = output.parent / 'runtime-configs'
    config_dir.mkdir(exist_ok=True)
    old = config_dir / 'old.json'
    new = config_dir / 'new.json'
    old.write_text('{"schema_version":1,"drop_score":0.5}\n', encoding='utf-8')
    new.write_text('{"schema_version":1,"drop_score":0.8}\n', encoding='utf-8')
    identity = _identity()
    expected_default = [
        {'page_index': 0, 'text': 'boundary', 'confidence': 0.5},
        {'page_index': 0, 'text': 'between', 'confidence': 0.7},
        {'page_index': 0, 'text': 'high', 'confidence': 0.9},
        {'page_index': 2, 'text': 'last', 'confidence': 0.95},
    ]
    expected_new = [
        {'page_index': 0, 'text': 'high', 'confidence': 0.9},
        {'page_index': 2, 'text': 'last', 'confidence': 0.95},
    ]
    cases = [
        _case('default', identity=identity, expected_rows=expected_default),
        _case('json_old', identity=identity, config_path=old, expected_rows=expected_default),
        _case('json_new', identity=identity, config_path=new, expected_rows=expected_new),
        _case('explicit_override', identity=identity, config_path=new, override=0.5, expected_rows=expected_default),
        _case('file_changed_not_adopted', identity=identity, config_path=new, override=0.5,
              expected_rows=expected_new, expected_status='FAILED'),
        _case('v3_default_entry_fails', identity=identity, shape='v3',
              expected_rows=V3_EXPECTED, expected_status='FAILED'),
        _case('v3_candidate_absorbs_shape', identity=identity, shape='v3', candidate=True,
              expected_rows=V3_EXPECTED),
        _case('unchanged_call_runtime_bad_type', identity=identity, bad_type=True,
              expected_rows=expected_new, expected_status='FAILED'),
    ]
    report = {'schema_version': 1, 'execution_kind': 'fixed_score_fixture', **identity,
              'cases': cases, 'expectations_met': all(row['expectation_met'] for row in cases),
              'observed_failed_cases': [row['case_id'] for row in cases if row['runtime_status'] == 'FAILED'],
              'scope': 'Original application only; fixed-score vendor shapes, not OCR inference, recognition accuracy, static review results or release approval. Candidate adapter is injected only for this fixture case.'}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def run_real_case(runtime, fixture_dir, output, *, configuration='default', timeout=900):
    """Supervise only the fixed original app, including native exit/timeout failures."""
    if runtime not in ('v2', 'v3') or configuration not in ('default', 'old', 'new', 'override'):
        raise ValueError('unsupported controlled OCR case')
    if runtime == 'v3' and configuration != 'default':
        raise ValueError('JSON drop_score cases require OCR 2.9.1')
    if type(timeout) not in (int, float) or not 1 <= timeout <= 1800:
        raise ValueError('controlled OCR timeout must be between 1 and 1800 seconds')
    output = Path(output)
    fixture_dir = Path(fixture_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity()
    child_output = output.with_name(output.stem + '.child-' + uuid.uuid4().hex + '.json')
    log_path = output.with_suffix('.log')
    interpreter = ROOT / '.local-demo-temp' / ('ocr-' + runtime) / 'Scripts/python.exe'
    command = [str(interpreter), str(APP_ROOT / 'run_upgrade_demo.py'),
               '--fixtures', str(fixture_dir.resolve()), '--output', str(child_output.resolve())]
    if configuration != 'default':
        config_dir = output.parent / 'runtime-configs'
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / (('old' if configuration == 'old' else 'new') + '.json')
        config_path.write_text('{"schema_version":1,"drop_score":' + ('0.5' if configuration == 'old' else '0.8') + '}\n', encoding='utf-8')
        command.extend(['--config', str(config_path.resolve())])
        if configuration == 'override':
            command.extend(['--drop-score', '0.5'])
    started = time.perf_counter()
    exit_code = None
    failure = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
        exit_code = child.returncode
    except subprocess.TimeoutExpired:
        failure = 'child_timeout'
    except OSError as exc:
        failure = 'child_start_failed:' + type(exc).__name__
    if child_output.is_file():
        try:
            report = json.loads(child_output.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            failure = 'child_report_invalid_json'
            report = None
    else:
        report = None
        failure = failure or 'child_exited_without_report'
    if report is None:
        manifest = fixture_dir / 'manifest.json'
        report = {'runtime_status': 'NOT_AVAILABLE', 'execution_kind': 'real_ocr', **identity,
                  'failure_reason': failure, 'actual_entrypoint_contract_passed': False,
                  'expected_dependency_version': '2.9.1' if runtime == 'v2' else '3.0.0',
                  'observed_dependency_version': None,
                  'input_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
                  'configuration_case': configuration,
                  'scope': 'Native exit or environment failure does not establish an application compatibility result.'}
    report.update(child_exit_code=exit_code, child_report_path=str(child_output.resolve()),
                  raw_log_path=str(log_path.resolve()), supervisor_seconds=round(time.perf_counter() - started, 2),
                  supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--real-runtime', choices=['v2', 'v3'])
    parser.add_argument('--fixtures', type=Path, default=ROOT / '.local-demo-temp/business-quality-20261114/ocr-fixtures')
    parser.add_argument('--configuration', choices=['default', 'old', 'new', 'override'], default='default')
    args = parser.parse_args()
    if args.real_runtime:
        result = run_real_case(args.real_runtime, args.fixtures, args.output, configuration=args.configuration)
        print(json.dumps({'runtime_status': result['runtime_status'], 'child_exit_code': result['child_exit_code']}, ensure_ascii=False))
        raise SystemExit(0 if result['runtime_status'] == 'PASSED' else 1)
    else:
        result = run_fixture_acceptance(args.output)
        print(json.dumps({'expectations_met': result['expectations_met'],
                          'observed_failed_cases': result['observed_failed_cases']}, ensure_ascii=False))
        raise SystemExit(0 if result['expectations_met'] else 1)
