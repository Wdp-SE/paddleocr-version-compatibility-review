"""Run the fixed original application against the pinned OCR 3.0 profiles.

Run using .local-demo-temp/ocr-v3/Scripts/python.exe. No submitted code, arbitrary
entry point, SDK upgrade, dependency download or automatic model download.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT/'examples/paddleocr_document_app'
sys.path.insert(0, str(APP_ROOT))


def stage(name, **data):
    print(json.dumps({'stage': name, **data}, ensure_ascii=False, default=str), flush=True)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, *, sdk_version='3.0.0'):
    import faulthandler
    faulthandler.enable()
    from run_upgrade_demo import source_identity, normalization_contract
    identity = source_identity()
    versions = {name: version(name) for name in ['paddleocr', 'paddlex', 'paddlepaddle', 'numpy']}
    if (sdk_version not in ('3.0.0', '3.0.3') or versions['paddlepaddle'] != '3.0.0'
            or any(versions[name] != sdk_version for name in ('paddleocr', 'paddlex'))):
        raise ValueError('requires the selected pinned PaddleOCR/PaddleX and PaddlePaddle 3.0.0')
    output = Path(output)
    if output.exists():
        raise FileExistsError('acceptance reports are immutable')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'.local-demo-temp/business-quality-20261114/ocr-fixtures/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    model_root = Path.home()/'.paddlex/official_models'
    model_names = ['PP-OCRv5_server_det', 'PP-OCRv5_server_rec']
    model_identity = {}
    for name in model_names:
        path = model_root/name
        files = [path/suffix for suffix in ['inference.json', 'inference.pdiparams', 'inference.yml']]
        if any(not p.is_file() for p in files):
            raise FileNotFoundError('all fixed local model files required')
        model_identity[name] = {p.name: _sha(p) for p in files}
    profile = dict(device='cpu', enable_mkldnn=False, cpu_threads=2,
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        text_rec_score_thresh=0.0,
        text_det_limit_type='max', text_det_limit_side_len=960,
        text_detection_model_name=model_names[0], text_detection_model_dir=str(model_root/model_names[0]),
        text_recognition_model_name=model_names[1], text_recognition_model_dir=str(model_root/model_names[1]))
    report = dict(schema_version=1, execution_kind='real_ocr', versions=versions,
        source_identity_at_start=identity, profile=profile, models_sha256=model_identity,
        fixture_manifest_sha256=_sha(manifest_path), input_sha256=[p['sha256'] for p in manifest['pages']],
        runtime_status='FAILED', patch_experiment=sdk_version == '3.0.3',
        scope='Original synthetic pages and explicit target application entry only; not recognition accuracy, enterprise data or release approval.')
    started = time.perf_counter()
    try:
        from PIL import Image
        import numpy as np
        from paddleocr import PaddleOCR
        from application import pipeline
        from application.consumer import to_json
        from result_adapter import normalize_v3
        images = []
        for row in manifest['pages']:
            path = Path(row['path'])
            if _sha(path) != row['sha256']:
                raise ValueError('fixture identity mismatch')
            images.append(np.ascontiguousarray(np.asarray(Image.open(path).convert('RGB'))[:,:,::-1]))
        stage('engine_init_enter', versions=versions, profile=profile)
        engine = PaddleOCR(lang='ch', **profile)
        stage('engine_init_exit', seconds=round(time.perf_counter()-started, 3))

        class RecordingEngine:
            def __init__(self):
                self.raw = None

            def ocr(self, image):
                stage('original_app_ocr_enter', page_count=len(image) if isinstance(image, list) else 1)
                self.raw = engine.ocr(image)
                stage('original_app_ocr_exit', raw_type=type(self.raw).__module__+'.'+type(self.raw).__qualname__,
                    page_types=[type(p).__module__+'.'+type(p).__qualname__ for p in self.raw])
                return self.raw

        recorder = RecordingEngine()

        def factory(**parameters):
            if parameters != {'lang': 'ch'}:
                raise ValueError('unexpected original application factory parameters')
            return recorder

        baseline = dict(entrypoint='application.pipeline.process_configured', runtime_status='FAILED')
        try:
            baseline['output'] = pipeline.process_configured(images[0], engine_factory=factory)
            baseline['runtime_status'] = 'PASSED'
        except Exception as exc:
            baseline.update(error_type=type(exc).__name__, error=str(exc))
        report['legacy_entry'] = baseline
        stage('legacy_entry_complete', **baseline)
        provenance = {}
        target_output = pipeline.process_target_configured(images, engine_factory=factory, provenance=provenance)
        raw = recorder.raw
        rows = normalize_v3(raw)
        contract = normalization_contract(raw, rows, target_output, major=3)
        probe_output = to_json(rows, page_count=len(raw))
        expected = all(all(fragment in target_output['pages'][index]['text'] for fragment in row['expected_fragments'])
                       if row['expected_fragments'] else target_output['pages'][index]['text'] == ''
                       for index, row in enumerate(manifest['pages']))
        raw_shape = [{'page_type':type(p).__module__+'.'+type(p).__qualname__,
                      'rec_texts_type':type(p['rec_texts']).__module__+'.'+type(p['rec_texts']).__qualname__,
                      'rec_scores_type':type(p['rec_scores']).__module__+'.'+type(p['rec_scores']).__qualname__,
                      'rec_texts':list(p['rec_texts']), 'rec_scores':[float(s) for s in p['rec_scores']]}
                     for p in raw]
        report.update(target_entry={'entrypoint':'application.pipeline.process_target_configured',
            'output':target_output, 'configuration':provenance,
            'configuration_notice':'No legacy drop_score is forwarded. The 2.x loader default record is not a 3.0 filter claim; actual 3.0 text_rec_score_thresh is explicit in profile.'}, raw_result_shapes=raw_shape,
            normalized_results=rows, normalization_contract_passed=contract,
            consumer_contract_passed=contract and target_output==probe_output,
            sample_regression_passed=expected,
            legacy_entry_failed_as_expected=baseline['runtime_status']=='FAILED' and baseline.get('error_type')=='ValueError')
        if contract and expected and report['legacy_entry_failed_as_expected']:
            report['runtime_status'] = 'PASSED'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['source_identity_at_end'] = source_identity()
        report['source_changed_during_run'] = report['source_identity_at_end'] != identity
        if report['source_changed_during_run']:
            report['runtime_status'] = 'INVALIDATED'
        report['seconds'] = round(time.perf_counter()-started, 3)
        report['acceptance_runner_sha256'] = _sha(Path(__file__))
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        stage('report_written', runtime_status=report['runtime_status'], path=str(output))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk-version', choices=['3.0.0', '3.0.3'], default='3.0.0')
    args = parser.parse_args()
    result = run(args.output, sdk_version=args.sdk_version)
    raise SystemExit(0 if result['runtime_status'] == 'PASSED' else 1)
