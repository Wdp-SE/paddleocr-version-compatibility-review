"""Stage-flushed developer probe for fixed PaddleOCR patch comparison."""
import argparse
import faulthandler
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
START = time.perf_counter()


def stage(name, **data):
    print(json.dumps({'stage': name, 'elapsed': round(time.perf_counter()-START, 3), **data},
                     ensure_ascii=False, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=['pipeline', 'det', 'rec'], default='pipeline')
    parser.add_argument('--sdk-version', choices=['3.0.0', '3.0.3'], default='3.0.0')
    parser.add_argument('--mkldnn', choices=['off', 'default'], default='off')
    parser.add_argument('--max-side', type=int, help='Documented max-side detector resize limit')
    parser.add_argument('--application-import-order', action='store_true')
    parser.add_argument('--original-runner', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    versions = {name: version(name) for name in ['paddleocr', 'paddlex', 'paddlepaddle', 'numpy']}
    if versions['paddleocr'] != args.sdk_version or versions['paddlex'] != args.sdk_version:
        raise RuntimeError('probe requires the selected pinned PaddleOCR and PaddleX versions')
    if args.original_runner and (args.sdk_version != '3.0.0' or args.mkldnn != 'off'):
        parser.error('--original-runner is reserved for the original 3.0.0 diagnostic')
    stage('start', versions=versions, executable=sys.executable, parameters=vars(args),
          environment={k:v for k,v in os.environ.items() if k.startswith(('FLAGS_', 'OMP_', 'MKL_', 'KMP_'))})
    if args.application_import_order:
        sys.path.insert(0, str(ROOT/'examples/paddleocr_document_app'))
        import run_upgrade_demo
        import paddleocr
        stage('application_and_paddleocr_imported')
    import paddle.inference
    stage('paddle_imported')
    original_create = paddle.inference.create_predictor
    count = 0

    def create(config):
        nonlocal count
        count += 1
        stage('create_predictor_enter', index=count, mkldnn=config.mkldnn_enabled())
        result = original_create(config)
        stage('create_predictor_exit', index=count)
        return result

    paddle.inference.create_predictor = create
    import paddleocr
    from paddlex.inference.models.common.static_infer import PaddleInferChainLegacy
    original_infer = PaddleInferChainLegacy.__call__

    def infer(self, inputs):
        stage('native_infer_enter', shapes=[list(v.shape) for v in inputs])
        result = original_infer(self, inputs)
        stage('native_infer_exit', shapes=[list(v.shape) for v in result])
        return result

    PaddleInferChainLegacy.__call__ = infer
    stage('paddleocr_imported', file=paddleocr.__file__)
    if args.original_runner:
        sys.path.insert(0, str(ROOT/'examples/paddleocr_document_app'))
        import run_upgrade_demo
        stage('original_runner_enter')
        result = run_upgrade_demo.run(ROOT/'.local-demo-temp/business-quality-20261114/ocr-fixtures',
            ROOT/'.local-demo-temp/resolve-20261114/v3-current-original-runner-result.json')
        stage('original_runner_exit', runtime_status=result['runtime_status'])
        faulthandler.cancel_dump_traceback_later()
        raise SystemExit(0 if result['runtime_status'] == 'PASSED' else 1)
    model_root = Path.home()/'.paddlex/official_models'
    names = {key:f'PP-OCRv5_server_{key}' for key in ['det', 'rec']}
    paths = {key:model_root/name for key,name in names.items()}
    hashes = {name:{str(p.relative_to(paths[key])): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths[key].glob('inference.*') if p.is_file()}
              for key,name in names.items()}
    stage('model_identity', files=hashes)
    common = dict(device='cpu', cpu_threads=2)
    if args.mkldnn == 'off':
        common['enable_mkldnn'] = False
    resize = dict(text_det_limit_type='max', text_det_limit_side_len=args.max_side) if args.max_side else {}
    stage('model_init_enter')
    if args.component == 'pipeline':
        engine = paddleocr.PaddleOCR(**common, **resize, lang='ch',
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            text_detection_model_name=names['det'], text_detection_model_dir=str(paths['det']),
            text_recognition_model_name=names['rec'], text_recognition_model_dir=str(paths['rec']))
    else:
        cls = paddleocr.TextDetection if args.component == 'det' else paddleocr.TextRecognition
        engine = cls(**common, model_name=names[args.component], model_dir=str(paths[args.component]))
    stage('model_init_exit')
    if args.infer:
        from PIL import Image
        import numpy as np
        fixture = ROOT/'.local-demo-temp/business-quality-20261114/ocr-fixtures/page-1.png'
        image = np.ascontiguousarray(np.asarray(Image.open(fixture).convert('RGB'))[:,:,::-1])
        if args.component == 'rec':
            image = image[28:100, 25:900]
        stage('predict_enter', input_shape=image.shape, input_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
        results = list(engine.predict(image))
        stage('predict_exit', results=[{'keys':list(r), 'rec_texts':r.get('rec_texts'),
                                       'rec_scores':r.get('rec_scores'), 'rec_text':r.get('rec_text'),
                                       'rec_score':r.get('rec_score')} for r in results])
    faulthandler.cancel_dump_traceback_later()
    stage('complete')


if __name__ == '__main__':
    main()
