"""Run only this original application, in separately pinned OCR environments."""
from pathlib import Path
import argparse,hashlib,json,time,os
import math
from importlib.metadata import version
from result_adapter import normalize_v2,normalize_v3


def check_application_entry(engine, image, *, expected_fragments=None, expected_page_count=1,
                            engine_factory=None, config_path=None, drop_score=None):
    """Run only the author's checked-out app; never accept an arbitrary source path."""
    from application import ocr_client, pipeline
    previous = ocr_client.create_engine
    entrypoint = 'application.pipeline.process' if engine_factory is None else 'application.pipeline.process_configured'
    configuration = {}
    if engine_factory is None:
        ocr_client.create_engine = lambda: engine
    try:
        if engine_factory is None:
            output = pipeline.process(image)
        else:
            output = pipeline.process_configured(image, config_path=config_path, drop_score=drop_score,
                                      engine_factory=engine_factory, provenance=configuration)
        pages = output.get('pages') if isinstance(output,dict) else None
        valid=(isinstance(output,dict) and set(output)=={'schema_version','pages'} and type(output['schema_version']) is int
            and output['schema_version']==1 and isinstance(pages,list) and len(pages)==expected_page_count
            and all(isinstance(p,dict) and set(p)=={'page_index','text'} and type(p['page_index']) is int
                    and p['page_index']==i and isinstance(p['text'],str) for i,p in enumerate(pages)))
        text='\n'.join(p['text'] for p in pages) if valid else ''
        text_valid=(expected_fragments is None or all(fragment in text for fragment in expected_fragments)
                    and (bool(expected_fragments) or text==''))
        return {'entrypoint': entrypoint, 'passed': valid and text_valid, 'output': output,
                'checks':{'output_contract':valid,'expected_text_or_empty_page':text_valid},
                'configuration': configuration,
                'engine_scope': 'original factory with controlled constructor' if engine_factory else 'explicit runner engine injection'}
    except Exception as exc:
        return {'entrypoint': entrypoint, 'passed': False, 'error': type(exc).__name__,
                'configuration': configuration,
                'engine_scope': 'original factory with controlled constructor' if engine_factory else 'explicit runner engine injection'}
    finally:
        ocr_client.create_engine = previous


def fixture_documents(folder):
    from PIL import Image,ImageDraw,ImageFont
    folder.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype(os.environ.get('OCR_DEMO_FONT','C:/Windows/Fonts/msyh.ttc'),42)
    pages=[['文档处理应用测试','合同编号 DEMO-2026-001','金额 1234 元'],
           ['文档处理应用第二页','交付日期 2026-10-04','数量 5 件'],[]]
    rows=[]
    for i,lines in enumerate(pages):
        image=Image.new('RGB',(1000,500),'white')
        draw=ImageDraw.Draw(image)
        for j,line in enumerate(lines):draw.text((40,40+j*100),line,font=font,fill='black')
        path=folder/f'page-{i+1}.png'
        image.save(path)
        rows.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'expected_fragments':['DEMO','1234'] if i==0 else ['2026','数量'] if i==1 else []})
    (folder/'manifest.json').write_text(json.dumps({'origin':'original synthetic Chinese documents, not customer scans','pages':rows},ensure_ascii=False,indent=2),encoding='utf-8')


def runner_parameters(runtime_version,drop_score):
    """Explicit controlled-run override, never evidence of configuration loading."""
    if drop_score is None:return {}
    if (runtime_version!='2.9.1' or type(drop_score) not in (int,float)
            or not 0<=drop_score<=1 or not math.isfinite(drop_score)):
        raise ValueError('drop_score override requires OCR 2.9.1 and a finite value between 0 and 1')
    return {'drop_score':float(drop_score)}


def normalization_contract(raw, rows, output, *, major):
    """Verify actual text/score pairs and page positions against the vendor result."""
    try:
        expected=[]
        for page_index,page in enumerate(raw):
            if major==2:
                if page is None:continue
                pairs=[line[1] for line in page]
            elif major==3:
                if len(page['rec_texts'])!=len(page['rec_scores']):return False
                pairs=zip(page['rec_texts'],page['rec_scores'])
            else:return False
            for text,score in pairs:
                expected.append({'page_index':page_index,'text':text,'confidence':float(score)})
        if rows!=expected:return False
        if any(set(row)!={'page_index','text','confidence'} or type(row['page_index']) is not int
               or not isinstance(row['text'],str) or type(row['confidence']) not in (int,float)
               or not math.isfinite(row['confidence']) or not 0<=row['confidence']<=1 for row in rows):return False
        if (not isinstance(output,dict) or set(output)!={'schema_version','pages'}
                or type(output['schema_version']) is not int or output['schema_version']!=1
                or not isinstance(output['pages'],list) or len(output['pages'])!=len(raw)):return False
        return all(isinstance(page,dict) and set(page)=={'page_index','text'}
                   and type(page['page_index']) is int and page['page_index']==index
                   and isinstance(page['text'],str)
                   and page['text']=='\n'.join(row['text'] for row in expected if row['page_index']==index)
                   for index,page in enumerate(output['pages']))
    except (KeyError,TypeError,ValueError,IndexError):
        return False


def application_identity():
    root = Path(__file__).parent
    return {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.joinpath('application').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


def source_identity():
    return {'application_files_sha256':application_identity(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'adapter_sha256':hashlib.sha256(Path(__file__).with_name('result_adapter.py').read_bytes()).hexdigest()}


def run(folder,output,*,drop_score=None,config_path=None):
    """Persist failure as well as success; a successful probe cannot hide a failure."""
    initial_identity=source_identity()
    try:
        report=_run(folder,output,drop_score=drop_score,config_path=config_path)
        final_identity=source_identity()
        report['source_identity_at_start']=initial_identity
        report['source_identity_at_end']=final_identity
        report['source_changed_during_run']=initial_identity!=final_identity
        report['application_files_sha256']=initial_identity['application_files_sha256']
        report['application_sha256']=initial_identity['runner_sha256']
        report['adapter_sha256']=initial_identity['adapter_sha256']
        if report['source_changed_during_run']:
            report['runtime_status']='INVALIDATED'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report
    except Exception as exc:
        output.parent.mkdir(parents=True,exist_ok=True)
        report = {'runtime_status':'FAILED','execution_kind':'real_ocr',
                  'actual_entrypoint_contract_passed':False,'error_type':type(exc).__name__,
                  'error':str(exc),'application_files_sha256':initial_identity['application_files_sha256'],
                  'source_identity_at_start':initial_identity,
                  'configuration': {'requested_path':str(config_path) if config_path else None,
                                    'requested_drop_score':str(drop_score) if drop_score is not None else None}}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise


def _run(folder,output,*,drop_score=None,config_path=None):
    from paddleocr import PaddleOCR
    runtime_version=version('paddleocr')
    parameters=runner_parameters(runtime_version,drop_score)
    if config_path is not None and runtime_version != '2.9.1':
        raise ValueError('JSON drop_score configuration requires OCR 2.9.1')
    major=int(runtime_version.split('.')[0])
    if runtime_version not in ('2.9.1','3.0.0'):raise ValueError('unsupported demonstration dependency')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if major==2:
        model_root=Path(os.environ.get('OCR_V2_MODEL_ROOT',str(Path.home()/'.paddleocr/whl')))
        required=[model_root/'det/ch/ch_PP-OCRv4_det_infer/inference.pdiparams',
                  model_root/'rec/ch/ch_PP-OCRv4_rec_infer/inference.pdiparams',
                  model_root/'cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams']
        if any(not p.is_file() for p in required):raise FileNotFoundError('pre-downloaded OCR 2.9.1 models required; runner does not download')
        def controlled_factory(**factory_parameters):
            return PaddleOCR(**factory_parameters,use_gpu=False,enable_mkldnn=False,cpu_threads=2,show_log=False,
                        det_model_dir=str(model_root/'det/ch/ch_PP-OCRv4_det_infer'),
                        rec_model_dir=str(model_root/'rec/ch/ch_PP-OCRv4_rec_infer'),
                        cls_model_dir=str(model_root/'cls/ch_ppocr_mobile_v2.0_cls_infer'))
    else:
        model_root=Path(os.environ.get('OCR_V3_MODEL_ROOT',str(Path.home()/'.paddlex/official_models')))
        required=[model_root/'PP-OCRv5_server_det/inference.pdiparams',
                  model_root/'PP-OCRv5_server_rec/inference.pdiparams']
        if any(not p.is_file() for p in required):raise FileNotFoundError('pre-downloaded OCR 3.0.0 models required; runner does not download')
        def controlled_factory(**factory_parameters):
            return PaddleOCR(**factory_parameters,device='cpu',enable_mkldnn=False,cpu_threads=2,
                        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
                        text_detection_model_name='PP-OCRv5_server_det',text_detection_model_dir=str(model_root/'PP-OCRv5_server_det'),
                        text_recognition_model_name='PP-OCRv5_server_rec',text_recognition_model_dir=str(model_root/'PP-OCRv5_server_rec'))
    from application.ocr_client import create_configured_engine
    configuration={}
    engine=create_configured_engine(config_path,drop_score=drop_score,engine_factory=controlled_factory,provenance=configuration)
    def reuse_controlled_engine(**factory_parameters):
        if factory_parameters != configuration['factory_parameters']:
            raise ValueError('application configuration changed during runtime')
        return engine
    started=time.perf_counter()
    raw=[]; entry_checks=[]
    for page in manifest['pages']:
        path=Path(page['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=page['sha256']:raise ValueError('fixture changed')
        if major==2:
            image_input=str(path)
            raw.extend(engine.ocr(image_input,cls=False))
        else:
            # PaddleX 3.0's cv2 path reader fails on Windows non-ASCII paths.
            # Decode locally and supply its documented numpy image input instead.
            from PIL import Image
            import numpy as np
            image_input=np.asarray(Image.open(path).convert('RGB'))[:,:,::-1]
            raw.extend(engine.predict(image_input))
        entry_checks.append({'input_sha256':page['sha256'], **check_application_entry(engine,image_input,
                            expected_fragments=page['expected_fragments'],engine_factory=reuse_controlled_engine,
                            config_path=config_path,drop_score=drop_score)})
    normalized=normalize_v2(raw) if major==2 else normalize_v3(raw)
    from application.consumer import to_json
    downstream=to_json(normalized,page_count=len(raw))
    contract_valid=normalization_contract(raw,normalized,downstream,major=major)
    legacy_error=None
    legacy_success=None
    try:
        legacy=[]
        for page in raw:
            if page is None:continue
            for line in page:
                legacy.append({'text':line[1][0],'confidence':float(line[1][1])})
        legacy_success=bool(legacy)
    except Exception as exc:
        legacy_success=False
        legacy_error=type(exc).__name__
    checks=[]
    for i,page in enumerate(manifest['pages']):
        text=' '.join(row['text'] for row in normalized if row['page_index']==i)
        checks.append({'page_index':i,'text':text,'expected_fragments':page['expected_fragments'],
                       'passed':all(fragment in text for fragment in page['expected_fragments']) if page['expected_fragments'] else text==''})
    report={'paddleocr_version':runtime_version,'paddle_version':version('paddlepaddle'),
            'numpy_version':version('numpy'),'paddlex_version':version('paddlex') if major==3 else None,
            'origin':manifest['origin'],'input_sha256':[p['sha256'] for p in manifest['pages']],
            'application_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'adapter_sha256':hashlib.sha256(Path(__file__).with_name('result_adapter.py').read_bytes()).hexdigest(),
            'normalization_checks':checks,'legacy_consumer_success':legacy_success,'legacy_error':legacy_error,
            'normalized_results':normalized,'inference_seconds':round(time.perf_counter()-started,2),
            'sample_regression_passed':all(c['passed'] for c in checks),
            'scope':'These three synthetic pages verify this adapter and runtime only, not document recognition accuracy.'}
    report['runner_parameters']=parameters
    report['configuration']=configuration
    report['configuration_scope']='Original application factory loads explicit JSON/default/override; controlled constructor sets only local hardware and model paths.'
    report['input_method']='file_path' if major==2 else 'locally_decoded_bgr_array_for_windows_unicode_paths'
    report['consumer_sha256']=hashlib.sha256(Path(__file__).with_name('application').joinpath('consumer.py').read_bytes()).hexdigest()
    report['application_files_sha256']={str(p.relative_to(Path(__file__).parent)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).with_name('application').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    report['downstream_json']=downstream
    report['downstream_contract_passed']=contract_valid
    report['normalization_contract_passed']=contract_valid
    report['entrypoint_regression']={'entrypoint':'application.pipeline.process_configured','results':entry_checks,
        'passed':all(row['passed'] for row in entry_checks),
        'notice':'Adapter-only success does not override a failed application entry point. The runner uses explicit local model configuration.'}
    report['actual_entrypoint_contract_passed']=report['entrypoint_regression']['passed']
    report['execution_kind']='real_ocr'
    report['runtime_status']='PASSED' if report['actual_entrypoint_contract_passed'] and contract_valid and report['sample_regression_passed'] else 'FAILED'
    report['sample_regression_scope']='separate adapter probe; not the default application entry point'
    import subprocess
    try:
        report['application_commit']=subprocess.check_output(['git','-C',str(Path(__file__).parent),'rev-parse','HEAD'],text=True).strip()
        report['application_worktree_dirty']=bool(subprocess.check_output(['git','-C',str(Path(__file__).parent),
            'status','--porcelain','--','.'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        report['application_commit']=None
        report['application_worktree_dirty']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='normalized_results'},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--create-fixtures',action='store_true')
    parser.add_argument('--drop-score',type=float,help='Explicit OCR 2.9.1 runner override; recorded separately from application configuration loading')
    parser.add_argument('--config',type=Path,help='Explicit bounded JSON loaded by the original OCR 2.9.1 application factory')
    args=parser.parse_args()
    if args.create_fixtures:fixture_documents(args.fixtures)
    else:
        if not args.output:parser.error('--output required')
        report=run(args.fixtures,args.output,drop_score=args.drop_score,config_path=args.config)
        if report['runtime_status'] != 'PASSED':raise SystemExit(1)
