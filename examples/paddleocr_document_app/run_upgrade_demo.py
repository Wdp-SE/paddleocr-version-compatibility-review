"""Run only this original application, in separately pinned OCR environments."""
from pathlib import Path
import argparse,hashlib,json,time,os
from importlib.metadata import version
from result_adapter import normalize_v2,normalize_v3


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


def run(folder,output):
    from paddleocr import PaddleOCR
    runtime_version=version('paddleocr')
    major=int(runtime_version.split('.')[0])
    if runtime_version not in ('2.9.1','3.0.0'):raise ValueError('unsupported demonstration dependency')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if major==2:
        model_root=Path(os.environ.get('OCR_V2_MODEL_ROOT',str(Path.home()/'.paddleocr/whl')))
        engine=PaddleOCR(lang='ch',use_gpu=False,enable_mkldnn=False,cpu_threads=2,show_log=False,
                        det_model_dir=str(model_root/'det/ch/ch_PP-OCRv4_det_infer'),
                        rec_model_dir=str(model_root/'rec/ch/ch_PP-OCRv4_rec_infer'),
                        cls_model_dir=str(model_root/'cls/ch_ppocr_mobile_v2.0_cls_infer'))
    else:
        model_root=Path(os.environ.get('OCR_V3_MODEL_ROOT',str(Path.home()/'.paddlex/official_models')))
        engine=PaddleOCR(lang='ch',device='cpu',enable_mkldnn=False,cpu_threads=2,
                        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
                        text_detection_model_name='PP-OCRv5_server_det',text_detection_model_dir=str(model_root/'PP-OCRv5_server_det'),
                        text_recognition_model_name='PP-OCRv5_server_rec',text_recognition_model_dir=str(model_root/'PP-OCRv5_server_rec'))
    started=time.perf_counter()
    raw=[]
    for page in manifest['pages']:
        path=Path(page['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=page['sha256']:raise ValueError('fixture changed')
        if major==2:raw.extend(engine.ocr(str(path),cls=False))
        else:
            # PaddleX 3.0's cv2 path reader fails on Windows non-ASCII paths.
            # Decode locally and supply its documented numpy image input instead.
            from PIL import Image
            import numpy as np
            raw.extend(engine.predict(np.asarray(Image.open(path).convert('RGB'))[:,:,::-1]))
    normalized=normalize_v2(raw) if major==2 else normalize_v3(raw)
    from application.consumer import to_json
    downstream=to_json(normalized)
    contract_valid=(downstream['schema_version']==1 and
                    all(set(page)=={'page_index','text'} and isinstance(page['text'],str) for page in downstream['pages']))
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
    report['input_method']='file_path' if major==2 else 'locally_decoded_bgr_array_for_windows_unicode_paths'
    report['consumer_sha256']=hashlib.sha256(Path(__file__).with_name('application').joinpath('consumer.py').read_bytes()).hexdigest()
    report['application_files_sha256']={str(p.relative_to(Path(__file__).parent)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).with_name('application').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    report['downstream_json']=downstream
    report['downstream_contract_passed']=contract_valid
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='normalized_results'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--create-fixtures',action='store_true')
    args=parser.parse_args()
    if args.create_fixtures:fixture_documents(args.fixtures)
    else:
        if not args.output:parser.error('--output required')
        run(args.fixtures,args.output)
