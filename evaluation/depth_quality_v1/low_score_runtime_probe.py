"""Actual OCR over original synthetic low-quality pages; never runs uploaded application code."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    from PIL import Image,ImageDraw,ImageFont,ImageFilter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if version('paddleocr')!='2.9.1':raise ValueError('This controlled probe requires PaddleOCR 2.9.1')
    from paddleocr import PaddleOCR
    sys.path.insert(0,str(ROOT/'examples/paddleocr_document_app'))
    from run_upgrade_demo import check_application_entry
    engine=PaddleOCR(lang='ch',drop_score=0.,use_angle_cls=False,use_gpu=False,show_log=False,cpu_threads=2,enable_mkldnn=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    report={'sample_origin':'author-created synthetic pages; no customer data','workflow':'same_version_threshold_comparison',
            'environment':{'paddleocr':version('paddleocr'),'paddlepaddle':version('paddlepaddle')},
            'runtime_verified':True,'configuration_loading_verified':False,'rows':[],
            'limitation':'Runtime parameter overrides exercise the original application entry; this does not certify JSON file loading or business quality.',
            'application_files':{str(p.relative_to(ROOT/'examples/paddleocr_document_app')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'examples/paddleocr_document_app/application').rglob('*.py')}}
    settings=[(0.,1.,0),(1.2,.7,6),(1.8,.55,10),(2.2,.45,14),(2.8,.4,18),(3.2,.35,22)]
    for i,(blur,scale,noise) in enumerate(settings):
        image=Image.new('RGB',(700,240),'white');draw=ImageDraw.Draw(image)
        draw.text((25,25),'合同编号 DEMO-2026-001',font=font,fill=(90,90,90))
        draw.text((25,110),'金额 1234 元 数量 5 件',font=font,fill=(120,120,120))
        if scale<1:
            image=image.resize((int(700*scale),int(240*scale))).resize((700,240))
        if blur:image=image.filter(ImageFilter.GaussianBlur(blur))
        if noise:
            rng=random.Random(20261026+i);pixels=image.load()
            for y in range(240):
                for x in range(700):
                    delta=rng.randint(-noise,noise);v=max(0,min(255,pixels[x,y][0]+delta));pixels[x,y]=(v,v,v)
        path=args.output/f'original-scan-{i}.png';image.save(path)
        engine.drop_score=0.;raw=engine.ocr(str(path),cls=False)
        observed=[{'text':line[1][0],'score':float(line[1][1])} for page in raw if page for line in page]
        row={'sample':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'degradation':{'blur':blur,'scale':scale,'noise':noise},
             'observed_without_score_filter':observed,'runs':{}}
        for threshold in (.5,.8):
            engine.drop_score=threshold;start=time.monotonic()
            result=check_application_entry(engine,str(path),expected_fragments=None)
            row['runs'][str(threshold)]={'seconds':round(time.monotonic()-start,3),**result}
        row['output_changed']=row['runs']['0.5'].get('output')!=row['runs']['0.8'].get('output')
        report['rows'].append(row)
        report['observed_changed_samples']=sum(r['output_changed'] for r in report['rows'])
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'sample':i,'scores':[round(r['score'],3) for r in observed],'changed':row['output_changed']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
