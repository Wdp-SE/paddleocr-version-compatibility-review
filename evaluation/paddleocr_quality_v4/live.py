"""Opt-in live answer probes; manually assess saved claims against pinned evidence."""
import json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from fastapi.testclient import TestClient
from src.public_server import create_app
from src.answer_generation import StructuredAnswerGenerator

QUESTIONS=[
 'PaddleOCR v3.0.0 的 OCR 结果如何逐个调用 save_to_json 保存到 output 目录？',
 'PaddleOCR v3.0.0 的 OCR 结果 rec_texts 和 rec_scores 分别表示什么，是否已过滤？',
 'PaddleOCR v3.0.0 的 PPStructure 应迁移到哪个接口？',
 'OCR 产线 v3.0.0 的 cpu_threads 默认值是多少？',
 'PaddleOCR v3.0.0 是否保证任何输入图片的文字识别准确率达到100%？',
]
def main():
    path=Path(__file__).parent/os.getenv('PADDLEOCR_EXPERIMENT_REVISION','.')/'live-report.json'
    if path.exists(): raise RuntimeError('Preserve previous runs')
    os.environ['APP_ENV']='public_demo'
    generator=StructuredAnswerGenerator(provider='deepseek',model=os.getenv('RD_V2_GENERATION_MODEL','deepseek-flash'))
    with TestClient(create_app(generator=generator)) as client:
        def query(question):
            start=time.perf_counter()
            response=client.post('/public/query',json={'query':question,'version':'v3.0.0','language':'zh',
                'retrieval_policy':'paddleocr_evidence','evidence_strategy':os.getenv('PADDLEOCR_LIVE_STRATEGY','contextual_bm25'),'top_k':5})
            result={'query':question,'seconds':round(time.perf_counter()-start,2),'http_status':response.status_code,
                    'response':response.json()}
            print(QUESTIONS.index(question)+1,result['response'].get('status'),result['seconds'],flush=True)
            return result
        rows=list(ThreadPoolExecutor(max_workers=2).map(query,QUESTIONS))
    path.write_text(json.dumps({'assessment':'live operational/claim audit probes, not population accuracy',
                               'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if __name__=='__main__': main()
