"""Actual provider smoke runs; judgments must be made against pinned originals."""
import json
import hashlib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from fastapi.testclient import TestClient
from src.public_server import create_app


def main():
    os.environ['APP_ENV']='public_demo'
    os.environ['RD_V2_ALLOW_EXTERNAL_GENERATION']='true'
    os.environ['RD_V2_GENERATION_PROVIDER']='deepseek'
    os.environ['RAG_PUBLIC_CORPUS_ROOT']=str(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    os.environ['RAG_PUBLIC_RETRIEVAL_CONFIG']=str(ROOT/'versioned-rag-service/public_corpus_paddleocr/public_retrieval_runtime.json')
    os.environ['RD_V2_GENERATION_MODEL']='deepseek-v4-flash'
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise ValueError('DeepSeek API key unavailable; no simulated success')
    folder=Path(__file__).parent/'live'/datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    profile=json.loads((ROOT/'versioned-rag-service/config/paddleocr_project.json').read_text(encoding='utf8'))['domain_profile']
    questions=profile['example_queries']+[
        'PaddleOCR 3.0.0 的官方 OCR 测试检查了哪些返回字段？',
        'v2.9.1 Python OCR 接口只跑识别，关闭检测，怎么调用？',
        'PaddleOCR 2.9.1 如何用 lang="ch" 初始化中文 OCR？',
    ]
    with TestClient(create_app()) as client:
        for i,q in enumerate(questions,1):
            versions=set(re.findall(r'\d+\.\d+\.\d+',q))
            version='all' if len(versions)>1 else 'v'+next(iter(versions)) if versions else 'v2.9.1'
            t=time.monotonic()
            response=client.post('/public/query',json={'query':q,'version':version,'language':'zh',
                'retrieval_policy':'paddleocr_evidence','evidence_strategy':'contextual_bm25',
                'candidate_budget':40,'top_k':5})
            result=response.json()
            record={'question':q,'version':version,'http_status':response.status_code,
                    'release_lock_sha256':hashlib.sha256((Path(__file__).parent/'final6/release-lock.json').read_bytes()).hexdigest(),
                    'elapsed_seconds':round(time.monotonic()-t,2),'result':result}
            (folder/f'question-{i}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(i,response.status_code,result.get('status'),record['elapsed_seconds'],flush=True)
            print(json.dumps(result.get('answer'),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
