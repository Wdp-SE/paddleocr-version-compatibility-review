"""Real tools and original demo files; no execution of submitted application."""
import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'change-review-agent'),str(ROOT/'demo-ui'),str(ROOT/'versioned-rag-service')]
from fastapi.testclient import TestClient
from src.public_server import create_app
from services.public_knowledge_client import PublicKnowledgeClient
from services.interview_examples import DEMO_CASES,demo_application
from app.paddleocr_review import PaddleOCRReviewAgent


class Gateway(PublicKnowledgeClient):
    def __init__(self,client):
        super().__init__('http://testserver')
        self.client=client
    def _request(self,method,endpoint,**options):
        response=self.client.request(method,endpoint,json=options.get('json'))
        response.raise_for_status()
        return response.json()


def main():
    os.environ['APP_ENV']='public_demo'
    os.environ['RD_V2_ALLOW_EXTERNAL_GENERATION']='true'
    os.environ['RD_V2_GENERATION_PROVIDER']='deepseek'
    os.environ['RD_V2_GENERATION_MODEL']='deepseek-v4-flash'
    folder=Path(__file__).parent/'agent'/datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    with TestClient(create_app()) as client:
        for i,label in enumerate(DEMO_CASES,1):
            # One genuine model-enabled positive run; the other two also run
            # the real static and evidence tools, without fabricated LLM output.
            result=PaddleOCRReviewAgent(Gateway(client)).analyze(demo_application(label),generate_advice=(i==1))
            lock=Path(__file__).parent/'final6/release-lock.json'
            (folder/f'case-{i}.json').write_text(json.dumps({'case':label,
                'release_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
                'result':result},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
            report=result['compatibility_report']
            print(i,label,report['status'],report['summary'],
                'searches',result['investigation']['search_calls'],'unresolved',len(result['investigation']['unresolved']),flush=True)
            if i==1:assert report['summary']['risk_count']>0
            if i==2:assert report['summary']['risk_count']==0
            if i==3:assert report['status']=='needs_verification' and report['gaps']
            assert report['runtime_verified'] is False


if __name__=='__main__':main()
