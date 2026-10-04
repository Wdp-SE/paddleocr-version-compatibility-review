"""Online checks use public official excerpts only; no uploaded/private application."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from fastapi.testclient import TestClient
from src.public_server import create_app


def run():
    os.environ['APP_ENV']='public_demo'
    os.environ['RD_V2_ALLOW_EXTERNAL_GENERATION']='true'
    os.environ['RD_V2_GENERATION_PROVIDER']='deepseek'
    folder=Path(__file__).resolve().parent
    cases=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    requests=[{'id':c['id'],'query':c['query'],'version':c.get('version','v3.0.0')}
              for c in cases['cases'] if c['id'] in ('d02','d05','h02','h11')]+cases['no_answer_probes']
    rows=[]
    with TestClient(create_app()) as client:
        for case in requests:
            response=client.post('/public/query',json={'query':case['query'],'version':case.get('version','v3.0.0'),
                                                       'language':'zh','top_k':5})
            payload=response.json()
            rows.append({'id':case['id'],'query':case['query'],'http_status':response.status_code,
                         **payload,'human_accuracy_label':None})
            print(case['id'],response.status_code,payload.get('status'),
                  payload.get('claim_verification',{}).get('status'),flush=True)
            output={'assessment_type':'online_smoke_not_accuracy_benchmark','cases':rows,
                    'generation_code_sha256':hashlib.sha256((ROOT/'versioned-rag-service/src/answer_generation.py').read_bytes()).hexdigest(),
                    'checker_code_sha256':hashlib.sha256((ROOT/'versioned-rag-service/src/claim_support.py').read_bytes()).hexdigest(),
                    'api_code_sha256':hashlib.sha256((ROOT/'versioned-rag-service/src/public_api.py').read_bytes()).hexdigest()}
            (folder/'online-report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':run()
