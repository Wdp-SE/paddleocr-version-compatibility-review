"""Developer-curated evidence control using the same generator and checker, not human accuracy."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.paddleocr_retrieval_views import materialize_evidence,generation_evidence_text


def curated_evidence(case,index):
    specs=case.get('oracle_sources',[])
    if not specs:raise ValueError('No predeclared evidence anchors for this case')
    selected=[]
    for spec in specs:
        matches=[r for r in index.chunks if r.get('version')==spec['version']
                 and r.get('document_path')==spec['path'] and all(t in r['content'] for t in spec['tokens'])]
        if not matches:raise ValueError('A predeclared source anchor is unavailable')
        for row in matches:
            selected.append({'parent_chunk_id':row['chunk_id'],'line_start':row['line_start'],'line_end':row['line_end']})
    hits=materialize_evidence(index.chunks,selected)
    if len(hits)>12 or sum(len(generation_evidence_text(h)) for h in hits)>24000:
        raise ValueError('Curated context exceeds fixed budget')
    return hits


def main():
    from fastapi.testclient import TestClient
    from starlette.requests import Request
    from src.public_server import create_app
    from src.public_api import SearchRequest,_index,_execute_public_search,_query_once
    from src.build_identity import service_identity_digest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    dataset=json.loads((Path(__file__).parent/'cases.json').read_text(encoding='utf8'))
    report={'experiment':'developer_curated_evidence_control','human_accuracy':None,
            'human_review_status':'PENDING_INDEPENDENT_REVIEW','rows':[],
            'limitation':'Curated anchors are developer-authored diagnostic controls, not independent expert gold labels.',
            'dataset_sha256':hashlib.sha256((Path(__file__).parent/'cases.json').read_bytes()).hexdigest()}
    with TestClient(create_app()) as client:
        identity=client.get('/public/workspace').json();report['identity']=identity
        if client.app.state.public_generator is None:raise ValueError('Configured generation is required for this control')
        request=Request({'type':'http','app':client.app,'state':{'request_id':'curated-context-control'}})
        for case in dataset['cases']:
            if not case.get('oracle_sources'):continue
            payload=SearchRequest(query=case['query'],version=case['version'],language='zh',
                                  retrieval_policy='paddleocr_evidence',evidence_strategy='contextual_bm25')
            normal=_execute_public_search(_index(request),payload,client.app.state.public_generator,top_k=5)
            oracle=curated_evidence(case,getattr(_index(request),'base_index',_index(request)))
            item={'id':case['id'],'modes':{}}
            for mode,hits in [('retrieved',normal['results']),('curated',oracle)]:
                execution={**normal,'results':hits,'final_evidence_count':len(hits)}
                start=time.monotonic()
                value=client.portal.call(partial(_query_once,payload,request,execution=execution))
                item['modes'][mode]={'seconds':round(time.monotonic()-start,3),'result':value,
                                      'context_chars':sum(len(generation_evidence_text(h)) for h in hits)}
                print(json.dumps({'id':case['id'],'mode':mode,'status':value.get('status'),
                                  'completeness':value.get('answer_completeness')},ensure_ascii=False),flush=True)
            report['rows'].append(item)
            (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        after=client.get('/public/workspace').json()
        report['identity_consistent']=service_identity_digest(identity)==service_identity_digest(after)
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
