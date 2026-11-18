"""Controlled context expansion with the real generator AND checker.

This is a selected-parent ablation, not gold-context accuracy or a blind test.
It never inserts expected answers, expands source scope, or changes production
endpoints. Run only with public-source cases and the configured provider.
"""
import argparse
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'versioned-rag-service'),str(ROOT/'demo-ui')]
from fastapi.testclient import TestClient
from starlette.requests import Request
from src.public_server import create_app
from src.public_api import SearchRequest,_index,_execute_public_search,_query_once
from src.paddleocr_retrieval_views import generation_evidence_text
from src.build_identity import service_identity_digest,review_execution_fingerprint
from services.knowledge_examples import EXAMPLES
from acceptance_report import case_fingerprint


def expand_selected_parents(hits,*,max_parent_chars=6000,max_total_chars=24000):
    """Only canonical selected parents; independent bounds and no mutation."""
    result=copy.deepcopy(hits)
    if sum(len(generation_evidence_text(h)) for h in result)>max_total_chars:
        raise ValueError('Original selected context exceeds probe budget')
    expanded=[]
    for row in result:
        if not row.get('evidence_spans') or len(row['content'])>max_parent_chars:
            continue
        remaining=sum(len(generation_evidence_text(h)) for h in result if h is not row)
        if remaining+len(row['content'])<=max_total_chars:
            row.pop('evidence_spans')
            expanded.append(row['chunk_id'])
    return result,expanded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',nargs='+',type=int,default=[7,11,13],help='1-based visible example numbers')
    args=parser.parse_args()
    if any(not 1<=i<=len(EXAMPLES) for i in args.only):parser.error('Unknown example')
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[{'id':f'example-{i:02}','query':EXAMPLES[i-1].question,'version':EXAMPLES[i-1].scope} for i in args.only]
    report={'experiment':'selected_parent_context_ablation','split':'known_regression_not_blind',
            'cases_sha256':case_fingerprint(cases),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':[],'human_accuracy':None,'limitation':'Expanded selected source blocks are not independently annotated gold evidence.'}
    code_before=review_execution_fingerprint(ROOT)
    with TestClient(create_app()) as client:
        identity=client.get('/public/workspace').json();report['identity']=identity
        request=Request({'type':'http','app':client.app,'state':{'request_id':'context-probe'}})
        for case in cases:
            payload=SearchRequest(query=case['query'],version=case['version'],language='zh',
                                  retrieval_policy='paddleocr_evidence',evidence_strategy='contextual_bm25',top_k=5)
            normal=_execute_public_search(_index(request),payload,client.app.state.public_generator,top_k=5)
            expanded,ids=expand_selected_parents(normal['results'])
            row={**case,'expanded_parent_ids':ids,'modes':{}}
            for mode,hits in [('retrieved_windows',normal['results']),('bounded_selected_parents',expanded)]:
                execution={**normal,'results':hits}
                started=time.monotonic()
                value=client.portal.call(partial(_query_once,payload,request,execution=execution))
                row['modes'][mode]={'seconds':round(time.monotonic()-started,3),
                    'context_chars':sum(len(generation_evidence_text(h)) for h in hits),'result':value}
                print(json.dumps({'id':case['id'],'mode':mode,'status':value.get('status'),
                    'completeness':value.get('answer_completeness'),'chars':row['modes'][mode]['context_chars']},ensure_ascii=False),flush=True)
            report['rows'].append(row)
            (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        after=client.get('/public/workspace').json()
    report['run_valid']=(service_identity_digest(identity)==service_identity_digest(after)
                         and code_before==review_execution_fingerprint(ROOT))
    (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
