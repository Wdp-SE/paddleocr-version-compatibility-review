"""Known-task evidence regression. No generation, answer grading or external model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))


def run():
    from src.public_knowledge import PublicKnowledgeIndex
    from src.paddleocr_evidence_search import configured_evidence_search
    from src.paddleocr_query_plan import plan_query
    from src.paddleocr_retrieval_views import generation_evidence_text
    from src.application_knowledge import search_application
    from src.public_scope import is_out_of_scope_public_request
    from src.paddleocr_compatibility import review_compatibility
    folder=Path(__file__).parent
    cases=json.loads((folder/'cases.json').read_text(encoding='utf8'))
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    search=configured_evidence_search(index,structure_windows=True)
    sdk=[]
    for case in cases['sdk_cases']:
        result=search.search(plan_query(case['query'],versions=tuple(case['versions'])),top_k=8,candidate_budget=120)
        hits=result['results']
        checks=[any(h['version']==e['version'] and all(t in generation_evidence_text(h) for t in e['tokens']) for h in hits)
                for e in case['evidence']]
        sdk.append({'id':case['id'],'covered':all(checks),'requirements':checks,
                    'wrong_versions':sum(h['version'] not in case['versions'] for h in hits)})
    app=ROOT/'examples/paddleocr_document_app'
    paths=['application/'+r['path'] for r in json.loads((app/'application/application_manifest.json').read_text(encoding='utf8'))['files']]
    files=[{'path':p,'content':(app/p).read_text(encoding='utf8')} for p in paths]+[
        {'path':'result_adapter.py','content':(app/'result_adapter.py').read_text(encoding='utf8')}]
    application=[]
    for case in cases['application_cases']:
        hits=search_application(files,case['query'],application_id='document-ocr',application_commit='0'*40,top_k=5)['results']
        application.append({'id':case['id'],'covered':set(case['paths']).issubset({h['path'] for h in hits}),
                            'paths':[h['path'] for h in hits]})
    static=review_compatibility(index,source_version='v2.9.1',target_version='v3.0.0',files=files)
    dig=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return {'dataset_id':cases['dataset_id'],'assessment_type':cases['assessment_type'],
        'cases_sha256':dig(folder/'cases.json'),'chunks_sha256':dig(index.root/'chunks.json'),
        'code_sha256':{p.name:dig(p) for p in (ROOT/'versioned-rag-service/src').glob('*.py')},
        'application_sha256':{r['path']:hashlib.sha256(r['content'].encode()).hexdigest() for r in files},
        'sdk_evidence':sdk,'application_evidence':application,
        'missing_internal_fact_rejected':is_out_of_scope_public_request(cases['missing_internal_fact']),
        'static':{'risk_count':static['summary']['risk_count'], 'gaps':len(static['gaps']),
                  'coverage':static['coverage_summary'],'status':static['status']},
        'answer_accuracy':None,'model_calls':0,'runtime_verified':False,
        'notice':'Literal evidence coverage and static findings on known tasks; not independent answer or business accuracy. Working files are identified by hashes, not claimed as a committed application snapshot.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in value.items() if k not in ('code_sha256','application_sha256')},ensure_ascii=False))
    raise SystemExit(0 if all(r['covered'] for r in value['sdk_evidence']+value['application_evidence']) and value['missing_internal_fact_rejected'] else 1)
