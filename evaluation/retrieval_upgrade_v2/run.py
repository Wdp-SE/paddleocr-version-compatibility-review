"""Comparable lexical/hybrid runs; never count model fallback as neural success."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from src.paddleocr_retrieval_views import evidence_text,generation_evidence_text
from src.paddleocr_impact_evaluation import assess_evidence,aggregate

FOLDER=Path(__file__).parent
CONFIGS={
 'raw_bm25':{'strategy':'legacy','structure_windows':False,'candidate_budget':40,'top_k':5},
 'window_bm25':{'strategy':'contextual_bm25','structure_windows':False,'candidate_budget':40,'top_k':5},
 'structure_bm25':{'strategy':'contextual_bm25','structure_windows':True,'candidate_budget':80,'top_k':5},
 'structure_hybrid':{'strategy':'contextual_rrf','structure_windows':True,'candidate_budget':80,'top_k':5},
 'structure_bm25_rerank':{'strategy':'contextual_rerank','structure_windows':True,'candidate_budget':80,'top_k':5},
 'structure_hybrid_rerank':{'strategy':'contextual_rrf_rerank','structure_windows':True,'candidate_budget':80,'top_k':5},
}


def sha(p):
    # Python's LF/CRLF checkout differences do not change the experiment code.
    # Corpus bodies retain exact upstream byte hashes.
    data=p.read_bytes()
    if p.suffix=='.py':data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()
def save(p,value):
    if p.exists():raise ValueError('do not overwrite measured file: '+str(p))
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(name,split,output):
    output.mkdir(parents=True,exist_ok=True)
    cfg=CONFIGS[name]
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    from src.paddleocr_quality import asset_identity
    import os
    identity={'embedding':asset_identity(os.getenv('PADDLEOCR_EMBEDDING_PATH','')),
              'reranker':asset_identity(os.getenv('PADDLEOCR_RERANKER_PATH',''))}
    files={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'versioned-rag-service/src').glob('*.py')}
    files.update({p.relative_to(ROOT).as_posix():sha(p) for p in index.root.rglob('*') if p.is_file()})
    files.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (Path(__file__),FOLDER/'dataset.json')})
    frozen={'files':files,'models':identity,'configs':CONFIGS}
    freeze_path=output/'freeze.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding='utf8'))!=frozen:raise ValueError('experiment identity drift')
    else:save(freeze_path,frozen)
    dataset=json.loads((FOLDER/'dataset.json').read_text(encoding='utf8'))
    search=None if cfg['strategy']=='legacy' else configured_evidence_search(index,cfg['strategy'],structure_windows=cfg['structure_windows'])
    rows=[];evaluation_status='complete'
    cases=[c for c in dataset['cases'] if c['split']==split]
    for case in cases:
        start=time.perf_counter()
        if search is None:
            hits=[h for h in index.search(case['query'],version=case['version'],language='zh',top_k=cfg['top_k']) if h.get('retrieval_score',0)>0]
            result={'diagnostics':{'actual_strategy':'legacy'},'candidate_evidence':hits}
        else:
            result=search.search(plan_query(case['query'],versions=(case['version'],)),top_k=cfg['top_k'],candidate_budget=cfg['candidate_budget'],strategy=cfg['strategy'])
            hits=result['results']
        def saved_hit(h):
            return {k:h[k] for k in ('chunk_id','version','document_path')}|{'text':evidence_text(h),
                'generation_text':generation_evidence_text(h),**({'evidence_spans':h['evidence_spans']} if 'evidence_spans' in h else {})}
        score=assess_evidence(case,hits);candidate_score=assess_evidence(case,result['candidate_evidence'])
        # Diagnostic stage names are observations, not presumed root causes.
        stage='covered' if score['complete'] else 'selection_or_ranking' if candidate_score['complete'] else 'candidate_or_representation'
        rows.append({'id':case['id'],'score':score,'candidate_score':candidate_score,'failure_stage':stage,
            'latency_ms':round((time.perf_counter()-start)*1000,2),'diagnostics':result['diagnostics'],
            'hits':[saved_hit(h) for h in hits],'candidates':[saved_hit(h) for h in result['candidate_evidence']]})
        print(name,split,case['id'],int(score['complete']),result['diagnostics'].get('fallback_reason'),flush=True)
        if result['diagnostics'].get('fallback_reason'):
            evaluation_status='rejected_model_output'
            break
    neural=[r for r in rows if r['diagnostics']['actual_strategy']==cfg['strategy'] and not r['diagnostics'].get('fallback_reason')]
    report={'strategy':name,'split':split,'evaluation_status':evaluation_status,'planned_count':len(cases),
        'freeze_sha256':sha(freeze_path),'metrics':aggregate([r['score'] for r in rows]),
        'candidate_metrics':aggregate([r['candidate_score'] for r in rows]),'execution_success_count':len(neural),
        'latency_p50_ms':float(np.percentile([r['latency_ms'] for r in rows],50)),
        'latency_p95_ms':float(np.percentile([r['latency_ms'] for r in rows],95)),
        'answer_correctness':None,'rows':rows}
    save(output/f'{name}-{split}.json',report)
    print('RESULT',name,split,json.dumps(report['metrics']),len(neural),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=CONFIGS,required=True)
    parser.add_argument('--split',choices=['regression','validation','sealed'],required=True)
    parser.add_argument('--output',type=Path,default=FOLDER/'experiment-01');args=parser.parse_args()
    run(args.name,args.split,args.output)
