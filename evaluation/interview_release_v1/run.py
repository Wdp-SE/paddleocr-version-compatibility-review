"""Re-measure current code. Existing questions are regressions, not blind tests."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from src.paddleocr_retrieval_views import module_for,evidence_text,generation_evidence_text
from src.paddleocr_impact_evaluation import assess_evidence,aggregate

FOLDER=Path(__file__).parent/'final6'
CONFIGS={'legacy':{'strategy':'legacy','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':False},
         'scoped_384':{'strategy':'contextual_bm25','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':True}}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=FOLDER/name
    if p.exists():raise ValueError('do not overwrite measured artifact: '+name)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    FOLDER.mkdir(exist_ok=True)
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    dataset=json.loads((ROOT/'evaluation/internal_workflow_v1/final/dataset.json').read_text(encoding='utf8'))
    for case in dataset['cases']:
        for fact in case['facts']:
            matching=[h for h in index.chunks if h['version']==case['version']
                      and h['document_path'] in fact['allowed_paths'] and all(t in h['content'] for t in fact['tokens'])]
            if not matching:raise ValueError('label no longer grounded: '+case['id'])
            fact['module']=module_for(matching[0])
    dataset['label_origin']='developer source-grounded labels; all 48 questions already known'
    dataset['split_method']='historical dev/validation/holdout groups retained only for regression comparability; no fresh blind set'
    save('dataset.json',dataset)
    names=['paddleocr_evidence_search.py','paddleocr_retrieval_views.py','paddleocr_query_plan.py',
        'paddleocr_quality.py','paddleocr_rag_release.py','paddleocr_internal_release.py','public_api.py',
        'public_knowledge.py','paddleocr_impact_evaluation.py','paddleocr_compatibility.py','paddleocr_application_roles.py',
        'paddleocr_function_results.py','public_scope.py','answer_generation.py','claim_support.py']
    paths=[ROOT/'versioned-rag-service/src'/n for n in names]
    paths += [p for p in index.root.rglob('*') if p.is_file()]
    paths += [Path(__file__),FOLDER/'dataset.json',ROOT/'change-review-agent/app/paddleocr_investigation.py']
    save('freeze.json',{'files':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},'configs':CONFIGS,
        'selection_rule':'retain prior 384/40/5 settings; compare current execution to legacy on known regressions; no parameter selection on these results'})
    frozen=digest(FOLDER/'freeze.json')
    search=configured_evidence_search(index)
    retrieval={}
    for name,cfg in CONFIGS.items():
        for split in ('dev','validation','holdout'):
            rows=[]
            for case in [c for c in dataset['cases'] if c['split']==split]:
                started=time.perf_counter()
                if name=='legacy':
                    hits=[h for h in index.search(case['query'],version=case['version'],language='zh',top_k=5) if h.get('retrieval_score',0)>0]
                    diag={'actual_strategy':'legacy'}
                else:
                    result=search.search(plan_query(case['query'],versions=(case['version'],)),top_k=5,candidate_budget=40)
                    hits=result['results'];diag=result['diagnostics']
                rows.append({'id':case['id'],'score':assess_evidence(case,hits),'latency_ms':round((time.perf_counter()-started)*1000,2),
                    'diagnostics':diag,'hits':[{k:h[k] for k in ('chunk_id','version','document_path')}|
                    {'text':evidence_text(h),'generation_text':generation_evidence_text(h),
                     **({'evidence_spans':h['evidence_spans']} if 'evidence_spans' in h else {})} for h in hits]})
            metrics=aggregate([r['score'] for r in rows])
            save(f'{name}-{split}.json',{'strategy':name,'split':split,'freeze_sha256':frozen,'metrics':metrics,'rows':rows})
            retrieval.setdefault(name,{})[split]={'metrics':metrics,'execution':'词法检索'}
            print(name,split,json.dumps(metrics),flush=True)
    base=retrieval['legacy']['validation']['metrics'];current=retrieval['scoped_384']['validation']['metrics']
    if current['complete_count']<base['complete_count'] or current['wrong_version_count'] or current['wrong_module_count']>base['wrong_module_count']:
        save('publication-status.json',{'promoted':False,'reason':'regression quality gate failed'})
        raise ValueError('current quality gate failed; do not publish new metrics as default')
    save('selection.json',{'selected':'scoped_384','validation_gate_passed':True,'parameters_retained':True})
    cfg=CONFIGS['scoped_384']
    save('release.json',{'schema_version':1,'dataset_id':'interview-release-v1-known-regression',
        'selected':'scoped_384','selected_strategy':cfg['strategy'],
        **{k:cfg[k] for k in ('top_k','window_budget','candidate_budget','use_context')},'retrieval':retrieval,
        'limitations':'48 条已知开发者标注题的重新回归，历史分组不是本轮盲测；完整证据覆盖不是回答准确率。本轮保留既有参数，无新神经选型或企业效果结论。'})
    files=dict(json.loads((FOLDER/'freeze.json').read_text(encoding='utf8'))['files'])
    files.update({p.relative_to(ROOT).as_posix():digest(p) for p in FOLDER.iterdir() if p.is_file()})
    save('release-lock.json',{'schema_version':1,'files':files})
    from src.paddleocr_internal_release import load_internal_release
    if load_internal_release(index.root) is None:raise ValueError('canonical release verification failed')
    save('publication-status.json',{'promoted':True,'selected':'scoped_384'})
    print('CANONICALLY_VERIFIED',flush=True)


if __name__=='__main__':main()
