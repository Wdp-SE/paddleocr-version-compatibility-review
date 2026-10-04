"""One-shot post-selection probes. No labels or strategies are adjusted here."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_quality import configured_quality_retriever
from src.paddleocr_evaluation import load_quality_comparison
from src.paddleocr_compatibility import review_compatibility
from src.quality_evaluation import assess_retrieval


def run():
    folder=Path(__file__).resolve().parent
    output=folder/'independent-report.json'
    if output.exists():raise ValueError('probes already executed; do not tune or overwrite results')
    cases=json.loads((folder/'independent-probes.json').read_text(encoding='utf-8'))
    corpus=ROOT/'versioned-rag-service/public_corpus_paddleocr'
    if hashlib.sha256((corpus/'chunks.json').read_bytes()).hexdigest()!=cases['chunks_sha256']:
        raise ValueError('independent probe labels belong to a different corpus')
    report=load_quality_comparison(corpus)
    if report is None:raise ValueError('current strategy comparison must finish and validate first')
    policy=report['selected_on_dev']
    index=PublicKnowledgeIndex(corpus)
    retriever=configured_quality_retriever(index) if policy!='bm25' else None
    retrieval=[]
    for case in cases['retrieval_cases']:
        hits=(retriever.search(case['query'],version=case['version'],language='zh',strategy=policy,top_k=5)
              if retriever else [c for c in index.search(case['query'],version=case['version'],language='zh',top_k=5,policy='bm25') if c['retrieval_score']>0])
        retrieval.append({'id':case['id'],**assess_retrieval(case,hits),'chunk_ids':[c['chunk_id'] for c in hits]})
    static=[]
    for case in cases['static_upgrade_cases']:
        result=review_compatibility(index,source_version=case['source_version'],target_version=case['target_version'],files=case['files'])
        rules={f['rule_id'] for f in result['findings']}
        gaps={g['code'] for g in result['gaps']}
        passed=(result['status']==case['expected_status'] and result['runtime_verified'] is False
                and set(case['required_rules'])<=rules and set(case['required_gap_codes'])<=gaps
                and not set(case.get('forbidden_rules',[]))&rules)
        static.append({'id':case['id'],'passed':passed,'status':result['status'],
                       'rules':sorted(rules),'gaps':sorted(gaps),'runtime_verified':False})
    result={'assessment_type':cases['assessment_type'],'selection_policy':cases['selection_policy'],
            'selected_strategy_frozen_before_probes':policy,
            'probes_sha256':hashlib.sha256((folder/'independent-probes.json').read_bytes()).hexdigest(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'comparison_sha256':hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
            'chunks_sha256':cases['chunks_sha256'],
            'static_code_sha256':hashlib.sha256((ROOT/'versioned-rag-service/src/paddleocr_compatibility.py').read_bytes()).hexdigest(),
            'retrieval':retrieval,'static':static,'limitations':cases['limitations']}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'strategy':policy,'complete_retrieval':sum(c['complete'] for c in retrieval),
                       'retrieval_count':len(retrieval),'static_pass':sum(c['passed'] for c in static),
                       'static_count':len(static)},ensure_ascii=False))


if __name__=='__main__':run()
