"""Freeze internal cases before running; never choose on the holdout."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_quality import configured_quality_retriever,STRATEGIES
from src.quality_evaluation import assess_retrieval,select_strategy


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    folder=Path(__file__).resolve().parent
    cases=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    corpus=ROOT/'versioned-rag-service/public_corpus_paddleocr'
    index=PublicKnowledgeIndex(corpus)
    for case in cases['cases']:
        for fact in case['facts']:
            if not any(c['version']==case['version'] and c['document_path']==fact['path']
                       and all(t in c['content'] for t in fact['tokens']) for c in index.chunks):
                raise ValueError(f"unverified source label: {case['id']}: {fact}")
    freeze={'cases_sha256':sha(folder/'cases.json'),'chunks_sha256':sha(corpus/'chunks.json'),
            'corpus_manifest_sha256':sha(corpus/'corpus_manifest.json'),
            'lexical_code_sha256':sha(ROOT/'versioned-rag-service/src/public_knowledge.py'),
            'fusion_code_sha256':sha(ROOT/'versioned-rag-service/src/retrieval_fusion.py'),
            'retriever_sha256':sha(ROOT/'versioned-rag-service/src/paddleocr_quality.py'),
            'metric_code_sha256':sha(ROOT/'versioned-rag-service/src/quality_evaluation.py'),
            'runner_sha256':sha(Path(__file__))}
    (folder/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    retriever=configured_quality_retriever(index)
    results={}
    for strategy in STRATEGIES:
        rows=[]
        for case in sorted(cases['cases'],key=lambda c:c['split']!='dev'):
            start=time.perf_counter()
            hits=retriever.search(case['query'],version=case['version'],language='zh',strategy=strategy,top_k=cases['top_k'])
            rows.append({'id':case['id'],'split':case['split'],**assess_retrieval(case,hits),
                         'latency_ms':round((time.perf_counter()-start)*1000,2),
                         'chunk_ids':[h['chunk_id'] for h in hits]})
        summary={}
        for split in ('dev','holdout'):
            subset=[row for row in rows if row['split']==split]
            summary[split]={'count':len(subset),'complete_rate':sum(r['complete'] for r in subset)/len(subset),
                            'fact_recall':sum(r['matched_fact_count'] for r in subset)/sum(r['required_fact_count'] for r in subset),
                            'mrr':sum(r['reciprocal_rank'] for r in subset)/len(subset),
                            'wrong_version_count':sum(r['wrong_version_count'] for r in subset),
                            'p95_ms':float(np.percentile([r['latency_ms'] for r in subset],95))}
        results[strategy]={**summary,'cases':rows}
        print(strategy,summary,flush=True)
    selected=select_strategy(results)
    output={'dataset_id':cases['dataset_id'],'label_origin':cases['label_origin'],**freeze,
            'strategies':results,'selected_on_dev':selected,'top_k':cases['top_k'],
            'answer_accuracy':None,'business_impact_accuracy':None,
            'embedding':{'model':'BAAI/bge-small-zh-v1.5','revision':'7999e1d3359715c523056ef9478215996d62a620'},
            'reranker':{'model':'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1','revision':'1427fd652930e4ba29e8149678df786c240d8825'},
            'limitations':['Internal developer-authored holdout, no independent domain expert labels.',
                           'Neural models truncate at their token limits; ranking does not rewrite source evidence.',
                           'No-answer probes require generation evaluation; not counted as retrieval accuracy.']}
    (folder/'report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SELECTED_ON_DEV',selected,flush=True)


if __name__=='__main__':run()
