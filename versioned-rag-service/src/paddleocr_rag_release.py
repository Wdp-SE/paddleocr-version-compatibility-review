"""Current RAG experiment release, invalidated by any pinned artifact/code drift."""
import hashlib
import json
from pathlib import Path

def load_rag_release(corpus: Path):
    from src.paddleocr_retrieval_comparison import load_retrieval_comparison
    comparison=load_retrieval_comparison(corpus)
    if comparison and comparison['promotion_eligible']:
        return comparison
    from src.paddleocr_internal_release import load_internal_release
    current=load_internal_release(corpus)
    if current is not None:
        return current
    root=Path(__file__).resolve().parents[2]
    folder=root/'evaluation/paddleocr_quality_v4/final'
    try:
        if corpus.resolve()!=(root/'versioned-rag-service/public_corpus_paddleocr').resolve():return None
        lock=json.loads((folder/'release-lock.json').read_text(encoding='utf8'))
        row=json.loads((folder/'release.json').read_text(encoding='utf8'))
        if (row['schema_version']!=1 or row['selected_strategy'] not in ('contextual_bm25','contextual_llm_rerank')
                or type(row['top_k']) is not int or row['top_k']!=5 or row['candidate_budget']!=40):return None
        names=set(row['retrieval'])
        chosen='llm' if row['selected_strategy']=='contextual_llm_rerank' else 'bm25'
        if not {'legacy','bm25',chosen}.issubset(names) or not names.issubset({'legacy','bm25','llm'}):return None
        required={'evaluation/paddleocr_quality_v4/final/'+name for name in (
            'release.json','freeze.json','dataset.json','legacy-dev.json','legacy-holdout.json',
            'bm25-dev.json','bm25-holdout.json')}
        required|={'versioned-rag-service/'+name for name in (
            'public_corpus_paddleocr/chunks.json','src/paddleocr_query_plan.py',
            'src/paddleocr_retrieval_views.py','src/paddleocr_evidence_search.py',
            'src/paddleocr_fact_checks.py','src/public_api.py','src/claim_support.py',
            'src/paddleocr_rag_release.py','src/paddleocr_impact_evaluation.py','src/public_knowledge.py')}
        required|={f'evaluation/paddleocr_quality_v4/final/{name}-{split}.json'
                   for name in names for split in ('dev','holdout')}
        if not required.issubset(lock['files']): return None
        for relative, expected in lock['files'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:return None
        freeze_hash=hashlib.sha256((folder/'freeze.json').read_bytes()).hexdigest()
        from src.public_knowledge import PublicKnowledgeIndex
        from src.paddleocr_retrieval_views import evidence_text, generation_evidence_text, module_for
        from src.paddleocr_impact_evaluation import aggregate, assess_evidence
        index=PublicKnowledgeIndex(corpus);by_id={h['chunk_id']:h for h in index.chunks}
        dataset=json.loads((folder/'dataset.json').read_text(encoding='utf8'))
        for name in row['retrieval']:
            for split in ('dev','holdout'):
                report=json.loads((folder/f'{name}-{split}.json').read_text(encoding='utf8'))
                if (report['strategy']!=name or report['split']!=split or report['freeze_sha256']!=freeze_hash
                        or set(row['retrieval'][name])!={'dev','holdout'}):return None
                cases={c['id']:c for c in dataset['cases'] if c['split']==split}
                if not cases or {r['id'] for r in report['rows']}!=set(cases) or len(report['rows'])!=len(cases):return None
                scores=[]
                for record in report['rows']:
                    hits=[]
                    if len(record['hits'])>5 or len({h['chunk_id'] for h in record['hits']})!=len(record['hits']):return None
                    for saved in record['hits']:
                        h=dict(by_id[saved['chunk_id']]); h['module']=module_for(h)
                        if saved['version']!=h['version'] or saved['document_path']!=h['document_path']:return None
                        if 'evidence_spans' in saved:h['evidence_spans']=saved['evidence_spans']
                        if saved['text']!=evidence_text(h) or saved['generation_text']!=generation_evidence_text(h):return None
                        hits.append(h)
                    score=assess_evidence(cases[record['id']],hits)
                    if score!=record['score']:return None
                    scores.append(score)
                if aggregate(scores)!=report['metrics'] or report['metrics']!=row['retrieval'][name][split]['metrics']:return None
        base=row['retrieval']['legacy']['holdout']['metrics']
        current=row['retrieval']['llm' if row['selected_strategy']=='contextual_llm_rerank' else 'bm25']['holdout']['metrics']
        if current['complete_count']<base['complete_count'] or current['wrong_version_count'] or current['wrong_module_count']>base['wrong_module_count']:return None
        return row
    except (OSError,ValueError,KeyError,TypeError):return None
