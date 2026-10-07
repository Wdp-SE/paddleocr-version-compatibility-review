"""Recompute published diagnostic metrics; artifact drift fails closed."""
import hashlib
import json
from pathlib import Path


def neural_execution_succeeded(rows):
    return bool(rows) and all(r.get('diagnostics',{}).get('actual_strategy')=='contextual_rerank'
        and r['diagnostics'].get('rerank_calls')==1 and not r['diagnostics'].get('fallback_reason') for r in rows)


def frozen_lock_matches(root, freeze, lock):
    try:
        root=Path(root).resolve()
        if not freeze.get('files'):return False
        for name,sha in freeze['files'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or lock.get('files',{}).get(name)!=sha or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def load_internal_release(corpus):
    root=Path(__file__).resolve().parents[2]
    folder=root/'evaluation/interview_release_v1/final6'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/interview_release_v1/final5'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/interview_release_v1/final4'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/interview_release_v1/final3'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/interview_release_v1/final2'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/interview_release_v1/final'
    if not (folder/'release.json').is_file():
        folder=root/'evaluation/internal_workflow_v1/final'
    relative_folder=folder.relative_to(root).as_posix()
    try:
        if Path(corpus).resolve()!=(root/'versioned-rag-service/public_corpus_paddleocr').resolve():return None
        release=json.loads((folder/'release.json').read_text(encoding='utf8'))
        lock=json.loads((folder/'release-lock.json').read_text(encoding='utf8'))
        freeze=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        if not frozen_lock_matches(root,freeze,lock):return None
        if release.get('schema_version')!=1 or release.get('selected') not in freeze['configs']:return None
        cfg=freeze['configs'][release['selected']]
        if cfg['strategy'] not in ('contextual_bm25','contextual_rerank') or cfg['top_k'] not in (5,8) or cfg['window_budget'] not in (256,384) or cfg['candidate_budget'] not in (40,80):return None
        required=set(freeze['files'])|{relative_folder+'/'+name for name in ('release.json','freeze.json','selection.json','dataset.json')}
        required|={f'{relative_folder}/{name}-{split}.json' for name,splits in release['retrieval'].items() for split in splits}
        if not required.issubset(lock['files']):return None
        for relative,sha in lock['files'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:return None
        from src.public_knowledge import PublicKnowledgeIndex
        from src.paddleocr_retrieval_views import module_for,evidence_text,generation_evidence_text
        from src.paddleocr_impact_evaluation import assess_evidence,aggregate
        index=PublicKnowledgeIndex(corpus);by_id={h['chunk_id']:h for h in index.chunks}
        dataset=json.loads((folder/'dataset.json').read_text(encoding='utf8'))
        freeze_sha=hashlib.sha256((folder/'freeze.json').read_bytes()).hexdigest()
        for name,splits in release['retrieval'].items():
            if name not in freeze['configs']:return None
            for split,metrics in splits.items():
                saved=json.loads((folder/f'{name}-{split}.json').read_text(encoding='utf8'))
                if name==release['selected'] and cfg['strategy']=='contextual_rerank' and not neural_execution_succeeded(saved['rows']):return None
                cases={c['id']:c for c in dataset['cases'] if c['split']==split}
                if saved['freeze_sha256']!=freeze_sha or saved['strategy']!=name or saved['split']!=split or len(saved['rows'])!=len(cases) or {r['id'] for r in saved['rows']}!=set(cases):return None
                scores=[]
                for row in saved['rows']:
                    if len(row['hits'])>freeze['configs'][name]['top_k'] or len({h['chunk_id'] for h in row['hits']})!=len(row['hits']):return None
                    hits=[]
                    for hit in row['hits']:
                        h=dict(by_id[hit['chunk_id']]);h['module']=module_for(h)
                        if hit['version']!=h['version'] or hit['document_path']!=h['document_path']:return None
                        if 'evidence_spans' in hit:h['evidence_spans']=hit['evidence_spans']
                        if evidence_text(h)!=hit['text'] or generation_evidence_text(h)!=hit['generation_text']:return None
                        hits.append(h)
                    score=assess_evidence(cases[row['id']],hits)
                    if score!=row['score']:return None
                    scores.append(score)
                if aggregate(scores)!=saved['metrics'] or metrics['metrics']!=saved['metrics']:return None
        current=release['retrieval'][release['selected']]['holdout']['metrics']
        base=release['retrieval']['legacy']['holdout']['metrics']
        if current['complete_count']<base['complete_count'] or current['wrong_version_count'] or current['wrong_module_count']>base['wrong_module_count']:return None
        if any(release.get(k)!=cfg[k] for k in ('top_k','candidate_budget','window_budget','use_context')) or release['selected_strategy']!=cfg['strategy']:return None
        return release
    except (OSError,ValueError,KeyError,TypeError):
        return None
