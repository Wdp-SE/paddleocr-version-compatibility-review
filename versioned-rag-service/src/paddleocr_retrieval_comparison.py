"""Verify exact current-source experiment results before exposing or selecting them."""
import hashlib
import json
from pathlib import Path


def digest(path):
    data=path.read_bytes()
    if path.suffix=='.py':data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()


def load_retrieval_comparison(corpus):
    root=Path(__file__).resolve().parents[2]
    folder=root/'evaluation/retrieval_upgrade_v2/final'
    try:
        if Path(corpus).resolve()!=(root/'versioned-rag-service/public_corpus_paddleocr').resolve():return None
        freeze=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        lock=json.loads((folder/'release-lock.json').read_text(encoding='utf8'))
        release=json.loads((folder/'release.json').read_text(encoding='utf8'))
        selection=json.loads((folder/'selection.json').read_text(encoding='utf8'))
        dataset=json.loads((folder/'dataset.json').read_text(encoding='utf8'))
        required=set(freeze['files'])
        relative=folder.relative_to(root).as_posix()
        required|={relative+'/'+name for name in ('freeze.json','release.json','selection.json','dataset.json')}
        required|={relative+f'/{name}-{split}.json' for name,splits in release['retrieval'].items() for split in splits}
        if not required.issubset(lock['files']):return None
        for name,value in lock['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or digest(p)!=value:return None
        if any(lock['files'].get(k)!=v for k,v in freeze['files'].items()):return None
        if digest(folder/'dataset.json')!=digest(root/'evaluation/retrieval_upgrade_v2/dataset.json'):return None
        configs=freeze['configs'];selected=release['selected']
        if (release.get('schema_version')!=1 or selected not in configs
                or release['selected_strategy']!=configs[selected]['strategy']
                or any(release.get(k)!=configs[selected][k] for k in ('top_k','candidate_budget','structure_windows'))
                or selection.get('selected_before_sealed')!=selected
                or selection.get('freeze_sha256')!=digest(folder/'freeze.json')
                or set(release['retrieval'])!=set(configs)):return None
        from src.public_knowledge import PublicKnowledgeIndex
        from src.paddleocr_retrieval_views import evidence_text,generation_evidence_text,module_for
        from src.paddleocr_impact_evaluation import assess_evidence,aggregate
        index=PublicKnowledgeIndex(corpus);by_id={c['chunk_id']:c for c in index.chunks}
        verified_rows=[]
        disqualified=release.get('disqualified',{})
        if set(disqualified)-set(configs):return None
        for name,splits in release['retrieval'].items():
            expected={'regression','validation','sealed'} if name in (selected,'window_bm25') else {'regression','validation'}
            if name in disqualified:
                expected={'regression'} if disqualified[name].get('split','regression')=='regression' else {'regression','validation'}
            if set(splits)!=expected:return None
            for split,summary in splits.items():
                report=json.loads((folder/f'{name}-{split}.json').read_text(encoding='utf8'))
                cases={c['id']:c for c in dataset['cases'] if c['split']==split}
                status=report.get('evaluation_status','complete')
                if (report['strategy']!=name or report['split']!=split or report['freeze_sha256']!=digest(folder/'freeze.json')
                        or report.get('planned_count',len(cases))!=len(cases)
                        or summary.get('evaluation_status','complete')!=status):return None
                if status=='complete':
                    if len(report['rows'])!=len(cases) or {r['id'] for r in report['rows']}!=set(cases):return None
                elif status=='rejected_model_output':
                    ordered=list(cases)
                    if (name not in disqualified or not report['rows'] or len(report['rows'])>len(cases)
                            or [r['id'] for r in report['rows']]!=ordered[:len(report['rows'])]
                            or not report['rows'][-1]['diagnostics'].get('fallback_reason')
                            or any(r['diagnostics'].get('fallback_reason') for r in report['rows'][:-1])):return None
                else:return None
                scores=[];success=0
                for record in report['rows']:
                    if len(record['hits'])>configs[name]['top_k'] or len({h['chunk_id'] for h in record['hits']})!=len(record['hits']):return None
                    hits=[]
                    for saved in record['hits']:
                        hit={**by_id[saved['chunk_id']]};hit['module']=module_for(hit)
                        if hit['version']!=saved['version'] or hit['document_path']!=saved['document_path']:return None
                        if 'evidence_spans' in saved:hit['evidence_spans']=saved['evidence_spans']
                        if evidence_text(hit)!=saved['text'] or generation_evidence_text(hit)!=saved['generation_text']:return None
                        hits.append(hit)
                    score=assess_evidence(cases[record['id']],hits)
                    if score!=record['score']:return None
                    scores.append(score)
                    d=record['diagnostics']
                    success+=int(d['actual_strategy']==configs[name]['strategy'] and not d.get('fallback_reason'))
                metrics=aggregate(scores)
                if (metrics!=report['metrics'] or metrics!=summary['metrics']
                        or success!=report['execution_success_count'] or summary['execution_success_count']!=success):return None
                verified_rows.append({'strategy':name,'split':split,**metrics,'execution_success_count':success,
                                      'evaluation_status':status,'planned_count':len(cases),
                                      'latency_p50_ms':report['latency_p50_ms']})
        # Reproduce the pre-sealed selection: coverage gates first, noise and complexity as tie breakers.
        eligible=[]
        for name,splits in release['retrieval'].items():
            base=release['retrieval']['window_bm25']
            if name in disqualified:continue
            if all(splits[s]['metrics']['wrong_version_count']==0
                   and splits[s]['execution_success_count']==splits[s]['metrics']['count']
                   and splits[s]['metrics']['complete_count']>=base[s]['metrics']['complete_count']
                   for s in ('regression','validation')):eligible.append(name)
        order=list(configs)
        def key(n):
            m=[release['retrieval'][n][s]['metrics'] for s in ('regression','validation')]
            return (min(x['complete_rate'] for x in m),sum(x['complete_rate'] for x in m),
                    -sum(x['wrong_module_count'] for x in m),-order.index(n))
        if not eligible or max(eligible,key=key)!=selected:return None
        chosen=release['retrieval'][selected]['sealed'];base=release['retrieval']['window_bm25']['sealed']
        release['promotion_eligible']=bool(chosen['metrics']['complete_count']>=base['metrics']['complete_count']
                and not chosen['metrics']['wrong_version_count']
                and chosen['execution_success_count']==chosen['metrics']['count'])
        release['rows']=verified_rows
        return release
    except (OSError,ValueError,KeyError,TypeError):return None
