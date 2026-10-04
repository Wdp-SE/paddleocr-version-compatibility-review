"""Source-grounded experiment scores, with no automatic answer-accuracy claim."""
from __future__ import annotations
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from src.paddleocr_retrieval_views import evidence_text, module_for


def assess_evidence(case:dict,hits:list[dict])->dict:
    facts=case.get('facts',[]);matched=[];source_hits=[]
    wrong_version=sum(h.get('version')!=case['version'] for h in hits)
    modules={f.get('module') for f in facts if f.get('module')}
    wrong_module=sum(bool(modules) and h.get('module',module_for(h)) not in modules|{'shared'} for h in hits)
    for fact in facts:
        valid=[h for h in hits if h.get('version')==case['version']
               and (not fact.get('module') or h.get('module',module_for(h)) in (fact['module'],'shared'))]
        source_hits.append(any(h.get('document_path') in fact['allowed_paths'] for h in valid))
        matched.append(any(h.get('document_path') in fact['allowed_paths']
                           and all(t in evidence_text(h) for t in fact['tokens'])
                           and not any(t in evidence_text(h) for t in fact.get('forbidden_tokens',[])) for h in valid))
    return {'complete':bool(facts) and all(matched),'fact_hit_count':sum(matched),'fact_count':len(facts),
            'fact_hits':matched,'source_hit_count':sum(source_hits),'wrong_version_count':wrong_version,
            'wrong_module_count':wrong_module,'assessment':'literal_fact_evidence_coverage_not_answer_correctness'}


def assess_review(case:dict,report:dict)->dict:
    risks={f['rule_id'] for f in report.get('findings',[]) if f.get('status')=='supported_risk'}
    expected=set(case.get('expected_rules',[]))
    return {'risk_hit_count':len(expected&risks),'expected_risk_count':len(expected),
            'false_negative_count':len(expected-risks),'false_positive_count':len(risks-expected),
            'gap_expected':bool(case.get('expect_gap')),'gap_present':bool(report.get('gaps')),
            'unknown_marked_compatible':bool(case.get('expect_gap')) and report.get('status')=='unaffected',
            'candidate_path_count':len(report.get('impact_paths',[])),
            'incorrect_relation_level_count':sum(p.get('level')!='candidate' or p.get('runtime_verified') is not False for p in report.get('impact_paths',[])),
            'regression_requirement_count':len(report.get('regression_requirements',[])),
            'runtime_success':None if report.get('runtime_verified') is not True else True}


def aggregate(rows:list[dict])->dict:
    output={'count':len(rows),'complete_count':sum(r['complete'] for r in rows)}
    for key in ('fact_hit_count','fact_count','source_hit_count','wrong_version_count','wrong_module_count'):
        output[key]=sum(r[key] for r in rows)
    output['complete_rate']=output['complete_count']/max(1,output['count'])
    output['fact_recall']=output['fact_hit_count']/max(1,output['fact_count'])
    return output


def promotion_allowed(baseline:dict,candidate:dict)->bool:
    return (candidate['complete_count']>baseline['complete_count']
            and candidate['fact_hit_count']>=baseline['fact_hit_count']
            and candidate['wrong_version_count']<=baseline['wrong_version_count']
            and candidate['wrong_module_count']<=baseline['wrong_module_count'])


def file_hash(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=256)
def _asset_hash(path:str,size:int,mtime:int)->str:
    return file_hash(Path(path))


def validate_frozen_models(freeze:dict)->None:
    for label,env in (('embedding','PADDLEOCR_EMBEDDING_PATH'),('reranker','PADDLEOCR_RERANKER_PATH')):
        record=freeze['models'][label]
        path=Path(os.environ.get(env) or record['local_path']).resolve()
        actual={str(p.relative_to(path)).replace('\\','/'):p for p in path.rglob('*')
                if p.is_file() and p.suffix in ('.json','.txt','.safetensors','.bin') and '.cache' not in p.relative_to(path).parts}
        if not actual or set(actual)!=set(record['files']):raise ValueError('frozen model asset set mismatch')
        for name,p in actual.items():
            stat=p.stat()
            if _asset_hash(str(p),stat.st_size,stat.st_mtime_ns)!=record['files'][name]:
                raise ValueError('frozen model asset identity mismatch')


def _canonical_hit(hit:dict,chunks:dict)->dict:
    if not isinstance(hit,dict) or set(hit)-{'chunk_id','version','module','evidence_spans'}:
        raise ValueError('saved hit contains noncanonical fields')
    original=chunks[hit['chunk_id']]
    if hit['version']!=original['version'] or hit['module']!=module_for(original):
        raise ValueError('saved hit identity mismatch')
    value={**original,'module':module_for(original)}
    if 'evidence_spans' in hit:value['evidence_spans']=hit['evidence_spans']
    evidence_text(value)  # All offsets and text must match the original body.
    return value


def load_impact_release(corpus:Path)->dict|None:
    root=Path(__file__).resolve().parents[2];folder=root/'evaluation/paddleocr_impact_v3'
    try:
        freeze=json.loads((folder/'freeze.json').read_text(encoding='utf-8'))
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        dataset=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
        if (report['schema_version']!=1 or report['dataset_id']!=dataset['dataset_id']
                or report['label_origin']!=dataset['label_origin'] or type(report['top_k']) is not int
                or report['top_k']!=5):return None
        if report['freeze_sha256']!=file_hash(folder/'freeze.json'):return None
        validate_frozen_models(freeze)
        for name,digest in freeze['files'].items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or file_hash(target)!=digest:return None
        if corpus.resolve()!=(root/'versioned-rag-service/public_corpus_paddleocr').resolve():return None
        chunks={r['chunk_id']:r for r in json.loads((corpus/'chunks.json').read_text(encoding='utf-8'))}
        if report['configs']!=freeze['configs']:return None
        marker=json.loads((folder/'sealed-opened.json').read_text(encoding='utf-8'))
        if marker!={'freeze_sha256':report['freeze_sha256'],'selected_before_opening':report['selected_strategy']}:return None
        cases={r['id']:r for r in dataset['retrieval_cases']}
        names=list(freeze['configs']);candidates=[n for n in names if n!='bm25']
        winner=report['development_candidate']
        if winner not in candidates or set(report['retrieval'])!=set(names):return None
        for strategy,splits in report['retrieval'].items():
            expected={'dev','validation','sealed'} if strategy in ('bm25',winner) else {'dev'}
            if set(splits)!=expected:return None
        recomputed={}
        for strategy,splits in report['retrieval'].items():
            recomputed[strategy]={}
            for split,raw in splits.items():
                ids=[c['id'] for c in dataset['retrieval_cases'] if c['split']==split]
                if {r['id'] for r in raw['rows']}!=set(ids) or len(raw['rows'])!=len(ids):return None
                scores=[];candidate_scores=[]
                for row in raw['rows']:
                    hits=[]
                    for hit in row['hits']:
                        hits.append(_canonical_hit(hit,chunks))
                    if len(hits)>report['top_k'] or len({h['chunk_id'] for h in hits})!=len(hits):return None
                    score=assess_evidence(cases[row['id']],hits)
                    if score!=row['score']:return None
                    scores.append(score)
                    if 'candidates' in row:
                        candidate_score=assess_evidence(cases[row['id']],[_canonical_hit(h,chunks) for h in row['candidates']])
                        if candidate_score!=row['candidate_score']:return None
                        candidate_scores.append(candidate_score)
                recomputed[strategy][split]=aggregate(scores)
                if recomputed[strategy][split]!=raw['metrics']:return None
                if candidate_scores and (len(candidate_scores)!=len(scores) or aggregate(candidate_scores)!=raw['candidate_metrics']):return None
        names=list(freeze['configs']);candidates=[n for n in names if n!='bm25']
        if set(recomputed)!=set(names):return None
        winner=max(candidates,key=lambda n:(recomputed[n]['dev']['complete_count'],recomputed[n]['dev']['fact_hit_count'],
                   -recomputed[n]['dev']['wrong_module_count'],-candidates.index(n)))
        if report['development_candidate']!=winner:return None
        allowed=promotion_allowed(recomputed['bm25']['validation'],recomputed[winner]['validation'])
        if report['promotion_passed'] is not allowed or report['selected_strategy']!=(winner if allowed else 'bm25'):return None
        from src.paddleocr_query_plan import plan_query
        probes=[{'id':c['id'],'expected':c['expected'],'scope_status':plan_query(c['query'],versions=(c['version'],))['status'],
                 'generation_status':'NOT_EXECUTED','answer_correctness':None} for c in dataset['negative_probes']]
        if report['negative_probes']!=probes:return None
        review_cases={c['id']:c for c in dataset['review_cases']}
        if len(report['review'])!=len(review_cases) or {r['id'] for r in report['review']}!=set(review_cases):return None
        if review_cases:
            from types import SimpleNamespace
            from src.paddleocr_compatibility import review_compatibility
            index=SimpleNamespace(chunks=list(chunks.values()),manifest=json.loads((corpus/'corpus_manifest.json').read_text(encoding='utf-8')))
            for row in report['review']:
                case=review_cases[row['id']]
                original=review_compatibility(index,files=case['files'],source_version='v2.9.1',target_version='v3.0.0')
                if row['report']!=original or row['split']!=case['split']:return None
                for stage,body in (('A_static',{**original,'impact_paths':[]}),('B_graph',original),('C_rag',original)):
                    if row[stage]!=assess_review(case,body):return None
                if row['D_model'].get('status')!='NOT_EXECUTED':return None
                for hit in row['investigation']['checked_evidence']:
                    canonical=chunks[hit['chunk_id']]
                    if any(hit.get(k)!=canonical.get(k) for k in ('content','version','source_id','source_sha256','document_path')):return None
        if any(report.get(k) is not None for k in ('answer_accuracy','hallucination_rate','business_impact_accuracy')):return None
        return report
    except (OSError,ValueError,KeyError,TypeError):return None
