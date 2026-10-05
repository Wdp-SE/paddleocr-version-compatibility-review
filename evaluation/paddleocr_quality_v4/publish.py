"""Publish only frozen, source-recomputed evidence results after holdout execution."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).parent/'final'
sys.path.insert(0,str(ROOT/'versioned-rag-service'))


def read(name):
    return json.loads((FOLDER/name).read_text(encoding='utf8'))


def write(name,value):
    path=FOLDER/name
    if path.exists():raise ValueError('Preserve prior published artifacts')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    selection=read('selection.json')
    frozen=read('freeze.json')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in frozen['files'].items()):
        raise ValueError('Frozen implementation changed')
    chosen=selection['selected']
    if chosen not in ('bm25','llm'):raise ValueError('Winner needs an explicitly supported release configuration')
    retrieval={name:{split:{'metrics':read(f'{name}-{split}.json')['metrics']}
                     for split in ('dev','holdout')} for name in ('legacy','bm25','llm')}
    base=retrieval['legacy']['holdout']['metrics'];scoped=retrieval['bm25']['holdout']['metrics']
    proposed=retrieval[chosen]['holdout']['metrics']
    candidate_gate=(proposed['complete_count']>=scoped['complete_count'] and
                    proposed['fact_hit_count']>=scoped['fact_hit_count'] and proposed['wrong_version_count']==0)
    if not candidate_gate:chosen='bm25'
    current=retrieval[chosen]['holdout']['metrics']
    gate=(current['complete_count']>=base['complete_count'] and current['fact_hit_count']>=base['fact_hit_count']
          and current['wrong_version_count']==0 and current['wrong_module_count']<=base['wrong_module_count'])
    if not gate:raise ValueError('Holdout gate failed; do not promote or retune holdout')
    limitations=('开发者依据固定原文标注的 12 条开发题和 8 条按产线资料分组留出题；'
                 '不是独立业务专家盲评。指标为 Top-5 原文事实覆盖，不是答案正确率、幻觉率或生产准确率。'
                 '本轮没有根据留出结果调整策略；模型排序会变化，版本过滤和引用核验仍是硬约束。')
    write('release.json',{'schema_version':1,'selected_strategy':'contextual_llm_rerank' if chosen=='llm' else 'contextual_bm25',
        'top_k':5,'candidate_budget':40,'retrieval':retrieval,'promotion_passed':gate,
        'development_candidate':selection['selected'],'candidate_promotion_passed':candidate_gate,
        'limitations':limitations,'deployment_status':'local_verified_not_publicly_deployed',
        'selection_record':'evaluation/paddleocr_quality_v4/final/selection.json'})
    names=set(frozen['files'])|{str((FOLDER/name).relative_to(ROOT)).replace('\\','/') for name in (
        'release.json','freeze.json','dataset.json','selection.json','promotion-rules.json','live-report.json',
        'legacy-dev.json','legacy-holdout.json','bm25-dev.json','bm25-holdout.json','llm-dev.json','llm-holdout.json',
        'minilm-dev.json','bge-dev.json','rrf-dev.json')}
    names.add('evaluation/paddleocr_quality_v4/publish.py')
    # Registry, source files and runtime metadata must not silently drift while
    # the UI still advertises a result measured on the prior knowledge space.
    names|={str(path.relative_to(ROOT)).replace('\\','/') for path in
            (ROOT/'versioned-rag-service/public_corpus_paddleocr').rglob('*') if path.is_file()}
    names.add('versioned-rag-service/src/paddleocr_quality.py')
    write('release-lock.json',{'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}})
    from src.paddleocr_rag_release import load_rag_release
    result=load_rag_release(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    if result is None:raise ValueError('Canonical release validation failed')
    print(json.dumps({'selected':result['selected_strategy'],'holdout':current},ensure_ascii=False))


if __name__=='__main__':main()
