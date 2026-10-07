"""Publish only a drift-free, canonically rechecked quality candidate."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('internal_eval',Path(__file__).with_name('run.py'))
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def main():
    experiment.verify()
    folder=experiment.FOLDER
    selection=experiment.read('selection.json')
    selected=selection['selected'];cfg=experiment.CONFIGS[selected]
    baseline=experiment.read('legacy-holdout.json')['metrics']
    measured=experiment.read(selected+'-holdout.json')
    current=measured['metrics']
    passed=(selection['validation_gate_passed'] and selected!='legacy'
            and current['complete_count']>=baseline['complete_count']
            and current['wrong_version_count']==0
            and current['wrong_module_count']<=baseline['wrong_module_count'])
    if selected=='bge_rerank' and not experiment.neural_execution_succeeded(measured['rows']):passed=False
    if not passed:
        experiment.write('publication-status.json',{'promoted':False,'reason':'validation/holdout safety gate failed; retain service baseline, do not repick on holdout'})
        print('NOT_PROMOTED');return
    retrieval={}
    for name in experiment.CONFIGS:
        for split in ('dev','validation','holdout'):
            path=folder/f'{name}-{split}.json'
            if path.is_file():
                report=json.loads(path.read_text(encoding='utf8'))
                fallbacks=sum(bool(r['diagnostics'].get('fallback_reason')) for r in report['rows'])
                reranks=sum(r['diagnostics'].get('rerank_calls')==1 for r in report['rows'])
                execution=f'重排 {reranks}/{len(report["rows"])}；回退 {fallbacks}' if name=='bge_rerank' else '词法检索'
                retrieval.setdefault(name,{})[split]={'metrics':report['metrics'],'execution':execution}
    experiment.write('release.json',{'schema_version':1,'dataset_id':'internal-workflow-v1-grouped-regression',
        'selected':selected,'selected_strategy':cfg['strategy'],
        **{k:cfg[k] for k in ('top_k','window_budget','candidate_budget','use_context')},
        'retrieval':retrieval,
        'limitations':'48 条开发者原文标注查询：24 开发 / 12 验证 / 12 分组回归留出；其中先前 28 条已打开，不是独立盲测。证据覆盖不等于回答准确率。BGE 开发只有 22/24 条成功重排，2 条无效输出后回退，混合成绩不作为稳定神经成绩；已排除默认选型，后续神经验证未完成。没有生产准确率或专家评审结论。'})
    files=dict(experiment.read('freeze.json')['files'])
    for path in folder.iterdir():
        if path.is_file() and path.name not in ('release-lock.json','publication-status.json'):
            files[str(path.relative_to(ROOT)).replace('\\','/')]=experiment.digest(path)
    experiment.write('release-lock.json',{'schema_version':1,'files':files})
    from src.paddleocr_internal_release import load_internal_release
    if load_internal_release(ROOT/'versioned-rag-service/public_corpus_paddleocr') is None:
        raise ValueError('release failed canonical verification; never advertise as current')
    experiment.write('publication-status.json',{'promoted':True,'selected':selected})
    print('CANONICALLY_VERIFIED',selected)


if __name__=='__main__':main()
