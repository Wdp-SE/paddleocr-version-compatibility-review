"""Select on known/validation sets before opening sealed questions; never overwrite."""
import argparse
import json
import importlib.metadata
from pathlib import Path
from run import ROOT,FOLDER,CONFIGS,sha,save


def select(folder):
    reports={n:{s:json.loads((folder/f'{n}-{s}.json').read_text(encoding='utf8'))
                for s in ('regression','validation') if (folder/f'{n}-{s}.json').is_file()} for n in CONFIGS}
    base=reports['window_bm25']
    eligible=[n for n in CONFIGS if set(reports[n])=={'regression','validation'} and all(
        reports[n][s]['evaluation_status']=='complete' and reports[n][s]['metrics']['wrong_version_count']==0
        and reports[n][s]['execution_success_count']==reports[n][s]['metrics']['count']
        and reports[n][s]['metrics']['complete_count']>=base[s]['metrics']['complete_count']
        for s in ('regression','validation'))]
    def key(n):
        m=[reports[n][s]['metrics'] for s in ('regression','validation')]
        return (min(x['complete_rate'] for x in m),sum(x['complete_rate'] for x in m),
                -sum(x['wrong_module_count'] for x in m),-list(CONFIGS).index(n))
    winner=max(eligible,key=key)
    save(folder/'selection.json',{'selected_before_sealed':winner,'freeze_sha256':sha(folder/'freeze.json'),
        'rule':'No fallback or wrong version; no decline against incumbent in either set. Maximize worst-set coverage, then mean coverage; noise and simplicity break ties.'})
    print('SELECTED',winner,flush=True)


def publish(folder):
    folder=folder.resolve()
    selection=json.loads((folder/'selection.json').read_text(encoding='utf8'));winner=selection['selected_before_sealed']
    freeze=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
    retrieval={};rows=[];failures={}
    disqualified={}
    for name in CONFIGS:
        splits=[s for s in ('regression','validation','sealed') if (folder/f'{name}-{s}.json').is_file()]
        retrieval[name]={}
        for split in splits:
            report=json.loads((folder/f'{name}-{split}.json').read_text(encoding='utf8'))
            retrieval[name][split]={'metrics':report['metrics'],'execution_success_count':report['execution_success_count'],
                'evaluation_status':report['evaluation_status'],'planned_count':report['planned_count'],
                'execution':f"实际策略执行 {report['execution_success_count']}/{report['metrics']['count']}"}
            if report['evaluation_status']!='complete':
                disqualified[name]={'split':split,'observed':len(report['rows']),'planned':report['planned_count'],
                    'reason':report['rows'][-1]['diagnostics'].get('fallback_reason'),
                    'quality_comparison':'提前中止，不提供全量质量指标，不纳入默认选型'}
            failures[f'{name}/{split}']=[{'id':r['id'],'stage':r['failure_stage'],
                'fallback_reason':r['diagnostics'].get('fallback_reason')} for r in report['rows'] if not r['score']['complete'] or r['diagnostics'].get('fallback_reason')]
    sealed=retrieval[winner]['sealed'];base=retrieval['window_bm25']['sealed']
    promoted=(sealed['metrics']['complete_count']>=base['metrics']['complete_count'] and not sealed['metrics']['wrong_version_count']
              and sealed['execution_success_count']==sealed['metrics']['count'])
    save(folder/'dataset.json',json.loads((FOLDER/'dataset.json').read_text(encoding='utf8')))
    save(folder/'release.json',{'schema_version':1,'selected':winner,**CONFIGS[winner],
        'selected_strategy':CONFIGS[winner]['strategy'],'window_budget':384,'use_context':True,
        'retrieval':retrieval,'selection_reason':'按完整证据覆盖优先选型，执行回退和错版本不得算作成功。'+('首次封存题未低于窗口基线，允许启用。' if promoted else '首次封存题未通过提升门槛，保留原默认。'),
        'conditions':{'top_k':5,'same_corpus':True,'known_regression':48,'new_validation':12,'first_frozen':12,
            'sealed_status':'原首次封存题在补查链路修复后复验，已不属于未见题；未据其更换问题、策略或参数。首次结果保存在 first-sealed-history.json。',
            'runtime_versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','sentence-transformers','numpy')},
            'cpu_kernel':'native_serial_v2：单线程，关闭 MKLDNN，eager attention；此前不稳定内核的成绩作废',
            'candidate_budget':{'window':40,'structured':80},'structured_unit':'完整参数行、段落和代码示例；原始字符位置引用',
            'latency':'本地 CPU，模型初始化不计；首个语义问题可能含文档向量计算。不是公网 P95 或企业 SLA。'},
        'models':freeze['models'],'disqualified':disqualified,'failures':failures,'generation_evaluation':None,
        'rerank_readiness':{'status':'EVALUATED' if all(n not in disqualified and all(s in retrieval[n] for s in ('regression','validation')) for n in ('structure_bm25_rerank','structure_hybrid_rerank')) else 'NOT_QUALIFIED',
            'reason':'前期 CPU 优化内核出现非有限输出和数值漂移；改用串行原生内核重新冻结测试。仅完整运行且无回退的结果参与选型，部分结果不作质量对照。'},
        'limitations':'开发者编写并按原文标注；48 题为已知回归，新增 24 题共享已收录手册。首次封存不等于未见文档或企业专家盲测。指标为字面事实完整证据覆盖，不是回答准确率。不同候选数是方案组合对照，不能将变化全部归因于 embedding。模型回退单列；不宣称企业用户收益。'})
    files=set(freeze['files'])|{p.relative_to(ROOT).as_posix() for p in folder.iterdir() if p.is_file()}
    save(folder/'release-lock.json',{'files':{p:sha(ROOT/p) for p in sorted(files)}})
    print('PUBLISHED',winner,'promotion',promoted,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','publish']);p.add_argument('--folder',type=Path,default=FOLDER/'final')
    args=p.parse_args();(select if args.action=='select' else publish)(args.folder)
