"""Exercise actual local Git commits, shared review, re-review and record binding.

Creates an isolated original-demo repository under --output. Does not execute
uploaded code or OCR models. Fixture checks are explicitly not runtime OCR proof.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'change-review-agent'), str(ROOT/'demo-ui')]
from services.public_knowledge_client import PublicKnowledgeClient
from services.registered_application import load_registry, collect_snapshot
from services.registered_review import run_registered_review
from services.interview_examples import DEMO_CASES, demo_application
from services.configuration_workflow import configuration_demo_inputs, run_configuration_review
from services.paddleocr_disposition import regression_template, validate_regression
from app.paddleocr_review import PaddleOCRReviewAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    repo = output/'original-demo-git'
    if repo.exists(): raise RuntimeError('Choose a new output directory; previous evidence must not be overwritten.')
    repo.mkdir()
    def git(*args):
        result = subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Original demo validation',
                                 '-c', 'user.email=demo@example.invalid', *args],
                                 check=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    git('init')
    app = {**load_registry()['document-ocr'], 'repository': str(repo), 'root': 'app'}
    def commit(files, message):
        target = repo/'app'
        previous = {p for p in target.rglob('*') if p.is_file()} if target.exists() else set()
        current = set()
        for file in files:
            path = target/file['path']; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(file['content'], encoding='utf-8', newline='\n'); current.add(path)
        for removed in previous-current: removed.unlink()
        git('add', 'app'); git('commit', '-m', message)
        return git('rev-parse', 'HEAD')
    client = PublicKnowledgeClient(args.url)
    context = {'application_name': '扫描文档处理应用', 'source_identity': 'student_poc_application',
               'business_scope': 'internal_document_processing'}
    def analyze(gateway, files, **kw): return PaddleOCRReviewAgent(gateway).analyze(files, **kw)
    baseline = None; rows = []
    for label in DEMO_CASES:
        sha = commit(demo_application(label), label)
        snapshot = collect_snapshot(app, sha, base_commit=baseline['head_commit'] if baseline else None)
        result = run_registered_review(client, snapshot, analyze=analyze, context=context,
            base_snapshot=baseline, source_version='v2.9.1', target_version='v3.0.0', generate_advice=False)
        if baseline is None: baseline = snapshot
        record = regression_template(result)
        record.update(environment='未执行 OCR；记录绑定检查夹具', actual_dependency='v3.0.0', execution_kind='fixture')
        parsed = validate_regression(result, record)
        changed = copy.deepcopy(result); changed['application_snapshot']['head_commit'] = baseline['head_commit']
        rejected = None
        if changed['application_snapshot']['head_commit'] != sha:
            try: validate_regression(changed, record)
            except ValueError: rejected = True
        rows.append({'case': label, 'commit': sha, 'result': result, 'record_status': parsed['record_status'],
                     'old_record_rejected_for_other_commit': rejected})
        print(json.dumps({'case': label, 'commit': sha[:12], 'risks': result['compatibility_report']['summary']['risk_count'],
                          'gaps': len(result['compatibility_report']['gaps']), 'runtime_verified': result['runtime_verified']}, ensure_ascii=False), flush=True)
    before, after = configuration_demo_inputs()
    before_sha=commit(before,'配置基准 drop_score=0.5')
    base_config=collect_snapshot(app,before_sha)
    after_sha=commit(after,'配置候选 drop_score=0.8')
    config_snapshot=collect_snapshot(app,after_sha,base_commit=before_sha)
    config=run_configuration_review(client,base_config['files'],config_snapshot['files'],
        snapshot=config_snapshot,base_snapshot=base_config)
    assert config['source_version']==config['target_version']=='v2.9.1'
    assert any(f['status']=='behavior_change' for f in config['compatibility_report']['findings'])
    config_record=regression_template(config)
    config_record.update(environment='未运行 OCR；配置记录绑定检查夹具',actual_dependency='v2.9.1',execution_kind='fixture')
    config_record_status=validate_regression(config,config_record)['record_status']
    # A developer-selected revision is another real commit, not an automatic fix.
    revised=copy.deepcopy(after)
    for file in revised:
        if file['path']=='application/ocr_client.py':
            file['content']=file['content'].replace('drop_score=0.8','drop_score=0.6')
    revised_sha=commit(revised,'开发者调整候选阈值为 0.6；仍待真实回归')
    revised_snapshot=collect_snapshot(app,revised_sha,base_commit=before_sha)
    revised_result=run_configuration_review(client,before,revised,snapshot=revised_snapshot,base_snapshot=base_config)
    try:validate_regression(revised_result,config_record)
    except ValueError:old_config_record_rejected=True
    else:raise AssertionError('Old configuration regression must not validate another commit')
    configuration_closure={'baseline_commit':before_sha,'candidate_commit':after_sha,'revised_commit':revised_sha,
        'initial':config,'recheck':revised_result,'record_status':config_record_status,
        'old_record_rejected_for_other_commit':old_config_record_rejected,
        'runtime_verified':False,'human_decision':'PENDING','actual_ocr':'NOT_EXECUTED'}
    json_before,json_after=configuration_demo_inputs(style='json')
    json_base_sha=commit(json_before,'JSON配置基准 0.5；未运行应用')
    json_base=collect_snapshot(app,json_base_sha)
    json_head_sha=commit(json_after,'JSON配置候选 0.8；未运行应用')
    json_head=collect_snapshot(app,json_head_sha,base_commit=json_base_sha)
    json_review=run_configuration_review(client,json_before,json_after,snapshot=json_head,base_snapshot=json_base)
    assert any(f['status']=='behavior_change' for f in json_review['compatibility_report']['findings'])
    (output/'review-results.json').write_text(json.dumps({'upgrade_cases': rows, 'configuration': config,
        'configuration_closure':configuration_closure,
        'json_configuration':{'baseline_commit':json_base_sha,'candidate_commit':json_head_sha,'result':json_review,
                              'configuration_loading_verified':False},
        'limitation': '真实本地 Git 快照及静态链路；未运行 OCR，未认证企业 CI。'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'workflow':'configuration_change','baseline':before_sha,'candidate':after_sha,
        'revised':revised_sha,'stale_record_rejected':old_config_record_rejected,'runtime_verified':False},ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
