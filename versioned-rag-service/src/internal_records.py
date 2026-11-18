"""Authenticated CI provenance and human dispositions; neither approves deployment."""
import re
import time
from src.internal_tasks import fingerprint
from src.internal_workflows import workflow_versions


def _hash(value):return isinstance(value,str) and bool(re.fullmatch('[a-f0-9]{64}',value))


def _regression_checks(task):
    checks = {}
    for requirement in task['result'].get('compatibility_report', {}).get('regression_requirements', []):
        finding = requirement['finding_id']
        conditions = [f'输入场景 · {label}' for label in requirement.get('input_conditions', [])]
        for label in [*requirement['checks'], *conditions]:
            check_id = fingerprint([finding, label])
            checks[check_id] = {'check_id': check_id, 'finding_id': finding, 'label': label}
    return list(checks.values())


def _current_ci_runs(task, records):
    _, _, target = workflow_versions(task['request'])
    groups = {}
    for record in records:
        row = record['payload']
        if (record['kind'] != 'CI' or row.get('provenance') != 'trusted_ci_identity'
                or row.get('application_commit') != task['request']['snapshot']['head_commit']
                or row.get('report_fingerprint') != task['result']['report_fingerprint']
                or row.get('environment', {}).get('paddleocr') != target.removeprefix('v')
                or not re.fullmatch('[0-9]{1,20}', str(row.get('run_id', '')))
                or type(row.get('run_attempt')) is not int or type(row.get('run_number')) is not int
                or not isinstance(row.get('caller_workflow_ref'), str)):
            continue
        group = row['caller_workflow_ref'].split('@', 1)[0]
        order = (row['run_number'], row['run_attempt'])
        latest, runs = groups.get(group, (order, []))
        if order >= latest:
            groups[group] = (order, [*runs, row] if order == latest else [row])
    return [run for _, runs in groups.values() for run in runs]


def regression_coverage(task, records):
    """Join report requirements with results from the latest attempt per workflow."""
    outcomes = {}
    rank = {'SKIPPED': 0, 'FIXTURE_PASSED': 1, 'PASSED': 2, 'FAILED': 3}
    for run in _current_ci_runs(task, records):
        for result in run.get('check_results', []):
            status = result['status']
            if status == 'PASSED' and result['execution_kind'] == 'FIXTURE':
                status = 'FIXTURE_PASSED'
            row = {**result, 'status': status, 'run_id': run['run_id'], 'run_attempt': run['run_attempt']}
            prior = outcomes.get(result['check_id'])
            if prior is None or rank[status] > rank[prior['status']]:
                outcomes[result['check_id']] = row
    return [{**check, **outcomes.get(check['check_id'], {'status': 'NOT_RUN'})}
            for check in _regression_checks(task)]


def validate_ci_record(task,payload,claims,application):
    _,_,target=workflow_versions(task['request'])
    policy=application.get('ci',{})
    snapshot=task['request']['snapshot']
    required = {'application_commit','report_fingerprint','run_id','status','input_sha256','environment'}
    if (task['state']!='DONE' or not policy or not required <= set(payload) <= required | {'check_results'}
            or claims.get('repository')!=policy.get('repository')
            or claims.get('job_workflow_ref')!=policy.get('workflow_ref')
            or not re.fullmatch('[a-f0-9]{40}',policy.get('workflow_sha',''))
            or claims.get('job_workflow_sha')!=policy['workflow_sha']
            or claims.get('sha')!=snapshot['head_commit']
            or payload['application_commit']!=snapshot['head_commit']
            or payload['report_fingerprint']!=task['result'].get('report_fingerprint')
            or not _hash(payload['report_fingerprint'])
            or str(claims.get('run_id',''))!=payload['run_id'] or not re.fullmatch('[0-9]{1,20}',payload['run_id'])
            or not re.fullmatch('[1-9][0-9]{0,5}',str(claims.get('run_attempt','')))
            or not re.fullmatch('[1-9][0-9]{0,19}',str(claims.get('run_number','')))
            or not re.fullmatch(re.escape(claims['repository'])+r'/\.github/workflows/[^/@]+\.ya?ml@.+',str(claims.get('workflow_ref','')))
            or payload['status'] not in ('PASSED','FAILED')):
        raise ValueError('CI identity or task mismatch')
    expected={r['path']:r['sha256'] for r in snapshot['inventory']}
    if payload['input_sha256']!=expected or not any('test' in p for p in expected) or not any('contract' in p for p in expected):
        raise ValueError('CI input snapshot incomplete')
    environment=payload['environment']
    if (not isinstance(environment,dict) or set(environment)!={'paddleocr','paddlepaddle','python','runner_os','model_sha256','container_digest'}
            or any(not isinstance(v,str) or not 1<=len(v)<=128 for v in environment.values())
            or environment['paddleocr']!=target.removeprefix('v') or not _hash(environment['model_sha256'])
            or not re.fullmatch('sha256:[a-f0-9]{64}',environment['container_digest'])):
        raise ValueError('Actual runtime identity missing')
    if 'check_results' in payload:
        results = payload['check_results']
        known = {row['check_id'] for row in _regression_checks(task)}
        if not isinstance(results, list) or len(results) > min(500, len(known)):
            raise ValueError('Invalid regression check results')
        seen = set()
        for result in results:
            if (not isinstance(result, dict)
                    or set(result) != {'check_id', 'status', 'execution_kind', 'artifact_sha256'}
                    or not isinstance(result['check_id'], str) or result['check_id'] not in known
                    or result['check_id'] in seen
                    or result['status'] not in ('PASSED', 'FAILED', 'SKIPPED')
                    or result['execution_kind'] not in ('REAL_OCR', 'FIXTURE')
                    or not _hash(result['artifact_sha256'])
                    or payload['status'] == 'PASSED' and result['status'] == 'FAILED'):
                raise ValueError('Regression check result does not bind a current requirement')
            seen.add(result['check_id'])
    return {**payload,'repository':claims['repository'],'workflow_sha':claims['job_workflow_sha'],
            'run_attempt':int(claims['run_attempt']),
            'run_number':int(claims['run_number']),'caller_workflow_ref':claims['workflow_ref'],
            'provenance':'trusted_ci_identity','upgrade_approved':False,
            'notice':'Authenticated runner declaration; does not establish test completeness or automatic compatibility approval.'}


def validate_decision(task,payload,actor,records):
    allowed={'decision','report_fingerprint','reason','accepted_risks','expires_at'}
    if (task['state']!='DONE' or not isinstance(payload,dict) or set(payload)-allowed
            or payload.get('decision') not in ('ACKNOWLEDGED','DEFERRED','CONTINUE_REVIEW')
            or payload.get('report_fingerprint')!=task['result'].get('report_fingerprint')
            or not isinstance(payload.get('reason'),str) or not 10<=len(payload['reason'].strip())<=2000):
        raise ValueError('Decision does not bind a current report')
    accepted=payload.get('accepted_risks',[])
    if not isinstance(accepted,list) or len(accepted)>100:raise ValueError('Invalid accepted risks')
    risks={f['finding_id'] for f in task['result'].get('compatibility_report',{}).get('findings',[])
           if f.get('status') in ('supported_risk','behavior_change')}
    if any(not isinstance(r,dict) or set(r)!={'finding_id','reason'} or not isinstance(r['finding_id'],str) or r['finding_id'] not in risks
           or not isinstance(r['reason'],str) or not 10<=len(r['reason'])<=1000 for r in accepted):
        raise ValueError('Risk acceptance requires a known finding and explanation')
    if payload['decision']=='CONTINUE_REVIEW':
        now=time.time();expires=payload.get('expires_at')
        if (actor==task['owner'] or type(expires) not in (int,float) or not now<expires<=now+30*86400
                or {r['finding_id'] for r in accepted}!=risks):
            raise ValueError('Independent reviewer and bounded residual-risk acceptance required')
        latest = _current_ci_runs(task, records)
        if not latest or any(run['status'] != 'PASSED' for run in latest):
            raise ValueError('Latest received target CI run and attempt must pass')
    return {**payload,'actor':actor,'application_commit':task['request']['snapshot']['head_commit'],
            'upgrade_approved':False,'scope':'human review record only; organization release process remains authoritative'}
