"""Bind human dispositions and external logs to one immutable static report.

Submitted logs are assertions by an unauthenticated demo user, never evidence
that this service executed an application or approved a production upgrade.
"""
import hashlib
import json


def _report(result):
    return result.get('compatibility_report') or {}


def binding(result):
    payload = {'report': result.get('report_fingerprint'),
               'workflow_type': result.get('workflow_type', 'dependency_upgrade'),
               'baseline_snapshot': result.get('baseline_application_snapshot'),
               'request': result.get('request_fingerprint'),
               'context': result.get('application_context'),
               'snapshot': result.get('application_snapshot'),
               'source_version': result.get('source_version'),
               'target_version': result.get('target_version'),
               'files': sorted((row['path'], row['sha256']) for row in _report(result).get('files', []))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def gap_id(gap):
    return hashlib.sha256(json.dumps(gap, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def regression_template(result):
    return {'schema_version': 1, 'report_binding': binding(result),
            'source_version': result.get('source_version'), 'target_version': result.get('target_version'),
            'application_id': (result.get('application_snapshot') or {}).get('application_id'),
            'application_commit': (result.get('application_snapshot') or {}).get('head_commit'),
            'actual_dependency': '', 'environment': '', 'execution_kind': 'developer_runtime',
            'application_files': _report(result).get('files', []),
            'results': [{'finding_id': row['finding_id'], 'input_conditions': row['input_conditions'],
                         'checks': [{'name': name, 'passed': False} for name in row['checks']],
                         'log_excerpt': ''} for row in _report(result).get('regression_requirements', [])]}


def validate_regression(result, data):
    if not isinstance(data, dict) or len(json.dumps(data, ensure_ascii=False).encode()) > 200_000:
        raise ValueError('回归记录必须是小于 200 KB 的 JSON 对象。')
    expected = regression_template(result)
    for key in ('application_id', 'application_commit'):
        if data.get(key) != expected.get(key):
            raise ValueError('回归记录的应用或提交不一致；请在当前候选提交上重新执行回归。')
    for key in ('schema_version', 'report_binding', 'source_version', 'target_version', 'application_files'):
        if data.get(key) != expected[key]:
            raise ValueError('回归记录与当前报告、版本或应用文件绑定不一致，请重新审查后导出模板。')
    if data.get('actual_dependency') != result.get('target_version'):
        raise ValueError('实际执行依赖必须与升级目标一致。')
    execution_kind = data.get('execution_kind', 'developer_runtime')
    if execution_kind not in ('developer_runtime', 'fixture'):
        raise ValueError('公网导入仅支持人工运行记录或夹具；不接受自称可信 CI 的认证。')
    if not isinstance(data.get('environment'), str) or not data['environment'].strip() or len(data['environment']) > 2000:
        raise ValueError('请填写实际测试环境。')
    rows = data.get('results')
    allowed = {row['finding_id'] for row in _report(result).get('regression_requirements', [])}
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError('回归结果最多 100 项。')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get('finding_id') not in allowed or row['finding_id'] in seen:
            raise ValueError('回归结果包含未知或重复核查项。')
        seen.add(row['finding_id'])
        if not isinstance(row.get('input_conditions'), list) or not all(isinstance(x, str) for x in row['input_conditions']):
            raise ValueError('请填写输入条件列表。')
        checks = row.get('checks')
        if not isinstance(checks, list) or len(checks) > 30:
            raise ValueError('核查结果必须是有界列表。')
        names = set()
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get('name'), str) or type(check.get('passed')) is not bool or check['name'] in names:
                raise ValueError('每项核查须有唯一名称及布尔 passed 值。')
            names.add(check['name'])
        if not isinstance(row.get('log_excerpt'), str) or len(row['log_excerpt']) > 10_000:
            raise ValueError('每项日志摘要必须是最多 10000 字的文本。')
    # Do not preserve caller-controlled verification/approval fields.
    return {**expected, 'environment': data['environment'], 'actual_dependency': data['actual_dependency'],
            'results': rows, 'execution_kind': execution_kind,
            'record_status': 'FIXTURE_ONLY' if execution_kind == 'fixture' else 'HUMAN_SUBMITTED_UNVERIFIED',
            'runtime_verified': False}


def assess_disposition(result, risks, gaps, regression):
    blockers = []
    if result.get('input_coverage', {}).get('status') == 'INCOMPLETE':
        blockers.append('登记范围内输入不完整，必须补齐并重新审查；不能仅以接受残余风险代替。')
    report = _report(result)
    findings = report.get('findings', [])
    if not findings:
        blockers.append('没有可处置的静态发现；不能据此认可变更。')
    if regression:
        try:
            regression = validate_regression(result, regression)
        except ValueError:
            regression = None
            blockers.append('外部回归记录已失效。')
    results = {row['finding_id']: row for row in (regression or {}).get('results', [])}
    if regression and regression.get('execution_kind') == 'fixture':
        blockers.append('夹具仅验证程序逻辑，不能替代实际 OCR 运行回归。')
    requirements = {row['finding_id']: row for row in report.get('regression_requirements', [])}
    for finding in findings:
        fid = finding['finding_id']
        if finding.get('status') == 'needs_verification':
            blockers.append(f'{fid}：静态分析未完成，需要补齐输入重新审查。')
        if finding.get('status') == 'supported_risk':
            row = risks.get(fid, {})
            if row.get('decision') not in ('mitigated', 'not_applicable') or not str(row.get('reason', '')).strip():
                blockers.append(f'{fid}：风险尚未记录处置理由。')
        if finding.get('status') == 'behavior_change':
            row = risks.get(fid, {})
            if row.get('decision') != 'accepted_change' or not str(row.get('reason', '')).strip():
                blockers.append(f'{fid}：需人工确认配置行为变化符合预期，并填写依据。')
        required = requirements.get(fid)
        actual = results.get(fid, {})
        passed = {row['name'] for row in actual.get('checks', []) if row['passed'] is True}
        if not required or not set(required['checks']).issubset(passed) or not set(required['input_conditions']).issubset(actual.get('input_conditions', [])) or not actual.get('log_excerpt', '').strip():
            blockers.append(f'{fid}：缺少覆盖输入条件与契约的外部回归记录。')
    all_gaps = [*report.get('gaps', []), *(result.get('investigation') or {}).get('unresolved', [])]
    for fid, required in requirements.items():
        if required.get('kind') != 'application_contract':
            continue
        actual = results.get(fid, {})
        passed = {row['name'] for row in actual.get('checks', []) if row['passed'] is True}
        if (not set(required['checks']).issubset(passed)
                or not set(required['input_conditions']).issubset(actual.get('input_conditions', []))
                or not actual.get('log_excerpt', '').strip()):
            blockers.append(f'{fid}：应用输出契约尚无覆盖完整条件的外部回归记录。')
    for gap in all_gaps:
        if gap.get('blocking') is False:
            continue
        row = gaps.get(gap_id(gap), {})
        if row.get('decision') != 'accepted_residual' or not str(row.get('reason', '')).strip():
            blockers.append(f'{gap_id(gap)}：缺口尚未由人记录残余风险。')
    return {'report_binding': binding(result), 'eligible_for_human_continuation': not blockers,
            'blockers': blockers, 'risk_dispositions': risks, 'gap_dispositions': gaps,
            'external_regression': regression, 'runtime_verified': False,
            'decision_scope': '人工决定是否继续变更流程；不是自动批准，也未验证提交者身份或日志真实性'}
