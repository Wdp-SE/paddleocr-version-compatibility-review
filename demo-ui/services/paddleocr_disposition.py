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
               'request': result.get('request_fingerprint'),
               'context': result.get('application_context'),
               'source_version': result.get('source_version'),
               'target_version': result.get('target_version'),
               'files': sorted((row['path'], row['sha256']) for row in _report(result).get('files', []))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def gap_id(gap):
    return hashlib.sha256(json.dumps(gap, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def regression_template(result):
    return {'schema_version': 1, 'report_binding': binding(result),
            'source_version': result.get('source_version'), 'target_version': result.get('target_version'),
            'actual_dependency': '', 'environment': '',
            'application_files': _report(result).get('files', []),
            'results': [{'finding_id': row['finding_id'], 'input_conditions': row['input_conditions'],
                         'checks': [{'name': name, 'passed': False} for name in row['checks']],
                         'log_excerpt': ''} for row in _report(result).get('regression_requirements', [])]}


def validate_regression(result, data):
    if not isinstance(data, dict) or len(json.dumps(data, ensure_ascii=False).encode()) > 200_000:
        raise ValueError('回归记录必须是小于 200 KB 的 JSON 对象。')
    expected = regression_template(result)
    for key in ('schema_version', 'report_binding', 'source_version', 'target_version', 'application_files'):
        if data.get(key) != expected[key]:
            raise ValueError('回归记录与当前报告、版本或应用文件绑定不一致，请重新审查后导出模板。')
    if data.get('actual_dependency') != result.get('target_version'):
        raise ValueError('实际执行依赖必须与升级目标一致。')
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
            'results': rows, 'record_status': 'HUMAN_SUBMITTED_UNVERIFIED', 'runtime_verified': False}


def assess_disposition(result, risks, gaps, regression):
    blockers = []
    report = _report(result)
    findings = report.get('findings', [])
    if not findings:
        blockers.append('没有可处置的静态发现；不能据此认定升级兼容。')
    if regression:
        try:
            regression = validate_regression(result, regression)
        except ValueError:
            regression = None
            blockers.append('外部回归记录已失效。')
    results = {row['finding_id']: row for row in (regression or {}).get('results', [])}
    requirements = {row['finding_id']: row for row in report.get('regression_requirements', [])}
    for finding in findings:
        fid = finding['finding_id']
        if finding.get('status') == 'needs_verification':
            blockers.append(f'{fid}：静态分析未完成，需要补齐输入重新审查。')
        if finding.get('status') == 'supported_risk':
            row = risks.get(fid, {})
            if row.get('decision') not in ('mitigated', 'not_applicable') or not str(row.get('reason', '')).strip():
                blockers.append(f'{fid}：风险尚未记录处置理由。')
        required = requirements.get(fid)
        actual = results.get(fid, {})
        passed = {row['name'] for row in actual.get('checks', []) if row['passed'] is True}
        if not required or not set(required['checks']).issubset(passed) or not set(required['input_conditions']).issubset(actual.get('input_conditions', [])) or not actual.get('log_excerpt', '').strip():
            blockers.append(f'{fid}：缺少覆盖输入条件与契约的外部回归记录。')
    all_gaps = [*report.get('gaps', []), *(result.get('investigation') or {}).get('unresolved', [])]
    for gap in all_gaps:
        row = gaps.get(gap_id(gap), {})
        if row.get('decision') != 'accepted_residual' or not str(row.get('reason', '')).strip():
            blockers.append(f'{gap_id(gap)}：缺口尚未由人记录残余风险。')
    return {'report_binding': binding(result), 'eligible_for_human_continuation': not blockers,
            'blockers': blockers, 'risk_dispositions': risks, 'gap_dispositions': gaps,
            'external_regression': regression, 'runtime_verified': False,
            'decision_scope': '人工决定是否继续升级流程；不是自动批准，也未验证提交者身份或日志真实性'}
