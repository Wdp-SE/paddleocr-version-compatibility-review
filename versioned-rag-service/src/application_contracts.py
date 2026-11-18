"""Submitted application requirements are declarations, never SDK or runtime facts."""
import json
import hashlib
from pathlib import PurePosixPath


def read_contracts(files):
    contracts, declarations, gaps = [], {}, []
    paths = {f['path'] for f in files}

    def gap(file, code, detail):
        gaps.append({'code': code, 'detail': detail, 'application': {'path': file['path'], 'line': 1,
            'end_line': min(4, len(file['lines'])), 'snippet': '\n'.join(file['lines'][:4])[:600]}})

    for file in files:
        if file['role'] != 'output_contract':
            continue
        try:
            value = json.loads(file['content'])
            if not isinstance(value, dict):
                raise ValueError()
        except (ValueError, RecursionError):
            gap(file, 'invalid_output_contract', '输出契约不是有效 JSON 对象，不能据此制定验收。')
            continue
        requirements = {}
        for key in ('required_line_fields', 'output_fields', 'page_fields'):
            if key not in value:
                continue
            fields = value[key]
            if (not isinstance(fields, list) or not 1 <= len(fields) <= 32
                    or any(not isinstance(v, str) or not 1 <= len(v) <= 80 for v in fields)
                    or len(set(fields)) != len(fields)):
                gap(file, 'invalid_contract_requirement', f'{key} 必须是有界、唯一的字段名称列表。')
            else:
                requirements[key] = fields
        if 'page_policy' in value:
            if value['page_policy'] in ('preserve_empty_pages', 'omit_empty_pages'):
                requirements['page_policy'] = value['page_policy']
            else:
                gap(file, 'unsupported_contract_requirement', '未支持的分页策略；需补充可执行验收定义。')
        contracts.append({'path': file['path'], 'sha256': file['sha256'], 'line_start': 1,
            'line_end': len(file['lines']), 'source_identity': 'submitted_application',
            'requirements': requirements, 'status': 'DECLARED_NOT_VERIFIED', 'runtime_verified': False})
        context = value.get('review_context', {})
        if not isinstance(context, dict) or set(context) - {'dependency_files'}:
            gap(file, 'invalid_review_context', '审查上下文格式无效或包含未支持设置。')
            continue
        roles = context.get('dependency_files', {})
        if not isinstance(roles, dict) or set(roles) - {'current', 'target', 'reference'}:
            gap(file, 'invalid_review_context', '依赖文件角色仅支持 current、target、reference。')
            continue
        for role, names in roles.items():
            if (not isinstance(names, list) or len(names) > 12
                    or any(not isinstance(p, str) or p not in paths or PurePosixPath(p).suffix == '.py' for p in names)):
                gap(file, 'invalid_review_context', '角色声明必须指向本次提交内的依赖文本文件。')
                continue
            for path in names:
                declarations.setdefault(path, set()).add(role)
    roles = {}
    for path, choices in declarations.items():
        if len(choices) == 1:
            roles[path] = next(iter(choices))
        else:
            gaps.append({'code': 'conflicting_dependency_role', 'detail': f'{path} 被声明为多个环境角色，不能选取有利角色消除冲突。'})
    return contracts, roles, gaps


def contract_regressions(contracts):
    rows = []
    for contract in contracts:
        req = contract['requirements']
        checks = [f'{key}: ' + ', '.join(values) for key, values in req.items() if isinstance(values, list)]
        if 'page_policy' in req:
            checks.append('分页策略: ' + req['page_policy'])
        if checks:
            rows.append({'kind': 'application_contract', 'rule_id': 'application_output_contract',
                'finding_id': 'contract-' + hashlib.sha256((contract['path']+contract['sha256']).encode()).hexdigest()[:16],
                'application': {'path': contract['path'], 'line': contract['line_start']},
                'source': contract, 'input_conditions': ['带文本页', '空页', '多页', '非法字段或长度不匹配'],
                'expected_contract': '按本次提交的应用输出契约验证；测试定义不代表测试已经执行。',
                'checks': checks, 'status': 'NOT_EXECUTED', 'runtime_verified': False})
    return rows


def classify_gaps(report, related_paths, test_paths=()):
    """Keep every limitation visible; only unrelated syntax limits are nonblocking."""
    syntax_limits = {'uncertain_control_flow', 'conditional_or_dynamic_flow', 'class_method_flow_not_resolved'}
    for gap in report['gaps']:
        path = gap.get('application', {}).get('path')
        if (gap['code']=='conditional_or_dynamic_flow'
                or gap['code'] in syntax_limits and not gap.get('binding_affected') and (path not in related_paths or path in test_paths)
                or path in test_paths and gap['code']=='decorated_function_flow_not_resolved'):
            gap.update(category='analysis_limitation', blocking=False,
                       next_action='影响图未证明该控制流；保留分析限制，不重复提升为已确认业务风险。')
        elif gap.get('blocking') is False:
            gap.setdefault('category', 'analysis_limitation')
        elif gap['code'] in ('unreviewed_configuration', 'actual_runtime_not_verified', 'runtime_not_verified'):
            gap.update(category='runtime_validation', blocking=True,
                       next_action='在对应应用提交及目标依赖环境执行回归并关联记录。')
        else:
            gap.update(category='analysis_gap', blocking=True,
                       next_action='核对该位置的绑定、必需资料或环境角色；补齐后重新审查。')
    report['coverage_summary'] = {
        'actionable_gaps': sum(g['blocking'] for g in report['gaps']),
        'analysis_limitations': sum(not g['blocking'] for g in report['gaps']),
        'runtime_verified': False,
    }
