"""Bounded configuration workflow over the same app and versioned evidence service."""
import hashlib
import json
import re

from app.paddleocr_review import (_workspace, _request_files, _location, _registered_source,
                                 _registered_chunk, _source_value, _analysis_policy)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _pinned_dependency(files, version):
    """Use the application's declared current role, never a target/reference file."""
    by_path = {r['path']: r['content'] for r in files}
    current = []
    try:
        for path, content in by_path.items():
            if path.endswith('/contract.json'):
                value = json.loads(content)
                current.extend(value.get('review_context', {}).get('dependency_files', {}).get('current', []))
        if not current or any(not isinstance(p, str) or p not in by_path for p in current):
            return False
        pins = []
        for path in set(current):
            for raw in by_path[path].splitlines():
                line = raw.split('#', 1)[0].strip()
                if re.match(r'paddleocr\b', line, re.I):
                    match = re.fullmatch(r'paddleocr\s*==\s*(\d+\.\d+\.\d+)', line, re.I)
                    if not match: return False
                    pins.append('v' + match[1])
                elif line.startswith(('-r', '-c')):
                    return False
        return bool(pins) and set(pins) == {version}
    except (ValueError, TypeError, AttributeError):
        return False


def _parameter_evidence(gateway, workspace, version):
    """Search, then verify against pinned registry and canonical document text."""
    response = gateway.search('drop_score 识别得分 阈值 过滤 默认', version=version,
                              retrieval_policy='paddleocr_evidence', evidence_strategy='contextual_bm25',
                              top_k=5, request_timeout=20)
    registry = {r['source_id']: r for r in workspace.get('source_registry', [])}
    documents, checked = {}, []
    for row in response.get('results', [])[:5]:
        if row.get('version') != version:
            continue
        citation = {k: _source_value(row, k) for k in ('source_id', 'chunk_id', 'version', 'sha256', 'commit', 'url', 'path')}
        citation.update(line_start=row.get('line_start'), line_end=row.get('line_end'))
        source = _registered_source(citation, registry)
        if row['source_id'] not in documents:
            documents[row['source_id']] = gateway.document(source.get('document_id') or row['source_id'], request_timeout=20)
        original = _registered_chunk(citation, source, documents[row['source_id']])
        if original.get('content') != row.get('content'):
            raise ValueError('配置查证原文与固定来源不一致')
        # Only the official definition row supports this rule, not arbitrary mentions.
        definition = next((line for line in original['content'].splitlines()
                           if 'drop_score' in line and re.search('过滤|丢弃|filter|discard', line, re.I)
                           and re.search('阈值|得分|score', line, re.I)), None)
        if definition:
            checked.append({**citation, 'title': row.get('document_title') or row.get('heading') or 'drop_score 参数定义',
                            'definition': definition})
    return checked


def review_configuration_change(gateway, before_files, after_files, *, dependency_version='v2.9.1', graph_runtime=None):
    from app.review_graph import run_review_graph
    validated_workspace = None

    def validate_input(state):
        before_files = state["data"]["before_files"]
        after_files = state["data"]["after_files"]
        before, after = _request_files(before_files), _request_files(after_files)
        workspace = validated_workspace if validated_workspace is not None else _workspace(gateway.workspace())
        task_identity = _digest({'workflow': 'paddleocr_configuration', 'before': before_files,
                                 'after': after_files, 'dependency_version': dependency_version,
                                 'policy': _analysis_policy(workspace),
                                 'corpus': workspace.get('corpus_fingerprint')})
        return {"data": {**state["data"], "before": before, "after": after, "workspace": workspace,
                         "task_identity": task_identity}, "completed_nodes": state["completed_nodes"] + ["validate_input"]}

    def static_review(state):
        before_files = state["data"]["before_files"]
        after_files = state["data"]["after_files"]
        dependency_version = state["data"]["dependency_version"]
        tool = gateway.configuration_review(before_files, after_files, dependency_version=dependency_version)
        return {"data": {**state["data"], "tool": tool}, "completed_nodes": state["completed_nodes"] + ["static_review"]}

    def evidence_validation(state):
        before_files = state["data"]["before_files"]
        after_files = state["data"]["after_files"]
        dependency_version = state["data"]["dependency_version"]
        before = state["data"]["before"]
        after = state["data"]["after"]
        tool = state["data"]["tool"]
        # The deterministic result is part of the trust boundary: independently match
        # both application inventories/locations before admitting any SDK conclusion.
        expected = lambda files: sorted((p, hashlib.sha256(t.encode()).hexdigest()) for p, t in files.items())
        for key, files in [('before_files', before), ('files', after)]:
            if sorted((r['path'], r['sha256']) for r in tool.get(key, [])) != expected(files):
                raise ValueError('配置工具返回的应用快照不一致')
        if (tool.get('workflow_type') != 'configuration_change' or tool.get('runtime_verified') is not False
                or tool.get('dependency_version') != dependency_version):
            raise ValueError('配置工具身份无效')
        gaps, findings, evidence = list(tool.get('gaps', [])), [], []
        dependency_verified = all(_pinned_dependency(files, dependency_version) for files in (before_files, after_files))
        if not dependency_verified:
            gaps.append({'code': 'dependency_context_unverified', 'blocking': True,
                         'detail': '两份快照的 current 依赖文件未同时明确固定到 v2.9.1；需补齐依赖角色与锁定版本，不能混用版本参数。'})
        for change in tool.get('changes', []):
            for key, files in [('before', before), ('after', after)]:
                _location(change[key], files)
                if change[key].get('sha256') != hashlib.sha256(files[change[key]['path']].encode()).hexdigest():
                    raise ValueError('配置位置哈希与应用不一致')
                for source in change[key].get('value_sources',[]):
                    _location(source,files)
                    if source.get('sha256')!=hashlib.sha256(files[source['path']].encode()).hexdigest():
                        raise ValueError('配置来源与提交文本不一致')
            old, new = change['before_value'], change['after_value']
            if (change.get('parameter') != 'drop_score' or type(old) not in (int, float)
                    or type(new) not in (int, float) or not 0 <= old <= 1 or not 0 <= new <= 1 or old == new):
                raise ValueError('配置变化不是受支持的阈值规则')
        return {"data": {**state["data"], "gaps": gaps, "findings": findings, "evidence": evidence, "dependency_verified": dependency_verified}, "completed_nodes": state["completed_nodes"] + ["evidence_validation"]}

    def official_lookup(state):
        dependency_version = state["data"]["dependency_version"]
        workspace = state["data"]["workspace"]
        tool = state["data"]["tool"]
        gaps = state["data"]["gaps"]
        evidence = state["data"]["evidence"]
        dependency_verified = state["data"]["dependency_verified"]
        if tool.get('changes') and dependency_verified:
            try:
                evidence = _parameter_evidence(gateway, workspace, dependency_version)
            except Exception:
                evidence = []
            if not evidence:
                gaps.append({'code': 'parameter_evidence_missing', 'blocking': True,
                             'detail': '未获得同版本、固定来源的 drop_score 过滤定义，不能确认行为影响。'})
        return {"data": {**state["data"], "gaps": gaps, "evidence": evidence}, "completed_nodes": state["completed_nodes"] + ["official_lookup"]}

    def report(state):
        after_files = state["data"]["after_files"]
        dependency_version = state["data"]["dependency_version"]
        workspace = state["data"]["workspace"]
        tool = state["data"]["tool"]
        gaps = state["data"]["gaps"]
        findings = state["data"]["findings"]
        evidence = state["data"]["evidence"]
        for change in tool.get('changes', []):
            old, new = change['before_value'], change['after_value']
            stricter = new > old
            findings.append({'finding_id': 'config-' + _digest(change)[:16], 'rule_id': 'ocr_drop_score_change',
                'status': 'behavior_change' if evidence else 'needs_verification',
                'title': f'drop_score：{old:g} → {new:g}', 'application': change['after'],
                'before_application': change['before'], 'evidence': evidence,
                'explanation': ('静态配置提高了保留分数门槛，原本保留的部分低分文字可能被过滤。' if stricter else
                                '静态配置降低了保留分数门槛，可能让更多低分文字进入输出。') +
                               '这不是已证实的质量改善或缺陷；实际生效与业务可接受性需回归。',
                'desired_check': '对比相同样本的文本、阈值边界、空页页序与应用 JSON 契约。',
                'before_value': old, 'after_value': new})
        contract_files = [r for r in after_files if r['path'].endswith('/contract.json')]
        # Read the actual submitted contract; SDK docs cannot supply internal fields.
        contracts = []
        for file in contract_files:
            try:
                value = json.loads(file['content'])
                if not isinstance(value, dict) or value.get('page_policy') not in ('preserve_empty_pages', 'omit_empty_pages'):
                    raise ValueError()
                if not all(isinstance(value.get(k), list) and value[k] and
                           all(isinstance(x, str) for x in value[k])
                           for k in ('required_line_fields', 'output_fields', 'page_fields')):
                    raise ValueError()
                contracts.append({'path': file['path'], 'sha256': hashlib.sha256(file['content'].encode()).hexdigest(),
                                  'requirements': {k: value[k] for k in ('required_line_fields', 'output_fields', 'page_fields', 'page_policy')},
                                  'source_identity': 'submitted_application', 'status': 'DECLARED_NOT_VERIFIED'})
            except (ValueError, TypeError):
                pass
        if not contracts:
            gaps.append({'code': 'application_contract_missing', 'blocking': True,
                         'detail': '缺少可解析的本应用输出字段与分页契约，官方资料不能替代。'})
        checks = ['比较同一输入的文本差异', '低于/等于/高于阈值的边界', '配置实际加载值',
                  '显式参数覆盖优先级', '空页输出', '多页页序', '本次提交 JSON 字段契约']
        for contract in contracts:
            for key, value in contract['requirements'].items():
                checks.append(key + ': ' + (', '.join(value) if isinstance(value, list) else value))
        requirements = [{'finding_id': f['finding_id'], 'kind': 'configuration_behavior',
                         'input_conditions': ['带文本页', '阈值附近低分文本', '空页', '多页'],
                         'checks': checks, 'status': 'NOT_EXECUTED', 'runtime_verified': False} for f in findings]
        status = 'needs_verification' if gaps or not findings else 'behavior_change'
        report = {'files': tool['files'], 'before_files': tool['before_files'], 'findings': findings, 'gaps': gaps,
                  'source_version': dependency_version, 'target_version': dependency_version, 'status': status,
                  'regression_requirements': requirements, 'application_contracts': contracts,
                  'verification_steps': checks, 'runtime_verified': False, 'configuration_delta': tool}
        policy = _analysis_policy(workspace)
        fingerprint = _digest({'workflow_type': 'configuration_change', 'report': report,
                               'corpus': workspace.get('corpus_fingerprint'), 'policy': policy})
        result = {'schema_version': 1, 'workflow_type': 'configuration_change', 'workspace_id': 'paddleocr',
                'request_mode': 'paddleocr_configuration', 'source_version': dependency_version, 'target_version': dependency_version,
                'status': status, 'task_id': 'config-' + fingerprint[:24], 'request_fingerprint': fingerprint,
                'report_fingerprint': _digest(report), 'analysis_policy_fingerprint': policy,
                'corpus_fingerprint': workspace.get('corpus_fingerprint'), 'compatibility_report': report,
                'runtime_verified': False, 'model_calls': 0, 'manual_review_required': True,
                'stage_status': {'static_review': 'completed', 'official_lookup': 'completed' if evidence else 'incomplete',
                                 'runtime_validation': 'not_executed', 'human_review': 'pending'}}
        return {"data": {**state["data"], "result": result}, "completed_nodes": state["completed_nodes"] + ["report"]}

    initial = {'data': {'before_files': before_files, 'after_files': after_files,
                        'dependency_version': dependency_version}, 'completed_nodes': []}
    validated = validate_input(initial)
    validated_workspace = validated["data"]["workspace"]
    initial['data'].update({key: validated['data'][key] for key in ('before', 'after', 'workspace', 'task_identity')})
    nodes = {'validate_input': validate_input, 'static_review': static_review,
             'evidence_validation': evidence_validation, 'official_lookup': official_lookup, 'report': report}
    return run_review_graph('paddleocr_configuration', nodes, initial,
                            lookup_route=lambda state: bool(state['data']['tool'].get('changes')
                                                           and state['data']['dependency_verified']),
                            runtime=graph_runtime)
