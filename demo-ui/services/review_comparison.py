"""Conservative rule/file comparison; disappearance is never runtime approval."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import defaultdict


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def capture_baseline(result, *, corpus_fingerprint):
    if (not isinstance(result, dict) or result.get('workspace_id') != 'paddleocr'
            or result.get('runtime_verified') is not False or not corpus_fingerprint):
        raise ValueError('只能保存身份明确的当前静态报告作为对照。')
    payload = {'result': copy.deepcopy(result), 'corpus_fingerprint': corpus_fingerprint}
    return {**payload, 'integrity': _digest(payload)}


def compare_reviews(baseline, result, *, corpus_fingerprint):
    refused = {'status': 'INCOMPARABLE', 'reason': '对照报告身份、版本或语料不一致，请重新保存基准。',
               'prior_records_reusable': False, 'runtime_verified': False, 'findings': [], 'files': []}
    try:
        payload = {key: baseline[key] for key in ('result', 'corpus_fingerprint')}
        previous = payload['result']
        old_snapshot = previous.get('application_snapshot') or {}
        new_snapshot = result.get('application_snapshot') or {}
        if any(old_snapshot.get(k) != new_snapshot.get(k) for k in ('application_id', 'registered_root', 'source_identity')):
            return refused
        if (baseline.get('integrity') != _digest(payload) or payload['corpus_fingerprint'] != corpus_fingerprint
                or not corpus_fingerprint or result.get('runtime_verified') is not False
                or any(previous.get(k) != result.get(k) for k in ('workspace_id', 'source_version', 'target_version',
                                                                  'analysis_policy_fingerprint', 'application_context'))
                or result.get('workspace_id') != 'paddleocr'):
            return refused
        before, after = previous['compatibility_report'], result['compatibility_report']
        old_files = {r['path']: r['sha256'] for r in before['files']}
        new_files = {r['path']: r['sha256'] for r in after['files']}
        if len(old_files) != len(before['files']) or len(new_files) != len(after['files']):
            return refused
        changes = [{'path': path, 'change': 'ADDED' if path not in old_files else
                    'REMOVED' if path not in new_files else 'UNCHANGED' if old_files[path] == new_files[path] else 'MODIFIED'}
                   for path in sorted(old_files.keys() | new_files.keys())]
        old, new = defaultdict(list), defaultdict(list)
        for row in before.get('findings', []):
            if row['status'] == 'supported_risk':
                old[(row['rule_id'], row['application']['path'])].append(row)
        for row in after.get('findings', []):
            if row['status'] == 'supported_risk':
                new[(row['rule_id'], row['application']['path'])].append(row)
        # Missing inputs or newly unresolvable flow cannot masquerade as a fix.
        missing_files = old_files.keys() - new_files.keys()
        blocking_gaps = [r for r in after.get('gaps', [])
                         if r.get('code') not in ('runtime_not_verified', 'actual_runtime_not_verified')]
        uncertain_paths = {r.get('application', {}).get('path') for r in after.get('findings', [])
                           if r.get('status') == 'needs_verification'}
        findings = []
        for key in sorted(old.keys() | new.keys()):
            was, now = old[key], new[key]
            if now:
                change = 'PERSISTS' if was else 'NEW'
            elif missing_files or blocking_gaps or key[1] in uncertain_paths or key[1] not in new_files:
                change = 'UNRESOLVED'
            else:
                change = 'NOT_OBSERVED_NOW'
            findings.append({'rule_id': key[0], 'path': key[1], 'change': change,
                             'previous_count': len(was), 'current_count': len(now),
                             'previous_locations': [r['application'] for r in was],
                             'current_locations': [r['application'] for r in now]})
        return {'status': 'COMPARED', 'files': changes, 'findings': findings,
                'previous_application_commit': old_snapshot.get('head_commit'),
                'current_application_commit': new_snapshot.get('head_commit'),
                'comparison_scope': 'rule_and_file_not_exact_site_identity',
                'prior_records_reusable': False, 'runtime_verified': False,
                'previous_request': previous.get('request_fingerprint'),
                'current_request': result.get('request_fingerprint'),
                'remaining_gaps': len(after.get('gaps', [])) + len((result.get('investigation') or {}).get('unresolved', [])),
                'notice': '当前未观察到原规则风险不等于修复已通过运行验证；原处置和回归记录不自动沿用。'}
    except (KeyError, TypeError, ValueError):
        return refused
