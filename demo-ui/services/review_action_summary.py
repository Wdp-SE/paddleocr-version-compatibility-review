"""Read-only grouping; original report records and disposition identities survive."""
import json


def finding_groups(result, *, statuses=None):
    """Group identical issues, retaining each finding identity and location."""
    groups={}
    for finding in (result.get('compatibility_report') or {}).get('findings',[]):
        if statuses is not None and finding.get('status') not in statuses:continue
        location=finding.get('application') or {}
        # Different source contracts must never be combined just by rule name.
        evidence=sorted(json.dumps(row,sort_keys=True,ensure_ascii=False) for row in finding.get('evidence',[]))
        key=(finding.get('rule_id'),finding.get('status'),location.get('path'),finding.get('title'),
             finding.get('explanation'),finding.get('desired_check'),tuple(evidence))
        group=groups.setdefault(key,{'finding':finding,'findings':[],'locations':[]})
        group['findings'].append(finding)
        if location not in group['locations']:group['locations'].append(location)
    return list(groups.values())


def review_overview(result, *, records=None):
    groups=finding_groups(result,statuses=('supported_risk',))
    changes=finding_groups(result,statuses=('behavior_change',))
    summary=summarize_review(result)
    runtime_label='未关联回归记录'
    head=(result.get('application_snapshot') or {}).get('head_commit')
    fingerprint=result.get('report_fingerprint')
    for record in records or []:
        payload=record.get('payload') if isinstance(record,dict) else None
        if (isinstance(payload,dict) and record.get('kind')=='CI' and head and fingerprint
                and payload.get('provenance')=='trusted_ci_identity'
                and payload.get('application_commit')==head
                and payload.get('report_fingerprint')==fingerprint
                and isinstance(payload.get('environment'),dict)
                and payload['environment'].get('paddleocr')==str(result.get('target_version','')).removeprefix('v')
                and payload.get('status') in ('PASSED','FAILED')):
            # Association only. Do not derive release eligibility in the UI.
            runtime_label='已关联可信 CI 记录'
    return {'risk_count':len(groups),'risk_location_count':sum(len(g['locations']) for g in groups),
            'change_count':len(changes),'gap_count':len(summary['input_gaps'])+len(summary['lookup_actions']),
            'runtime_label':runtime_label,'risk_groups':groups,'change_groups':changes,
            'summary':summary}


def summarize_review(result):
    report = result.get('compatibility_report') or {}
    buckets = {name: {} for name in ('input_gaps', 'limitations', 'runtime_actions', 'lookup_actions')}
    records = [(g, 'limitations' if g.get('blocking') is False else 'input_gaps')
               for g in report.get('gaps', [])]
    records += [(g, 'runtime_actions' if g.get('reason') == 'application_validation_required' else 'lookup_actions')
                for g in (result.get('investigation') or {}).get('unresolved', [])]
    for gap, category in records:
        location = gap.get('application') or {}
        key = (gap.get('code') or gap.get('reason'), location.get('path'),
               gap.get('next_action'), gap.get('detail') if not (gap.get('code') or gap.get('reason')) else None)
        group = buckets[category].setdefault(key, {'code': key[0], 'detail': gap.get('detail') or gap.get('reason'),
            'next_action': gap.get('next_action'), 'count': 0, 'locations': [], 'versions': [], 'records': []})
        group['count'] += 1
        group['records'].append(gap)
        if location and location not in group['locations']:
            group['locations'].append(location)
        for v in gap.get('versions', []):
            if v not in group['versions']:
                group['versions'].append(v)
    return {**{name: list(rows.values()) for name, rows in buckets.items()}, 'record_count': len(records)}
