"""Attach collection provenance without changing the authority of static findings."""
import copy
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def finalize_application_report(value):
    report = value['compatibility_report']
    findings, gaps = report.get('findings', []), report.get('gaps', [])
    risks = sum(row.get('status') == 'supported_risk' for row in findings)
    report['summary'] = {'finding_count': len(findings), 'risk_count': risks, 'gap_count': len(gaps)}
    if risks:
        report['status'] = value['status'] = 'supported_risk'
    elif gaps or not findings:
        report['status'] = value['status'] = 'needs_verification'
    value['evidence_gaps'] = list(dict.fromkeys([*value.get('evidence_gaps', []),
                                               *(row['detail'] for row in gaps if isinstance(row.get('detail'), str))]))
    value['report_fingerprint'] = _digest(report)
    return value


def attach_application_context(result, snapshot):
    value = copy.deepcopy(result)
    value['application_snapshot'] = copy.deepcopy({k: v for k, v in snapshot.items() if k != 'files'})
    report = value['compatibility_report']
    report['gaps'] = [*report.get('gaps', []), *copy.deepcopy(snapshot['gaps'])]
    value['input_coverage'] = copy.deepcopy(snapshot['coverage'])
    value['request_fingerprint'] = _digest({'request': result.get('request_fingerprint'),
                                          'snapshot': value['application_snapshot']})
    value['task_id'] = 'paddleocr-' + value['request_fingerprint'][:24]
    value.setdefault('stage_status', {})['input_collection'] = (
        'completed' if snapshot['coverage']['status'] == 'REGISTERED_SCOPE_COMPLETE' else 'incomplete')
    value['runtime_verified'] = False
    return finalize_application_report(value)
