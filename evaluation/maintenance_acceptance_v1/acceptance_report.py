"""Report observations without presenting model labels as human accuracy."""
import hashlib
import json
import statistics
from collections import Counter


def case_fingerprint(cases):
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def identity_status(before, after, rows):
    observed = [r.get('response_identity_sha256') for r in rows]
    if before != after or any(value and value != before for value in observed):
        return 'CHANGED'
    if not before or not after or any(not value for value in observed):
        return 'UNVERIFIED'
    return 'CONSISTENT'


def summarize(rows):
    labels = Counter(r['result'].get('answer_completeness', 'NOT_ASSESSED') for r in rows)
    failures, leaks, empty = [], [], []
    for row in rows:
        value = row['result']
        status = value.get('status')
        search_result = isinstance(value.get('results'), list)
        if (status is None and not search_result or
                status not in (None, 'OK', 'OUT_OF_SCOPE', 'ABSTAINED', 'NEEDS_CLARIFICATION', 'NO_EVIDENCE','NEEDS_APPLICATION_CONTEXT')):
            failures.append(row['id'])
        if status == 'NO_EVIDENCE' or (status in (None, 'OK') and search_result and not value['results']):
            empty.append(row['id'])
        refs = value.get('sources') or value.get('results') or []
        if row['version'] != 'all' and any(r.get('version') != row['version'] for r in refs):
            leaks.append(row['id'])
    seconds = [r['seconds'] for r in rows if r['result'].get('status') != 'OUT_OF_SCOPE']
    types={}
    for row in rows:
        group=types.setdefault(row.get('question_type','unclassified'),{'count':0,'model_complete':0,'partial':0,'boundary':0})
        group['count']+=1
        group['model_complete']+=row['result'].get('answer_completeness')=='MODEL_ASSESSED_COMPLETE'
        group['partial']+=row['result'].get('answer_completeness')=='PARTIAL_SUPPORTED'
        group['boundary']+=row['result'].get('status')=='NEEDS_APPLICATION_CONTEXT'
    return {'case_count': len(rows), 'model_complete': labels['MODEL_ASSESSED_COMPLETE'],
            'boundary_cases':[r['id'] for r in rows if r['result'].get('status')=='NEEDS_APPLICATION_CONTEXT'],
            'by_question_type':types,
            'model_partial': labels['PARTIAL_SUPPORTED'], 'completeness_labels': dict(labels),
            'with_claims': sum(bool(r['result'].get('claims')) for r in rows),
            'request_failures': failures, 'scope_violation_cases': leaks,
            'no_evidence_cases': empty,
            'clarification_cases': [r['id'] for r in rows if r['result'].get('status') == 'NEEDS_CLARIFICATION'],
            'median_seconds': round(statistics.median(seconds), 3) if seconds else None,
            'max_seconds': max(seconds, default=None), 'human_accuracy': None,
            'human_review_status': 'PENDING_INDEPENDENT_REVIEW',
            'assessment_type': 'runtime_observation_not_answer_accuracy'}
