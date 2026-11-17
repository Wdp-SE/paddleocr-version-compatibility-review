"""Retain generator concerns unless a complete, scoped review retires each one.

This is an auditable model assessment, never an independent truth judgement.
Malformed or absent gap reviews preserve every original gap.
"""
from __future__ import annotations


def assess_gaps(gaps, rows, claims, supported, coverage):
    original = list(gaps or [])
    fallback = {'status': 'NOT_ASSESSED', 'active_gaps': original, 'items': [],
                'assessment_type': 'model_assessment_not_human_validation'}
    if not original:
        return {**fallback, 'status': 'NOT_NEEDED'}
    if not isinstance(rows, list) or len(rows) != len(original):
        return fallback
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'gap_index', 'status', 'claim_indexes', 'reason'}:
            return fallback
        i, refs = row['gap_index'], row['claim_indexes']
        if (type(i) is not int or not 0 <= i < len(original) or i in indexed
                or row['status'] not in ('RESOLVED', 'OUTSIDE_QUESTION', 'REQUIRED_MISSING')
                or not isinstance(refs, list) or len(refs) > 5
                or any(type(n) is not int or not 0 <= n < len(claims) for n in refs)
                or len(set(refs)) != len(refs)
                or not isinstance(row['reason'], str) or not 0 < len(row['reason'].strip()) <= 800):
            return fallback
        indexed[i] = row
    covered_ids = {cid for r in coverage.get('items', []) if r.get('status') == 'COVERED'
                   for cid in r.get('evidence_ids', [])}
    complete = (coverage.get('status') == 'MODEL_ASSESSED_COMPLETE'
                and len(supported) == len(claims))
    items, active = [], []
    for i, gap in enumerate(original):
        row = indexed[i]
        valid_refs = (bool(row['claim_indexes']) and all(n in supported and
            any(cid in covered_ids for cid in claims[n]['evidence_ids']) for n in row['claim_indexes']))
        retired = row['status'] != 'REQUIRED_MISSING' and complete and valid_refs
        if not retired:
            active.append(gap)
        items.append({'gap_index': i, 'original_gap': gap, 'status': row['status'] if retired else 'REQUIRED_MISSING',
                      'proposed_status': row['status'], 'retired': retired, 'reason': row['reason'],
                      'evidence_ids': list(dict.fromkeys(cid for n in row['claim_indexes'] if n in supported
                                                         for cid in claims[n]['evidence_ids']))})
    return {**fallback, 'status': 'REVIEWED', 'active_gaps': active, 'items': items}
