"""Validate model coverage labels against surviving claims and evidence scope."""
from __future__ import annotations

ASSESSMENT = 'model_assessment_not_human_validation'


def assess_requirement_coverage(requirements, verdicts, claims, supported, evidence):
    unknown = {'status': 'NOT_ASSESSED', 'items': [], 'assessment_type': ASSESSMENT}
    if not requirements:
        return unknown
    try:
        by_requirement = {r['id']: r for r in requirements}
        if len(by_requirement) != len(requirements) or not 1 <= len(requirements) <= 4:
            return unknown
        if not isinstance(verdicts, list) or len(verdicts) != len(requirements):
            return unknown
        by_evidence = {r['chunk_id']: r for r in evidence}
        indexed, seen = {}, set()
        for row in verdicts:
            if not isinstance(row, dict) or set(row) != {'requirement_id', 'status', 'claim_indexes', 'reason'}:
                return unknown
            rid = row['requirement_id']
            if rid not in by_requirement or rid in seen or row['status'] not in ('COVERED', 'PARTIAL', 'MISSING'):
                return unknown
            seen.add(rid)
            indexes = row['claim_indexes']
            if (not isinstance(indexes, list) or len(indexes) > 5 or len(set(indexes)) != len(indexes)
                    or any(type(i) is not int or not 0 <= i < len(claims) for i in indexes)
                    or not isinstance(row['reason'], str) or not 0 < len(row['reason']) <= 800):
                return unknown
            requirement = by_requirement[rid]
            valid = []
            for i in indexes:
                refs = [by_evidence[cid] for cid in claims[i]['evidence_ids']]
                # A label cannot override scope validation or resurrect a rejected claim.
                if i in supported and any(
                    (not requirement.get('version') or ref.get('version') == requirement['version'])
                    and (not requirement.get('module') or ref.get('module', module_for(ref))
                         in (requirement['module'], 'shared', 'general')) for ref in refs
                ):
                    valid.append(i)
            status = row['status'] if valid or row['status'] == 'MISSING' else 'MISSING'
            if status == 'MISSING':
                valid = []
            ids = list(dict.fromkeys(cid for i in valid for cid in claims[i]['evidence_ids']))
            indexed[rid] = {'requirement_id': rid, 'query': requirement['query'],
                            'version': requirement.get('version'), 'status': status,
                            'evidence_ids': ids, 'reason': row['reason'] if status == row['status']
                            else '对应主张未通过支持度或适用范围核验，不能关闭此要点。'}
        items = [indexed[r['id']] for r in requirements]
        return {'status': 'MODEL_ASSESSED_COMPLETE' if all(r['status'] == 'COVERED' for r in items)
                else 'PARTIAL_SUPPORTED', 'items': items, 'assessment_type': ASSESSMENT}
    except (KeyError, TypeError, ValueError):
        return unknown


def module_for(row):
    from src.paddleocr_retrieval_views import module_for as resolve
    return resolve(row)
