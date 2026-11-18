"""Conservative agreement and repair selection over already verified answers."""
import copy


def has_new_relevant_evidence(previous,candidate,requirements,*,covered=()):
    from src.paddleocr_retrieval_views import generation_evidence_text,module_for
    from src.paddleocr_applicability import version_applies
    from src.paddleocr_evidence_units import retrieval_tokens
    missing=[r for r in requirements if r['id'] not in covered]
    if not requirements:return has_new_evidence(previous,candidate)
    for row in candidate:
        if not has_new_evidence(previous,[row]):continue
        text=generation_evidence_text(row)
        for req in missing:
            if req.get('version') and not version_applies(row,req['version']):continue
            if req.get('module') and row.get('module',module_for(row)) not in (req['module'],'shared','general'):continue
            symbols=[s for s in req.get('symbols',[]) if s!='PaddleOCR']
            if symbols:
                if any(s in text for s in symbols):return True
            elif len(set(retrieval_tokens(req.get('retrieval_query',req['query'])))&set(retrieval_tokens(text)))>=2:
                return True
    return False


def has_new_evidence(previous, candidate):
    """Compare the actual bounded generation context, not ranks or chunk IDs.

    A wider window in the same canonical chunk can add facts. Reordering,
    duplicate hits or a smaller window must not trigger another paid answer.
    """
    from src.paddleocr_retrieval_views import generation_evidence_text

    def identity(row):
        return tuple(row.get(k) for k in ('version', 'source_id', 'document_path', 'commit'))

    def text(row):
        return '\n'.join(line.strip() for line in generation_evidence_text(row).splitlines() if line.strip())

    known = {}
    for row in previous:
        known.setdefault(identity(row), []).append(text(row))
    return any(text(row) and not any(text(row) in old for old in known.get(identity(row), []))
               for row in candidate)


def reconcile_coverage(coverage,generator_gaps):
    result=copy.deepcopy(coverage)
    if result.get('status')=='MODEL_ASSESSED_COMPLETE' and generator_gaps:
        result.update(status='CONFLICTING_ASSESSMENTS',checker_status='MODEL_ASSESSED_COMPLETE',
                      conflicts=list(generator_gaps))
    return result


def _covered(answer):
    return {r['requirement_id'] for r in answer.get('requirement_coverage',{}).get('items',[])
            if r.get('status')=='COVERED'}


def choose_repair(first,second):
    if second.get('status')!='OK':
        return first,'REPAIR_FAILED'
    if first.get('status')!='OK':
        return second,'SUPPORTED_ANSWER_RECOVERED'
    old,new=_covered(first),_covered(second)
    if old-new:
        return first,'WOULD_LOSE_COVERED_REQUIREMENTS'
    levels = {'COVERED': 2, 'PARTIAL': 1}
    previous = {r['requirement_id']: levels.get(r.get('status'), 0)
                for r in first.get('requirement_coverage', {}).get('items', [])}
    current = {r['requirement_id']: levels.get(r.get('status'), 0)
               for r in second.get('requirement_coverage', {}).get('items', [])}
    if any(current.get(key, 0) < value for key, value in previous.items()):
        return first, 'WOULD_LOSE_PARTIAL_SUPPORT'
    if second.get('answer_completeness')=='MODEL_ASSESSED_COMPLETE':
        return second,'COMPLETE_ANSWER_RECOVERED'
    if new-old:
        return second,'MORE_REQUIREMENTS_COVERED'
    if any(value > previous.get(key, 0) for key, value in current.items()):
        return second, 'MORE_REQUIREMENTS_PARTIALLY_SUPPORTED'
    # Compatibility for non-PaddleOCR clients without an assessed requirement plan.
    if not first.get('requirement_coverage') and not second.get('requirement_coverage'):
        return second,'LEGACY_UNASSESSED_REPAIR'
    return first,'NO_VERIFIED_COVERAGE_GAIN'
