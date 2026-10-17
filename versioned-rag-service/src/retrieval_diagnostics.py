"""Stage-local observations, never automatic semantic correctness labels."""


def diagnose_evidence_flow(*, source_complete, candidate_complete, context_complete,
                           accepted=None, oracle_accepted=None, human_supported=None):
    if not source_complete:
        return 'SOURCE_GAP'
    if not candidate_complete:
        return 'CANDIDATE_MISS'
    if not context_complete:
        return 'CONTEXT_SELECTION_LOSS'
    if accepted is None:
        return 'GENERATION_NOT_EVALUATED'
    if accepted:
        return 'ACCEPTED_REQUIRES_HUMAN_REVIEW'
    if human_supported is True:
        return 'HUMAN_CONFIRMED_FALSE_REJECTION'
    if oracle_accepted is True:
        return 'CONTEXT_SENSITIVITY_REQUIRES_REVIEW'
    return 'GENERATION_OR_VERIFICATION_FAILURE'
