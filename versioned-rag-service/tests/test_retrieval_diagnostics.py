import pytest
from src.retrieval_diagnostics import diagnose_evidence_flow


@pytest.mark.parametrize('values,expected', [
    ({'source_complete': False}, 'SOURCE_GAP'),
    ({'candidate_complete': False}, 'CANDIDATE_MISS'),
    ({'context_complete': False}, 'CONTEXT_SELECTION_LOSS'),
    ({}, 'GENERATION_NOT_EVALUATED'),
    ({'accepted': True}, 'ACCEPTED_REQUIRES_HUMAN_REVIEW'),
    ({'accepted': False, 'oracle_accepted': True}, 'CONTEXT_SENSITIVITY_REQUIRES_REVIEW'),
    ({'accepted': False, 'oracle_accepted': False}, 'GENERATION_OR_VERIFICATION_FAILURE'),
    ({'accepted': False, 'human_supported': True}, 'HUMAN_CONFIRMED_FALSE_REJECTION'),
])
def test_stage_labels_never_treat_model_rejection_as_a_proven_checker_error(values, expected):
    args = dict(source_complete=True, candidate_complete=True, context_complete=True)
    assert diagnose_evidence_flow(**{**args, **values}) == expected
