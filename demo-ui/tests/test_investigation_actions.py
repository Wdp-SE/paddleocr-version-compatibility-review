from services.investigation_actions import can_resume


def test_runtime_and_missing_application_inputs_do_not_offer_automatic_resume():
    assert not can_resume({'unresolved':[{'reason':'application_validation_required'}],'elapsed_seconds':2,'search_calls':0})


def test_official_evidence_gap_is_resumable_with_budget():
    assert can_resume({'unresolved':[{'reason':'content_requirements_not_covered'}],'elapsed_seconds':2,'search_calls':2})


def test_exhausted_budget_is_not_resumable():
    assert not can_resume({'unresolved':[{'reason':'required_source_not_verified'}],'elapsed_seconds':240,'search_calls':2})
