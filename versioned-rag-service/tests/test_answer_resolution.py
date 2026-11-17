import copy


def coverage(*covered):
    return {'status':'MODEL_ASSESSED_COMPLETE' if len(covered)==2 else 'PARTIAL_SUPPORTED',
            'items':[{'requirement_id':key,'status':'COVERED' if key in covered else 'MISSING'} for key in ('r1','r2')]}


def answer(*covered,gaps=None):
    return {'status':'OK','answer_completeness':'MODEL_ASSESSED_COMPLETE' if len(covered)==2 else 'PARTIAL_SUPPORTED',
            'requirement_coverage':coverage(*covered),'evidence_gaps':gaps or [],'claims':[{'text':'supported'}]}


def test_generator_and_checker_disagreement_stays_visible_without_erasing_gap():
    from src.answer_resolution import reconcile_coverage
    original=coverage('r1','r2');snapshot=copy.deepcopy(original)
    result=reconcile_coverage(original,['原文未明确批处理单位'])
    assert result['status']=='CONFLICTING_ASSESSMENTS'
    assert result['checker_status']=='MODEL_ASSESSED_COMPLETE'
    assert result['conflicts']==['原文未明确批处理单位']
    assert original==snapshot


def test_complete_coverage_without_gaps_is_preserved():
    from src.answer_resolution import reconcile_coverage
    original=coverage('r1','r2')
    assert reconcile_coverage(original,[])==original


def test_repair_cannot_drop_an_already_supported_requirement():
    from src.answer_resolution import choose_repair
    first,second=answer('r1'),answer('r2')
    chosen,reason=choose_repair(first,second)
    assert chosen is first and reason=='WOULD_LOSE_COVERED_REQUIREMENTS'


def test_complete_repair_can_replace_partial_answer():
    from src.answer_resolution import choose_repair
    first,second=answer('r1'),answer('r1','r2')
    chosen,_=choose_repair(first,second)
    assert chosen is second


def test_rewording_without_coverage_gain_does_not_replace_valid_partial():
    from src.answer_resolution import choose_repair
    first,second=answer('r1',gaps=['缺空页']),answer('r1',gaps=['缺空页'])
    chosen,reason=choose_repair(first,second)
    assert chosen is first and reason=='NO_VERIFIED_COVERAGE_GAIN'


def test_failed_repair_preserves_first_answer():
    from src.answer_resolution import choose_repair
    first=answer('r1')
    assert choose_repair(first,{'status':'ABSTAINED'})[0] is first


def test_new_supported_partial_facts_count_as_progress_without_claiming_complete():
    from src.answer_resolution import choose_repair
    first, second = answer(), answer()
    second['requirement_coverage']['items'][0]['status'] = 'PARTIAL'
    chosen, reason = choose_repair(first, second)
    assert chosen is second and reason == 'MORE_REQUIREMENTS_PARTIALLY_SUPPORTED'


def test_partial_coverage_cannot_be_lost_for_rewording_another_requirement():
    from src.answer_resolution import choose_repair
    first, second = answer(), answer('r2')
    first['requirement_coverage']['items'][0]['status'] = 'PARTIAL'
    assert choose_repair(first, second)[0] is first
