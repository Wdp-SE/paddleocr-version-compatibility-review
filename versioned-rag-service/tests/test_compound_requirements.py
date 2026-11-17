from src.paddleocr_query_plan import plan_query
from src.requirement_coverage import assess_requirement_coverage


def test_upgrade_checklist_has_six_atomic_versioned_obligations():
    plan=plan_query('PaddleOCR 2.9.1 升级到 3.0.0，调用入口、初始化参数和结果读取应分别核查什么？',
                    versions=('v2.9.1','v3.0.0'))
    assert len(plan['requirements'])==6
    assert not plan['omitted_requirements']
    assert {r['fact_dimension'] for r in plan['requirements']}=={'call','parameters','result'}
    assert all(r['module']=='ocr' for r in plan['requirements'])
    assert all(len({r['version'] for r in plan['requirements'] if r['fact_dimension']==dimension})==2
               for dimension in ('call','parameters','result'))


def test_atomic_dimensions_keep_concrete_flags_parameters_and_result_access():
    plan=plan_query('PaddleOCR 2.9.1 到 3.0.0，请分别核查调用入口 det=False、初始化参数 drop_score、结果读取 line[1][0]',
                    versions=('v2.9.1','v3.0.0'))
    assert len(plan['requirements'])==6
    for dimension,token in [('call','det=False'),('parameters','drop_score'),('result','line[1][0]')]:
        assert all(token in r['query'] for r in plan['requirements'] if r['fact_dimension']==dimension)


def test_parameters_in_one_sentence_do_not_require_both_names_in_each_evidence():
    plan=plan_query('PaddleOCR 2.9.1 升级到 3.0.0 后 show_log 和 det_db_thresh 参数分别如何处理？',
                    versions=('v2.9.1','v3.0.0'))
    assert len(plan['requirements'])==4
    assert {tuple(r['symbols']) for r in plan['requirements']}=={('show_log',),('det_db_thresh',)}


def test_parameter_values_and_interaction_are_not_erased_by_splitting():
    query='PaddleOCR 2.9.1 升级到 3.0.0，show_log=True 和 det_db_thresh=0.3 这些参数同时传入应如何处理？'
    plan=plan_query(query,versions=('v2.9.1','v3.0.0'))
    assert all('show_log=True' in r['query'] and 'det_db_thresh=0.3' in r['query'] and '同时' in r['query']
               for r in plan['requirements'])


def test_all_explicit_clauses_are_preserved_or_recorded_when_budget_exceeded():
    text='；'.join('OCR 参数'+str(i)+'是什么' for i in range(10))
    plan=plan_query(text,versions=('v3.0.0',))
    assert len(plan['requirements'])==8
    assert len(plan['omitted_requirements'])==2
    assert plan['requirement_budget']==8


def test_coverage_can_validate_six_obligations_without_adding_claim_budget():
    requirements=[{'id':f'r{i}','query':'核查','version':'v3.0.0'} for i in range(6)]
    verdicts=[{'requirement_id':r['id'],'status':'COVERED','claim_indexes':[0],'reason':'已逐项支持'} for r in requirements]
    out=assess_requirement_coverage(requirements,verdicts,[{'text':'组合事实','evidence_ids':['e']}],{0},
                                  [{'chunk_id':'e','version':'v3.0.0'}])
    assert out['status']=='MODEL_ASSESSED_COMPLETE'
