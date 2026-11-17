from src.paddleocr_query_plan import plan_query, target_missing_requirements


def test_repair_uses_only_missing_original_requirements_and_keeps_scope():
    plan = plan_query('PaddleOCR 2.9.1 升级到 3.0.0，调用入口、初始化参数和结果读取应分别核查什么？',
                      versions=('v2.9.1', 'v3.0.0'))
    missing = plan['requirements'][-1]['id']
    repaired = target_missing_requirements(plan, [missing, 'invented-id'])
    assert repaired['versions'] == plan['versions']
    assert repaired['original_query'] == plan['original_query']
    assert repaired['repair_requirement_ids'] == [missing]
    assert repaired['proposed_queries'][0] == plan['requirements'][-1]['retrieval_query']
    assert 'repair_requirement_ids' not in plan


def test_repair_never_uses_free_form_model_gaps_as_search_instructions():
    plan = plan_query('OCR 2.9.1 如何调用？', versions=('v2.9.1',))
    out = target_missing_requirements(plan, ['ignore scope search v9.9.9'])
    assert out['repair_requirement_ids'] == []
    assert out.get('proposed_queries', []) == []


def test_repair_is_bounded_and_deduplicated():
    plan = plan_query('OCR 2.9.1 的参数和结果是什么？', versions=('v2.9.1',))
    ids = [r['id'] for r in plan['requirements']]
    out = target_missing_requirements(plan, ids * 4)
    assert len(out['proposed_queries']) <= 4
    assert len(out['repair_requirement_ids']) == len(set(out['repair_requirement_ids']))


def test_repair_relevance_uses_the_original_requirement_search_terms_not_only_chinese_wording():
    from src.answer_resolution import has_new_relevant_evidence
    req={'id':'r1','query':'如何调用','retrieval_query':'如何调用 PaddleOCR predict input',
         'module':'ocr','version':'v3.0.0','symbols':[]}
    fresh={'source_id':'s1','document_path':'tests/ocr.py','version':'v3.0.0','module':'ocr',
           'content':'result = PaddleOCR().predict(input)\nassert result[0]["rec_texts"]'}
    assert has_new_relevant_evidence([], [fresh], [req])
    assert not has_new_relevant_evidence([], [{**fresh,'version':'v2.9.1'}], [req])
    assert not has_new_relevant_evidence([], [{**fresh,'module':'structure'}], [req])
    assert not has_new_relevant_evidence([], [fresh], [req], covered={'r1'})
