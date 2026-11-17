from src.answer_resolution import has_new_evidence


def row(text, **kw):
    return {'chunk_id': 'a', 'version': 'v2.9.1', 'content': text, **kw}


def test_reordering_and_duplicate_chunks_do_not_justify_another_generation():
    before = [row('same'), row('other', chunk_id='b')]
    assert not has_new_evidence(before, list(reversed(before)) + [before[0]])


def test_new_window_in_same_source_counts_but_smaller_context_does_not():
    assert has_new_evidence([row('definition')], [row('definition\nusage')])
    assert not has_new_evidence([row('definition\nusage')], [row('definition')])


def test_different_version_or_source_identity_is_not_equivalent_evidence():
    assert has_new_evidence([row('same')], [row('same', version='v3.0.0')])


def test_new_but_irrelevant_context_does_not_justify_paid_repair():
    from src.answer_resolution import has_new_relevant_evidence
    requirements=[{'id':'r1','query':'drop_score 的过滤边界','symbols':['drop_score'],'version':'v2.9.1','module':'ocr'}]
    first=[row('drop_score 参数定义',module='ocr')]
    unrelated=[*first,row('部署服务的线程设置',chunk_id='b',module='ocr')]
    assert not has_new_relevant_evidence(first,unrelated,requirements)
    relevant=[*first,row('if score >= self.drop_score: append(text)',chunk_id='b',module='ocr')]
    assert has_new_relevant_evidence(first,relevant,requirements)
    assert not has_new_relevant_evidence(first,relevant,requirements,covered={'r1'})


def test_no_new_evidence_skips_paid_generation_and_preserves_supported_first_answer(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from src import public_api as api
    calls = []
    first = {'status': 'OK', 'answer': 'valid partial', 'answer_completeness': 'PARTIAL_SUPPORTED',
             'claim_verification': {'status': 'SUPPORTED'}, 'evidence': [row('same context')], 'generation': {}}
    async def once(*args, **kwargs):
        calls.append(1)
        return first
    monkeypatch.setattr(api, '_query_once', once)
    monkeypatch.setattr(api, '_index', lambda _: SimpleNamespace(manifest={'workspace_id': 'paddleocr'}))
    monkeypatch.setattr(api, '_execute_public_search', lambda *a, **k: {'results': [row('same context')]})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(public_generator=None)))
    out = asyncio.run(api.query(api.SearchRequest(query='test', version='v2.9.1'), request))
    assert len(calls) == 1 and out['answer'] == 'valid partial'
    assert out['correction']['status'] == 'SKIPPED_NO_NEW_EVIDENCE'
