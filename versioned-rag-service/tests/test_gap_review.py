"""A generator concern may be retired only by a scoped, supported review."""
from src.claim_support import check_claim_support


def review(rows, *, requirement_status='COVERED', supported=True):
    claims = [{'text': 'False 显式关闭方向分类。', 'evidence_ids': ['e1']}]
    evidence = [{'chunk_id': 'e1', 'content': 'False disables classification', 'version': 'v3.0.0'}]
    result = {'verdicts': [{'claim_index': 0, 'supported': supported, 'reason': '原文'}],
              'requirement_verdicts': [{'requirement_id': 'r1', 'status': requirement_status,
                                       'claim_indexes': [0] if supported else [], 'reason': '本题条件'}],
              'gap_verdicts': rows}
    return check_claim_support(claims, evidence, lambda _: (result, {}),
        requirements=[{'id': 'r1', 'query': '如何关闭方向分类？', 'version': 'v3.0.0'}],
        reported_gaps=['缺少分类模型的完整类型定义'])


def verdict(**kw):
    return {'gap_index': 0, 'status': 'OUTSIDE_QUESTION', 'claim_indexes': [0],
            'reason': '用户只问关闭开关，不要求完整类型。', **kw}


def test_out_of_question_gap_can_retire_with_complete_supported_answer_and_audit():
    result = review([verdict()])
    assert result['gap_review']['active_gaps'] == []
    assert result['gap_review']['items'][0]['original_gap'] == '缺少分类模型的完整类型定义'
    assert result['gap_review']['items'][0]['status'] == 'OUTSIDE_QUESTION'


def test_partial_or_rejected_claims_never_retire_generator_gap():
    for kw in ({'requirement_status': 'PARTIAL'}, {'supported': False}):
        assert review([verdict()], **kw)['gap_review']['active_gaps']


def test_foreign_duplicate_or_missing_gap_verdicts_preserve_original_concern():
    for rows in ([], [verdict(gap_index=99)], [verdict(), verdict()],
                 [verdict(claim_indexes=[99])], [verdict(reason='')]):
        assert review(rows)['gap_review']['active_gaps'] == ['缺少分类模型的完整类型定义']


def test_unresolved_gap_is_never_removed_by_complete_label_alone():
    result = review([verdict(status='REQUIRED_MISSING', claim_indexes=[])])
    assert result['gap_review']['active_gaps']
