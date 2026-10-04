import pytest


def test_claim_checker_drops_unsupported_even_when_citation_exists():
    from src.claim_support import check_claim_support
    claims=[{'text':'默认线程数是8','evidence_ids':['a']},{'text':'默认线程数是4','evidence_ids':['a']}]
    evidence=[{'chunk_id':'a','content':'cpu_threads 默认值为 4。','version':'v3.0.0'}]
    def judge(payload):
        return {'verdicts':[{'claim_index':0,'supported':False,'reason':'原文为4而非8'},
                            {'claim_index':1,'supported':True,'reason':'原文直接支持'}]},{'model':'test-judge'}
    result=check_claim_support(claims,evidence,judge)
    assert result['claims']==[claims[1]]
    assert result['rejected'][0]['claim_index']==0
    assert result['status']=='PARTIAL_SUPPORTED'


@pytest.mark.parametrize('verdicts',[
    [], [{'claim_index':0,'supported':'true','reason':'bad'}],
    [{'claim_index':0,'supported':True,'reason':'ok'},{'claim_index':0,'supported':True,'reason':'dup'}],
    [{'claim_index':9,'supported':True,'reason':'foreign'}],
])
def test_invalid_verification_does_not_release_unchecked_claims(verdicts):
    from src.claim_support import check_claim_support
    result=check_claim_support([{'text':'事实','evidence_ids':['a']}],[{'chunk_id':'a','content':'原文'}],
                              lambda payload:({'verdicts':verdicts},{}))
    assert result['claims']==[]
    assert result['status']=='CHECK_FAILED'


def test_checker_receives_only_cited_excerpts_and_never_accepts_foreign_ids():
    from src.claim_support import check_claim_support
    seen=[]
    def judge(payload):
        seen.append(payload)
        return {'verdicts':[{'claim_index':0,'supported':True,'reason':'支持'}]},{}
    result=check_claim_support([{'text':'事实','evidence_ids':['a']}],
                              [{'chunk_id':'a','content':'原文'},{'chunk_id':'b','content':'无关敏感内容'}],judge)
    assert '无关敏感内容' not in str(seen)
    assert result['claims']
    result=check_claim_support([{'text':'事实','evidence_ids':['foreign']}],[{'chunk_id':'a','content':'原文'}],judge)
    assert result['claims']==[]


def test_checker_exception_preserves_failure_and_hides_answer():
    from src.claim_support import check_claim_support
    def judge(payload):raise TimeoutError()
    result=check_claim_support([{'text':'事实','evidence_ids':['a']}],[{'chunk_id':'a','content':'原文'}],judge)
    assert result['status']=='CHECK_FAILED'
    assert result['claims']==[]


def test_oversized_evidence_is_not_sent_to_checker():
    from src.claim_support import check_claim_support
    def judge(payload):pytest.fail('over-budget evidence must not be sent')
    result=check_claim_support([{'text':'事实','evidence_ids':['a']}],
                              [{'chunk_id':'a','content':'原文'*40000}],judge)
    assert result['status']=='CHECK_FAILED'
    assert result['failure_type']=='EvidenceBudgetExceeded'


def test_invalid_checker_retains_safe_provider_usage():
    from src.claim_support import check_claim_support
    result=check_claim_support([{'text':'事实','evidence_ids':['a']}],[{'chunk_id':'a','content':'原文'}],
                              lambda payload:({'verdicts':[]},{'usage':{'total_tokens':10}}))
    assert result['claims']==[]
    assert result['failure_type']=='InvalidCheckerResponse'
    assert result['diagnostics']['usage']['total_tokens']==10
