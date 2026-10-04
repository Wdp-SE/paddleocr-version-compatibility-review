import pytest


def test_repeated_query_stops_without_spending_another_search():
    from app.paddleocr_investigation import investigate
    class Gateway:
        calls=0
        def search(self,question,**kwargs):
            self.calls+=1
            return {'results':[]}
    gateway=Gateway(); check={'check_id':'r1','query':'结果字段','versions':['v3.0.0']}
    result=investigate([check,check],gateway,clock=lambda:0)
    assert gateway.calls==1
    assert result['unresolved'] and result['stop_reason'] in ('NO_NEW_EVIDENCE','DUPLICATE_QUERY')


def test_gateway_without_timeout_support_is_not_called():
    from app.paddleocr_investigation import investigate
    class Gateway:
        def search(self,question,*,version,language,top_k):
            pytest.fail('unbounded gateway must not be called')
    result=investigate([{'check_id':'r','query':'OCR','versions':['v3.0.0']}],Gateway(),clock=lambda:0)
    assert result['stop_reason']=='DEADLINE_NOT_ENFORCEABLE'


def test_resume_rejects_changed_checks_and_does_not_repeat_completed_search():
    from app.paddleocr_investigation import investigate
    class Gateway:
        calls=0
        def search(self,question,**kw):self.calls+=1;return {'results':[]}
    g=Gateway(); checks=[{'check_id':'r','query':'OCR','versions':['v3.0.0']}]
    first=investigate(checks,g,clock=lambda:0)
    second=investigate(checks,g,clock=lambda:0,resume=first['resume'])
    assert g.calls==1 and second['unresolved']
    with pytest.raises(ValueError,match='identity'):
        investigate([{'check_id':'r','query':'其他','versions':['v3.0.0']}],g,clock=lambda:0,resume=first['resume'])


def test_forged_search_hit_cannot_become_checked_evidence():
    from app.paddleocr_investigation import investigate
    class Gateway:
        def search(self,question,**kw):return {'results':[{'chunk_id':'fake','version':'v3.0.0'}]}
    result=investigate([{'check_id':'r','query':'OCR','versions':['v3.0.0']}],Gateway(),clock=lambda:0)
    assert result['checked_evidence']==[]
    assert any(t['status']=='EVIDENCE_VALIDATION_REQUIRED' for t in result['trace'])
def test_cancel_preserves_unprocessed_checks_as_gaps():
    from app.paddleocr_investigation import investigate
    class Gateway:
        def search(self, question, **kwargs): return {'results': []}
    checks=[{'check_id':str(i),'query':'结果'+str(i),'versions':['v3.0.0']} for i in range(5)]
    result=investigate(checks,Gateway(),clock=lambda:0,cancelled=lambda:True)
    assert {r['check_id'] for r in result['unresolved']}=={str(i) for i in range(5)}
