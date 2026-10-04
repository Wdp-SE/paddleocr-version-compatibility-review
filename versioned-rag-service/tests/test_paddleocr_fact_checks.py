def test_same_name_from_other_module_does_not_satisfy_requirement():
    from src.paddleocr_fact_checks import check_mechanical_facts
    req={'id':'r1','module':'ocr','symbol':'cpu_threads'}
    hit={'chunk_id':'c1','module':'doc_preprocessor','version':'v3.0.0','content':'cpu_threads: 8'}
    result=check_mechanical_facts([], [hit],requirements=[req])
    assert result['requirements']['r1']['status']=='MISSING_APPLICABLE_EVIDENCE'


def test_wrong_numeric_default_is_rejected_but_explanations_not_auto_approved():
    from src.paddleocr_fact_checks import check_mechanical_facts
    hit={'chunk_id':'c1','module':'ocr','version':'v3.0.0',
         'content':'| 参数 | 默认值 |\n| --- | --- |\n| cpu_threads | 8 |'}
    claims=[{'text':'cpu_threads 默认值为 99','evidence_ids':['c1']},
            {'text':'CPU 推理速度更快','evidence_ids':['c1']}]
    result=check_mechanical_facts(claims,[hit],requirements=[])
    assert result['claims'][0]['status']=='CONTRADICTED'
    assert result['claims'][1]['status']=='NOT_MECHANICALLY_CHECKABLE'
def test_call_argument_assignment_is_not_a_documented_default():
    from src.paddleocr_fact_checks import check_mechanical_facts
    out=check_mechanical_facts([{'text':'cpu_threads 默认值为 8','evidence_ids':['c']}],
        [{'chunk_id':'c','content':'engine = PaddleOCR(cpu_threads=10)','document_path':'docs/ocr.md'}],requirements=[])
    assert out['claims'][0]['status']=='NOT_MECHANICALLY_CHECKABLE'

def test_default_from_other_module_cannot_support_an_ocr_claim():
    from src.paddleocr_fact_checks import check_mechanical_facts
    req={'id':'r1','module':'ocr','symbols':['cpu_threads'],'version':'v3.0.0'}
    claim={'text':'OCR cpu_threads 默认值为 8','evidence_ids':['c']}
    hit={'chunk_id':'c','module':'text_detection','version':'v3.0.0',
         'content':'| 参数 | 默认值 |\n| cpu_threads | 8 |'}
    assert check_mechanical_facts([claim],[hit],requirements=[req])['claims'][0]['status']=='SCOPE_MISMATCH'

def test_explicit_claim_version_cannot_use_another_versions_default():
    from src.paddleocr_fact_checks import check_mechanical_facts
    reqs=[{'id':v,'module':'ocr','symbols':['cpu_threads'],'version':v} for v in ('v2.9.1','v3.0.0')]
    claim={'text':'v3.0.0 OCR cpu_threads 默认值为 8','evidence_ids':['old']}
    hit={'chunk_id':'old','module':'ocr','version':'v2.9.1','content':'| 参数 | 默认值 |\n| cpu_threads | 8 |'}
    assert check_mechanical_facts([claim],[hit],requirements=reqs)['claims'][0]['status']=='SCOPE_MISMATCH'

def test_versioned_comparison_accepts_each_claims_own_version_evidence():
    from src.paddleocr_fact_checks import check_mechanical_facts
    reqs=[{'id':v,'module':'ocr','symbols':['cpu_threads'],'version':v} for v in ('v2.9.1','v3.0.0')]
    hits=[{'chunk_id':v,'module':'ocr','version':v,'content':f'| 参数 | 默认值 |\n| cpu_threads | {n} |'}
          for v,n in [('v2.9.1',8),('v3.0.0',10)]]
    claims=[{'text':f'{v} OCR cpu_threads 默认值为 {n}','evidence_ids':[v]} for v,n in [('v2.9.1',8),('v3.0.0',10)]]
    assert [r['status'] for r in check_mechanical_facts(claims,hits,requirements=reqs)['claims']]==['CONSISTENT_PARSED_FACT']*2
