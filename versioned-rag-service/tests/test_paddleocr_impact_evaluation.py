from src.paddleocr_impact_evaluation import assess_evidence, assess_review, promotion_allowed
import json
import pytest


@pytest.fixture
def release_files(tmp_path,monkeypatch):
    import src.paddleocr_impact_evaluation as mod
    root=tmp_path; folder=root/'evaluation/paddleocr_impact_v3';folder.mkdir(parents=True)
    corpus=root/'versioned-rag-service/public_corpus_paddleocr';corpus.mkdir(parents=True)
    monkeypatch.setattr(mod,'__file__',str(root/'versioned-rag-service/src/paddleocr_impact_evaluation.py'))
    for env in ('PADDLEOCR_EMBEDDING_PATH','PADDLEOCR_RERANKER_PATH'):monkeypatch.delenv(env,raising=False)
    model=root/'model';model.mkdir();(model/'model.bin').write_bytes(b'original model')
    chunks=[{'chunk_id':'c','version':'v3.0.0','document_path':'docs/ocr.md','line_start':1,'content':'real_answer'}]
    cases=[{'id':s,'split':s,'version':'v3.0.0','facts':[{'module':'ocr','allowed_paths':['docs/ocr.md'],'tokens':['real_answer']}]} for s in ('dev','validation','sealed')]
    def write(name,value):
        p=folder/name;p.write_text(json.dumps(value),encoding='utf-8');return p
    (corpus/'chunks.json').write_text(json.dumps(chunks),encoding='utf-8')
    write('dataset.json',{'dataset_id':'test','label_origin':'test','retrieval_cases':cases,'review_cases':[],'negative_probes':[]})
    freeze={'files':{'evaluation/paddleocr_impact_v3/dataset.json':mod.file_hash(folder/'dataset.json')},
            'models':{'embedding':{'local_path':str(model),'files':{'model.bin':mod.file_hash(model/'model.bin')}},
                      'reranker':{'local_path':str(model),'files':{'model.bin':mod.file_hash(model/'model.bin')}}},
            'configs':{'bm25':['bm25',80],'candidate':['contextual_bm25',80]}}
    write('freeze.json',freeze)
    score=assess_evidence(cases[0],chunks)
    splits={s:{'rows':[{'id':s,'hits':[{'chunk_id':'c','version':'v3.0.0','module':'ocr'}],'score':score}],
                 'metrics':mod.aggregate([score])} for s in ('dev','validation','sealed')}
    report={'freeze_sha256':mod.file_hash(folder/'freeze.json'),'retrieval':{'bm25':splits,'candidate':splits},
            'schema_version':1,'dataset_id':'test','label_origin':'test','top_k':5,
            'configs':freeze['configs'],'selected_strategy':'bm25','development_candidate':'candidate',
            'promotion_passed':False,'review':[],'negative_probes':[], 'answer_accuracy':None,'hallucination_rate':None}
    write('sealed-opened.json',{'freeze_sha256':report['freeze_sha256'],'selected_before_opening':'bm25'})
    write('report.json',report)
    return mod,folder,corpus,model,report,write


def test_current_release_recomputes_and_accepts_canonical_evidence(release_files):
    mod,folder,corpus,model,report,write=release_files
    assert mod.load_impact_release(corpus) is not None

@pytest.mark.parametrize('tamper',['sealed','top_k','wrong_top_k','dataset_id','business_accuracy'])
def test_release_rejects_incomplete_or_mislabelled_report(release_files,tamper):
    mod,folder,corpus,model,report,write=release_files
    if tamper=='sealed':
        for splits in report['retrieval'].values():splits.pop('sealed',None)
    elif tamper=='top_k':report.pop('top_k')
    elif tamper=='wrong_top_k':report['top_k']=20
    elif tamper=='dataset_id':report['dataset_id']='other'
    else:report['business_impact_accuracy']=1.0
    write('report.json',report)
    assert mod.load_impact_release(corpus) is None


@pytest.mark.parametrize('tamper',['content','span','model','marker','review','negative','config'])
def test_release_rejects_identity_and_unbound_report_tampering(release_files,tamper):
    mod,folder,corpus,model,report,write=release_files
    if tamper=='content':report['retrieval']['bm25']['dev']['rows'][0]['hits'][0]['content']='real_answer'
    elif tamper=='span':report['retrieval']['bm25']['dev']['rows'][0]['hits'][0]['evidence_spans']=[{'line_start':1,'line_end':1,'content':'fabricated'}]
    elif tamper=='model':(model/'model.bin').write_bytes(b'changed model')
    elif tamper=='marker':write('sealed-opened.json',{'selected_before_opening':'candidate'})
    elif tamper=='review':report['review']=[{'id':'invented','C_rag':{'risk_hit_count':999}}]
    elif tamper=='negative':report['negative_probes']=[{'id':'invented','answer_correctness':True}]
    else:report['configs']['candidate']=['contextual_rerank',120]
    write('report.json',report)
    assert mod.load_impact_release(corpus) is None


def test_equivalent_fact_from_wrong_module_is_not_a_hit():
    case={'version':'v3.0.0','facts':[{'module':'ocr','allowed_paths':['ocr.md'],'tokens':['cpu_threads']}]}
    row={'version':'v3.0.0','module':'other','document_path':'ocr.md','content':'cpu_threads'}
    result=assess_evidence(case,[row])
    assert result['complete'] is False and result['wrong_module_count']==1


def test_canonical_tail_outside_selected_span_is_not_a_hit():
    case={'version':'v3.0.0','facts':[{'module':'ocr','allowed_paths':['ocr.md'],'tokens':['secret_tail']}]}
    row={'chunk_id':'c','version':'v3.0.0','module':'ocr','document_path':'ocr.md',
         'line_start':1,'content':'head\nsecret_tail','evidence_spans':[{'line_start':1,'line_end':1,'content':'head'}]}
    assert assess_evidence(case,[row])['complete'] is False


def test_speed_never_compensates_for_wrong_version_or_risks():
    baseline={'complete_count':3,'fact_hit_count':4,'wrong_version_count':0,'wrong_module_count':0}
    candidate={**baseline,'complete_count':4,'wrong_version_count':1,'p95_ms':1}
    assert not promotion_allowed(baseline,candidate)


def test_runtime_unexecuted_never_counts_as_regression_success():
    result=assess_review({'expected_rules':['removed_structure_api'],'expect_gap':False},
        {'findings':[{'rule_id':'removed_structure_api','status':'supported_risk'}],
         'regression_requirements':[{'status':'NOT_EXECUTED'}],'runtime_verified':False})
    assert result['runtime_success'] is None and result['risk_hit_count']==1
