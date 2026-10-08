import json
from types import SimpleNamespace
import pytest
from src.paddleocr_impact_evaluation import assess_evidence,aggregate
from src.paddleocr_retrieval_views import module_for


@pytest.fixture
def comparison(tmp_path,monkeypatch):
    import src.paddleocr_retrieval_comparison as loader
    import src.public_knowledge as knowledge
    monkeypatch.setattr(loader,'__file__',str(tmp_path/'versioned-rag-service/src/paddleocr_retrieval_comparison.py'))
    corpus=tmp_path/'versioned-rag-service/public_corpus_paddleocr'
    folder=tmp_path/'evaluation/retrieval_upgrade_v2/final';folder.mkdir(parents=True)
    hit={'chunk_id':'c','content':'cpu_threads 默认 8','version':'v3.0.0',
        'document_path':'docs/version3.x/pipeline_usage/OCR.md','line_start':1,'line_end':1}
    hit['module']=module_for(hit)
    monkeypatch.setattr(knowledge,'PublicKnowledgeIndex',lambda path:SimpleNamespace(chunks=[hit]))
    def write(p,v):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False),encoding='utf8')
    cases=[{'id':s,'split':s,'version':'v3.0.0','facts':[{'module':'ocr',
        'allowed_paths':[hit['document_path']],'tokens':['cpu_threads','8']}]} for s in ('regression','validation','sealed')]
    dataset={'cases':cases}
    write(folder/'dataset.json',dataset);write(folder.parent/'dataset.json',dataset)
    code=tmp_path/'versioned-rag-service/src/code.py';write(code,{})
    cfgs={n:{'strategy':s,'top_k':5,'candidate_budget':40,'structure_windows':False}
          for n,s in [('window_bm25','contextual_bm25'),('structure_hybrid','contextual_rrf')]}
    write(folder/'freeze.json',{'files':{code.relative_to(tmp_path).as_posix():loader.digest(code)},'configs':cfgs})
    retrieval={}
    for n,cfg in cfgs.items():
        splits=('regression','validation','sealed') if n=='window_bm25' else ('regression','validation')
        retrieval[n]={}
        for s in splits:
            case=next(c for c in cases if c['split']==s);score=assess_evidence(case,[hit]);metrics=aggregate([score])
            report={'strategy':n,'split':s,'freeze_sha256':loader.digest(folder/'freeze.json'),
                'metrics':metrics,'execution_success_count':1,'latency_p50_ms':1,
                'rows':[{'id':s,'score':score,'diagnostics':{'actual_strategy':cfg['strategy'],'fallback_reason':None},
                    'hits':[{k:hit[k] for k in ('chunk_id','version','document_path')}|{'text':hit['content'],'generation_text':hit['content']}]}]}
            write(folder/f'{n}-{s}.json',report)
            retrieval[n][s]={'metrics':metrics,'execution_success_count':1}
    write(folder/'selection.json',{'selected_before_sealed':'window_bm25','freeze_sha256':loader.digest(folder/'freeze.json')})
    write(folder/'release.json',{'schema_version':1,'selected':'window_bm25','selected_strategy':'contextual_bm25',
        **{k:v for k,v in cfgs['window_bm25'].items() if k!='strategy'},'retrieval':retrieval})
    def resign():
        paths=[code]+[p for p in folder.iterdir() if p.name!='release-lock.json']
        write(folder/'release-lock.json',{'files':{p.relative_to(tmp_path).as_posix():loader.digest(p) for p in paths}})
    resign()
    return loader,corpus,folder,code,resign


def test_verified_comparison_recomputes_counts_and_invalidates_changed_code(comparison):
    loader,corpus,folder,code,resign=comparison
    result=loader.load_retrieval_comparison(corpus)
    assert result and result['promotion_eligible']
    assert len(result['rows'])==5
    code.write_text('changed',encoding='utf8')
    assert loader.load_retrieval_comparison(corpus) is None


def test_partial_model_failure_is_disclosed_and_cannot_win(comparison):
    loader,corpus,folder,code,resign=comparison
    p=folder/'structure_hybrid-regression.json';report=json.loads(p.read_text(encoding='utf8'))
    report['evaluation_status']='rejected_model_output';report['planned_count']=1;report['execution_success_count']=0
    report['rows'][0]['diagnostics']={'actual_strategy':'contextual_bm25','fallback_reason':'SEMANTIC_INVALID_OUTPUT'}
    p.write_text(json.dumps(report),encoding='utf8')
    release_path=folder/'release.json';release=json.loads(release_path.read_text(encoding='utf8'))
    release['retrieval']['structure_hybrid']={'regression':{'metrics':report['metrics'],
        'execution_success_count':0,'evaluation_status':'rejected_model_output','planned_count':1}}
    release['disqualified']={'structure_hybrid':{'split':'regression','reason':'SEMANTIC_INVALID_OUTPUT'}}
    release_path.write_text(json.dumps(release),encoding='utf8');resign()
    result=loader.load_retrieval_comparison(corpus)
    assert result and result['selected']=='window_bm25'
    assert result['disqualified']


@pytest.mark.parametrize('damage',['citation','score','empty','neural_fallback','selection'])
def test_resigning_does_not_make_false_evidence_or_selection_valid(comparison,damage):
    loader,corpus,folder,code,resign=comparison
    p=folder/'window_bm25-validation.json';report=json.loads(p.read_text(encoding='utf8'))
    if damage=='citation':report['rows'][0]['hits'][0]['text']='invented'
    elif damage=='score':report['rows'][0]['score']['complete']=False
    elif damage=='empty':report['rows']=[]
    elif damage=='neural_fallback':report['rows'][0]['diagnostics']['fallback_reason']='MODEL_UNAVAILABLE'
    else:
        p=folder/'selection.json';report=json.loads(p.read_text(encoding='utf8'));report['selected_before_sealed']='structure_hybrid'
    p.write_text(json.dumps(report,ensure_ascii=False),encoding='utf8');resign()
    assert loader.load_retrieval_comparison(corpus) is None
