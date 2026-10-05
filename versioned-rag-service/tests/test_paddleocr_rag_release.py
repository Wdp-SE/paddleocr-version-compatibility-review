"""A published score must be complete, pinned, and reproducible from source."""
import hashlib
import json
from types import SimpleNamespace
import pytest

from src.paddleocr_impact_evaluation import assess_evidence, aggregate
from src.paddleocr_retrieval_views import evidence_text, generation_evidence_text, module_for


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    import src.paddleocr_rag_release as release
    import src.public_knowledge as knowledge
    root=tmp_path
    monkeypatch.setattr(release, '__file__', str(root/'versioned-rag-service/src/paddleocr_rag_release.py'))
    corpus=root/'versioned-rag-service/public_corpus_paddleocr'
    folder=root/'evaluation/paddleocr_quality_v4/final'
    folder.mkdir(parents=True)
    hit={'chunk_id':'c1','content':'cpu_threads 默认 8','version':'v3.0.0',
         'document_path':'docs/version3.x/pipeline_usage/OCR.md','line_start':1,'line_end':1}
    hit['module']=module_for(hit)
    monkeypatch.setattr(knowledge,'PublicKnowledgeIndex',lambda path:SimpleNamespace(chunks=[hit]))
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8')
    cases=[{'id':split,'split':split,'version':'v3.0.0','facts':[
        {'module':'ocr','allowed_paths':[hit['document_path']],'tokens':['cpu_threads','8']}]} for split in ('dev','holdout')]
    dataset='evaluation/paddleocr_quality_v4/final/dataset.json'
    write(dataset,{'cases':cases})
    files={'versioned-rag-service/'+name for name in ('public_corpus_paddleocr/chunks.json',
        'src/paddleocr_query_plan.py','src/paddleocr_retrieval_views.py','src/paddleocr_evidence_search.py',
        'src/paddleocr_fact_checks.py','src/public_api.py','src/claim_support.py',
        'src/paddleocr_rag_release.py','src/paddleocr_impact_evaluation.py','src/public_knowledge.py')}
    for name in files: write(name,{})
    files.add(dataset)
    write('evaluation/paddleocr_quality_v4/final/freeze.json',{'files':{}})
    files.add('evaluation/paddleocr_quality_v4/final/freeze.json')
    retrieval={}
    for name in ('legacy','bm25','llm'):
        retrieval[name]={}
        for case in cases:
            score=assess_evidence(case,[hit]);metrics=aggregate([score])
            report={'strategy':name,'split':case['split'],
                    'freeze_sha256':hashlib.sha256((folder/'freeze.json').read_bytes()).hexdigest(),
                    'metrics':metrics,'rows':[{'id':case['id'],'score':score,'hits':[
                        {k:hit[k] for k in ('chunk_id','version','document_path')}|
                        {'text':evidence_text(hit),'generation_text':generation_evidence_text(hit)}]}]}
            path=f'evaluation/paddleocr_quality_v4/final/{name}-{case["split"]}.json'
            write(path,report);files.add(path)
            retrieval[name][case['split']]={'metrics':metrics}
    write('evaluation/paddleocr_quality_v4/final/release.json',{'schema_version':1,
        'selected_strategy':'contextual_llm_rerank','candidate_budget':40,'top_k':5,'retrieval':retrieval})
    files.add('evaluation/paddleocr_quality_v4/final/release.json')
    def resign(exclude=()):
        write('evaluation/paddleocr_quality_v4/final/release-lock.json',{'files':{
            name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files if name not in exclude}})
    resign()
    return release,corpus,folder,resign


def test_current_release_requires_every_report_it_reads(release_fixture):
    release,corpus,folder,resign=release_fixture
    assert release.load_rag_release(corpus)
    resign(['evaluation/paddleocr_quality_v4/final/llm-holdout.json'])
    assert release.load_rag_release(corpus) is None


@pytest.mark.parametrize('damage',['empty_rows','duplicate_source','fabricated_score','source_text','freeze_identity'])
def test_release_rejects_incomplete_or_noncanonical_report(release_fixture,damage):
    release,corpus,folder,resign=release_fixture
    path=folder/'llm-holdout.json';report=json.loads(path.read_text(encoding='utf8'))
    if damage=='empty_rows':report['rows']=[]
    elif damage=='duplicate_source':report['rows'][0]['hits']*=2
    elif damage=='fabricated_score':report['rows'][0]['score']['complete']=False
    elif damage=='source_text':report['rows'][0]['hits'][0]['generation_text']='伪造内容'
    else:report['freeze_sha256']='wrong'
    path.write_text(json.dumps(report,ensure_ascii=False),encoding='utf8');resign()
    assert release.load_rag_release(corpus) is None
