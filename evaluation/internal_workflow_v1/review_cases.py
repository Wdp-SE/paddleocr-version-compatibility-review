"""Fresh structural probes, not a trained benchmark or proof of runtime success."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'versioned-rag-service'),str(ROOT/'change-review-agent')]
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_compatibility import review_compatibility
from src.paddleocr_impact_evaluation import assess_review
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from app.paddleocr_investigation import build_checks,investigate

CASES=[
 ('legacy-inline','from paddleocr import PaddleOCR\ne=PaddleOCR()\nr=e.ocr("page.png")\ntext=r[0][0][1][0]\n',['legacy_ocr_result'],False),
 ('legacy-function','from paddleocr import PaddleOCR\ndef scan(path):\n    engine=PaddleOCR()\n    pages=engine.ocr(path)\n    return pages[0][0][1][0]\n',['legacy_ocr_result'],False),
 ('legacy-alias','from paddleocr import PaddleOCR as Engine\nengine=Engine()\nresult=engine.ocr("page.png")\nvalue=result[0][0][1][0]\n',['legacy_ocr_result'],False),
 ('legacy-loop','from paddleocr import PaddleOCR\nengine=PaddleOCR()\nresult=engine.ocr("page.png")\nfor page in result:\n    for line in page:\n        value=line[1][0]\n',['legacy_ocr_result'],False),
 ('structure-direct','from paddleocr import PPStructure\nengine=PPStructure()\nresult=engine("page.png")\n',['removed_ppstructure'],False),
 ('structure-alias','from paddleocr import PPStructure as Layout\nengine=Layout()\nresult=engine("page.png")\n',['removed_ppstructure'],False),
 ('basic-direct','from paddleocr import PaddleOCR\nengine=PaddleOCR()\nresult=engine.ocr("page.png")\n',[],False),
 ('basic-function','from paddleocr import PaddleOCR\ndef scan(image):\n    engine=PaddleOCR()\n    return engine.ocr(image)\n',[],False),
 ('unrelated-list','records=[[[("hello",0.9)]]]\ntext=records[0][0][0][0]\n',[],True),
 ('shadow-sdk','from paddleocr import PaddleOCR\nPaddleOCR=object\nengine=PaddleOCR()\n',[],True),
 ('dynamic-load','import importlib\nm=importlib.import_module("paddleocr")\nengine=getattr(m,"PaddleOCR")()\nresult=engine.ocr("page.png")\n',[],True),
 ('dynamic-kwargs','from paddleocr import PaddleOCR\nsettings=load_settings()\nengine=PaddleOCR(**settings)\n',[],True),
]


def main():
    folder=Path(__file__).parent/'final'
    dataset=[{'id':name,'files':[{'path':'application/main.py','content':body}],'expected_rules':rules,'expect_gap':gap} for name,body,rules,gap in CASES]
    def add(name,files,rules,gap=False):
        dataset.append({'id':name,'files':[{'path':p,'content':c} for p,c in files], 'expected_rules':rules,'expect_gap':gap})
    for name,body,rules,gap in CASES[:6]:
        add(name+'-with-consumer', [('application/ocr_client.py',body),('application/output.py','def serialize(text):\n    return {"text": text}\n')],rules,gap)
    call='from paddleocr import PaddleOCR\ndef recognize(path):\n    engine=PaddleOCR()\n    return engine.ocr(path)\n'
    add('wrapper-direct-consumer',[('application/ocr_client.py',call),('application/main.py','from .ocr_client import recognize\ndef process(path):\n    pages=recognize(path)\n    return pages[0][0][1][0]\n')],['legacy_ocr_result'])
    add('wrapper-export-consumer',[('application/ocr_client.py',call),('application/consumer.py','from .ocr_client import recognize\ndef extract(image):\n    result=recognize(image)\n    return result[0][0][1][0]\n')],['legacy_ocr_result'])
    add('basic-with-contract',[('application/ocr_client.py',call),('application/contracts/result.json','{"text": "string", "schema_version": 1}')],[])
    add('basic-with-tests',[('application/ocr_client.py',call),('application/tests/test_contract.py','def test_empty_output():\n    result={"text": ""}\n    assert isinstance(result["text"],str)\n')],[])
    add('missing-wrapper',[('application/main.py','from .missing import recognize\ndef scan(path):\n    return recognize(path)\n')],[],True)
    add('basic-malformed-contract',[('application/ocr_client.py',call),('application/contracts/result.json','{"text":')],[],True)
    locations={'legacy-inline':4,'legacy-function':5,'legacy-alias':4,'legacy-loop':6,'structure-direct':2,'structure-alias':2}
    for case in dataset:
        name=case['id'];base=name.removesuffix('-with-consumer')
        if base in locations:
            path='application/ocr_client.py' if name.endswith('-with-consumer') else 'application/main.py'
            case['expected_locations']=[{'rule_id':case['expected_rules'][0],'path':path,'line':locations[base]}]
        elif name in ('wrapper-direct-consumer','wrapper-export-consumer'):
            case['expected_locations']=[{'rule_id':'legacy_ocr_result','path':case['files'][1]['path'],'line':4}]
        else:case['expected_locations']=[]
    data_path=folder/'review-dataset.json';result_path=folder/'review-results.json'
    if data_path.exists() or result_path.exists():raise ValueError('preserve opened cases and results')
    data_path.write_text(json.dumps({'label_origin':'developer-authored structural probes; no independent blind audit','cases':dataset},ensure_ascii=False,indent=2),encoding='utf8')
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr');search=configured_evidence_search(index)
    registry={h['chunk_id']:h for h in index.chunks}
    class Gateway:
        def search(self,query,*,version,request_timeout,**_):
            return search.search(plan_query(query,versions=(version,)),top_k=5,candidate_budget=80)
    rows=[]
    for case in dataset:
        report=review_compatibility(index,source_version='v2.9.1',target_version='v3.0.0',files=case['files'])
        investigation=investigate(build_checks(report),Gateway(),clock=time.monotonic,
            validate_evidence=lambda h,remaining:registry[h['chunk_id']])
        expected=case['expected_locations']
        found=[f for f in report['findings'] if f['status']=='supported_risk']
        matched=sum(any(f['rule_id']==loc['rule_id'] and f['application']['path']==loc['path']
            and f['application']['line']<=loc['line']<=f['application']['end_line'] for f in found) for loc in expected)
        rows.append({'id':case['id'],'score':assess_review(case,report),'static_report':report,
            'risk_locations':{'expected':len(expected),'matched':matched,'missed':len(expected)-matched,
                'false_positive':sum(not any(f['rule_id']==loc['rule_id'] and f['application']['path']==loc['path']
                and f['application']['line']<=loc['line']<=f['application']['end_line'] for loc in expected) for f in found)},
            'check_coverage':investigation['check_assessments'],'unresolved':investigation['unresolved'],
            'search_calls':investigation['search_calls'],'stop_reason':investigation['stop_reason']})
    result_path.write_text(json.dumps({'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'runtime_executed':False,'model_planner_executed':False,'rows':rows},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'cases':len(rows),'false_negative':sum(r['score']['false_negative_count'] for r in rows),
        'false_positive':sum(r['score']['false_positive_count'] for r in rows),
        'missed_risk_locations':sum(r['risk_locations']['missed'] for r in rows),
        'unknown_marked_compatible':sum(r['score']['unknown_marked_compatible'] for r in rows),
        'unresolved_checks':sum(len(r['unresolved']) for r in rows)},ensure_ascii=False))


if __name__=='__main__':main()
