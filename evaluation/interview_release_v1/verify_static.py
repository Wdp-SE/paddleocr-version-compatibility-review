"""Re-run known structural probes. Never execute supplied application files."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'versioned-rag-service'),str(ROOT/'change-review-agent')]
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_compatibility import review_compatibility
from src.paddleocr_impact_evaluation import assess_review
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from app.paddleocr_investigation import build_checks,investigate


def main():
    folder=Path(__file__).parent/'final6'
    dataset=json.loads((ROOT/'evaluation/internal_workflow_v1/final/review-dataset.json').read_text(encoding='utf8'))
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    search=configured_evidence_search(index)
    registry={h['chunk_id']:h for h in index.chunks}
    class Gateway:
        def search(self,query,*,version,request_timeout,**_):
            return search.search(plan_query(query,versions=(version,)),top_k=5,candidate_budget=40)
    rows=[]
    for case in dataset['cases']:
        report=review_compatibility(index,source_version='v2.9.1',target_version='v3.0.0',files=case['files'])
        investigation=investigate(build_checks(report),Gateway(),clock=time.monotonic,
            validate_evidence=lambda h,remaining:registry[h['chunk_id']])
        expected=case['expected_locations']
        found=[f for f in report['findings'] if f['status']=='supported_risk']
        def matches(f,loc):
            return f['rule_id']==loc['rule_id'] and f['application']['path']==loc['path'] and f['application']['line']<=loc['line']<=f['application']['end_line']
        matched=sum(any(matches(f,loc) for f in found) for loc in expected)
        rows.append({'id':case['id'],'score':assess_review(case,report),'static_report':report,
            'risk_locations':{'expected':len(expected),'matched':matched,'missed':len(expected)-matched,
                'false_positive':sum(not any(matches(f,loc) for loc in expected) for f in found)},
            'check_coverage':investigation['check_assessments'],'unresolved':investigation['unresolved'],
            'search_calls':investigation['search_calls'],'stop_reason':investigation['stop_reason']})
    summary={'cases':len(rows),'expected_risk_locations':sum(r['risk_locations']['expected'] for r in rows),
        'matched_risk_locations':sum(r['risk_locations']['matched'] for r in rows),
        'false_positive':sum(r['risk_locations']['false_positive'] for r in rows),
        'unknown_marked_compatible':sum(r['score']['unknown_marked_compatible'] for r in rows)}
    result={'release_lock_sha256':hashlib.sha256((folder/'release-lock.json').read_bytes()).hexdigest(),
        'label_origin':'24 known developer-authored structural probes; not blind or runtime tests',
        'runtime_executed':False,'model_planner_executed':False,'summary':summary,'rows':rows}
    output=folder/'static-regression.json'
    if output.exists():raise ValueError('preserve measured result')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
