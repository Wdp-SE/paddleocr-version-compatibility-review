"""Read-only Git/CI precheck for the registered original application. No LLM or OCR execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
for folder in ('demo-ui','versioned-rag-service'):
    sys.path.insert(0,str(ROOT/folder))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application',default='document-ocr')
    parser.add_argument('--head',default='HEAD')
    parser.add_argument('--base')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--advisory',action='store_true',help='Record findings without a failing exit code; never approves an upgrade.')
    args=parser.parse_args(argv)
    from services.registered_application import load_registry,collect_snapshot
    from src.paddleocr_compatibility import review_compatibility
    registry=load_registry()
    if args.application not in registry:
        parser.error('Application is not registered')
    snapshot=collect_snapshot(registry[args.application],args.head,base_commit=args.base)
    corpus=ROOT/'versioned-rag-service/public_corpus_paddleocr'
    index=SimpleNamespace(manifest=json.loads((corpus/'corpus_manifest.json').read_text(encoding='utf8')),
                          chunks=json.loads((corpus/'chunks.json').read_text(encoding='utf8')))
    def review(current):
        return review_compatibility(index,files=current['files'],source_version='v2.9.1',target_version='v3.0.0')
    result=review(snapshot)
    code=2 if result['summary']['risk_count'] else 3
    # Static absence of a finding is never sufficient for approval.
    payload={'schema_version':1,'application_id':snapshot['application_id'],
        'application_commit':snapshot['head_commit'],'base_application_commit':snapshot['base_commit'],
        'source_version':'v2.9.1','target_version':'v3.0.0',
        'gate':'ADVISORY_NOT_APPROVAL' if args.advisory else 'HUMAN_AND_RUNTIME_REVIEW_REQUIRED',
        'stage':'static_precheck_only','runtime_verified':False,'model_calls':0,
        'input_scope':snapshot['coverage'],'collection_gaps':snapshot['gaps'],
        'inventory':snapshot['inventory'],'review':result,
        'corpus_sha256':hashlib.sha256((corpus/'chunks.json').read_bytes()).hexdigest()}
    if snapshot['base_commit']:
        baseline=collect_snapshot(registry[args.application],snapshot['base_commit'])
        payload['baseline_review']=review(baseline)
        payload['changed_paths']=snapshot['changed_paths']
        payload['comparison_notice']='Both complete registered snapshots are reviewed; risk disappearance is not repair or runtime certification.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'application_commit':payload['application_commit'],'status':result['status'],
                      'risk_count':result['summary']['risk_count'],'gate':payload['gate']},ensure_ascii=False))
    return 0 if args.advisory else code


if __name__=='__main__':
    raise SystemExit(main())
