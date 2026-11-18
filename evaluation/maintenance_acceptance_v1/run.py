"""Local acceptance runner. Model assessment and source coverage are not accuracy.

Outputs contain public-source excerpts and original demo data. Use an ignored
output directory. --generate calls the already configured external provider.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'demo-ui'))
sys.path.insert(0, str(ROOT/'versioned-rag-service'))
from src.build_identity import service_identity_digest
from services.knowledge_examples import EXAMPLES
from cases import FACTS, PROBES
from acceptance_report import case_fingerprint, summarize, identity_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    parser.add_argument('--output', required=True)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--suite', choices=['demos', 'maintenance30', 'all','depth12'], default='demos')
    parser.add_argument('--only', nargs='*', help='Recheck selected case IDs without repeating the full set')
    args = parser.parse_args()
    out = Path(args.output)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Output directory already exists; use a new directory to preserve previous evidence.')
    session = requests.Session(); session.trust_env = False
    initial = session.get(args.url+'/public/workspace', timeout=30)
    initial.raise_for_status(); identity = initial.json()
    expected_identity = service_identity_digest(identity)
    cases = [{'id': f'example-{i+1:02}', 'split': 'known_demo', 'query': e.question,
              'version': e.scope, 'expected_facts': facts, 'purpose': e.purpose}
             for i, (e, facts) in enumerate(zip(EXAMPLES, FACTS))]
    cases += [{'id': f'probe-{i+1:02}', 'split': 'wording_probe_not_independent_holdout',
               'query': q, 'version': v, 'expected_facts': []} for i, (v, q) in enumerate(PROBES)]
    if args.suite=='depth12':
        cases=json.loads((ROOT/'evaluation/depth_quality_v1/cases.json').read_text(encoding='utf8'))['cases']
    elif args.suite != 'demos':
        extended = json.loads((Path(__file__).parent/'maintenance30.json').read_text(encoding='utf-8'))
        extra = [{**r, 'split': 'developer_authored_regression_not_independent_blind'} for r in extended['cases']]
        cases = extra if args.suite == 'maintenance30' else cases + extra
    from src.paddleocr_query_plan import plan_query
    from src.public_scope import application_boundary_answer
    for case in cases:
        if 'question_type' not in case:
            versions=('v2.9.1','v3.0.0') if case['version']=='all' else (case['version'],)
            plan=plan_query(case['query'],versions=versions)
            case['question_type']=('application_boundary' if application_boundary_answer(case['query']) else
                'compound_fact' if len(plan['requirements'])>1 else 'component_fact')
    if args.only:
        known = {case['id'] for case in cases}
        if set(args.only)-known: parser.error('Unknown case IDs')
        cases = [case for case in cases if case['id'] in args.only]
    (out/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    case_hash = case_fingerprint(cases)
    def check(case):
        local = requests.Session(); local.trust_env = False
        started = time.monotonic()
        payload = {'query': case['query'], 'version': case['version'], 'language': 'zh', 'top_k': 5,
                   'retrieval_policy': 'paddleocr_evidence', 'evidence_strategy': 'contextual_bm25'}
        response_identity = None
        try:
            response = local.post(args.url+('/public/query' if args.generate else '/public/search'), json=payload, timeout=180)
            response_identity = response.headers.get('X-Service-Identity')
            response.raise_for_status(); value = response.json()
        except Exception as exc:
            value = {'status': 'CLIENT_ERROR', 'failure_type': type(exc).__name__}
        return {**case, 'seconds': round(time.monotonic()-started, 3), 'result': value,
                'response_identity_sha256': response_identity,
                'human_verdict': 'PENDING', 'limitation': 'Model completeness is not human fact accuracy.'}
    rows = []
    def save_report(status, after=None):
        report = {'identity': identity, 'identity_after': after, 'service_identity_status': status,
                  'run_valid': status == 'CONSISTENT' and len(rows) == len(cases),
                  'case_set_sha256': case_hash,
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'rows': sorted(rows, key=lambda r:r['id']), 'summary': summarize(rows),
                  'mode': 'generated' if args.generate else 'retrieval_only'}
        temporary = out/'results.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(out/'results.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check, c) for c in cases]):
            row = future.result(); rows.append(row); result = row['result']
            print(json.dumps({'id': row['id'], 'seconds': row['seconds'], 'status': result.get('status'),
                'complete': result.get('answer_completeness'), 'correction': result.get('correction', {}).get('status')}, ensure_ascii=False), flush=True)
            save_report('RUNNING')
    try:
        final = session.get(args.url+'/public/workspace', timeout=30)
        final.raise_for_status(); after = final.json()
    except (requests.RequestException, ValueError):
        save_report('UNVERIFIED')
        return
    save_report(identity_status(expected_identity, service_identity_digest(after), rows), after)


if __name__ == '__main__': main()
