"""Actual public/original-data checks; generated output still needs human review."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'change-review-agent'), str(ROOT / 'demo-ui')]
from app.paddleocr_review import PaddleOCRReviewAgent
from services.interview_examples import DEMO_CASES, demo_application
from services.public_knowledge_client import PublicKnowledgeClient
from services.query_experience import interview_queries


def run(base_url, output):
    if output.exists():
        raise ValueError('Do not overwrite measured checks')
    session = requests.Session(); session.trust_env = False
    client = PublicKnowledgeClient(base_url, session=session, retry_limit=0)
    workspace = client.workspace()
    assert workspace['workspace_id'] == 'paddleocr'
    rows = []
    report = {'assessment': 'structural_and_operational_regression_not_semantic_accuracy',
              'workspace': workspace, 'rows': rows, 'source_fingerprints': {
                  p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  for folder in ('versioned-rag-service/src', 'demo-ui/services', 'change-review-agent/app')
                  for p in (ROOT / folder).glob('*.py')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save(row):
        rows.append(row)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(row['kind'], row['label'], row['response'].get('status'), row['seconds'], flush=True)
    questions = list(zip(interview_queries(workspace), ('v2.9.1', 'v3.0.0', 'all', 'v3.0.0')))
    data = json.loads((ROOT / 'evaluation/retrieval_upgrade_v2/final/dataset.json').read_text(encoding='utf8'))
    questions += [(c['query'], c['version']) for c in data['cases'] if c['id'] in ('fresh-17', 'fresh-23')]
    for number, (question, version) in enumerate(questions, 1):
        started = time.perf_counter()
        result = client.query_official(question, version=version, language='zh', top_k=5,
                                       retrieval_policy='paddleocr_evidence')
        save({'kind': 'rag', 'label': number, 'question': question, 'version': version,
              'seconds': round(time.perf_counter() - started, 2), 'response': result})
    for label in DEMO_CASES:
        started = time.perf_counter()
        result = PaddleOCRReviewAgent(client).analyze(demo_application(label), generate_advice=True)
        save({'kind': 'agent', 'label': label, 'seconds': round(time.perf_counter() - started, 2), 'response': result})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8786')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base_url, args.output)
