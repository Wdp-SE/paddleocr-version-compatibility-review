"""Repeat actual UI examples. Operational regression, not answer accuracy."""
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


def run(base_url, output, repeats):
    if output.exists():
        raise ValueError('Preserve existing measured reports; choose a new output')
    session = requests.Session()
    session.trust_env = False
    client = PublicKnowledgeClient(base_url, session=session, retry_limit=0)
    workspace = client.workspace()
    assert workspace['workspace_id'] == 'paddleocr'
    questions = interview_queries(workspace)
    assert len(questions) == 4
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                    for folder in ('versioned-rag-service/src', 'demo-ui/services', 'change-review-agent/app')
                    for p in (ROOT / folder).glob('*.py')}
    report = {'assessment': 'operational_regression_not_answer_accuracy',
              'workspace': workspace, 'source_fingerprints': fingerprints, 'rows': rows}
    for repeat in range(1, repeats + 1):
        for question, version in zip(questions, ('v2.9.1', 'v3.0.0', 'all', 'v3.0.0')):
            started = time.perf_counter()
            try:
                result = client.query_official(question, version=version, language='zh', top_k=5,
                                               retrieval_policy='paddleocr_evidence')
            except Exception as exc:
                result = {'error': type(exc).__name__, 'message': str(exc)}
            rows.append({'kind': 'rag', 'repeat': repeat, 'question': question, 'version': version,
                         'seconds': round(time.perf_counter() - started, 2), 'response': result})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print('RAG', repeat, version, result.get('status'), rows[-1]['seconds'], flush=True)
        for label in DEMO_CASES:
            started = time.perf_counter()
            files = demo_application(label)
            try:
                result = PaddleOCRReviewAgent(client).analyze(files, generate_advice=True)
            except Exception as exc:
                result = {'error': type(exc).__name__, 'message': str(exc)}
            rows.append({'kind': 'agent', 'repeat': repeat, 'case': label, 'files': files,
                         'seconds': round(time.perf_counter() - started, 2), 'response': result})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print('AGENT', repeat, label, result.get('status'), rows[-1]['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8782')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    run(args.base_url, args.output, args.repeats)
