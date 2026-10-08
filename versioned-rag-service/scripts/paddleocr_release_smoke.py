"""Verify a deployed revision and pinned PaddleOCR corpus; never infer UI revision from HTTP 200."""
import argparse
import json
from pathlib import Path

import requests


def run(base_url, expected_revision, ui_url, output):
    session = requests.Session()
    session.trust_env = False
    def get(path):
        response = session.get(base_url.rstrip('/') + path, timeout=60)
        response.raise_for_status()
        return response.json()
    health, workspace = get('/health'), get('/public/workspace')
    if not health.get('alive') or not health.get('rag_ready'):
        raise RuntimeError('Backend not ready')
    for payload in (health, workspace):
        if payload.get('build_revision') != expected_revision:
            raise RuntimeError('Backend revision mismatch')
        if payload.get('workspace_id') != 'paddleocr':
            raise RuntimeError('Unexpected workspace')
    local = json.loads((Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr/corpus_manifest.json').read_text(encoding='utf8'))
    if workspace.get('repository') != 'PaddlePaddle/PaddleOCR' or workspace.get('languages') != ['zh']:
        raise RuntimeError('Unexpected publisher or language scope')
    documents = get('/public/documents')['documents']
    if len(documents) != len(local['sources']):
        raise RuntimeError('Source count mismatch')
    for version in ('v2.9.1', 'v3.0.0'):
        response = session.post(base_url.rstrip('/') + '/public/search', timeout=60, json={
            'query': 'PaddleOCR OCR 结果 rec_texts', 'version': version, 'language': 'zh', 'top_k': 5})
        response.raise_for_status()
        hits = response.json()['results']
        if not hits or any(h['version'] != version or h['repository'] != 'PaddlePaddle/PaddleOCR'
                           or h['commit'] != local['versions'][version]['commit'] for h in hits):
            raise RuntimeError('Version or provenance mismatch')
    ui = session.get(ui_url, timeout=60)
    ui.raise_for_status()
    report = {'backend_verification': 'PASS', 'expected_revision': expected_revision,
              'health': health, 'workspace': workspace, 'document_count': len(documents),
              'ui_http_status': ui.status_code, 'ui_revision_verification': 'NOT_VERIFIED_BY_HTTP',
              'note': 'The UI source revision must be checked separately in the running application.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError('Do not overwrite previous deployment records')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ('backend_verification', 'document_count', 'ui_http_status', 'ui_revision_verification')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--ui-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base_url, args.expected_revision, args.ui_url, args.output)
