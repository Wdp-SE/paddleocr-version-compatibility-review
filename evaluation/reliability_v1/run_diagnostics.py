"""Known regression questions: stage diagnosis and optional oracle intervention.

This reuses previously observed validation questions. It is not a new blind set.
The oracle supplies exact source units; it changes context size and distractors,
so a successful oracle run does not prove a single retrieval root cause.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'versioned-rag-service'))
from fastapi.testclient import TestClient
from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app
from src.paddleocr_query_plan import plan_query
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_impact_evaluation import assess_evidence
from src.paddleocr_retrieval_views import build_structure_views, materialize_evidence
from src.retrieval_diagnostics import diagnose_evidence_flow


def run(output, generate_ids, *, structure_windows=True, max_windows=4, candidate_budget=80, top_k=5,
        splits=('validation',)):
    if output.exists():
        raise ValueError('Do not overwrite measured runs')
    index = PublicKnowledgeIndex(ROOT / 'versioned-rag-service/public_corpus_paddleocr')
    data_path = ROOT / 'evaluation/retrieval_upgrade_v2/final/dataset.json'
    cases = [r for r in json.loads(data_path.read_text(encoding='utf8'))['cases'] if r['split'] in splits]
    files = [data_path, Path(__file__), *sorted((ROOT / 'versioned-rag-service/src').glob('*.py')),
             index.root / 'corpus_manifest.json', index.root / 'chunks.json']
    report = {'assessment': 'known_literal_regression_and_model_acceptance_not_answer_accuracy',
              'config': {'strategy': 'contextual_bm25', 'structure_windows': structure_windows,
                         'max_windows_per_source': max_windows, 'candidate_budget': candidate_budget, 'top_k': top_k},
              'source_fingerprints': {
                  p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')
                    if p.suffix == '.py' else p.read_bytes()).hexdigest() for p in files}, 'rows': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    search = configured_evidence_search(index, 'contextual_bm25', structure_windows=structure_windows)
    with TestClient(create_app(index=index)) as client:
        if generate_ids and client.app.state.public_generator is None:
            raise ValueError('Configured model required; no fake fallback acceptance')
        for case in cases:
            plan = plan_query(case['query'], versions=(case['version'],))
            result = search.search(plan, top_k=top_k, candidate_budget=candidate_budget,
                                   strategy='contextual_bm25', max_windows_per_source=max_windows)
            scores = {name: assess_evidence(case, hits) for name, hits in (
                ('source', index.chunks), ('candidates', result['candidate_evidence']), ('context', result['results']))}
            row = {'id': case['id'], 'split': case['split'], 'query': case['query'], 'scores': scores, 'responses': {}}
            if case['id'] in generate_ids:
                oracle_views = []
                for fact in case['facts']:
                    for view in build_structure_views(index.chunks):
                        parent = search.parents[view['parent_chunk_id']]
                        if (parent['version'] == case['version'] and parent['document_path'] in fact['allowed_paths']
                                and all(token in view['content'] for token in fact['tokens'])):
                            oracle_views.append(view); break
                oracle = materialize_evidence(index.chunks, oracle_views)
                if not assess_evidence(case, oracle)['complete']:
                    raise ValueError('Oracle source units do not cover the existing labels')
                for name, hits in (('normal', result['results']), ('oracle', oracle)):
                    def execution(*args, **kwargs):
                        return {'results': copy.deepcopy(hits), 'requested_policy': 'paddleocr_evidence',
                                'actual_policy': 'contextual_bm25', 'requirements': plan['requirements'],
                                'route': 'LOOKUP', 'rerank_status': 'NOT_REQUESTED', 'rerank_diagnostics': {},
                                'candidate_count': len(result['candidate_evidence']), 'final_evidence_count': len(hits),
                                'retrieval_latency_ms': 0, 'rerank_latency_ms': 0}
                    started = time.perf_counter()
                    with patch('src.public_api._execute_public_search', execution):
                        response = client.post('/public/query', json={'query': case['query'], 'version': case['version'],
                                                                    'language': 'zh', 'retrieval_policy': 'paddleocr_evidence'})
                    response.raise_for_status()
                    row['responses'][name] = {'seconds': round(time.perf_counter() - started, 2), 'result': response.json()}
            accepted = lambda name: row['responses'][name]['result'].get('status') == 'OK' if name in row['responses'] else None
            row['diagnosis'] = diagnose_evidence_flow(source_complete=scores['source']['complete'],
                candidate_complete=scores['candidates']['complete'], context_complete=scores['context']['complete'],
                accepted=accepted('normal'), oracle_accepted=accepted('oracle'))
            report['rows'].append(row)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print(case['id'], row['diagnosis'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generate-ids', default='')
    parser.add_argument('--structure-windows', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-windows', type=int, default=4)
    parser.add_argument('--candidate-budget', type=int, default=80)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--splits', default='validation')
    args = parser.parse_args()
    run(args.output, set(filter(None, args.generate_ids.split(','))), structure_windows=args.structure_windows,
        max_windows=args.max_windows, candidate_budget=args.candidate_budget, top_k=args.top_k,
        splits=tuple(args.splits.split(',')))
