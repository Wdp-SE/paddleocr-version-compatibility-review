"""Source-bound acceptance of the actual default BM25 path, with wording variants.

Developer labels score saved canonical spans after retrieval; they never enter the
search request. This is necessary-fact regression coverage, not answer accuracy.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'versioned-rag-service'))
sys.path.insert(0, str(FOLDER))
import retrieval_benchmark as grader

# New developer wording of the same source tasks, explicitly not a blind set.
VARIANTS = (
    ('colloquial-text-score', 'OCR 识别出的文字以及对应置信度，分别存在哪个字段？低分结果是否已经过滤？'),
    ('regression-v4-03', '拿到 OCR predict 的结果后，我需要逐条打印，再把可视化图片导出，怎么写？'),
    ('colloquial-save-json', '怎样把 OCR 每页输出保存成 JSON 文件，放到指定目录？'),
    ('colloquial-old-crop', 'OCR 2.9.1 已有裁好的文字图片，只做识别，不再检测，该怎么调用？'),
    ('compound-six-obligations', 'OCR 从 2.9.1 迁到 3.0.0，构造器、如何调用、返回结构分别要检查哪些内容？请附上两版出处。'),
    ('module-structure-fields', '不是 OCR，是 PPStructureV3 的 overall_ocr_res 字段，识别文字与置信度分别放在哪里？'),
    ('module-ocr-fields', 'OCR 的 rec_texts 和 rec_scores 是什么意思？别用 PPStructureV3 结果作依据。'),
    ('version-conflict', '资料范围仍选 2.9.1，请解释 3.0.0 的 rec_texts。'),
    ('missing-internal-approval', '只查 PaddleOCR 官方资料，能否断定本公司应用已完成回归、可以上线？'),
)


def run(output: Path, baseline: Path | None = None):
    from src import public_api as api
    from src.public_knowledge import PublicKnowledgeIndex
    from src.public_scope import is_out_of_scope_public_request, application_boundary_answer
    from src.paddleocr_query_plan import plan_query
    from src.paddleocr_evidence_search import configured_evidence_search

    output = output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; never overwrite a run')
    output.mkdir(parents=True)
    files = [p.relative_to(ROOT).as_posix() for folder in
             ('versioned-rag-service/src', 'versioned-rag-service/config',
              'versioned-rag-service/public_corpus_paddleocr', 'evaluation/business_quality_v1')
             for p in (ROOT / folder).iterdir() if p.suffix in ('.py', '.json') and p.is_file()]
    files += ['versioned-rag-service/requirements.txt']
    settings = {'entrypoint': '_execute_public_search after query admission',
                'strategy': 'auto (must resolve to contextual_bm25)', 'top_k': 5,
                'candidate_budget': 80, 'structure_windows': True, 'window_budget': 384,
                'max_windows_per_source': 4, 'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in
                             ('fastapi', 'pydantic', 'numpy', 'langgraph')}}
    frozen = grader.freeze_identity(ROOT, files, {}, settings)
    (output / 'freeze.json').write_text(json.dumps(frozen, indent=2), encoding='utf8')
    dataset = json.loads((FOLDER / 'cases_retrieval.json').read_text(encoding='utf8'))
    cases = dataset['cases']
    by_id = {c['id']: c for c in cases}
    for i, (base, query) in enumerate(VARIANTS):
        case = copy.deepcopy(by_id[base])
        case.update(id=f'wording-{i+1:02}', query=query, split='additional_developer_wording',
                    label_origin=f'same-source-facts-as:{base}; not independently blind')
        cases.append(case)
    index = PublicKnowledgeIndex(ROOT / 'versioned-rag-service/public_corpus_paddleocr')
    chunks = {h['chunk_id']: h for h in index.chunks}
    rows = []
    for case in cases:
        payload = api.SearchRequest(query=case['query'], language='zh', top_k=5,
                    version='all' if len(case['versions']) > 1 else case['versions'][0],
                    retrieval_policy='paddleocr_evidence')
        reason = ('OUT_OF_SCOPE' if is_out_of_scope_public_request(case['query']) else
                  'INTERNAL_FACT_BOUNDARY' if application_boundary_answer(case['query']) else
                  'NEEDS_CLARIFICATION' if api._paddleocr_version_clarification(index, payload) else None)
        row = {'id': case['id'], 'query': case['query'], 'split': case['split'],
               'case': case, 'status': reason or 'OK'}
        if reason:
            expected = ('NEEDS_CLARIFICATION' if case['expected_behavior'] == 'version_clarification'
                        else ('OUT_OF_SCOPE', 'INTERNAL_FACT_BOUNDARY')
                        if case['expected_behavior'] == 'missing_internal_evidence' else ())
            row['expected_guard_met'] = reason == expected if isinstance(expected, str) else reason in expected
            row['evidence'] = []
        else:
            started = time.perf_counter()
            actual = api._execute_public_search(index, payload, None, top_k=payload.top_k)
            plan = plan_query(case['query'], versions=tuple(case['versions']))
            direct = configured_evidence_search(index, 'contextual_bm25', window_budget=384,
                        structure_windows=True).search(plan, top_k=5, candidate_budget=80,
                                                        strategy='contextual_bm25')
            if actual['actual_policy'] != 'contextual_bm25' or grader.pack_hits(actual['results']) != grader.pack_hits(direct['results']):
                raise ValueError('Actual default policy no longer matches the declared experiment')
            row.update(seconds=round(time.perf_counter()-started, 4),
                       assessment=grader.assess_hits(case, actual['results'], chunks, assembled=True),
                       candidate_assessment=grader.assess_hits(case, direct['candidate_evidence'], chunks, assembled=True),
                       requirements=actual['requirements'], diagnostics=actual['rerank_diagnostics'],
                       evidence=grader.pack_hits(actual['results']))
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('id', 'status')} | {'facts': row.get('assessment', {}).get('fact_hit_count')}, ensure_ascii=True), flush=True)
    grader.validate_identity(frozen, ROOT)
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'identity_sha256': frozen['identity_sha256'], 'settings': settings,
              'assessment_type': 'developer necessary literal source facts; not blind or answer accuracy',
              'external_model_calls': 0, 'rows': rows}
    for label, subset in (('fixed', rows[:len(dataset['cases'])-len(VARIANTS)]),
                          ('additional_wording', rows[-len(VARIANTS):])):
        report[label] = {'questions': len(subset),
                         'fact_hits': sum(r.get('assessment', {}).get('fact_hit_count', 0) for r in subset),
                         'fact_count': sum(len(r['case']['facts']) for r in subset),
                         'wrong_version_or_module': sum(r.get('assessment', {}).get('wrong_version_count', 0) + r.get('assessment', {}).get('wrong_module_count', 0) for r in subset),
                         'guards_met': all(r.get('expected_guard_met', True) for r in subset)}
    if baseline:
        previous = json.loads(baseline.read_text(encoding='utf8'))
        old = {r['id']: r for r in previous['rows']}
        regressions = []
        for row in rows[:report['fixed']['questions']]:
            prior = old[row['id']]
            if row['case']['query'] != prior['case']['query'] or row['case']['facts'] != prior['case']['facts']:
                raise ValueError('Baseline questions or labels were changed')
            prior_hits = prior.get('assessment', {}).get('fact_hits', [])
            after_hits = row.get('assessment', {}).get('fact_hits', [])
            if any(hit and not after_hits[i] for i, hit in enumerate(prior_hits)):
                regressions.append(row['id'])
        report['baseline'] = {'sha256': grader.sha256(baseline),
            'note': 'Captured default-path diagnostic before repair; no full pre-run source freeze, not a strategy promotion benchmark',
            'fact_hits': sum(r.get('assessment', {}).get('fact_hit_count', 0) for r in old.values()),
            'fact_count': sum(len(r['case']['facts']) for r in old.values()), 'lost_existing_facts': regressions}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({key: report[key] for key in ('identity_sha256', 'fixed', 'additional_wording')}, ensure_ascii=True), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    report = run(args.output, args.baseline)
    failed = any(v['fact_hits'] != v['fact_count'] or not v['guards_met'] or v['wrong_version_or_module']
                 for v in (report['fixed'], report['additional_wording']))
    raise SystemExit(1 if failed else 0)
