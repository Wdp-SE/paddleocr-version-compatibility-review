"""Current-corpus, offline, source-bound retrieval experiment. Never publishes policy.

Labels are developer regressions / developer-authored acceptance wording, not blind
business accuracy. Each strategy runs in its own process against the same freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'versioned-rag-service'))
STRATEGIES = ('contextual_bm25', 'contextual_rrf', 'contextual_rrf_rerank')
FOLDER = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / '.local-demo-temp/business-quality-20261114/retrieval-final'
SETTINGS = {'top_k': 5, 'candidate_budget': 40, 'window_budget': 384,
            'max_windows_per_source': 4, 'structure_windows': False,
            'warm_repeats': 2, 'warm_p95_budget_ms': 5000.0,
            'warm_latency_ratio_budget': 10.0, 'first_pass_p95_budget_ms': 5000.0,
            'first_pass_latency_ratio_budget': 10.0, 'minimum_fact_gain': 1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf8')).hexdigest()


def model_files(folder):
    if not folder or not Path(folder).is_dir():
        return {}
    root = Path(folder)
    # Include tokenizer binary (.model) as well as weights and configuration.
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.cache' not in p.relative_to(root).parts}


def freeze_identity(root, files, models, settings):
    root = Path(root).resolve()
    value = {'schema_version': 1,
             'files': {str(p).replace('\\', '/'): sha256(root / p) for p in sorted(files)},
             'models': {name: {'local_path': str(Path(path).resolve()) if path else '',
                               'files': model_files(path)} for name, path in models.items()},
             'settings': settings}
    value['identity_sha256'] = canonical_digest(value)
    return value


def validate_identity(frozen, root):
    unsigned = {k: v for k, v in frozen.items() if k != 'identity_sha256'}
    if canonical_digest(unsigned) != frozen['identity_sha256']:
        raise ValueError('frozen identity payload was altered')
    settings = frozen['settings']
    if settings.get('python', platform.python_version()) != platform.python_version():
        raise ValueError('frozen runtime Python identity changed')
    for package, pinned in settings.get('packages', {}).items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = 'NOT_INSTALLED'
        if actual != pinned:
            raise ValueError(f'frozen runtime dependency identity changed: {package}')
    root = Path(root).resolve()
    for name, digest in frozen['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f'frozen source identity changed: {name}')
    for label, record in frozen['models'].items():
        if model_files(record['local_path']) != record['files']:
            raise ValueError(f'frozen model identity changed: {label}')


def resolve_hits(hits, chunks):
    """Reconstruct saved evidence from corpus; never trust supplied parent text."""
    from src.paddleocr_retrieval_views import evidence_text, module_for
    resolved = []
    for hit in hits:
        parent = chunks.get(hit.get('chunk_id'))
        if parent is None:
            raise ValueError('unknown evidence identity')
        keys = ('version', 'document_path', 'source_sha256', 'source_id', 'source_url', 'line_start', 'line_end', 'content')
        if any(key in hit and hit[key] != parent.get(key) for key in keys):
            raise ValueError('evidence parent identity mismatch')
        module = parent.get('module', module_for(parent))
        if 'module' in hit and hit['module'] != module:
            raise ValueError('evidence module identity mismatch')
        row = {**parent, 'module': module}
        if 'evidence_spans' in hit:
            row['evidence_spans'] = hit['evidence_spans']
        evidence_text(row)  # validates offsets, original text and line positions
        resolved.append(row)
    return resolved


def assess_hits(case, hits, chunks, *, assembled=False):
    from src.paddleocr_retrieval_views import evidence_text, generation_evidence_text
    resolved = resolve_hits(hits, chunks)
    text = generation_evidence_text if assembled else evidence_text
    matched = []
    for fact in case.get('facts', []):
        valid = [h for h in resolved if h['version'] == fact['version']
                 and h['module'] in (fact['module'], 'shared')
                 and h['document_path'] in fact['allowed_paths']]
        matched.append(any(all(t in text(h) for t in fact['tokens'])
                           and not any(t in text(h) for t in fact.get('forbidden_tokens', []))
                           for h in valid))
    behavior = case.get('expected_behavior', 'supported')
    wrong_version = sum(h['version'] not in case['versions'] for h in resolved)
    wrong_module = sum(bool(case.get('modules')) and h['module'] not in case['modules'] + ['shared', 'general']
                       for h in resolved)
    return {'complete': bool(matched) and all(matched), 'fact_hits': matched,
            'fact_hit_count': sum(matched), 'fact_count': len(matched),
            'wrong_version_count': wrong_version, 'wrong_module_count': wrong_module,
            'unclassified_module_count': sum(h['module'] == 'general' for h in resolved),
            'expected_behavior_met': (not resolved if behavior == 'version_clarification'
                                      else wrong_version == 0 and wrong_module == 0),
            'answerability': ('NOT_ANSWERABLE_FROM_COMPONENT_CORPUS'
                              if behavior == 'missing_internal_evidence' else
                              'VERSION_CLARIFICATION_REQUIRED' if behavior == 'version_clarification' else
                              'REQUIRED_SOURCE_FACTS_PRESENT' if matched and all(matched) else 'RETRIEVAL_GAP'),
            'generation_abstention': 'NOT_RUN',
            'assessment_type': 'literal_source_fact_coverage_not_semantic_or_answer_accuracy'}


def execution_status(strategy, result):
    diag = result['diagnostics']
    if diag.get('status') == 'NEEDS_CLARIFICATION':
        return 'SKIPPED_SCOPE_GUARD'
    if diag.get('actual_strategy') != strategy or diag.get('fallback_reason'):
        return 'NOT_AVAILABLE'
    if strategy in ('contextual_rrf', 'contextual_rrf_rerank') and not diag.get('semantic_candidates'):
        return 'NOT_AVAILABLE'
    if 'rerank' in strategy and diag.get('rerank_calls') != 1:
        return 'NOT_AVAILABLE'
    return 'OK'


def percentile(values, quantile=.95):
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def promotion_gate(baseline, candidate, settings=None):
    settings = settings or SETTINGS
    reasons = []
    if baseline['identity_sha256'] != candidate['identity_sha256']:
        reasons.append('IDENTITY_MISMATCH')
    if baseline.get('status') != 'OK' or candidate.get('status') != 'OK':
        reasons.append('INCOMPLETE_OR_STALE_REPORT')
    old = {r['id']: r for r in baseline['rows']}
    new = {r['id']: r for r in candidate['rows']}
    if not old or set(old) != set(new) or len(old) != len(baseline['rows']) or len(new) != len(candidate['rows']):
        reasons.append('CASE_SET_MISMATCH')
    if any(r['execution_status'] not in ('OK', 'SKIPPED_SCOPE_GUARD')
           or r.get('diagnostics', {}).get('fallback_reason')
           or (r['execution_status'] == 'OK' and r.get('diagnostics', {}).get('actual_strategy') != candidate['strategy'])
           for r in candidate['rows']):
        reasons.append('NEURAL_EXECUTION_NOT_AVAILABLE')
    if any(not all(r.get('warm_identical_evidence', [True])) for r in candidate['rows']):
        reasons.append('WARM_EVIDENCE_CHANGED')
    gains = {'dev': 0, 'acceptance': 0}
    for cid in old.keys() & new.keys():
        a, b = old[cid]['assessment'], new[cid]['assessment']
        if len(a['fact_hits']) != len(b['fact_hits']) or any(x and not y for x, y in zip(a['fact_hits'], b['fact_hits'])):
            reasons.append('PER_CASE_FACT_REGRESSION')
        if any(b[k] > a[k] for k in ('wrong_version_count', 'wrong_module_count')):
            reasons.append('SCOPE_REGRESSION')
        if not b['expected_behavior_met']:
            reasons.append('NEGATIVE_CONTROL_FAILED')
        split = new[cid]['split']
        gains[split] = gains.get(split, 0) + sum(b['fact_hits']) - sum(a['fact_hits'])
    if gains['acceptance'] < settings['minimum_fact_gain']:
        reasons.append('NO_FACT_COVERAGE_GAIN')
    a_latency = percentile([v for r in baseline['rows'] for v in r['warm_latency_ms']])
    b_latency = percentile([v for r in candidate['rows'] for v in r['warm_latency_ms']])
    ratio = b_latency / max(a_latency, .01) if a_latency and b_latency is not None else None
    if b_latency is None or b_latency > settings['warm_p95_budget_ms'] or ratio is None or ratio > settings['warm_latency_ratio_budget']:
        reasons.append('LATENCY_BUDGET_EXCEEDED')
    a_first = percentile([r['first_pass_latency_ms'] for r in baseline['rows']])
    b_first = percentile([r['first_pass_latency_ms'] for r in candidate['rows']])
    first_ratio = b_first / max(a_first, .01) if a_first and b_first is not None else None
    if (b_first is None or b_first > settings['first_pass_p95_budget_ms']
            or first_ratio is None or first_ratio > settings['first_pass_latency_ratio_budget']):
        reasons.append('FIRST_PASS_LATENCY_BUDGET_EXCEEDED')
    reasons = list(dict.fromkeys(reasons))
    return {'recommend_enable': not reasons, 'reasons': reasons, 'fact_gain_by_split': gains,
            'warm_p95_ms': b_latency, 'warm_p95_ratio': ratio,
            'first_pass_p95_ms': b_first, 'first_pass_p95_ratio': first_ratio,
            'policy_changed': False, 'gate': 'per-case facts and scope non-regression; acceptance gain; first-pass and warm local latency budgets'}


def pack_hits(hits):
    fields = ('chunk_id', 'version', 'document_path', 'source_id', 'source_sha256',
              'source_url', 'module', 'line_start', 'line_end', 'evidence_spans')
    return [{key: h[key] for key in fields if key in h} for h in hits]


def summarize(rows):
    scores = [r['assessment'] for r in rows]
    facts = sum(s['fact_count'] for s in scores)
    times = [v for r in rows for v in r['warm_latency_ms']]
    return {'case_count': len(rows), 'fact_hit_count': sum(s['fact_hit_count'] for s in scores),
            'fact_count': facts, 'fact_coverage': sum(s['fact_hit_count'] for s in scores) / max(1, facts),
            'supported_case_count': sum(bool(s['fact_count']) for s in scores),
            'supported_complete_count': sum(s['complete'] for s in scores),
            'wrong_version_count': sum(s['wrong_version_count'] for s in scores),
            'wrong_module_count': sum(s['wrong_module_count'] for s in scores),
            'scope_failures': sum(not s['expected_behavior_met'] for s in scores),
            'unclassified_module_count': sum(s['unclassified_module_count'] for s in scores),
            'not_available_count': sum(r['execution_status'] == 'NOT_AVAILABLE' for r in rows),
            'warm_p50_ms': statistics.median(times) if times else None,
            'warm_p95_ms': percentile(times),
            'first_pass_p50_ms': statistics.median(r['first_pass_latency_ms'] for r in rows) if rows else None,
            'first_pass_p95_ms': percentile([r['first_pass_latency_ms'] for r in rows])}


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def default_models():
    embedding = os.environ.get('PADDLEOCR_EMBEDDING_PATH', '')
    reranker = os.environ.get('PADDLEOCR_RERANKER_PATH', '')
    if not embedding:
        hub = Path.home() / '.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots'
        snapshots = sorted(hub.glob('*')) if hub.is_dir() else []
        embedding = str(snapshots[-1]) if snapshots else ''
    if not reranker:
        local = ROOT / '.local-demo-temp/quality-models/bge-reranker-base'
        reranker = str(local) if local.is_dir() else ''
    return {'embedding': embedding, 'reranker': reranker}


def make_freeze(output):
    corpus = ROOT / 'versioned-rag-service/public_corpus_paddleocr'
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'versioned-rag-service/src').rglob('*.py')]
    files += [p.relative_to(ROOT).as_posix() for p in corpus.rglob('*') if p.is_file()]
    files += [(FOLDER / name).relative_to(ROOT).as_posix() for name in
              ('retrieval_benchmark.py', 'cases_retrieval.json')]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'docs').glob('business_*.md')]
    packages = {}
    for package in ('numpy', 'torch', 'transformers', 'sentence-transformers', 'tokenizers'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'NOT_INSTALLED'
    settings = {**SETTINGS, 'strategies': list(STRATEGIES), 'python': platform.python_version(),
                'packages': packages, 'platform': platform.platform(),
                'model_execution': 'local_files_only eager attention native_serial_v2 CPU',
                'vector_cache': 'shared exact text plus model identity cache in output parent; query vectors computed by real encoder; cache input hashes recorded per run'}
    frozen = freeze_identity(ROOT, files, default_models(), settings)
    # Validate source labels against full canonical parents before any model run.
    from src.public_knowledge import PublicKnowledgeIndex
    index = PublicKnowledgeIndex(corpus)
    chunks = {h['chunk_id']: h for h in index.chunks}
    for case in load(FOLDER / 'cases_retrieval.json')['cases']:
        for fact in case['facts']:
            witnesses = resolve_hits(fact['oracle_evidence'], chunks)
            probe = {**case, 'facts': [fact]}
            if not assess_hits(probe, witnesses, chunks)['complete']:
                raise ValueError(f"oracle fact has no canonical witness: {case['id']} / {fact['id']}")
    dump_new(output / 'retrieval_freeze.json', frozen)
    print(f"FROZEN {frozen['identity_sha256']}", flush=True)


def run_strategy(output, strategy, split='all'):
    frozen = load(output / 'retrieval_freeze.json')
    validate_identity(frozen, ROOT)
    for label, env in (('embedding', 'PADDLEOCR_EMBEDDING_PATH'), ('reranker', 'PADDLEOCR_RERANKER_PATH')):
        relevant = strategy != 'contextual_bm25' and (label == 'embedding' or 'rerank' in strategy)
        os.environ[env] = frozen['models'][label]['local_path'] if relevant else ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    vector_folder = output.parent / 'retrieval-vector-cache'
    vector_existed = vector_folder.exists()
    cache_inputs = {p.name: sha256(p) for p in vector_folder.glob('*.npy')} if vector_existed else {}
    os.environ['PADDLEOCR_VECTOR_CACHE'] = str(vector_folder)
    from src.public_knowledge import PublicKnowledgeIndex
    from src.paddleocr_evidence_search import configured_evidence_search
    from src.paddleocr_query_plan import plan_query
    settings = frozen['settings']
    start = time.perf_counter()
    index = PublicKnowledgeIndex(ROOT / 'versioned-rag-service/public_corpus_paddleocr')
    preflight = {'status': 'NOT_REQUESTED'}
    if strategy != 'contextual_bm25':
        from src.paddleocr_quality import configured_quality_retriever, ModelUnavailable
        try:
            configured_quality_retriever(index, load_embedding=True,
                load_reranker='rerank' in strategy, build_parent_vectors=False)
            preflight = {'status': 'LOADED_LOCAL_ASSETS'}
        except ModelUnavailable as exc:
            preflight = {'status': 'NOT_AVAILABLE', 'reason': str(exc),
                         'cause': type(exc.__cause__).__name__ + ': ' + str(exc.__cause__) if exc.__cause__ else None}
    search = configured_evidence_search(index, strategy, window_budget=settings['window_budget'],
                                        structure_windows=settings['structure_windows'])
    initialization_ms = (time.perf_counter() - start) * 1000
    chunks = {h['chunk_id']: h for h in index.chunks}
    dataset = load(FOLDER / 'cases_retrieval.json')
    cases = [c for c in dataset['cases'] if split == 'all' or c['split'] == split]
    rows = []
    for case in cases:
        plan = plan_query(case['query'], versions=tuple(case['versions']))
        def retrieve():
            before = time.perf_counter()
            result = search.search(plan, strategy=strategy, top_k=settings['top_k'],
                                   candidate_budget=settings['candidate_budget'],
                                   max_windows_per_source=settings['max_windows_per_source'])
            return result, (time.perf_counter() - before) * 1000
        result, first_ms = retrieve()
        warm_times, warm_checks, warm_status = [], [], []
        for _ in range(settings['warm_repeats']):
            warm, warm_ms = retrieve()
            warm_times.append(round(warm_ms, 3))
            warm_checks.append(pack_hits(warm['results']) == pack_hits(result['results']))
            warm_status.append(execution_status(strategy, warm))
        final = pack_hits(result['results'])
        candidates = pack_hits(result.get('candidate_evidence', []))
        oracle = list({h['chunk_id']: h for f in case['facts'] for h in f['oracle_evidence']}.values())
        assessment = assess_hits(case, final, chunks)
        row = {'id': case['id'], 'split': case['split'], 'case': case, 'plan': plan,
               'first_pass_latency_ms': round(first_ms, 3), 'warm_latency_ms': warm_times,
               'warm_identical_evidence': warm_checks, 'warm_execution_status': warm_status,
               'execution_status': execution_status(strategy, result),
               'diagnostics': result['diagnostics'], 'assessment': assessment,
               'candidate_assessment': assess_hits(case, candidates, chunks),
               'assembled_assessment': assess_hits(case, final, chunks, assembled=True),
               'oracle_assessment': assess_hits(case, oracle, chunks),
               'final_evidence': final, 'candidate_evidence': candidates}
        rows.append(row)
        print(f"{strategy} {case['id']} {row['execution_status']} facts={assessment['fact_hit_count']}/{assessment['fact_count']} first_ms={first_ms:.1f}", flush=True)
    # Concurrent source/model changes invalidate the run rather than publishing old scores.
    validate_identity(frozen, ROOT)
    status = 'NOT_AVAILABLE' if any(r['execution_status'] == 'NOT_AVAILABLE' or
                  'NOT_AVAILABLE' in r['warm_execution_status'] for r in rows) else 'OK'
    report = {'schema_version': 1, 'identity_sha256': frozen['identity_sha256'],
              'created_at': datetime.now(timezone.utc).isoformat(), 'status': status,
              'strategy': strategy, 'split': split, 'label_origin': dataset['label_origin'],
              'initialization_ms': round(initialization_ms, 3), 'disk_vector_cache_existed': vector_existed,
              'disk_vector_cache_inputs': cache_inputs,
              'disk_vector_cache_outputs': {p.name: sha256(p) for p in vector_folder.glob('*.npy')},
              'cold_initialization_and_first_request_ms': round(initialization_ms + rows[0]['first_pass_latency_ms'], 3) if rows else None,
              'model_preflight': preflight,
              'latency_note': 'first_pass includes lazy vectors on first eligible query; warm repeats reuse exact model caches; not all case times are process-cold',
              'metrics': summarize(rows), 'metrics_by_split': {
                  s: summarize([r for r in rows if r['split'] == s]) for s in ('dev', 'acceptance')},
              'coverage_pipeline': {stage: {
                  'fact_hit_count': sum(r[key]['fact_hit_count'] for r in rows),
                  'fact_count': sum(r[key]['fact_count'] for r in rows),
                  'complete_count': sum(r[key]['complete'] for r in rows)}
                  for stage, key in (('candidates', 'candidate_assessment'), ('final_spans', 'assessment'),
                                     ('assembled_context', 'assembled_assessment'), ('full_source_oracle', 'oracle_assessment'))},
              'generation': {'status': 'NOT_RUN', 'external_model_calls': 0,
                             'reason': 'retrieval / assembled-context / full-source oracle split only; no answer accuracy or abstention claim'},
              'rows': rows}
    dump_new(output / f'retrieval_{strategy}_{split}.json', report)


def compare(output):
    frozen = load(output / 'retrieval_freeze.json')
    validate_identity(frozen, ROOT)
    dataset = load(FOLDER / 'cases_retrieval.json')
    cases = {c['id']: c for c in dataset['cases']}
    from src.public_knowledge import PublicKnowledgeIndex
    index = PublicKnowledgeIndex(ROOT / 'versioned-rag-service/public_corpus_paddleocr')
    chunks = {h['chunk_id']: h for h in index.chunks}
    reports = {}
    for strategy in STRATEGIES:
        saved = load(output / f'retrieval_{strategy}_all.json')
        if saved['identity_sha256'] != frozen['identity_sha256'] or saved['strategy'] != strategy:
            raise ValueError('report identity / requested strategy mismatch')
        if {r['id'] for r in saved['rows']} != set(cases) or len(saved['rows']) != len(cases):
            raise ValueError('report case set mismatch')
        for row in saved['rows']:
            if row['case'] != cases[row['id']] or row['split'] != cases[row['id']]['split']:
                raise ValueError('report source labels mismatch')
            actual = assess_hits(cases[row['id']], row['final_evidence'], chunks)
            if actual != row['assessment']:
                raise ValueError('saved report coverage does not match canonical evidence')
            result = {'results': row['final_evidence'], 'diagnostics': row['diagnostics']}
            if execution_status(strategy, result) != row['execution_status']:
                raise ValueError('saved report neural execution mismatch')
        reports[strategy] = saved
    decisions = {s: promotion_gate(reports[STRATEGIES[0]], reports[s], frozen['settings']) for s in STRATEGIES[1:]}
    comparison = {'schema_version': 1, 'identity_sha256': frozen['identity_sha256'],
                  'label_origin': dataset['label_origin'], 'strategies': {
                      s: {k: r[k] for k in ('status', 'initialization_ms', 'cold_initialization_and_first_request_ms',
                          'metrics', 'metrics_by_split', 'coverage_pipeline', 'model_preflight', 'latency_note')}
                      for s, r in reports.items()}, 'decisions': decisions,
                  'generation': reports[STRATEGIES[0]]['generation'],
                  'business_boundary': 'PaddleOCR official proxy corpus; student authored POC; no production users or internal company facts',
                  'policy_changed': False}
    dump_new(output / 'retrieval_comparison.json', comparison)
    (output / 'retrieval_comparison.md').write_text(markdown_report(comparison), encoding='utf8')
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


def markdown_report(report):
    lines = ['# 当前资料检索对照', '', f"冻结身份：`{report['identity_sha256']}`", '',
             f"题目标注：{report['label_origin']}", '',
             '|策略|实际状态|事实命中|错版本|错模块|暖 P95 ms|初始化 ms|',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, row in report.get('strategies', {}).items():
        m = row['metrics']
        lines.append(f"|{name}|{row['status']}|{m['fact_hit_count']}/{m['fact_count']}|{m['wrong_version_count']}|{m['wrong_module_count']}|{m['warm_p95_ms']}|{row['initialization_ms']}|")
    for strategy, gate in report.get('decisions', {}).items():
        lines += ['', f"{strategy}：建议启用={gate['recommend_enable']}；原因={', '.join(gate['reasons']) or '门槛通过'}。"]
    lines += ['', '|策略|候选池事实|最终跨度事实|组装上下文事实|完整来源 oracle|',
              '|---|---:|---:|---:|---:|']
    for name, row in report.get('strategies', {}).items():
        pipeline = row.get('coverage_pipeline', {})
        cells = [f"{pipeline[key]['fact_hit_count']}/{pipeline[key]['fact_count']}" if key in pipeline else '未记录'
                 for key in ('candidates', 'final_spans', 'assembled_context', 'full_source_oracle')]
        lines.append('|' + name + '|' + '|'.join(cells) + '|')
    lines += ['', f"生成评测：{report.get('generation', {}).get('status', 'NOT_RUN')}。检索、组装上下文和完整来源 oracle 分开；未调用付费 API，不报告回答正确率或生成拒答通过。",
              '', '暖延迟包含模型原有精确缓存；每个策略独立进程。按完整文本和模型身份键控的文档向量缓存可复用，输入/输出文件哈希逐运行记录；查询仍由真实模型编码。首个请求可能包含全量向量构建，后续首遍请求不是独立进程冷启动。',
              '', '这是已见开发回归与开发者新写题的本地对照，不是业务专家盲评。字面事实覆盖不等于语义支持，也不证明原创应用兼容；官方来源不能证明未提供的内部运行与审批事实。',
              '', '门槛逐题不降低已有事实覆盖、错版本/模块不增加；冻结验收题至少新增一个事实，first-pass 与 warm P95均≤5000 ms且≤各自基线10倍，防止缓存掩盖未缓存推理成本。首次文档建库/进程初始化另列，不等同常驻服务请求成本。仅形成建议，不更改发布配置。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'compare'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--strategy', choices=STRATEGIES, default=STRATEGIES[0])
    parser.add_argument('--split', choices=('all', 'dev', 'acceptance'), default='all')
    args = parser.parse_args()
    if args.command == 'freeze':
        make_freeze(args.output)
    elif args.command == 'run':
        run_strategy(args.output, args.strategy, args.split)
    else:
        compare(args.output)


if __name__ == '__main__':
    main()
