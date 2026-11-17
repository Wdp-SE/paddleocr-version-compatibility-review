"""Guard real benchmark conclusions against fallback, scope and stale identities."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'evaluation/business_quality_v1/retrieval_benchmark.py'


@pytest.fixture
def benchmark():
    assert SCRIPT.is_file(), 'The current-corpus benchmark runner is missing'
    spec = importlib.util.spec_from_file_location('business_retrieval_benchmark', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(chunk_id='correct', version='v3.0.0', module='ocr', text='rec_texts and rec_scores'):
    return {'chunk_id': chunk_id, 'version': version, 'module': module,
            'document_path': 'docs/ocr.md', 'source_sha256': 'source-sha',
            'line_start': 10, 'line_end': 10, 'content': text}


def case():
    return {'id': 'ocr-fields', 'versions': ['v3.0.0'], 'modules': ['ocr'],
            'expected_behavior': 'supported', 'facts': [
                {'id': 'fields', 'version': 'v3.0.0', 'module': 'ocr',
                 'allowed_paths': ['docs/ocr.md'], 'tokens': ['rec_texts', 'rec_scores']}]}


def report(strategy='contextual_bm25'):
    return {'identity_sha256': 'frozen', 'strategy': strategy, 'status': 'OK',
            'rows': [{'id': 'ocr-fields', 'split': 'acceptance',
                      'assessment': {'fact_hits': [True, False], 'wrong_version_count': 0,
                                     'wrong_module_count': 0, 'expected_behavior_met': True},
                      'first_pass_latency_ms': 12.0,
                      'warm_latency_ms': [10.0, 12.0], 'execution_status': 'OK',
                      'diagnostics': {'requested_strategy': strategy, 'actual_strategy': strategy,
                                      'fallback_reason': None, 'semantic_candidates': 20,
                                      'rerank_calls': int('rerank' in strategy)}}]}


def test_scope_mismatches_never_satisfy_required_facts(benchmark):
    wrong_version = source('old', version='v2.9.1')
    wrong_module = source('table', module='table_recognition_v2')
    chunks = {h['chunk_id']: h for h in [wrong_version, wrong_module]}
    result = benchmark.assess_hits(case(), list(chunks.values()), chunks)
    assert result['fact_hits'] == [False]
    assert result['wrong_version_count'] == 1
    assert result['wrong_module_count'] == 1
    assert result['complete'] is False


def test_general_source_is_unclassified_rather_than_a_known_wrong_module(benchmark):
    general = source(module='general')
    result = benchmark.assess_hits(case(), [general], {'correct': general})
    assert result['wrong_module_count'] == 0
    assert result['unclassified_module_count'] == 1
    assert result['fact_hits'] == [False]


def test_forged_span_and_parent_identity_are_rejected(benchmark):
    original = source()
    forged = {**original, 'content': 'invented rec_texts rec_scores'}
    with pytest.raises(ValueError, match='identity'):
        benchmark.assess_hits(case(), [forged], {'correct': original})
    forged = {**original, 'evidence_spans': [
        {'char_start': 0, 'char_end': 9, 'line_start': 10, 'line_end': 10,
         'content': 'invented!'}]}
    with pytest.raises(ValueError):
        benchmark.assess_hits(case(), [forged], {'correct': original})


def test_publisher_source_link_cannot_be_relabelled(benchmark):
    original = {**source(), 'source_url': 'https://github.com/PaddlePaddle/PaddleOCR/source'}
    forged = {**original, 'source_url': 'https://example.invalid/internal-original-document'}
    with pytest.raises(ValueError, match='identity'):
        benchmark.assess_hits(case(), [forged], {'correct': original})


def test_empty_facts_never_turn_missing_internal_knowledge_into_success(benchmark):
    negative = {**case(), 'expected_behavior': 'missing_internal_evidence', 'facts': []}
    result = benchmark.assess_hits(negative, [source()], {'correct': source()})
    assert result['complete'] is False
    assert result['answerability'] == 'NOT_ANSWERABLE_FROM_COMPONENT_CORPUS'
    assert result['fact_hits'] == []
    assert result['generation_abstention'] == 'NOT_RUN'


def test_neural_fallback_is_not_available_even_with_matching_fact_text(benchmark):
    result = {'results': [source()], 'diagnostics': {
        'actual_strategy': 'contextual_bm25', 'fallback_reason': 'SEMANTIC_NONFINITE_OUTPUT',
        'status': 'OK', 'semantic_candidates': 0, 'rerank_calls': 0}}
    assert benchmark.execution_status('contextual_rrf', result) == 'NOT_AVAILABLE'


def test_claimed_hybrid_without_real_dense_candidates_is_not_available(benchmark):
    result = {'results': [source()], 'diagnostics': {
        'actual_strategy': 'contextual_rrf', 'fallback_reason': None,
        'status': 'OK', 'semantic_candidates': 0, 'rerank_calls': 0}}
    assert benchmark.execution_status('contextual_rrf', result) == 'NOT_AVAILABLE'


def test_gate_rejects_warm_results_that_change_evidence(benchmark):
    baseline, candidate = report(), report('contextual_rrf')
    candidate['rows'][0]['assessment']['fact_hits'] = [True, True]
    candidate['rows'][0]['warm_identical_evidence'] = [True, False]
    assert 'WARM_EVIDENCE_CHANGED' in benchmark.promotion_gate(baseline, candidate)['reasons']


def test_gate_requires_gain_without_losing_any_question_fact_or_scope(benchmark):
    baseline = report()
    candidate = report('contextual_rrf')
    candidate['rows'][0]['assessment']['fact_hits'] = [True, True]
    assert benchmark.promotion_gate(baseline, candidate)['recommend_enable'] is True
    candidate['rows'][0]['assessment']['fact_hits'] = [False, True]
    decision = benchmark.promotion_gate(baseline, candidate)
    assert decision['recommend_enable'] is False
    assert 'PER_CASE_FACT_REGRESSION' in decision['reasons']


@pytest.mark.parametrize('mutation,reason', [
    ('fallback', 'NEURAL_EXECUTION_NOT_AVAILABLE'),
    ('stale', 'IDENTITY_MISMATCH'),
    ('scope', 'SCOPE_REGRESSION'),
    ('slow', 'LATENCY_BUDGET_EXCEEDED'),
    ('slow_first_pass', 'FIRST_PASS_LATENCY_BUDGET_EXCEEDED'),
    ('no_gain', 'NO_FACT_COVERAGE_GAIN'),
    ('missing_case', 'CASE_SET_MISMATCH'),
])
def test_gate_rejects_unsupported_promotions(benchmark, mutation, reason):
    baseline = report()
    candidate = report('contextual_rrf')
    candidate['rows'][0]['assessment']['fact_hits'] = [True, True]
    if mutation == 'fallback':
        candidate['rows'][0]['execution_status'] = 'NOT_AVAILABLE'
    elif mutation == 'stale':
        candidate['identity_sha256'] = 'other-corpus-or-code'
    elif mutation == 'scope':
        candidate['rows'][0]['assessment']['wrong_module_count'] = 1
    elif mutation == 'slow':
        candidate['rows'][0]['warm_latency_ms'] = [10001.0, 10002.0]
    elif mutation == 'slow_first_pass':
        candidate['rows'][0]['first_pass_latency_ms'] = 10002.0
    elif mutation == 'no_gain':
        candidate['rows'][0]['assessment']['fact_hits'] = [True, False]
    elif mutation == 'missing_case':
        candidate['rows'] = []
    assert reason in benchmark.promotion_gate(baseline, candidate)['reasons']


def test_identity_detects_source_and_tokenizer_changes(benchmark, tmp_path):
    (tmp_path / 'code.py').write_text('first', encoding='utf8')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'sentencepiece.bpe.model').write_bytes(b'tokenizer-one')
    (model / 'model.safetensors').write_bytes(b'weights')
    frozen = benchmark.freeze_identity(tmp_path, ['code.py'], {'embedding': model}, {'top_k': 5})
    assert len(frozen['models']['embedding']['files']) == 2
    benchmark.validate_identity(frozen, tmp_path)
    (model / 'sentencepiece.bpe.model').write_bytes(b'tokenizer-two')
    with pytest.raises(ValueError, match='model'):
        benchmark.validate_identity(frozen, tmp_path)
    (model / 'sentencepiece.bpe.model').write_bytes(b'tokenizer-one')
    (tmp_path / 'code.py').write_text('second', encoding='utf8')
    with pytest.raises(ValueError, match='source'):
        benchmark.validate_identity(frozen, tmp_path)


def test_identity_rejects_changed_dependency_environment(benchmark, tmp_path):
    (tmp_path / 'code.py').write_text('source', encoding='utf8')
    frozen = benchmark.freeze_identity(tmp_path, ['code.py'], {}, {
        'packages': {'a-deliberately-nonexistent-benchmark-library': '1.0'}})
    with pytest.raises(ValueError, match='runtime'):
        benchmark.validate_identity(frozen, tmp_path)


def test_markdown_carries_frozen_identity_and_limitations(benchmark):
    value = {'identity_sha256': 'frozen-corpus-code-model-sha', 'strategies': {},
             'decisions': {}, 'generation': {'status': 'NOT_RUN'},
             'label_origin': 'developer-authored; not blind'}
    rendered = benchmark.markdown_report(value)
    assert 'frozen-corpus-code-model-sha' in rendered
    assert 'developer-authored; not blind' in rendered
    assert 'NOT_RUN' in rendered
