import copy
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.maintenance_acceptance_v1.acceptance_report import summarize, case_fingerprint


def test_report_never_converts_model_complete_or_anchor_hits_to_human_accuracy():
    rows = [{'id': 'x', 'seconds': 1.0, 'version': 'v2.9.1',
             'result': {'status': 'OK', 'answer_completeness': 'MODEL_ASSESSED_COMPLETE',
                        'claims': [{'text': '事实', 'evidence_ids': ['a']}],
                        'sources': [{'version': 'v2.9.1'}]}}]
    before = copy.deepcopy(rows)
    out = summarize(rows)
    assert out['human_accuracy'] is None
    assert out['model_complete'] == 1
    assert out['human_review_status'] == 'PENDING_INDEPENDENT_REVIEW'
    assert rows == before


def test_version_leaks_and_provider_failures_are_visible_even_when_complete():
    rows = [{'id': 'x', 'seconds': 1.0, 'version': 'v2.9.1', 'result': {'status': 'OK',
        'answer_completeness': 'MODEL_ASSESSED_COMPLETE', 'sources': [{'version': 'v3.0.0'}]}},
        {'id': 'y', 'seconds': 2.0, 'version': 'all', 'result': {'status': 'CLIENT_ERROR'}}]
    out = summarize(rows)
    assert out['scope_violation_cases'] == ['x']
    assert out['request_failures'] == ['y']


def test_case_fingerprint_changes_when_question_or_rubric_changes():
    cases = [{'id': 'x', 'query': 'q', 'expected_facts': ['a']}]
    changed = copy.deepcopy(cases); changed[0]['expected_facts'] = ['b']
    assert case_fingerprint(cases) != case_fingerprint(changed)


def test_boundary_answers_are_separate_from_factual_accuracy_and_provider_errors():
    rows=[{'id':'scope','version':'all','seconds':.01,'question_type':'application_boundary',
           'result':{'status':'NEEDS_APPLICATION_CONTEXT',
                     'answer_completeness':'BOUNDARY_EXPLAINED_NOT_APPLICATION_VERIFIED'}},
          {'id':'fact','version':'v2.9.1','seconds':2,'question_type':'compound_fact',
           'result':{'status':'OK','answer_completeness':'PARTIAL_SUPPORTED'}}]
    out=summarize(rows)
    assert out['request_failures']==[]
    assert out['boundary_cases']==['scope']
    assert out['by_question_type']['compound_fact']['partial']==1
    assert out['human_accuracy'] is None


def test_empty_search_is_no_evidence_and_missing_response_schema_is_failure():
    out = summarize([
        {'id': 'empty', 'version': 'all', 'seconds': .1, 'result': {'results': []}},
        {'id': 'malformed', 'version': 'all', 'seconds': .1, 'result': {}},
        {'id': 'clarify', 'version': 'all', 'seconds': .1,
         'result': {'status': 'NEEDS_CLARIFICATION', 'results': []}},
    ])
    assert out['no_evidence_cases'] == ['empty']
    assert out['request_failures'] == ['malformed']


def test_acceptance_refuses_existing_output_before_network_and_preserves_old_report(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'evaluation/maintenance_acceptance_v1'))
    from evaluation.maintenance_acceptance_v1 import run
    folder = tmp_path/'accepted'; folder.mkdir()
    original = '{"old_complete_run":true}'
    (folder/'results.json').write_text(original)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--output', str(folder), '--only', 'example-01'])
    monkeypatch.setattr(run.requests, 'Session', lambda: pytest.fail('must reject before network'))
    with pytest.raises(SystemExit):
        run.main()
    assert (folder/'results.json').read_text() == original


def test_response_identity_mismatch_or_missing_header_invalidates_observations():
    from evaluation.maintenance_acceptance_v1.acceptance_report import identity_status
    rows = [{'response_identity_sha256': 'a'*64}]
    assert identity_status('a'*64, 'a'*64, rows) == 'CONSISTENT'
    assert identity_status('a'*64, 'b'*64, rows) == 'CHANGED'
    assert identity_status('b'*64, 'b'*64, rows) == 'CHANGED'
    assert identity_status('a'*64, 'a'*64, [{}]) == 'UNVERIFIED'
