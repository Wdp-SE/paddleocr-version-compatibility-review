"""Business probes for the complete internal OCR upgrade path, not example names."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.paddleocr_compatibility import review_compatibility

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def index():
    corpus = ROOT / 'versioned-rag-service/public_corpus_paddleocr'
    return SimpleNamespace(manifest=json.loads((corpus/'corpus_manifest.json').read_text(encoding='utf8')),
                           chunks=json.loads((corpus/'chunks.json').read_text(encoding='utf8')))


def review(index, rows):
    return review_compatibility(index, files=[{'path': p, 'content': t} for p, t in rows.items()],
                                source_version='v2.9.1', target_version='v3.0.0')


def chain(adapter='return raw[0][0][1][0]', argument='raw', shadow=''):
    return {
        'team/client.py': 'from paddleocr import PaddleOCR\ndef scan(path):\n    return PaddleOCR(lang="ch").ocr(path)\n',
        'team/adapter.py': f'def convert(raw):\n    {adapter}\n',
        'team/facade.py': 'from .adapter import convert as project\ndef adapt(value):\n    return project(raw=value)\n',
        'team/main.py': ('from .client import scan\nfrom .facade import adapt as normalize\n'
                         f'def process(path{shadow}):\n    raw=scan(path)\n    return normalize({argument})\n'),
    }


def risks(report):
    return [f for f in report['findings'] if f['rule_id'] == 'legacy_ocr_result']


def test_argument_flow_through_alias_and_keyword_wrapper(index):
    result = review(index, chain())
    assert any(f['application']['path']=='team/adapter.py' for f in risks(result))
    assert all({e['version'] for e in f['evidence']} == {'v2.9.1','v3.0.0'} for f in risks(result))


def test_enumeration_and_guards_in_adapter_preserve_known_input_origin(index):
    rows=chain()
    rows['team/adapter.py']=('def convert(raw):\n    if not isinstance(raw,list):\n        raise ValueError("shape")\n'
        '    rows=[]\n    for number,page in enumerate(raw):\n        if page is None:\n            continue\n'
        '        for line in page:\n            rows.append(line[1][0])\n    return rows\n')
    assert any(f['application']['path']=='team/adapter.py' for f in risks(review(index,rows)))


@pytest.mark.parametrize('kwargs', [dict(argument='[[[1,2]]]'), dict(shadow=', normalize'),
                                  dict(shadow=', scan')])
def test_unrelated_or_shadowed_argument_is_not_ocr_result(index, kwargs):
    assert not risks(review(index,chain(**kwargs)))


def test_target_field_adapter_is_not_legacy_index_risk(index):
    assert not risks(review(index,chain("return [page['rec_texts'] for page in raw]")))


@pytest.mark.parametrize('expression', [
    "[line[1][0] for n,page in enumerate(raw) if page is not None for line in page]",
    "(line[1][0] for n,page in enumerate(raw) for line in page)",
    "{n: line[1][0] for n,page in enumerate(raw) for line in page}",
])
def test_enumerated_comprehensions_preserve_result_origin(index, expression):
    assert risks(review(index,chain('return '+expression)))


def test_shadowed_enumerate_does_not_invent_result_origin(index):
    rows=chain('return [line[1][0] for n,page in enumerate(raw) for line in page]')
    rows['team/adapter.py']='def convert(raw, enumerate):\n    return [line[1][0] for n,page in enumerate(raw) for line in page]\n'
    assert not risks(review(index,rows))


def test_actual_ui_old_and_adapted_examples_are_distinguishable(index):
    import sys
    sys.path.insert(0,str(ROOT/'demo-ui'))
    from services.interview_examples import DEMO_CASES,demo_application
    labels=list(DEMO_CASES)
    old=review_compatibility(index,files=demo_application(labels[0]),source_version='v2.9.1',target_version='v3.0.0')
    adapted=review_compatibility(index,files=demo_application(labels[1]),source_version='v2.9.1',target_version='v3.0.0')
    missing=review_compatibility(index,files=demo_application(labels[2]),source_version='v2.9.1',target_version='v3.0.0')
    assert risks(old)
    assert not risks(adapted) and adapted['runtime_verified'] is False
    assert missing['status']=='needs_verification'


def test_recursive_adapter_stops_with_explicit_gap(index):
    rows=chain();rows['team/adapter.py']='def convert(raw):\n    return convert(raw)\n'
    result=review(index,rows)
    assert not risks(result)
    assert any(g['code']=='argument_flow_limit' for g in result['gaps'])


def test_published_application_adapter_risk_is_found(index):
    root=ROOT/'examples/paddleocr_document_app'
    names=['result_adapter.py', *['application/'+p for p in (
        '__init__.py','ocr_client.py','normalizer.py','pipeline.py','consumer.py','contract.json',
        'requirements-v2.txt','requirements-v3.txt','design.md','upgrade.md','tests/test_contract.py')]]
    result=review(index,{p:(root/p).read_text(encoding='utf8') for p in names})
    assert any(f['application']['path']=='result_adapter.py' for f in risks(result))


def test_declared_target_manifest_is_compared_to_target_not_source(index):
    rows=chain();rows.update({'current.txt':'paddleocr==2.9.1\n','future.txt':'paddleocr==3.0.0\n',
        'contract.json': json.dumps({'review_context':{'dependency_files':{'current':['current.txt'],'target':['future.txt']}}})})
    result=review(index,rows)
    assert not any(g['code']=='dependency_version_mismatch' for g in result['gaps'])
    assert {f.get('dependency_role') for f in result['files'] if f['path'].endswith('.txt')} == {'current','target'}


def test_explicit_wrong_target_manifest_is_still_reported(index):
    rows=chain();rows.update({'future.txt':'paddleocr==2.9.1\n','contract.json':json.dumps({
        'review_context':{'dependency_files':{'target':['future.txt']}}})})
    assert any(g['code']=='dependency_version_mismatch' for g in review(index,rows)['gaps'])


def test_unknown_manifest_role_does_not_guess_current_from_filename(index):
    result=review(index,{**chain(),'requirements-next.txt':'paddleocr==3.0.0\n'})
    assert not any(g['code']=='dependency_version_mismatch' for g in result['gaps'])
    assert any(g['code']=='dependency_role_unknown' for g in result['gaps'])


def test_unrelated_validation_is_a_visible_nonblocking_limitation(index):
    rows=chain(); rows['unrelated.py']='def validate(x):\n    if x:\n        return x\n    return None\n'
    gaps=review(index,rows)['gaps']
    unrelated=[g for g in gaps if g.get('application',{}).get('path')=='unrelated.py']
    assert unrelated and all(g.get('category')=='analysis_limitation' and g.get('blocking') is False for g in unrelated)


def test_contract_requirements_have_application_source_not_official_source(index):
    rows=chain();rows['contract.json']=json.dumps({'required_line_fields':['text','confidence','page_index'],
        'output_fields':['schema_version','pages'],'page_policy':'preserve_empty_pages'})
    result=review(index,rows)
    assert result['application_contracts'][0]['source_identity']=='submitted_application'
    assert any(r['kind']=='application_contract' and r['status']=='NOT_EXECUTED'
               for r in result['regression_requirements'])


def test_conflicting_or_missing_role_references_cannot_silently_pick_a_version(index):
    rows={**chain(),'requirements.txt':'paddleocr==3.0.0\n','contract.json':json.dumps({
        'review_context':{'dependency_files':{'current':['requirements.txt'],'target':['requirements.txt']}}})}
    result=review(index,rows)
    assert any(g['code']=='conflicting_dependency_role' for g in result['gaps'])
    assert not any(f.get('dependency_role') for f in result['files'])
    rows['contract.json']=json.dumps({'review_context':{'dependency_files':{'target':['missing.txt']}}})
    assert any(g['code']=='invalid_review_context' for g in review(index,rows)['gaps'])


@pytest.mark.parametrize('adapter',[
    'def convert(raw):\n    def convert(raw):\n        return raw[0][0][1][0]\n    return []\n',
    'def convert(raw):\n    def unused():\n        return raw[0][0][1][0]\n    return []\n',
])
def test_uncalled_nested_function_does_not_inherit_argument_proof(index,adapter):
    rows=chain();rows['team/adapter.py']=adapter
    assert not risks(review(index,rows))


def test_local_definition_does_not_bind_same_named_module_function(index):
    rows=chain();rows['team/main.py']=('from .client import scan\nfrom .facade import adapt\n'
        'def process(path):\n    def adapt(raw):\n        return []\n    raw=scan(path)\n    return adapt(raw)\n')
    assert not risks(review(index,rows))


def test_test_fixture_branches_do_not_become_production_blockers(index):
    rows=chain();rows['tests/test_fixture.py']='def test_fixture(x):\n    if x:\n        assert x[0] == 1\n'
    result=review(index,rows)
    assert all(g.get('blocking') is False for g in result['gaps'] if g.get('application',{}).get('path')=='tests/test_fixture.py')


def test_test_only_old_result_use_is_not_reported_as_production_migration(index):
    rows=chain(adapter="return [p['rec_texts'] for p in raw]")
    rows['tests/test_old_result.py']=('from team.client import scan\ndef test_old_shape():\n'
        '    raw=scan("fixture")\n    assert raw[0][0][1][0] == "old"\n')
    findings=risks(review(index,rows))
    assert findings and all(f['status']=='needs_verification' for f in findings)


def test_conditional_wrapper_result_keeps_actionable_unknown(index):
    rows=chain()
    rows['team/main.py']=('from .client import scan\ndef process(path,flag):\n'
        '    if flag:\n        raw=scan(path)\n    else:\n        raw=[]\n    return raw[0][0][1][0]\n')
    result=review(index,rows)
    assert any(g.get('blocking') is True and g.get('application',{}).get('path')=='team/main.py'
               for g in result['gaps'])
