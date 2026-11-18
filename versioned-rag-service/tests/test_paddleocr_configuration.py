import pytest
from src.paddleocr_configuration import review_configuration


def files(value='.5', prefix=''):
    return [{'path': 'application/ocr_client.py', 'content': prefix +
             'from paddleocr import PaddleOCR\n'
             'def create_engine():\n'
             f'    return PaddleOCR(lang="ch", drop_score={value})\n'}]


def test_constant_assignment_is_traced_without_executing_application():
    before=files('threshold',prefix='threshold=.5\n')
    after=files('threshold',prefix='threshold=.8\n')
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert out['changes'][0]['after_value']==.8
    assert out['changes'][0]['after']['value_sources']
    assert not out['gaps']


@pytest.mark.parametrize('suffix,text', [('json','{"ocr":{"drop_score":0.8}}'),('yaml','ocr:\n  drop_score: 0.8\n')])
def test_submitted_configuration_chain_is_read_as_data_not_opened(suffix,text):
    module,loader=('json','load') if suffix=='json' else ('yaml','safe_load')
    code=f'import {module}\nfrom paddleocr import PaddleOCR\nsettings={module}.{loader}(open("application/ocr.{suffix}"))\nengine=PaddleOCR(drop_score=settings["ocr"]["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},
            {'path':f'application/ocr.{suffix}','content':text.replace('0.8','0.5')}]
    after=[before[0],{**before[1],'content':text}]
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert out['changes'][0]['after_value']==.8
    assert any(s['path'].endswith(suffix) and s['key_path']==['ocr','drop_score']
               for s in out['changes'][0]['after']['value_sources'])
    assert not out['gaps']
    assert out['runtime_verified'] is False


def test_literal_factory_argument_is_resolved_but_unknown_runtime_input_is_not():
    code='from paddleocr import PaddleOCR\ndef make(score):\n return PaddleOCR(drop_score=score)\nengine=make(.5)\n'
    before=[{'path':'application/ocr_client.py','content':code}]
    after=[{**before[0],'content':code.replace('make(.5)','make(.8)')}]
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert out['changes'][0]['after_value']==.8 and not out['gaps']
    after[0]['content']=code.replace('make(.5)','make(runtime_score)')
    assert not review_configuration(before,after,dependency_version='v2.9.1')['changes']


def test_configuration_other_field_change_and_source_rebinding_remain_gaps():
    code='import json\nfrom paddleocr import PaddleOCR\ncfg=json.load(open("application/ocr.json"))\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},
            {'path':'application/ocr.json','content':'{"drop_score":0.5,"lang":"ch"}'}]
    after=[before[0],{**before[1],'content':'{"drop_score":0.8,"lang":"en"}'}]
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert any(g['code']=='additional_code_changes' for g in out['gaps'])
    after[0]={**before[0],'content':code+'json=other\n'}
    assert not review_configuration(before,after,dependency_version='v2.9.1')['changes']


@pytest.mark.parametrize('extra',[
    'cfg["drop_score"]=.2',
    'cfg.update({"drop_score":.2})',
    'def open(*args): return other',
    'def json(): pass',
])
def test_mutated_config_or_shadowed_loader_is_not_resolved(extra):
    code='import json\nfrom paddleocr import PaddleOCR\ncfg=json.load(open("application/ocr.json"))\n'+extra+'\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},
            {'path':'application/ocr.json','content':'{"drop_score":0.5}'}]
    after=[before[0],{**before[1],'content':'{"drop_score":0.8}'}]
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


@pytest.mark.parametrize('body,tail',[
    (' score=.2\n return PaddleOCR(drop_score=score)','engine=make(.8)'),
    (' return PaddleOCR(drop_score=score)','alias=make\nengine=make(.8)\nother=alias(.2)'),
])
def test_ambiguous_factory_binding_cannot_use_a_convenient_call(body,tail):
    code='from paddleocr import PaddleOCR\ndef make(score):\n'+body+'\n'+tail+'\n'
    out=review_configuration(files(),[{'path':'application/ocr_client.py','content':code}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_malformed_yaml_becomes_a_gap_instead_of_crashing_review():
    code='import yaml\nfrom paddleocr import PaddleOCR\ncfg=yaml.safe_load(open("application/ocr.yaml"))\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},{'path':'application/ocr.yaml','content':'drop_score: 0.5'}]
    out=review_configuration(before,[before[0],{**before[1],'content':'drop_score: [invalid'}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_relative_open_does_not_guess_working_directory_from_python_file_path():
    code='import json\nfrom paddleocr import PaddleOCR\ncfg=json.load(open("ocr.json"))\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},{'path':'application/ocr.json','content':'{"drop_score":0.5}'}]
    out=review_configuration(before,[before[0],{**before[1],'content':'{"drop_score":0.8}'}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


@pytest.mark.parametrize('definition,call',[
    ('def make(*, score):\n return PaddleOCR(drop_score=score)','make(score=.9)'),
    ('def outer():\n score=.9\n def make():\n  return PaddleOCR(drop_score=score)\n return make()','outer()'),
])
def test_unsupported_parameter_or_closure_must_not_fall_back_to_module_constant(definition,call):
    code='from paddleocr import PaddleOCR\nscore=.8\n'+definition+'\nengine='+call+'\n'
    out=review_configuration(files(),[{'path':'application/ocr_client.py','content':code}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


@pytest.mark.parametrize('extra',[
    'alias=cfg\nalias["drop_score"]=.9',
    'unknown(target=cfg)',
    'unknown(target=cfg["ocr"])',
    'from other import json',
])
def test_data_alias_keyword_escape_or_loader_import_shadowing_stays_unknown(extra):
    code='import json\nfrom paddleocr import PaddleOCR\ncfg=json.load(open("application/ocr.json"))\n'+extra+'\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},
            {'path':'application/ocr.json','content':'{"drop_score":0.5,"ocr":{"drop_score":0.5}}'}]
    out=review_configuration(before,[before[0],{**before[1],'content':'{"drop_score":0.8,"ocr":{"drop_score":0.8}}'}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_a_different_functions_parameter_named_like_factory_is_not_a_factory_call():
    code='from paddleocr import PaddleOCR\ndef make(score):\n return PaddleOCR(drop_score=score)\ndef other(make):\n return make(.8)\n'
    out=review_configuration(files(),[{'path':'application/ocr_client.py','content':code}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_shared_scalar_source_keeps_threshold_delta_but_flags_other_consumers():
    code='from paddleocr import PaddleOCR\nscore=.5\nexport_filter=score\nengine=PaddleOCR(drop_score=score)\n'
    before=[{'path':'application/ocr_client.py','content':code}]
    after=[{**before[0],'content':code.replace('score=.5','score=.8')}]
    out=review_configuration(before,after,dependency_version='v2.9.1')
    assert out['changes'][0]['after_value']==.8
    assert any(g['code']=='shared_configuration_value' for g in out['gaps'])
    assert out['status']=='NEEDS_CONTEXT'


def test_loader_import_in_unrelated_function_is_not_visible_to_module_call():
    code='from paddleocr import PaddleOCR\ndef unused():\n import json\ncfg=json.load(open("application/ocr.json"))\nengine=PaddleOCR(drop_score=cfg["drop_score"])\n'
    before=[{'path':'application/ocr_client.py','content':code},{'path':'application/ocr.json','content':'{"drop_score":0.5}'}]
    out=review_configuration(before,[before[0],{**before[1],'content':'{"drop_score":0.8}'}],dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_same_sdk_threshold_increase_is_behavior_change_not_proven_quality_gain():
    out = review_configuration(files(), files('.8'), dependency_version='v2.9.1')
    assert out['changes'][0]['before_value'] == .5
    assert out['changes'][0]['after_value'] == .8
    assert out['changes'][0]['direction'] == 'stricter_filter'
    assert out['runtime_verified'] is False
    assert out['changes'][0]['after']['line'] == 3


@pytest.mark.parametrize('value', ['threshold', 'float(".8")', 'True', '-.1', '1.1', 'None'])
def test_dynamic_or_invalid_values_are_gaps_not_effective_configuration(value):
    out = review_configuration(files(), files(value), dependency_version='v2.9.1')
    assert not out['changes']
    assert out['gaps']


def test_unpacking_shadowing_and_missing_entry_never_prove_effective_parameter():
    for content in [
        'from paddleocr import PaddleOCR\nPaddleOCR(drop_score=.8, **options)',
        'from paddleocr import PaddleOCR\nPaddleOCR = other\nPaddleOCR(drop_score=.8)',
        'def PaddleOCR(**kw): pass\nPaddleOCR(drop_score=.8)',
        'from paddleocr import PaddleOCR\ndef f(PaddleOCR):\n return PaddleOCR(drop_score=.8)',
        'from paddleocr import PaddleOCR\ndef f():\n return PaddleOCR(drop_score=.8)\nPaddleOCR = other',
    ]:
        out = review_configuration(files(), [{'path': 'application/ocr_client.py', 'content': content}],
                                   dependency_version='v2.9.1')
        assert not out['changes'] and out['gaps']


def test_unchanged_threshold_is_not_a_new_impact():
    out = review_configuration(files(), files(), dependency_version='v2.9.1')
    assert not out['changes'] and out['status'] == 'NO_SUPPORTED_CHANGE'


def test_original_app_function_local_import_has_a_resolvable_binding():
    old = [{'path': 'application/ocr_client.py', 'content':
            'def create_engine():\n from paddleocr import PaddleOCR\n return PaddleOCR(lang="ch", drop_score=.5)\n'}]
    new = [{**old[0], 'content': old[0]['content'].replace('=.5', '=.8')}]
    out = review_configuration(old, new, dependency_version='v2.9.1')
    assert len(out['changes']) == 1 and not out['gaps']


def test_multiple_constructors_and_removed_file_stay_unknown():
    multi = files()[0]['content'] + '\nother = PaddleOCR(drop_score=.2)\n'
    out = review_configuration(files(), [{'path': 'application/ocr_client.py', 'content': multi}], dependency_version='v2.9.1')
    assert not out['changes'] and out['gaps']


def test_other_sdk_version_is_rejected_until_rules_are_implemented():
    with pytest.raises(ValueError):
        review_configuration(files(), files('.8'), dependency_version='v3.0.0')


def test_unrelated_edits_cannot_be_silently_approved_by_a_parameter_only_review():
    before, after = files(), files('.8')
    after[0]['content'] += '\nexport_mode = "new"\n'
    out = review_configuration(before, after, dependency_version='v2.9.1')
    assert any(g['code'] == 'additional_code_changes' for g in out['gaps'])
