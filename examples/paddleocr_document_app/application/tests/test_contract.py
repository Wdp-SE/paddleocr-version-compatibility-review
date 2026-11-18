import importlib.util
from pathlib import Path
import sys
import pytest
import importlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))


def load(name):
    sys.path.insert(0,str(ROOT.parent))
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_normalized_results_work_with_downstream_contract():
    normalizer=load('normalizer'); consumer=load('consumer')
    old=[[[[0,0],[1,0],[1,1],[0,1]],('合同编号 DEMO',.98)]]
    result=normalizer.normalize_legacy([old])
    assert consumer.to_json(result)['pages'][0]['text']=='合同编号 DEMO'
    assert consumer.to_json(result)['schema_version']==1


def test_v3_dictionary_cannot_be_consumed_as_legacy_list():
    normalizer=load('normalizer')
    with pytest.raises((KeyError,TypeError,IndexError,ValueError)):
        normalizer.normalize_legacy([{'rec_texts':['合同编号 DEMO'],'rec_scores':[.98]}])


class FixtureEngine:
    """Test-only OCR results; no model or submitted source is executed."""
    def __init__(self, raw): self.raw=raw
    def ocr(self, path): return self.raw


def entry_modules():
    return importlib.import_module('application.pipeline'), importlib.import_module('application.ocr_client')


def test_real_entrypoint_preserves_empty_page_positions(monkeypatch):
    pipeline,client=entry_modules()
    raw=[[[None,('first',.98)]], [], [[None,('last',.97)]]]
    monkeypatch.setattr(client,'create_engine',lambda:FixtureEngine(raw))
    result=pipeline.process('original-fixture')
    assert result=={'schema_version':1,'pages':[{'page_index':0,'text':'first'},
        {'page_index':1,'text':''},{'page_index':2,'text':'last'}]}


def test_real_entrypoint_rejects_new_shape_until_adapter_is_changed(monkeypatch):
    pipeline,client=entry_modules()
    monkeypatch.setattr(client,'create_engine',lambda:FixtureEngine([{'rec_texts':['first'],'rec_scores':[.98]}]))
    with pytest.raises(ValueError): pipeline.process('original-fixture')


def test_proposed_field_adapter_through_same_entrypoint_is_fixture_only(monkeypatch):
    pipeline,client=entry_modules()
    from application.normalizer import normalize_target
    monkeypatch.setattr(client,'create_engine',lambda:FixtureEngine([
        {'rec_texts':['first'],'rec_scores':[.98]}, {'rec_texts':[],'rec_scores':[]}]))
    # Explicit fixture injection is not a committed candidate or real OCR inference.
    monkeypatch.setattr(pipeline,'normalize_legacy',normalize_target)
    result=pipeline.process('original-fixture')
    assert result['pages']==[{'page_index':0,'text':'first'},{'page_index':1,'text':''}]


def test_consumer_rejects_impossible_page_count():
    from application.consumer import to_json
    with pytest.raises(ValueError): to_json([{'page_index':1,'text':'x','confidence':.9}],page_count=1)


def test_runtime_runner_checks_real_entry_and_reports_failure_without_fallback():
    from run_upgrade_demo import check_application_entry
    result=check_application_entry(FixtureEngine([{'rec_texts':['x'],'rec_scores':[.9]}]),'fixture')
    assert result['entrypoint']=='application.pipeline.process'
    assert result['passed'] is False and result['error']=='ValueError'


@pytest.mark.parametrize('output',[
    {'schema_version':1,'pages':[]},
    {'schema_version':1,'pages':[{'page_index':0,'text':'wrong text'}]},
    {'schema_version':1,'pages':[{'page_index':2,'text':'expected'}]},
])
def test_runtime_runner_rejects_wrong_entry_output_even_without_exception(monkeypatch,output):
    from run_upgrade_demo import check_application_entry
    pipeline,_=entry_modules()
    monkeypatch.setattr(pipeline,'process',lambda _:output)
    result=check_application_entry(FixtureEngine([]),'fixture',expected_fragments=['expected'])
    assert result['passed'] is False
