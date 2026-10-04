import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


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
