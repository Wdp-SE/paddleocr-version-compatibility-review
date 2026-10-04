from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'examples/paddleocr_document_app'))


def test_old_and_new_outputs_normalize_to_same_downstream_contract():
    from result_adapter import normalize_v2,normalize_v3
    old=[[[[[0,0],[1,0],[1,1],[0,1]],('合同编号',.99)]]]
    new=[{'rec_texts':['合同编号'],'rec_scores':[.99]}]
    assert normalize_v2(old)==normalize_v3(new)==[{'text':'合同编号','confidence':.99,'page_index':0}]


def test_blank_pages_and_multi_pages_keep_page_identity():
    from result_adapter import normalize_v2,normalize_v3
    assert normalize_v2([None])==[]
    assert normalize_v3([{'rec_texts':[],'rec_scores':[]}])==[]
    rows=normalize_v3([{'rec_texts':[],'rec_scores':[]},{'rec_texts':['第二页'],'rec_scores':[.8]}])
    assert rows[0]['page_index']==1


@pytest.mark.parametrize('output',[
    [{'rec_texts':['文本'],'rec_scores':[]}],
    [{'rec_texts':['文本'],'rec_scores':[float('nan')]}],
    [{'rec_texts':['文本'],'rec_scores':[1.1]}],
    [{'text':'没有结果字段'}],
    [{'rec_texts':'ABC','rec_scores':[.8,.9,.7]}],
    [{'rec_texts':['A','B'],'rec_scores':'01'}],
])
def test_invalid_output_is_not_silent_empty_success(output):
    from result_adapter import normalize_v3
    with pytest.raises(ValueError):normalize_v3(output)
