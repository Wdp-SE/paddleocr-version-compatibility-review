import pytest
from services.evidence_span_display import display_spans

def row():
    return {'content':'first\nsecond\nthird','line_start':10,'line_end':12,
            'evidence_spans':[{'line_start':11,'line_end':11,'char_start':6,'char_end':12,'content':'second'}]}

def test_only_actual_selected_original_text_is_displayed():
    assert display_spans(row())==[{'content':'second','line_start':11,'line_end':11}]

@pytest.mark.parametrize('field,value',[('content','invented'),('line_start',10),('char_end',90)])
def test_fabricated_or_outside_spans_cannot_be_displayed(field,value):
    value_row=row();value_row['evidence_spans'][0][field]=value
    with pytest.raises(ValueError):display_spans(value_row)

def test_legacy_result_without_windows_keeps_full_body():
    value=row();value.pop('evidence_spans')
    assert display_spans(value)==[{'content':value['content'],'line_start':10,'line_end':12}]
