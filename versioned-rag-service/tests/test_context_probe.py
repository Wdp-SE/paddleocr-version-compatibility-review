import copy
import importlib.util
from pathlib import Path
import sys


def test_probe_expands_only_selected_canonical_parents_with_budget_and_without_mutation():
    folder=Path(__file__).resolve().parents[2]/'evaluation/maintenance_acceptance_v1'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('parent_context_probe',folder/'parent_context_probe.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    text='one\ntwo\nthree\n'
    hit={'chunk_id':'selected','content':text,'line_start':1,
         'evidence_spans':[{'line_start':2,'line_end':2,'char_start':4,'char_end':8,'content':'two\n'}]}
    original=copy.deepcopy(hit)
    rows,ids=probe.expand_selected_parents([hit])
    assert ids==['selected'] and rows[0]['content']==text and 'evidence_spans' not in rows[0]
    assert hit==original
    rows,ids=probe.expand_selected_parents([hit],max_parent_chars=4)
    assert ids==[] and rows==[original]
