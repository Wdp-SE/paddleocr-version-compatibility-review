import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from evaluation.depth_quality_v1.curated_context_probe import curated_evidence


def test_curated_control_cannot_use_another_version_or_insert_rubric_text():
    chunk={'chunk_id':'a','source_id':'s','version':'v2.9.1','document_path':'api.py',
           'content':'def ocr(): pass','line_start':1,'line_end':1}
    index=SimpleNamespace(chunks=[chunk])
    case={'expected_facts':['SECRET_REFERENCE_ANSWER'],'oracle_sources':[
        {'version':'v3.0.0','path':'api.py','tokens':['def ocr']}]}
    with pytest.raises(ValueError,match='unavailable'):curated_evidence(case,index)
    case['oracle_sources'][0]['version']='v2.9.1'
    result=curated_evidence(case,index)
    assert result[0]['content']==chunk['content'] and 'SECRET' not in str(result)


def test_declared_diagnostic_anchors_exist_in_current_frozen_corpus():
    root=Path(__file__).resolve().parents[2]
    chunks=json.loads((root/'versioned-rag-service/public_corpus_paddleocr/chunks.json').read_text(encoding='utf8'))
    cases=json.loads((root/'evaluation/depth_quality_v1/cases.json').read_text(encoding='utf8'))['cases']
    for case in cases:
        if case.get('oracle_sources'):assert curated_evidence(case,SimpleNamespace(chunks=chunks))
