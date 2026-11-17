"""Result-origin coverage must survive ordinary wrapping or become explicitly unknown."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.paddleocr_compatibility import review_compatibility


@pytest.fixture(scope='module')
def index():
    root=Path(__file__).resolve().parents[1]/'public_corpus_paddleocr'
    return SimpleNamespace(manifest=json.loads((root/'corpus_manifest.json').read_text(encoding='utf8')),
        chunks=json.loads((root/'chunks.json').read_text(encoding='utf8')))


def review(index,body,arguments='path'):
    code='from paddleocr import PaddleOCR\nengine=PaddleOCR(lang="ch")\n'+f'def process({arguments}):\n'+body
    return review_compatibility(index,source_version='v2.9.1',target_version='v3.0.0',
        files=[{'path':'application/ocr_client.py','content':code}])


@pytest.mark.parametrize('expression',[
    'list(engine.ocr(path))','engine.ocr(path)[:]',
    '[page for page in engine.ocr(path)]',
])
def test_proven_sequence_wrappers_keep_legacy_read_risk(index,expression):
    report=review(index,f'    raw={expression}\n    return raw[0][0][1][0]\n')
    assert any(f['rule_id']=='legacy_ocr_result' and f['status']=='supported_risk' for f in report['findings'])
    assert report['runtime_verified'] is False


@pytest.mark.parametrize('body',[
    '    envelope={"payload":engine.ocr(path)}\n    raw=envelope["payload"]\n    return raw[0][0][1][0]\n',
    '    page=next(iter(engine.ocr(path)))\n    return page[0][1][0]\n',
    '    raw=unknown_transform(engine.ocr(path))\n    return raw[0][0][1][0]\n',
])
def test_unknown_transform_cannot_silently_be_unaffected(index,body):
    report=review(index,body)
    risk=any(f['rule_id']=='legacy_ocr_result' and f['status']=='supported_risk' for f in report['findings'])
    gaps=[g for g in report['gaps'] if g.get('blocking') and g.get('application',{}).get('path')=='application/ocr_client.py']
    assert risk or gaps
    assert report['status']!='unaffected'


def test_async_return_keeps_explicit_analysis_gap(index):
    code=('from paddleocr import PaddleOCR\nengine=PaddleOCR()\n'
          'async def scan(path):\n    return engine.ocr(path)\n'
          'async def process(path):\n    raw=await scan(path)\n    return raw[0][0][1][0]\n')
    r=review_compatibility(index,source_version='v2.9.1',target_version='v3.0.0',files=[{'path':'application/a.py','content':code}])
    assert r['status']=='needs_verification'
    assert any(g.get('blocking') and g.get('application',{}).get('path')=='application/a.py' for g in r['gaps'])


@pytest.mark.parametrize('body,args',[
    ('    raw=list([[[[1]]]])\n    return raw[0][0][0][0]\n','path'),
    ('    raw=engine.ocr(path)\n    raw=[[[[1]]]]\n    return raw[0][0][0][0]\n','path'),
    ('    raw=list(engine.ocr(path))\n    return raw[0][0][1][0]\n','path,list'),
])
def test_plain_values_reassignments_and_shadowed_builtins_not_proven_ocr_results(index,body,args):
    r=review(index,body,args)
    assert not any(f['rule_id']=='legacy_ocr_result' and f['status']=='supported_risk' for f in r['findings'])


def test_rereview_preserves_unknown_flow_as_unresolved(index):
    before=review(index,'    raw=engine.ocr(path)\n    return raw[0][0][1][0]\n')
    after=review(index,'    raw=unknown_transform(engine.ocr(path))\n    return raw[0][0][1][0]\n')
    assert before['summary']['risk_count']>0
    assert after['status']=='needs_verification' and any(g.get('blocking') for g in after['gaps'])


def test_unknown_mutator_cannot_leave_stale_shape_proof(index):
    r=review(index,'    raw=engine.ocr(path)\n    mutate(raw)\n    return raw[0][0][1][0]\n')
    assert not any(f['rule_id']=='legacy_ocr_result' and f['status']=='supported_risk' for f in r['findings'])
    assert any(g['code']=='ocr_result_flow_unresolved' and g.get('blocking') for g in r['gaps'])


def test_named_field_projection_is_not_an_unknown_sequence_transform(index):
    r=review(index,"    raw=engine.ocr(path)\n    return [page['rec_texts'] for page in raw]\n")
    assert not any(g['code']=='ocr_result_flow_unresolved' for g in r['gaps'])
    assert not any(f['rule_id']=='legacy_ocr_result' for f in r['findings'])
    assert r['runtime_verified'] is False
