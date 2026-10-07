"""Business regressions: source-grounded retrieval and same-app consumers."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.paddleocr_compatibility import review_compatibility
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from src.paddleocr_retrieval_views import generation_evidence_text
from src.public_knowledge import PublicKnowledgeIndex

CORPUS = Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr'


@pytest.fixture(scope='module')
def index():
    return PublicKnowledgeIndex(CORPUS)


@pytest.mark.parametrize('question', [
    'PaddleOCR 2.9.1 的 OCR 调用如何只识别文字而不检测？',
    'v2.9.1 Python OCR 接口只跑识别，关闭检测，怎么调用？',
    '旧版 v2.9.1 OCR 识别已有裁剪文本图片时传什么参数？',
])
def test_api_intent_returns_call_arguments_not_service_deployment(index, question):
    out = configured_evidence_search(index).search(plan_query(question, versions=('v2.9.1',)))
    assert any('det=False' in generation_evidence_text(h)
               and h['document_path'] in ('doc/doc_ch/whl.md', 'tests/test_paddleocr_api.py', 'paddleocr.py')
               for h in out['results'])


@pytest.mark.parametrize('question', [
    'PaddleOCR 3.0.0 的官方 OCR 测试检查了哪些返回字段？',
    'v3.0.0 OCR predict 的测试有哪些结果断言？',
])
def test_test_intent_retrieves_assertions_not_only_fixture(index, question):
    out = configured_evidence_search(index).search(plan_query(question, versions=('v3.0.0',)))
    assert any('assert isinstance(res["rec_texts"], list)' in generation_evidence_text(h)
               for h in out['results'] if h['document_path'] == 'tests/pipelines/test_ocr.py')


def run_review(index, consumer, wrapper=None):
    wrapper = wrapper or ('from paddleocr import PaddleOCR\ndef recognize(path):\n'
                          '    engine=PaddleOCR(lang="ch")\n    return engine.ocr(path)\n')
    return review_compatibility(index, source_version='v2.9.1', target_version='v3.0.0', files=[
        {'path': 'application/ocr_client.py', 'content': wrapper},
        {'path': 'application/consumer.py', 'content': consumer},
    ])


@pytest.mark.parametrize('binding, call', [('recognize', 'recognize'), ('recognize as scan', 'scan')])
def test_cross_file_raw_result_consumer_is_located(index, binding, call):
    report = run_review(index, f'from .ocr_client import {binding}\ndef process(path):\n'
                        f'    pages={call}(path)\n    return pages[0][0][1][0]\n')
    risk = [f for f in report['findings'] if f['rule_id'] == 'legacy_ocr_result']
    assert any(f['application']['path'] == 'application/consumer.py' and f['application']['line'] == 4 for f in risk)
    assert report['runtime_verified'] is False


@pytest.mark.parametrize('consumer, wrapper', [
    ('from .ocr_client import recognize\ndef process(path, recognize):\n    pages=recognize(path)\n    return pages[0][0][1][0]\n', None),
    ('from .ocr_client import recognize\nrecognize=lambda path: [[[(0,"other")]]]\ndef process(path):\n    pages=recognize(path)\n    return pages[0][0][1][0]\n', None),
    ('from .ocr_client import recognize\ndef process(path):\n    pages=recognize(path)\n    return pages[0][0][1][0]\n',
     'from paddleocr import PaddleOCR\ndef recognize(path):\n    if path:\n        return PaddleOCR().ocr(path)\n    return []\n'),
])
def test_unproven_wrappers_do_not_create_supported_consumer_risks(index, consumer, wrapper):
    report = run_review(index, consumer, wrapper)
    assert not any(f['rule_id'] == 'legacy_ocr_result' and f['application']['path'] == 'application/consumer.py'
                   for f in report['findings'])
    assert report['runtime_verified'] is False


def test_wrapper_summary_depth_has_a_real_six_layer_bound(index):
    wrapper='from paddleocr import PaddleOCR\ndef f0(path):\n    engine=PaddleOCR()\n    return engine.ocr(path)\n'
    for i in range(1,8):wrapper+=f'def f{i}(path):\n    return f{i-1}(path)\n'
    report=run_review(index,'from .ocr_client import f7\ndef process(path):\n    pages=f7(path)\n    return pages[0][0][1][0]\n',wrapper)
    assert not any(f['rule_id']=='legacy_ocr_result' and f['application']['path']=='application/consumer.py'
                   for f in report['findings'])


def test_missing_internal_contract_is_not_answered_from_sdk_docs():
    from src.public_scope import is_out_of_scope_public_request
    assert is_out_of_scope_public_request('仅凭官方资料，能否确认我们应用升级后下游 JSON 输出兼容？')
    assert not is_out_of_scope_public_request('PaddleOCR 3.0.0 的结果如何保存为 JSON？')


def test_migration_answer_has_both_result_contracts(index):
    out=configured_evidence_search(index).search(plan_query(
        'PaddleOCR 2.9.1 升级到 3.0.0，OCR 旧结果读取方式为什么需要调整？',
        versions=('v2.9.1','v3.0.0')))
    assert any(h['version']=='v2.9.1' and ('line[1]' in generation_evidence_text(h) or 'ocr_res.append' in generation_evidence_text(h)) for h in out['results'])
    assert any(h['version']=='v3.0.0' and 'rec_texts' in generation_evidence_text(h) and 'rec_scores' in generation_evidence_text(h) for h in out['results'])
    assert any(h['version']=='v3.0.0' and 'def ocr(' in generation_evidence_text(h)
               and 'self.predict' in generation_evidence_text(h) for h in out['results'])


def test_repaired_answer_replaces_instead_of_concatenating_partial_paraphrases(monkeypatch):
    import asyncio
    from src import public_api
    first={'status':'OK','answer_completeness':'PARTIAL_SUPPORTED','claims':[{'text':'旧式索引取文字','evidence_ids':['a']}],
           'generation':{},'claim_verification':{'status':'PARTIAL_SUPPORTED'}}
    second={'status':'OK','answer_completeness':'PARTIAL_SUPPORTED','claims':[{'text':'旧版通过索引读取文字','evidence_ids':['a']}],
            'generation':{},'claim_verification':{'status':'PARTIAL_SUPPORTED'}}
    for result in (first,second):result.update(sources=[{'chunk_id':'a'}],evidence=[])
    expected=list(second['claims'])
    results=iter([first,second])
    async def once(*args,**kwargs):return next(results)
    monkeypatch.setattr(public_api,'_query_once',once)
    monkeypatch.setattr(public_api,'_index',lambda request:SimpleNamespace(manifest={'workspace_id':'paddleocr'}))
    monkeypatch.setattr(public_api,'_answer_evidence_support',lambda *args: {})
    out=asyncio.run(public_api.query(public_api.SearchRequest(query='旧结果读取',version='v2.9.1'),None))
    assert out['claims']==expected and len(out['claims'])==1
