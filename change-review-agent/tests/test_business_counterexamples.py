"""Business negatives against actual pinned sources and the actual review graph."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.paddleocr_review import PaddleOCRReviewAgent

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.paddleocr_compatibility import review_compatibility


@pytest.fixture(scope='module')
def gateway():
    corpus=ROOT/'versioned-rag-service/public_corpus_paddleocr'
    index=SimpleNamespace(manifest=json.loads((corpus/'corpus_manifest.json').read_text(encoding='utf-8')),
                          chunks=json.loads((corpus/'chunks.json').read_text(encoding='utf-8')))
    class Gateway:
        def workspace(self):
            return {'workspace_id':'paddleocr','domain_profile':{'id':'paddleocr'},
                'repository':'PaddlePaddle/PaddleOCR','repositories':['PaddlePaddle/PaddleOCR'],
                'languages':['zh'],'available_versions':['v2.9.1','v3.0.0'], 'rag_ready':True,
                'source_registry':index.manifest['sources'], 'corpus_fingerprint':{'fingerprint_sha256':'a'*64}}
        def compatibility_review(self, files, request_timeout=None, **versions):
            return review_compatibility(index, files=files, **versions)
        def document(self, source_id, **kw):
            return [r for r in index.chunks if source_id in (r['source_id'],r['document_id'])]
        def search(self, query, **kw): return {'results':[]}
    return Gateway()


def run(gateway, files):
    return PaddleOCRReviewAgent(gateway).analyze(files,generate_advice=False)


def test_adapter_absorbs_return_change_without_changing_internal_consumer(gateway):
    consumer={'path':'application/consumer.py','content':'def consume(rows):\n return [r["text"] for r in rows]\n'}
    legacy={'path':'application/adapter.py','content':'from paddleocr import PaddleOCR\ndef adapt(path):\n pages=PaddleOCR(lang="ch").ocr(path)\n return [{"text":line[1][0]} for line in pages[0]]\n'}
    adapted={'path':'application/adapter.py','content':'from paddleocr import PaddleOCR\ndef adapt(path):\n results=PaddleOCR(lang="ch").predict(path)\n return [{"text":text} for page in results for text in page["rec_texts"]]\n'}
    before=copy.deepcopy(consumer)
    old=run(gateway,[legacy,consumer])
    new=run(gateway,[adapted,consumer])
    assert old['compatibility_report']['summary']['risk_count']>=1
    assert new['compatibility_report']['summary']['risk_count']==0
    assert consumer==before
    assert new['runtime_verified'] is False
    assert new['graph_execution']['engine']=='langgraph'


def test_unrelated_indexing_and_readme_change_do_not_manufacture_sdk_risk(gateway):
    files=[{'path':'application/local.py','content':'def first(items):\n return items[0][1][0]\n'},
           {'path':'application/README.md','content':'内部维护说明已更新。'}]
    out=run(gateway,files)
    assert out['compatibility_report']['summary']['risk_count']==0
    assert out['runtime_verified'] is False


def test_missing_call_wrapper_is_not_interpreted_as_fixed(gateway):
    files=[{'path':'application/normalizer.py','content':'from .ocr_client import recognize\ndef adapt(path):\n return [line[1][0] for line in recognize(path)[0]]\n'}]
    out=run(gateway,files)
    report=out['compatibility_report']
    assert report['gaps']
    assert out['runtime_verified'] is False
    assert out['status']!='unaffected'
