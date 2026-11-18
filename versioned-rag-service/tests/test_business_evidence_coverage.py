from pathlib import Path
import ast
import textwrap
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_query_plan import plan_query
from src.paddleocr_retrieval_views import generation_evidence_text
import pytest


@pytest.mark.parametrize('query', [
    '对照 OCR 2.9.1 与 3.0.0，初始化参数、调用入口、结果怎么读需要知道哪些？',
    'OCR 从 2.9.1 迁到 3.0.0，构造器、如何调用、返回结构分别要检查哪些内容？请附上两版出处。',
])
def test_upgrade_pack_has_both_callable_and_initialization_contracts_and_target_fields(query):
    index=PublicKnowledgeIndex(Path(__file__).parents[1]/'public_corpus_paddleocr')
    plan=plan_query(query, versions=('v2.9.1','v3.0.0'))
    out=configured_evidence_search(index,'contextual_bm25',structure_windows=True).search(
        plan,strategy='contextual_bm25',candidate_budget=80,top_k=5)
    by_version={v:'\n'.join(generation_evidence_text(h) for h in out['results'] if h['version']==v)
                for v in ('v2.9.1','v3.0.0')}
    assert 'def __init__' in by_version['v2.9.1']
    assert 'def ocr(' in by_version['v2.9.1']
    assert 'def __init__' in by_version['v3.0.0']
    assert 'def predict(' in by_version['v3.0.0']
    assert all(s in by_version['v3.0.0'] for s in ('rec_texts','rec_scores','text_rec_score_thresh'))
    # A signature plus the first half of return list(self.predict_iter(...))
    # is not a complete callable contract. Both actual functions must parse.
    for version,path in [('v2.9.1','paddleocr.py'),('v3.0.0','paddleocr/_pipelines/ocr.py')]:
        hits=[h for h in out['results'] if h['version']==version and h['document_path']==path
              and 'def '+('ocr(' if version=='v2.9.1' else 'predict(') in generation_evidence_text(h)]
        assert hits
        for h in hits:
            ast.parse(textwrap.dedent(generation_evidence_text(h)))


def test_method_relationship_question_retrieves_the_real_forwarding_implementation():
    index=PublicKnowledgeIndex(Path(__file__).parents[1]/'public_corpus_paddleocr')
    plan=plan_query('OCR 3.0.0 的 ocr() 方法还保留吗，它与 predict() 是什么关系？',
                    versions=('v3.0.0',))
    out=configured_evidence_search(index,'contextual_bm25',structure_windows=True).search(
        plan,strategy='contextual_bm25',candidate_budget=80,top_k=5)
    assert any('def ocr(' in generation_evidence_text(h) and 'self.predict(' in generation_evidence_text(h)
               for h in out['results'] if h['document_path']=='paddleocr/_pipelines/ocr.py')


def test_request_for_field_evidence_keeps_ocr_scope_and_the_field_definition():
    index=PublicKnowledgeIndex(Path(__file__).parents[1]/'public_corpus_paddleocr')
    plan=plan_query('请提供 OCR 3.0.0 中 rec_texts 和 rec_scores 含义的依据。',versions=('v3.0.0',))
    out=configured_evidence_search(index,'contextual_bm25',structure_windows=True).search(
        plan,strategy='contextual_bm25',candidate_budget=80,top_k=5)
    assert out['results']
    assert all(h.get('module') in ('ocr','shared','general') for h in out['results'])
    assert all('PaddleOCR' in h.get('document_title','')+h['content']
               for h in out['results'] if h.get('module')=='general')
    assert any(all(t in generation_evidence_text(h) for t in ('rec_texts','rec_scores','text_rec_score_thresh'))
               for h in out['results'] if h['document_path'].endswith('pipeline_usage/OCR.md'))
