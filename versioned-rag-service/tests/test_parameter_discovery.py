import pytest
from pathlib import Path
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_query_plan import plan_query
from src.paddleocr_evidence_search import configured_evidence_search


@pytest.mark.parametrize('question', [
    'v2.9.1 旧版 OCR 一次识别多张裁剪图，默认是多少？',
    'v2.9.1 OCR 文本识别一批可以处理几张图片，默认配置怎么查？',
])
def test_colloquial_batch_question_retrieves_parameter_definition(question):
    index = PublicKnowledgeIndex(Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr')
    plan = plan_query(question, versions=('v2.9.1',))
    search = configured_evidence_search(index, 'contextual_bm25', structure_windows=True)
    result = search.search(plan, candidate_budget=120, top_k=8, max_windows_per_source=8)
    assert any('rec_batch_num' in r['content'] and '识别' in r['content'] and '6' in r['content'] for r in result['results'])
    assert all(r['version'] == 'v2.9.1' for r in result['results'])


def test_query_aliases_are_search_terms_not_default_values_or_answers():
    plan = plan_query('OCR 识别一批图片默认张数是多少？', versions=('v2.9.1',))
    aliases = plan['retrieval_intent']['aliases']
    assert aliases and any('batch' in q for q in aliases)
    assert all('rec_batch_num' not in q and '6' not in q for q in aliases)
