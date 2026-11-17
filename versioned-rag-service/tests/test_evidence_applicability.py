import copy
import json
from pathlib import Path
import pytest


def upgrade_chunks():
    rows=json.loads((Path(__file__).parents[1]/'public_corpus_paddleocr/chunks.json').read_text(encoding='utf8'))
    return [r for r in rows if r['document_path']=='docs/update/upgrade_notes.md']


def test_migration_code_keeps_publication_version_but_has_explicit_described_version():
    from src.paddleocr_retrieval_views import materialize_evidence
    chunks=upgrade_chunks(); original=copy.deepcopy(chunks)
    selected=[{'parent_chunk_id':r['chunk_id'],'line_start':r['line_start'],'line_end':r['line_end']}
              for r in chunks if r['line_start'] in (29,53)]
    old,new=materialize_evidence(chunks,selected)
    assert old['version']==new['version']=='v3.0.0'
    assert old['applicability']['described_version_family']=='2.x'
    assert old['applicability']['role']=='baseline_example'
    assert new['applicability']['described_version_family']=='3.x'
    assert new['applicability']['role']=='target_example'
    assert old['applicability']['basis']['line']==27
    assert chunks==original


def test_old_example_cannot_be_promoted_to_new_contract_by_model_judge():
    from src.paddleocr_retrieval_views import materialize_evidence
    from src.claim_support import check_claim_support
    chunks=upgrade_chunks(); old=next(r for r in chunks if r['line_start']==29)
    row=materialize_evidence(chunks,[{'parent_chunk_id':old['chunk_id'],'line_start':29,'line_end':49}])[0]
    claims=[{'text':'PaddleOCR 3.0.0 仍然保证使用 line[1][0] 读取文本。','evidence_ids':[old['chunk_id']]}]
    result=check_claim_support(claims,[row],lambda payload:({'verdicts':[
        {'claim_index':0,'supported':True,'reason':'代码出现此访问方式'}]},{}))
    assert result['claims']==[]
    assert result['rejected'][0]['reason']=='EXAMPLE_VERSION_SCOPE_MISMATCH'


def test_family_example_cannot_prove_an_exact_baseline_patch_release():
    from src.paddleocr_retrieval_views import materialize_evidence
    from src.paddleocr_applicability import version_applies
    chunks=upgrade_chunks();old=next(r for r in chunks if r['line_start']==29)
    row=materialize_evidence(chunks,[{'parent_chunk_id':old['chunk_id'],'line_start':29,'line_end':49}])[0]
    assert not version_applies(row,'v2.9.1')
    assert row['version']=='v3.0.0'


@pytest.mark.parametrize('query',[
    '同一个扫描件应用准备由 PaddleOCR 2.9.1 升到 3.0.0，只保留旧的 line[1][0] 读取方式为什么不稳妥？',
    'OCR 2.9.1 到 3.0.0，调用名 ocr 还存在就能认定结果归一化不用修改吗？',
])
def test_colloquial_upgrade_wording_uses_result_migration_route(query):
    from src.paddleocr_query_plan import retrieval_intent
    assert retrieval_intent(query)['kind']=='result_migration'
