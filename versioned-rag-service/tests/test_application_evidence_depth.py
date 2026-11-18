"""Application knowledge retrieval must preserve structures and source identity."""
from src.application_knowledge import search_application


def search(files,query,top_k=5):
    return search_application(files,query,application_id='ocr',application_commit='a'*40,top_k=top_k)


def test_contract_and_test_intent_has_both_source_roles():
    files=[{'path':'contract.json','content':'{"page_policy":"preserve_empty_pages","required_line_fields":["page_index","text"]}'},
        {'path':'tests/test_output.py','content':'def test_empty_page():\n    assert actual["pages"][0]["text"] == ""\n'},
        {'path':'noise.py','content':'page = "空页 空页 空页 测试 测试 测试"\n'*12}]
    rows=search(files,'空页的输出契约和测试断言是什么？',top_k=2)['results']
    assert {r['path'] for r in rows}=={'contract.json','tests/test_output.py'}
    assert {r['document_role'] for r in rows}=={'output_contract','application_test'}


def test_function_window_keeps_signature_and_assertion_together():
    content='\n'*18+'def test_pages():\n    result=process()\n    assert result["pages"] == []\n'
    rows=search([{'path':'tests/test_pages.py','content':content}],'test_pages pages assert')['results']
    assert rows and 'def test_pages' in rows[0]['content'] and 'assert result' in rows[0]['content']
    assert rows[0]['content']=='\n'.join(content.splitlines()[rows[0]['line_start']-1:rows[0]['line_end']])


def test_unknown_query_has_no_fabricated_application_evidence():
    assert not search([{'path':'x.py','content':'n=12\n'}],'unrelated_missing_identifier')['results']
