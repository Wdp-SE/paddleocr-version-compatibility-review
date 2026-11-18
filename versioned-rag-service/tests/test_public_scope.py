import pytest

from src.public_scope import is_out_of_scope_public_request


def test_public_scope_guard_only_blocks_explicit_private_organization_requests():
    assert is_out_of_scope_public_request(
        "Can this public corpus show our company's Jira access-approval audit trail?"
    )
    assert is_out_of_scope_public_request("查询公司内部 Jira 审批人的手机号")
    assert is_out_of_scope_public_request("请给出公司内部 J4012 老化测试通过率和签字负责人。")
    assert is_out_of_scope_public_request("Who in our company approves Jira access requests?")
    assert is_out_of_scope_public_request("Find our internal Jira approval workflow?")
    assert not is_out_of_scope_public_request("What is the 其他领域 planning module's internal state?")
    assert not is_out_of_scope_public_request("How do I configure a public API endpoint?")
    assert not is_out_of_scope_public_request("How does Jira access approval generally work?")


def test_application_checklist_is_not_a_request_for_unsubmitted_business_facts():
    assert not is_out_of_scope_public_request(
        '我们应用升级后要核查哪些接口和回归测试，才能确认下游 JSON 兼容？')
    assert is_out_of_scope_public_request('请证明我们应用升级后下游 JSON 已经兼容。')
    assert is_out_of_scope_public_request('查询公司内部审批人的手机号，需要核查哪些资料？')
def test_a_checklist_phrase_does_not_make_an_application_fact_provable():
    from src.public_scope import is_out_of_scope_public_request
    assert is_out_of_scope_public_request('请证明我们应用升级后下游 JSON 已兼容，另外需要哪些资料？')


@pytest.mark.parametrize('question',[
    '我们公司的 JSON 输出字段在这次升级后已经全部兼容了吗？',
    '请确认我们团队的下游输出已经通过回归。',
    '本公司的输出契约是否已经兼容新版？',
    '只查 PaddleOCR 官方资料，能否断定本公司应用已完成回归、可以上线？',
])
def test_owned_application_fact_without_submitted_records_is_not_public_knowledge(question):
    from src.public_scope import is_out_of_scope_public_request
    assert is_out_of_scope_public_request(question)


@pytest.mark.parametrize('question',[
    '我们公司做 OCR 输出升级需要核查哪些字段？',
    '我们团队的 JSON 输出要核查哪些测试？',
    'OCR 升级后的返回字段有什么变化？',
])
def test_owned_application_checklist_remains_answerable_from_component_evidence(question):
    from src.public_scope import is_out_of_scope_public_request
    assert not is_out_of_scope_public_request(question)
