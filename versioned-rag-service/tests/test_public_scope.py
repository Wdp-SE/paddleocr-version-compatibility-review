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
