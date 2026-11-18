from services.knowledge_examples import examples_for_scope, sync_scope_examples


def test_business_examples_have_explicit_version_scope_and_maintenance_purpose():
    rows = [row for scope in ('v2.9.1', 'v3.0.0', 'all') for row in examples_for_scope(scope)]
    assert len(rows) == 15
    assert len({row.question for row in rows}) == 15
    for scope in ('v2.9.1', 'v3.0.0', 'all'):
        scoped = examples_for_scope(scope)
        assert len(scoped) == 5
        assert all(row.purpose and row.category for row in scoped)
        assert all(row.scope == scope for row in scoped)
        if scope != 'all':
            other = '3.0.0' if scope == 'v2.9.1' else '2.9.1'
            assert all(scope[1:] in row.question and other not in row.question for row in scoped)
    assert examples_for_scope('unsupported') == ()


def test_scope_changes_replace_old_example_but_preserve_user_written_question():
    old = examples_for_scope('v2.9.1')[0].question
    state = {'official_example': old, 'official_question': old, 'official_result': ('stale',)}
    sync_scope_examples(state, 'v3.0.0')
    assert state['official_question'] == state['official_example'] == examples_for_scope('v3.0.0')[0].question
    assert 'official_result' not in state
    state['official_question'] = '我的自定义问题'
    sync_scope_examples(state, 'all')
    assert state['official_question'] == '我的自定义问题'


def test_unchanged_scope_does_not_reset_answer_or_custom_question():
    row = examples_for_scope('v2.9.1')[0]
    state = {'official_example': row.question, 'official_question': '自定义', 'official_result': ('keep',)}
    sync_scope_examples(state, row.scope)
    assert state['official_result'] == ('keep',)
    assert state['official_question'] == '自定义'
