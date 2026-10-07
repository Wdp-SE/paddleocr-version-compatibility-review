import pytest
from services.internal_application_scope import dependency_scope


def test_default_is_application_dependency_not_latest_sdk():
    assert dependency_scope(['v3.0.0', 'v2.9.1'], 'current') == 'v2.9.1'


def test_explicit_selection_and_target_are_separate():
    versions = ['v3.0.0', 'v2.9.1']
    assert dependency_scope(versions, 'target') == 'v3.0.0'
    assert dependency_scope(versions, 'manual', 'v2.9.1') == 'v2.9.1'
    assert dependency_scope(versions, 'manual', 'all') == 'all'


def test_unavailable_dependency_never_falls_back_to_wrong_version():
    with pytest.raises(ValueError, match='未收录'):
        dependency_scope(['v3.0.0'], 'current')
    with pytest.raises(ValueError):
        dependency_scope(['v3.0.0'], 'manual', 'v9')
