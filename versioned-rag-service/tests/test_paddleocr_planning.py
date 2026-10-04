import pytest


def test_planning_cannot_change_version_or_parent_scope():
    from src.paddleocr_planning import validate_proposal
    items=[{'check_id':'c','versions':['v2.9.1','v3.0.0'],'query':'结果'}]
    with pytest.raises(ValueError):validate_proposal({'queries':[{'parent_check_id':'c','versions':['v3.0.0'],'query':'结果'}]},items)
    with pytest.raises(ValueError):validate_proposal({'queries':[{'parent_check_id':'c','versions':items[0]['versions'],'query':'v9.9.9 结果'}]},items)
