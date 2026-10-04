import pytest


def test_duplicate_casefold_paths_are_rejected():
    from services.paddleocr_application_inputs import parse_application_inputs
    with pytest.raises(ValueError,match='duplicate'):
        parse_application_inputs([{'path':'app/A.py','content':'pass'},{'path':'app/a.py','content':'pass'}])


def test_unicode_bytes_and_relative_paths_are_bounded():
    from services.paddleocr_application_inputs import parse_application_inputs
    with pytest.raises(ValueError):parse_application_inputs([{'path':'../escape.py','content':'pass'}])
    with pytest.raises(ValueError):parse_application_inputs([{'path':'app.py','content':'中'*20000}])


def test_default_application_is_integral_and_fingerprint_changes_on_configuration():
    from services.paddleocr_application_inputs import load_original_application, application_fingerprint
    files=load_original_application()
    assert 4<len(files)<=12 and any(f['path'].endswith('consumer.py') for f in files)
    a=application_fingerprint(files,('v2.9.1','v3.0.0'),{'generate':False},'corpus-a')
    assert a!=application_fingerprint(files,('v2.9.1','v3.0.0'),{'generate':True},'corpus-a')
    assert a!=application_fingerprint(files,('v2.9.1','v3.0.0'),{'generate':False},'corpus-b')
