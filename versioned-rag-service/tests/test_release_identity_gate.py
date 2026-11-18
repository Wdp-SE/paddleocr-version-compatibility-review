import pytest
from scripts.paddleocr_release_smoke import validate_release_identity


def test_unknown_or_uncommitted_revision_cannot_pass_release_gate():
    with pytest.raises(ValueError):
        validate_release_identity({'build_revision': 'unknown'}, {'build_revision': 'unknown'})


def test_same_source_count_does_not_hide_a_changed_corpus_or_strategy():
    identity = {'build_revision': 'a'*40, 'corpus_fingerprint': {'fingerprint_sha256': 'b'*64},
                'service_code_fingerprint': 'c'*64, 'review_code_fingerprint': 'd'*64,
                'retrieval_config_fingerprint': 'e'*64}
    validate_release_identity(identity, identity)
    for key in ('corpus_fingerprint','service_code_fingerprint','review_code_fingerprint','retrieval_config_fingerprint'):
        changed = {**identity, key: {'fingerprint_sha256':'f'*64} if key=='corpus_fingerprint' else 'f'*64}
        with pytest.raises(ValueError): validate_release_identity(changed, identity)
