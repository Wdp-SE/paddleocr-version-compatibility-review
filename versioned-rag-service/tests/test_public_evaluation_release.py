from __future__ import annotations

from pathlib import Path

from src.build_identity import public_build_identity
from src.public_evaluation_release import validate_public_evaluation_release
from src.public_knowledge import PublicKnowledgeIndex


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "versioned-rag-service"
CORPUS = SERVICE / "public_corpus_paddleocr"


def _identity():
    return public_build_identity(
        repo_root=ROOT,
        corpus_root=CORPUS,
        retrieval_config_path=CORPUS / "public_retrieval_runtime.json",
    )


def test_new_pphuman_corpus_does_not_reuse_a_historical_evaluation():
    index = PublicKnowledgeIndex(CORPUS)
    release = validate_public_evaluation_release(index.manifest, CORPUS, _identity())
    assert release is None


def test_wrong_workspace_cannot_use_a_pphuman_evaluation():
    manifest = dict(PublicKnowledgeIndex(CORPUS).manifest)
    manifest["workspace_id"] = "unrelated_workspace"
    assert validate_public_evaluation_release(manifest, CORPUS, _identity()) is None


def test_changed_corpus_identity_invalidates_evaluation():
    identity = _identity()
    identity["corpus_fingerprint"] = dict(identity["corpus_fingerprint"])
    identity["corpus_fingerprint"]["manifest_sha256"] = "0" * 64
    assert validate_public_evaluation_release(PublicKnowledgeIndex(CORPUS).manifest, CORPUS, identity) is None


def test_changed_evaluation_fingerprint_invalidates_evaluation():
    identity = _identity()
    identity["evaluation_fingerprint"] = "0" * 64
    assert validate_public_evaluation_release(PublicKnowledgeIndex(CORPUS).manifest, CORPUS, identity) is None
