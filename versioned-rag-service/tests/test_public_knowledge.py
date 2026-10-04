from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.public_knowledge import (
    PublicKnowledgeIndex, _parts, build_index, verified_consistency_notes,
)

CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_pphuman"


@pytest.fixture(scope="module")
def index() -> PublicKnowledgeIndex:
    return PublicKnowledgeIndex(CORPUS)


def test_pinned_chinese_pphuman_corpus_metadata_and_source_hashes(index):
    manifest = index.manifest
    assert manifest["workspace_id"] == "pphuman"
    assert manifest["repository"] == "PaddlePaddle/PaddleDetection"
    assert manifest["current_version"] == "v2.9.0"
    assert manifest["available_versions"] == ["v2.5.0", "v2.6.0", "v2.7.0", "v2.8.0", "v2.8.1", "v2.9.0"]
    assert len(manifest["sources"]) == 83
    assert {row["language"] for row in manifest["sources"]} == {"zh"}
    assert {row["locale"] for row in manifest["sources"]} == {"zh-CN"}
    for row in manifest["sources"]:
        assert row["source_url"].startswith("https://github.com/PaddlePaddle/PaddleDetection/blob/")
        assert row["commit"] == manifest["versions"][row["version"]]["commit"]
        assert hashlib.sha256((CORPUS / row["local_path"]).read_bytes()).hexdigest() == row["sha256"]


def test_chinese_and_snapshot_filters_keep_one_workspace(index):
    snapshot = "v2.9.0"
    hits = index.search("PP-Human 行人跟踪模型 推理配置", language="zh", version="latest")
    assert hits
    assert all(row["locale"] == "zh-CN" and row["version"] == snapshot for row in hits)
    assert all(row["repository"] == "PaddlePaddle/PaddleDetection" for row in hits)
    assert index.search("行人跟踪", language="zh", version=snapshot)
    with pytest.raises(ValueError, match="unsupported public corpus scope"):
        index.search("行人跟踪", language="zh", version="not-a-pinned-release")


def test_consistency_warning_only_reports_verifiable_version_text_difference():
    base = {
        "document_key": "guide/parameter/priority", "heading": "Parameter Priority",
        "locale": "zh-CN", "source_url": "https://example.invalid/source",
    }
    notes = verified_consistency_notes([
        {**base, "version": "snapshot-old", "content": "context > local"},
        {**base, "version": "snapshot-current", "content": "context > startup > local"},
    ])
    assert len(notes) == 1
    assert notes[0]["kind"] == "verified_version_text_difference"
    assert verified_consistency_notes([
        {**base, "version": "snapshot-current", "content": "same"},
        {**base, "version": "snapshot-current", "content": "different"},
    ]) == []


def test_consistency_warning_does_not_compare_different_languages_as_version_conflicts():
    base = {
        "document_key": "device-documentation/design/planning",
        "heading": "Planning Architecture", "source_url": "https://example.invalid/source",
    }

    assert verified_consistency_notes([
        {**base, "version": "docs-main", "language": "en", "content": "Plan a safe trajectory."},
        {**base, "version": "docs-main", "language": "zh", "content": "规划安全轨迹。"},
    ]) == []


def test_current_scope_is_only_the_pinned_document_snapshot(index):
    snapshot = index.manifest["current_version"]
    current = index.search("行人跟踪", version="latest", language="zh")
    exact = index.search("Jetson", version=snapshot, language="zh")
    assert current and exact
    assert {row["version"] for row in current + exact} == {snapshot}
    assert len(index.manifest["available_versions"]) == 6


def test_parts_keep_inherited_heading_path():
    parts = _parts("# API\n## Workflow\n### Recovery\nDefault: retry")
    assert parts[0] == ("Recovery", ["API", "Workflow", "Recovery"], "Default: retry")


def test_heading_match_ranks_above_body_repetition_with_bm25_fields(tmp_path):
    source_root = tmp_path / "sources" / "snapshot-current" / "zh"
    source_root.mkdir(parents=True)
    documents = [
        ("guide/missed-fire", "# Scheduling Guide\n## Missed Fire Policy\nDefault: CONTINUE"),
        ("guide/operations", "# Operations Guide\n## General Notes\n" + "missed fire policy " * 20),
    ]
    sources = []
    for key, content in documents:
        local_path = f"sources/snapshot-current/zh/{key.rsplit('/', 1)[-1]}.md"
        path = tmp_path / local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
        sources.append({
            "version": "snapshot-current", "language": "zh", "locale": "zh-CN",
            "document_key": key, "document_path": f"docs/docs/zh/{key.rsplit('/', 1)[-1]}.md",
            "local_path": local_path, "source_type": "official_documentation",
            "source_url": "https://example.invalid/docs.md",
            "repository": "example/docs", "commit": "pinned",
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    manifest = {
        "workspace": "Generic engineering docs", "repository": "example/docs",
        "baseline_version": "snapshot-previous", "current_version": "snapshot-current",
        "commits": {"snapshot-current": "pinned"}, "sources": sources,
    }
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "retrieval_policy.json").write_text(json.dumps({
        "default_policy": "bm25", "benchmark_query_count": 0,
        "benchmark_corpus_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "index_artifacts_sha256": {},
    }), encoding="utf-8")

    build_index(tmp_path)
    policy_path = tmp_path / "retrieval_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["default_policy"] == "bm25"
    assert policy["benchmark_corpus_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert policy["index_artifacts_sha256"] == {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("chunks.json", "dense_vectors.npy")
    }

    index = PublicKnowledgeIndex(tmp_path)
    baseline = index.search("missed fire policy", version="snapshot-current", language="zh", policy="bm25")
    candidate = index.search("missed fire policy", version="snapshot-current", language="zh", policy="bm25_fields")
    assert baseline[0]["document_key"] == "guide/operations"
    assert candidate[0]["document_key"] == "guide/missed-fire"
    assert candidate[0]["heading_path"] == ["Scheduling Guide", "Missed Fire Policy"]


def test_fielded_bm25_does_not_compute_dense_or_baseline_scores(index, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("bm25_fields must calculate only its selected score")

    monkeypatch.setattr(index, "_bm25", unexpected)
    monkeypatch.setattr("src.public_knowledge.dense_vector", unexpected)

    hits = index.search(
        "PP-Human 行人跟踪推理配置", version="latest", language="zh", policy="bm25_fields"
    )

    assert hits
    assert all(hit["retrieval_policy"] == "bm25_fields" for hit in hits)


def test_technical_expansion_recovers_tracker_config_for_chinese_paraphrase(index):
    query = "跟踪器参数具体放在哪个配置文件里？"
    baseline = index.search(query, top_k=5, version="v2.9.0", language="zh", policy="bm25")
    hits = index.search(
        query,
        top_k=5,
        version="v2.9.0",
        language="zh",
        policy="bm25_pphuman_term_expansion_rrf",
    )

    assert any(
        row["source_id"] == "v2-9-0-deploy-pipeline-config-tracker-config-yml-0b088dd"
        for row in hits
    )
    assert [row["chunk_id"] for row in hits[:2]] == [row["chunk_id"] for row in baseline[:2]]
    assert all(row["version"] == "v2.9.0" and row["language"] == "zh" for row in hits)


def test_source_diverse_bm25_exposes_more_relevant_documents_without_changing_default(monkeypatch):
    candidate_index = object.__new__(PublicKnowledgeIndex)
    candidate_index.manifest = {
        "current_version": "snapshot-current",
        "sources": [{"version": "snapshot-current"}, {"version": "snapshot-old"}],
    }
    candidate_index.policy = {"default_policy": "bm25"}
    candidate_index.project_status = None
    candidate_index.ready = True
    candidate_index.chunks = [
        {"chunk_id": "a:1", "document_id": "snapshot-current:zh:a", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "a:2", "document_id": "snapshot-current:zh:a", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "a:3", "document_id": "snapshot-current:zh:a", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "b:1", "document_id": "snapshot-current:zh:b", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "c:1", "document_id": "snapshot-current:zh:c", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "old-a:1", "document_id": "snapshot-old:zh:a", "version": "snapshot-old", "language": "zh"},
    ]
    monkeypatch.setattr(candidate_index, "_bm25", lambda _query: np.array([10, 9, 8, 7, 6, 5], dtype=np.float32))

    default = candidate_index.search("recovery", top_k=3, version="current", language="zh")
    diverse = candidate_index.search(
        "recovery", top_k=3, version="current", language="zh", policy="bm25_source_diverse"
    )
    top2 = candidate_index.search(
        "recovery", top_k=4, version="current", language="zh", policy="bm25_top2_diverse"
    )
    top3 = candidate_index.search(
        "recovery", top_k=5, version="current", language="zh", policy="bm25_top3_diverse"
    )
    across_versions = candidate_index.search(
        "recovery", top_k=6, version="all", language="zh", policy="bm25_source_diverse"
    )

    assert [row["chunk_id"] for row in default] == ["a:1", "a:2", "a:3"]
    assert [row["chunk_id"] for row in diverse] == ["a:1", "b:1", "c:1"]
    assert [row["retrieval_score"] for row in diverse] == [10, 7, 6]
    assert all(row["retrieval_policy"] == "bm25_source_diverse" for row in diverse)
    assert [row["chunk_id"] for row in top2] == ["a:1", "a:2", "b:1", "c:1"]
    assert [row["chunk_id"] for row in top3] == ["a:1", "a:2", "a:3", "b:1", "c:1"]
    assert [row["chunk_id"] for row in across_versions[:4]] == ["a:1", "b:1", "c:1", "old-a:1"]


def test_source_diverse_bm25_keeps_matching_siblings_before_zero_score_documents(monkeypatch):
    candidate_index = object.__new__(PublicKnowledgeIndex)
    candidate_index.manifest = {"current_version": "snapshot-current", "sources": [{"version": "snapshot-current"}]}
    candidate_index.policy = {"default_policy": "bm25"}
    candidate_index.project_status = None
    candidate_index.ready = True
    candidate_index.chunks = [
        {"chunk_id": "a:1", "document_id": "a", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "a:2", "document_id": "a", "version": "snapshot-current", "language": "zh"},
        {"chunk_id": "b:1", "document_id": "b", "version": "snapshot-current", "language": "zh"},
    ]
    monkeypatch.setattr(candidate_index, "_bm25", lambda _query: np.array([10, 9, 0], dtype=np.float32))

    hits = candidate_index.search("recovery", top_k=2, policy="bm25_source_diverse")

    assert [row["chunk_id"] for row in hits] == ["a:1", "a:2"]



def test_consistency_warning_for_explicit_same_parameter_values():
    base = {
        "document_key": "architecture/configuration", "heading": "Configuration",
        "locale": "zh-CN", "source_url": "https://example.invalid/source",
    }
    notes = verified_consistency_notes([
        {**base, "version": "snapshot-old", "content": "worker.threads=500"},
        {**base, "version": "snapshot-current", "content": "worker.threads=1000"},
    ])
    assert any(note["kind"] == "verified_literal_value_difference" and note["values"] == ["500", "1000"] for note in notes)


def test_public_index_rejects_policy_from_a_different_corpus(tmp_path):
    for name in ("corpus_manifest.json", "chunks.json", "dense_vectors.npy", "retrieval_policy.json"):
        (tmp_path / name).write_bytes((CORPUS / name).read_bytes())
    policy_path = tmp_path / "retrieval_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["benchmark_corpus_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    shutil.copytree(CORPUS / "sources", tmp_path / "sources")
    with pytest.raises(ValueError, match="does not match pinned corpus"):
        PublicKnowledgeIndex(tmp_path)


@pytest.mark.parametrize("artifact", ["chunks.json", "dense_vectors.npy"])
def test_public_index_rejects_tampered_prebuilt_artifact(tmp_path, artifact):
    for name in ("corpus_manifest.json", "chunks.json", "dense_vectors.npy", "retrieval_policy.json"):
        (tmp_path / name).write_bytes((CORPUS / name).read_bytes())
    shutil.copytree(CORPUS / "sources", tmp_path / "sources")

    artifact_path = tmp_path / artifact
    if artifact == "chunks.json":
        chunks = json.loads(artifact_path.read_text(encoding="utf-8"))
        chunks[0]["content"] = "伪造的官方资料内容"
        artifact_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    else:
        tampered = bytearray(artifact_path.read_bytes())
        tampered[-1] ^= 1
        artifact_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="public corpus index artifact hash mismatch"):
        PublicKnowledgeIndex(tmp_path)
