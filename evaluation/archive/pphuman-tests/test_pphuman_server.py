from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


SERVICE = Path(__file__).resolve().parents[1]
CORPUS = SERVICE / "public_corpus_pphuman"


def test_pphuman_workspace_exposes_chinese_release_history_and_latest_scope():
    index = PublicKnowledgeIndex(CORPUS)
    with TestClient(create_app(
        index=index,
        retrieval_config_path=CORPUS / "public_retrieval_runtime.json",
    )) as client:
        workspace = client.get("/public/workspace")
        health = client.get("/health")
        documents = client.get("/public/documents")

    assert workspace.status_code == 200
    data = workspace.json()
    assert data["workspace_id"] == "pphuman"
    assert data["repository"] == "PaddlePaddle/PaddleDetection"
    assert data["repositories"] == ["PaddlePaddle/PaddleDetection"]
    assert data["current_version"] == "v2.9.0"
    assert data["available_versions"] == [
        "v2.9.0", "v2.8.1", "v2.8.0", "v2.7.0", "v2.6.0", "v2.5.0",
    ]
    assert data["languages"] == ["zh"]
    assert data["rag_ready"] is True
    assert data["source_count"] == 83
    assert data["retrieval_evaluation_status"] == "new_corpus_pending_rebenchmark"
    assert any(row["document_family"] == "tracking" for row in data["source_breakdown"])
    assert health.json()["rag_ready"] is True
    assert documents.status_code == 200
    assert len(documents.json()["documents"]) == 83
    assert all(row["repository"] == "PaddlePaddle/PaddleDetection" for row in documents.json()["documents"])
    assert any(row["title"] == "PP-Human 推理流水线配置" for row in documents.json()["documents"])
    assert all(row["publisher"] == "PaddlePaddle" for row in documents.json()["documents"])
    assert all(row["license"] == "Apache-2.0" for row in documents.json()["documents"])
    assert all(row["license_url"].endswith("/release/2.9/LICENSE") for row in documents.json()["documents"])


def test_pphuman_search_observes_release_filter_and_immutable_citations():
    index = PublicKnowledgeIndex(CORPUS)
    with TestClient(create_app(
        index=index,
        retrieval_config_path=CORPUS / "public_retrieval_runtime.json",
    )) as client:
        latest = client.post("/public/search", json={
            "query": "PP-Human 行人属性识别模型怎么开启",
            "version": "latest",
            "language": "zh",
            "top_k": 5,
        })
        historical = client.post("/public/search", json={
            "query": "PP-Human 行人属性识别模型怎么开启",
            "version": "v2.8.1",
            "language": "zh",
            "top_k": 5,
        })
        english_filter = client.post("/public/search", json={
            "query": "PP-Human 行人属性识别模型怎么开启",
            "version": "latest",
            "language": "en",
            "top_k": 5,
        })

    assert latest.status_code == 200
    assert historical.status_code == 200
    assert english_filter.status_code == 200
    current_version = [row for row in latest.json()["results"] if row["retrieval_score"] > 0]
    old_version = [row for row in historical.json()["results"] if row["retrieval_score"] > 0]
    assert current_version
    assert old_version
    assert {row["version"] for row in current_version} == {"v2.9.0"}
    assert {row["version"] for row in old_version} == {"v2.8.1"}
    for row in current_version + old_version:
        assert row["source_url"].startswith(
            "https://github.com/PaddlePaddle/PaddleDetection/blob/"
        )
        assert row["publisher"] == "PaddlePaddle"
        assert row["license"] == "Apache-2.0"
        assert row["commit"] == json.loads((CORPUS / "corpus_manifest.json").read_text(encoding="utf-8"))["versions"][row["version"]]["commit"]
    assert english_filter.json()["results"] == []
