from pathlib import Path

from fastapi.testclient import TestClient

from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_pphuman"
ROOT_ID = "v2.9.0:zh:deploy/pipeline/config/infer_cfg_pphuman"


def test_trace_returns_pinned_same_version_relations_without_calling_a_model():
    with TestClient(create_app(index=PublicKnowledgeIndex(CORPUS))) as client:
        response = client.post("/public/config-trace", json={
            "document_ids": [ROOT_ID], "version": "latest",
        })
    assert response.status_code == 200
    report = response.json()
    assert report["version"] == "v2.9.0"
    assert report["relations"]
    assert all(row["version"] == "v2.9.0" for row in report["relations"])
    assert all(row["source_url"].startswith(
        "https://github.com/PaddlePaddle/PaddleDetection/blob/"
    ) for row in report["relations"])
    assert report["model_calls"] == 0
    assert report["impact_status"] == "REQUIRES_HUMAN_REVIEW"


def test_trace_does_not_resolve_a_historical_root_in_the_latest_scope():
    with TestClient(create_app(index=PublicKnowledgeIndex(CORPUS))) as client:
        response = client.post("/public/config-trace", json={
            "document_ids": [ROOT_ID.replace("v2.9.0", "v2.8.1")],
            "version": "v2.9.0",
        })
    assert response.status_code == 422
    assert response.json()["detail"] == "INVALID_CONFIG_TRACE"


def test_trace_rejects_unbounded_roots_before_querying_sources():
    with TestClient(create_app(index=PublicKnowledgeIndex(CORPUS))) as client:
        response = client.post("/public/config-trace", json={
            "document_ids": [ROOT_ID] * 9, "version": "v2.9.0",
        })
    assert response.status_code == 422
