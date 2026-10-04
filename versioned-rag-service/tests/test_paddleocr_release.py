"""Public release contract; legacy fixtures must not become the active workspace."""
from pathlib import Path

from fastapi.testclient import TestClient

from src.public_server import create_app


def test_public_default_is_chinese_paddleocr_and_ignores_retired_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_PUBLIC_CORPUS_ROOT", "public_corpus_pphuman")
    monkeypatch.setenv("RD_V2_ALLOW_EXTERNAL_GENERATION", "false")
    with TestClient(create_app()) as client:
        workspace = client.get("/public/workspace").json()
        health = client.get("/health").json()
        assert workspace["workspace_id"] == "paddleocr"
        assert workspace["repository"] == "PaddlePaddle/PaddleOCR"
        assert workspace["languages"] == ["zh"]
        assert workspace["current_version"] == "v3.0.0"
        assert set(workspace["available_versions"]) == {"v2.9.1", "v3.0.0"}
        assert "已收录" in workspace["version_labels"]["v3.0.0"]
        assert workspace["compatibility_review"]["runtime_verified"] is False
        assert health["rag_ready"] is True


def test_public_lookup_filters_exact_version_and_does_not_mix_domains(monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RD_V2_ALLOW_EXTERNAL_GENERATION", "false")
    with TestClient(create_app()) as client:
        response = client.post("/public/search", json={
            "query": "PPStructure 表格识别返回 res html", "version": "v2.9.1",
            "language": "zh", "top_k": 5,
        })
        assert response.status_code == 200
        rows = response.json()["results"]
        assert rows
        assert all(row["version"] == "v2.9.1" for row in rows)
        assert all(row["repository"] == "PaddlePaddle/PaddleOCR" for row in rows)
        assert client.post("/public/search", json={
            "query": "PPStructure", "language": "en",
        }).status_code == 422


def test_compatibility_review_checks_application_without_executing_it(monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RD_V2_ALLOW_EXTERNAL_GENERATION", "false")
    with TestClient(create_app()) as client:
        response = client.post("/public/compatibility-review", json={
            "source_version": "v2.9.1", "target_version": "v3.0.0",
            "files": [{"path": "app.py", "content":
                       "from paddleocr import PPStructure\nengine = PPStructure()\n"}],
        })
        assert response.status_code == 200
        report = response.json()
        assert report["workspace_id"] == "paddleocr"
        assert report["source_version"] == "v2.9.1"
        assert report["target_version"] == "v3.0.0"
        assert report["runtime_verified"] is False
        assert report["findings"]
        assert report["model_calls"] == 0
        traversal = client.post("/public/compatibility-review", json={
            "source_version": "v2.9.1", "target_version": "v3.0.0",
            "files": [{"path": "../secrets.py", "content": "print('never executed')"}],
        })
        assert traversal.status_code == 422
