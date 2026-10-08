import importlib.util
import sys
from pathlib import Path

import httpx
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "public_release_smoke.py"
spec = importlib.util.spec_from_file_location("public_release_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = smoke
spec.loader.exec_module(smoke)


EXPECTED_SHA = "a" * 40
CORPUS_HASH = "b" * 64
POLICY_HASH = "c" * 64
EVALUATION_HASH = "d" * 64


@pytest.fixture(autouse=True)
def independent_manifest(tmp_path, monkeypatch):
    # Legacy smoke protocol is tested with explicit synthetic metadata, never
    # with the retired production corpus or the current PaddleOCR deployment.
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(__import__('json').dumps({
        'workspace_id': 'pphuman', 'repository': 'PaddlePaddle/PaddleDetection',
        'current_version': 'v2.9.0',
        'available_versions': ['v2.5.0', 'v2.6.0', 'v2.7.0', 'v2.8.0', 'v2.8.1', 'v2.9.0'],
        'versions': {'v2.9.0': {'commit': 'b25522a0f4bde8c80603f3ba5e3472059972e3b5'}},
    }), encoding='utf8')
    monkeypatch.setattr(smoke, '_CORPUS_MANIFEST_PATH', manifest)


def _client(*, wrong_version=False, unhealthy=False, noanswer_status="OUT_OF_SCOPE"):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "ui.example":
            return httpx.Response(200, text="Streamlit")
        if path == "/health":
            return httpx.Response(200, json={
                "alive": not unhealthy, "rag_ready": True,
                "build_revision": EXPECTED_SHA,
                "corpus_fingerprint": {"fingerprint_sha256": CORPUS_HASH},
                "retrieval_config_fingerprint": POLICY_HASH,
                "evaluation_fingerprint": EVALUATION_HASH,
            })
        if path == "/public/workspace":
            return httpx.Response(200, json={
                "build_revision": EXPECTED_SHA,
                "corpus_fingerprint": {"fingerprint_sha256": CORPUS_HASH},
                "retrieval_config_fingerprint": POLICY_HASH,
                "evaluation_fingerprint": EVALUATION_HASH,
                "workspace_id": "pphuman",
                "repository": "PaddlePaddle/PaddleDetection",
                "repositories": ["PaddlePaddle/PaddleDetection"],
                "languages": ["zh"], "current_version": "v2.9.0",
                "available_versions": ["v2.9.0", "v2.8.1", "v2.8.0", "v2.7.0", "v2.6.0", "v2.5.0"],
                "source_status": "ready",
                "public_body_indexing_enabled": True,
                "retrieval_evaluation_status": "new_corpus_pending_rebenchmark",
                "frozen_benchmark_query_count": 0,
                "snapshots": [{"version": "v2.9.0", "commit": "b25522a0f4bde8c80603f3ba5e3472059972e3b5"}],
                "version_scopes": {"latest": {"versions": ["v2.9.0"]}},
            })
        if path == "/public/search":
            body = request.read().decode("utf-8")
            request_json = __import__("json").loads(body)
            version = "v2.8.1" if wrong_version else "v2.9.0"
            return httpx.Response(200, json={"status": "OK", "results": [{
                "version": version, "language": "zh", "locale": "zh-CN",
                "repository": "PaddlePaddle/PaddleDetection", "publisher": "PaddlePaddle",
                "commit": "b25522a0f4bde8c80603f3ba5e3472059972e3b5",
            }]})
        if path == "/public/query":
            return httpx.Response(200, json={"status": noanswer_status, "evidence": []})
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_release_smoke_passes_when_ui_api_and_all_public_probes_match():
    with _client() as client:
        result = smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision=EXPECTED_SHA,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )

    assert result["status"] == "PASS"
    assert result["probes"]["tracking_configuration"]["versions"] == ["v2.9.0"]
    assert result["probes"]["behavior_configuration"]["versions"] == ["v2.9.0"]
    assert result["probes"]["no_answer_scope"]["status"] == "OUT_OF_SCOPE"
    assert "ui_http_ms" in result["remote_latency_ms"]


def test_release_smoke_rejects_a_ui_revision_mismatch_before_network_calls():
    with _client() as client, pytest.raises(smoke.ReleaseSmokeError, match="UI revision"):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision="f" * 40,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )


def test_release_smoke_rejects_unknown_revision():
    with _client() as client, pytest.raises(smoke.ReleaseSmokeError, match="40-character"):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision="unknown",
            api_url="https://api.example", expected_sha="unknown",
        )


def test_release_smoke_rejects_unhealthy_api_and_wrong_version_results():
    with _client(unhealthy=True) as client, pytest.raises(smoke.ReleaseSmokeError, match="unhealthy"):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision=EXPECTED_SHA,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )
    with _client(wrong_version=True) as client, pytest.raises(smoke.ReleaseSmokeError, match="wrong-version"):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision=EXPECTED_SHA,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )


def test_release_smoke_rejects_no_answer_query_if_it_is_not_guarded():
    with _client(noanswer_status="OK") as client, pytest.raises(smoke.ReleaseSmokeError, match="no-answer scope"):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision=EXPECTED_SHA,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )


def test_release_smoke_rejects_non_pphuman_workspace():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ui.example":
            return httpx.Response(200, text="Streamlit")
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "alive": True, "rag_ready": True, "build_revision": EXPECTED_SHA,
                "corpus_fingerprint": {"fingerprint_sha256": CORPUS_HASH},
                "retrieval_config_fingerprint": POLICY_HASH,
                "evaluation_fingerprint": EVALUATION_HASH,
            })
        if request.url.path == "/public/workspace":
            return httpx.Response(200, json={"workspace_id": "unrelated_workspace"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(
        smoke.ReleaseSmokeError, match="PP-Human Chinese corpus",
    ):
        smoke.run_smoke(
            client, ui_url="https://ui.example", ui_revision=EXPECTED_SHA,
            api_url="https://api.example", expected_sha=EXPECTED_SHA,
        )
