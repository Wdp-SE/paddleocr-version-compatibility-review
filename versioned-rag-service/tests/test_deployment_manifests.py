from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_deployment_manifests_use_public_profile_and_lightweight_dependencies() -> None:
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "versioned-rag-service" in render
    assert "uvicorn src.public_server:app --host 0.0.0.0 --port $PORT" in render
    assert "APP_ENV" in render and "public_demo" in render
    assert "RD_V2_ALLOW_EXTERNAL_GENERATION" in render
    assert "RD_V2_GENERATION_PROVIDER" in render and "deepseek" in render
    assert "RD_V2_GENERATION_MODEL" in render and "deepseek-v4-flash" in render
    assert "DEEPSEEK_API_KEY" in render and "sync: false" in render
    assert "public_corpus_paddleocr" in render
    assert "public_corpus_pphuman" not in render
    assert "public_corpus_paddleocr" in (ROOT / "start_prototype.ps1").read_text(encoding="utf-8")
    assert "MAX_LLM_CALLS_PER_SESSION" not in render
    requirements = (
        ROOT / "versioned-rag-service/requirements-render.txt"
    ).read_text(encoding="utf-8").casefold()
    for forbidden in ("torch", "transformers", "pytest", "notebook"):
        assert forbidden not in requirements
    ui_requirements = (ROOT / "demo-ui/requirements.txt").read_text(encoding="utf-8")
    for required in ("streamlit", "requests", "pydantic", "httpx", "python-docx"):
        assert required in ui_requirements


def test_public_smoke_is_read_only_and_example_secrets_are_placeholders() -> None:
    smoke = (
        ROOT / "versioned-rag-service/scripts/public_demo_smoke.py"
    ).read_text(encoding="utf-8")
    assert "/health" in smoke
    assert "/retrieve" in smoke
    for forbidden in ("/query", "/engineering/candidates", "activate", "publish"):
        assert forbidden not in smoke
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "RAG_API_BASE_URL=https://<render-service>" in example
    assert "DEEPSEEK_API_KEY=<set-in-platform-secret-store>" in example
    assert "MAX_LLM_CALLS_PER_SESSION=unlimited" in example
    service_example = (ROOT / "versioned-rag-service/.env.example").read_text(encoding="utf-8")
    assert "RAG_PUBLIC_CORPUS_ROOT=public_corpus_paddleocr" in service_example
