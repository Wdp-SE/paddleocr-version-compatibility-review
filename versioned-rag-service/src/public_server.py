"""Public FastAPI entrypoint serving only pinned official-source knowledge."""

from __future__ import annotations

import os
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.answer_generation import (
    StructuredAnswerGenerator,
    default_generation_model,
    generation_api_key_env,
)
from src.build_identity import public_build_identity
from src.engineering_change import (
    EngineeringImpactService, EngineeringItem, TraceLink, compare_engineering_items,
)
from src.figure_sidecar_integrity import validate_reviewed_sidecar
from src.public_api import router as public_router
from src.public_knowledge import PublicKnowledgeIndex
from src.public_retrieval_runtime import PublicRetrievalRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _is_public_demo() -> bool:
    return os.environ.get("APP_ENV", "local").strip().casefold() == "public_demo"


def _configured_public_corpus_root() -> Path:
    if _is_public_demo():
        return SERVICE_ROOT / "public_corpus_paddleocr"
    configured = os.environ.get("RAG_PUBLIC_CORPUS_ROOT", "").strip()
    if not configured:
        return SERVICE_ROOT / "public_corpus_paddleocr"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (SERVICE_ROOT / path).resolve()


def _configured_public_retrieval_config() -> Path:
    if _is_public_demo():
        return _configured_public_corpus_root() / "public_retrieval_runtime.json"
    configured = os.environ.get("RAG_PUBLIC_RETRIEVAL_CONFIG", "").strip()
    if not configured:
        return _configured_public_corpus_root() / "public_retrieval_runtime.json"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (SERVICE_ROOT / path).resolve()


class DiffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old_items: list[EngineeringItem]
    new_items: list[EngineeringItem]


class ImpactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changed_item_id: str = Field(min_length=1)
    items: list[EngineeringItem]
    trace_links: list[TraceLink] = Field(default_factory=list)
    dense_item_ids: list[str] = Field(default_factory=list)
    evidence_by_item: dict[str, list[str]] = Field(default_factory=dict)


def create_app(*, index: PublicKnowledgeIndex | None = None, generator=None,
               retrieval_config_path=None, figure_sidecar_path=None,
               figure_sidecar_lock_path=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        base_index = index or PublicKnowledgeIndex(root=_configured_public_corpus_root())
        sidecar_path = Path(figure_sidecar_path or base_index.root / "figure_evidence_reviewed.json")
        lock_path = Path(figure_sidecar_lock_path or base_index.root / "figure_evidence_reviewed.lock.json")
        validate_reviewed_sidecar(
            sidecar_path=sidecar_path,
            lock_path=lock_path,
            manifest_path=base_index.root / "corpus_manifest.json",
        )
        active_retrieval_config_path = Path(
            retrieval_config_path or (
                base_index.root / "public_retrieval_runtime.json" if index is not None
                else _configured_public_retrieval_config()
            )
        ).resolve()
        app.state.public_base_knowledge_index = base_index
        app.state.public_knowledge_index = PublicRetrievalRuntime(
            base_index,
            config_path=active_retrieval_config_path,
            sidecar_path=sidecar_path,
        )
        app.state.public_build_identity = public_build_identity(
            repo_root=SERVICE_ROOT.parent,
            corpus_root=base_index.root,
            retrieval_config_path=active_retrieval_config_path,
        )
        app.state.public_generator = generator
        generation_allowed = os.environ.get(
            "RD_V2_ALLOW_EXTERNAL_GENERATION", "false"
        ).strip().casefold() in ("1", "true", "yes", "on")
        generation_provider = os.environ.get(
            "RD_V2_GENERATION_PROVIDER", "dashscope"
        ).strip().casefold()
        try:
            api_key_env = generation_api_key_env(generation_provider)
            generation_model = (
                os.environ.get("RD_V2_GENERATION_MODEL", "").strip()
                or default_generation_model(generation_provider)
            )
        except ValueError:
            api_key_env = None
            generation_model = None
        api_key_configured = bool(api_key_env and os.environ.get(api_key_env, "").strip())
        if app.state.public_generator is None and generation_allowed and api_key_configured:
            app.state.public_generator = StructuredAnswerGenerator(
                provider=generation_provider,
                model=generation_model,
        )
        if app.state.public_generator is not None:
            status = "CONFIGURED_UNVERIFIED"
            generation_provider = getattr(app.state.public_generator, "provider", None)
            generation_model = getattr(app.state.public_generator, "model", None)
        elif not generation_allowed:
            status = "DISABLED"
        elif api_key_env is None:
            status = "INVALID_PROVIDER"
        else:
            status = "API_KEY_MISSING"
        app.state.public_generation_config = {
            "status": status,
            "provider": generation_provider if status == "CONFIGURED_UNVERIFIED" or api_key_env else None,
            "model": generation_model if status == "CONFIGURED_UNVERIFIED" or api_key_env else None,
        }
        yield

    app = FastAPI(
        title="Versioned Public Engineering Knowledge",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(public_router)

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            logger.error(
                "api_request request_id=%s method=%s route=%s status=500 duration_ms=%s "
                "stage=request safe_error_code=UNHANDLED_EXCEPTION exception_type=%s",
                request_id, request.method, route, duration_ms, type(exc).__name__,
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "api_request request_id=%s method=%s route=%s status=%s duration_ms=%s stage=request",
            request_id, request.method, route, response.status_code, duration_ms,
        )
        return response

    @app.get("/health")
    def health() -> dict:
        identity = app.state.public_build_identity
        base_index = app.state.public_base_knowledge_index
        return {
            "alive": True, "rag_ready": bool(getattr(base_index, "ready", True)),
            "workspace": base_index.manifest.get("workspace", base_index.manifest.get("project_name", "未配置知识空间")),
            "workspace_id": base_index.manifest.get("project_id", base_index.manifest.get("workspace_id")),
            "source_status": base_index.manifest.get("source_status"),
            "project_status": getattr(base_index, "project_status", None),
            "retrieval_policy": app.state.public_knowledge_index.policy["default_policy"],
            "runtime_retrieval_policy": app.state.public_knowledge_index.runtime_policy,
            "approved_image_chunk_count": len(app.state.public_knowledge_index._images),
            "generation": app.state.public_generation_config,
            **identity,
        }

    @app.post("/engineering/items/diff")
    def engineering_diff(payload: DiffRequest) -> dict:
        try:
            return {"changes": [
                row.model_dump(mode="json")
                for row in compare_engineering_items(payload.old_items, payload.new_items)
            ]}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="ENGINEERING_DIFF_INVALID") from exc

    @app.post("/engineering/impacts/discover")
    def engineering_impacts(payload: ImpactRequest) -> dict:
        try:
            rows = EngineeringImpactService().discover(
                changed_item_id=payload.changed_item_id,
                items=payload.items,
                trace_links=payload.trace_links,
                dense_item_ids=payload.dense_item_ids,
                evidence_by_item=payload.evidence_by_item,
            )
            return {"impacts": [row.model_dump(mode="json") for row in rows]}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="ENGINEERING_IMPACT_INVALID") from exc

    return app


app = create_app()
