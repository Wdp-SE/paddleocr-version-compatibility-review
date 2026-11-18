"""Thin HTTP client for the version-aware RAG public API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ServiceError(RuntimeError):
    public_message: str
    detail: str = ""
    code: str = "SERVICE_ERROR"

    def __str__(self) -> str:
        return self.public_message


def service_identity_digest(identity: dict) -> str:
    """Match the backend response identity, including code and evaluation assets."""
    keys = ('build_revision', 'corpus_fingerprint', 'retrieval_config_fingerprint',
            'service_code_fingerprint', 'review_code_fingerprint', 'evaluation_fingerprint')
    values = {key: identity.get(key) for key in keys}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class RAGClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 45.0,
        session: requests.Session | None = None,
        *,
        session_id: str | None = None,
        retry_limit: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session_id = session_id
        self.retry_limit = retry_limit
        self._service_identity_digest: str | None = None

    def bind_service_identity(self, workspace: dict) -> None:
        """Pin a task to one service build; rebinding cannot accept deployment drift."""
        digest = service_identity_digest(workspace)
        if self._service_identity_digest is not None and digest != self._service_identity_digest:
            self._identity_changed()
        self._service_identity_digest = digest

    @staticmethod
    def _identity_changed() -> None:
        raise ServiceError(
            "知识服务版本在任务期间发生变化，已停止审查，请重新发起任务。",
            "Missing or changed service identity", "SERVICE_IDENTITY_CHANGED",
        )

    def _validate_response_identity(self, response) -> None:
        if self._service_identity_digest is None:
            return
        headers = getattr(response, 'headers', {})
        if headers.get('X-Service-Identity') != self._service_identity_digest:
            self._identity_changed()

    def _request(
        self, method: str, endpoint: str, *, retry_limit: int | None = None,
        request_timeout: float | None = None, **kwargs
    ) -> dict[str, Any]:
        retries = self.retry_limit if retry_limit is None else retry_limit
        if retries < 0:
            raise ValueError("retry_limit must be nonnegative")
        if self.session_id:
            headers = dict(kwargs.pop("headers", {}))
            headers["X-Demo-Session-ID"] = self.session_id
            kwargs["headers"] = headers
        response = None
        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method, f"{self.base_url}{endpoint}",
                    timeout=self.timeout if request_timeout is None else request_timeout,
                    **kwargs,
                )
                self._validate_response_identity(response)
                response.raise_for_status()
                break
            except requests.Timeout as exc:
                if attempt < retries:
                    continue
                raise ServiceError(
                    "RAG 请求超时，公共免费演示后端可能正在启动，请稍后重试。",
                    type(exc).__name__,
                    "RAG_TIMEOUT",
                ) from exc
            except requests.ConnectionError as exc:
                if attempt < retries:
                    continue
                raise ServiceError(
                    "知识服务暂不可用。公共免费演示后端可能正在启动，请稍后重试。",
                    type(exc).__name__,
                    "RAG_UNAVAILABLE",
                ) from exc
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                if attempt < retries and isinstance(status, int) and status >= 500:
                    continue
                if status == 429 and endpoint in (
                    "/engineering/versions/query", "/public/query", "/public/review-advice"
                ):
                    raise ServiceError(
                        "在线生成额度已用完，仍可继续检索引用依据。",
                        "HTTPStatus=429", "LLM_BUDGET_EXHAUSTED",
                    ) from exc
                raise ServiceError(
                    f"RAG 服务返回 HTTP {status}。", f"HTTPStatus={status}", "RAG_HTTP_ERROR"
                ) from exc
        assert response is not None
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceError("RAG 返回了无法解析的响应。", "Malformed JSON", "RAG_MALFORMED") from exc
        if not isinstance(payload, dict):
            raise ServiceError("RAG 返回结构不符合契约。", "Expected JSON object", "RAG_SCHEMA_INVALID")
        return payload

    def documents(self) -> dict[str, Any]:
        payload = self._request("GET", "/documents")
        if not isinstance(payload.get("documents"), list):
            raise ServiceError("文档目录响应不符合契约。", "Invalid /documents schema", "RAG_SCHEMA_INVALID")
        return payload

    def versions(self, document_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/documents/{document_id}/versions")
        if payload.get("document_id") != document_id or not isinstance(payload.get("versions"), list):
            raise ServiceError("版本目录响应不符合契约。", "Invalid versions schema", "RAG_SCHEMA_INVALID")
        return payload

    def diff(self, document_id: str, from_version_id: str, to_version_id: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            f"/documents/{document_id}/diff",
            params={"from_version_id": from_version_id, "to_version_id": to_version_id},
        )
        if not {"document_id", "from_version", "to_version", "summary", "sections"}.issubset(payload):
            raise ServiceError("版本对比响应不符合契约。", "Invalid diff schema", "RAG_SCHEMA_INVALID")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def artifact_status(self) -> dict[str, Any]:
        payload = self._request("GET", "/artifacts/status")
        required = {"artifact_status", "retrieval_policy", "dense_representation"}
        if not required.issubset(payload):
            raise ServiceError("Artifact 状态响应不完整。", "Missing required fields", "RAG_SCHEMA_INVALID")
        return payload

    def query(self, question: str, scope: dict[str, Any] | None = None) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("问题不能为空")
        request_payload: dict[str, Any] = {"question": question.strip()}
        if scope is not None:
            request_payload["scope"] = scope
        payload = self._request("POST", "/query", json=request_payload)
        if not {"answer", "sources", "status"}.issubset(payload) or not isinstance(payload["sources"], list):
            raise ServiceError("RAG 问答响应不符合契约。", "Invalid /query schema", "RAG_SCHEMA_INVALID")
        return payload

    def search_candidate_versions(
        self, question: str, top_k: int, scope: dict[str, Any]
    ) -> dict[str, Any]:
        question = question.strip()
        if not question or not 3 <= top_k <= 10:
            raise ValueError("问题不能为空，返回数量必须为 3 到 10")
        payload = self._request(
            "POST", "/engineering/versions/search",
            json={"query": question, "top_k": top_k, "scope": scope},
        )
        results = payload.get("results")
        required = {
            "rank", "document_id", "version_id", "version_label", "version_status",
            "project_id", "section_id", "section_path", "page_number", "text",
            "similarity", "chunk_id",
        }
        if not isinstance(results, list) or len(results) > top_k or any(
            not isinstance(row, dict) or not required.issubset(row) for row in results
        ):
            raise ServiceError("知识检索结果不完整。", "Invalid candidate search response", "RAG_SCHEMA_INVALID")
        return payload

    def query_candidate_versions(
        self, question: str, scope: dict[str, Any]
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        payload = self._request(
            "POST", "/engineering/versions/query",
            json={"question": question, "scope": scope},
        )
        if not {"answer", "sources", "status"}.issubset(payload) or not isinstance(
            payload["sources"], list
        ):
            raise ServiceError("知识问答结果不完整。", "Invalid candidate query response", "RAG_SCHEMA_INVALID")
        return payload

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question or not 3 <= top_k <= 10:
            raise ValueError("问题不能为空，Top K 必须为 3 到 10")
        request_payload: dict[str, Any] = {"query": question, "top_k": top_k}
        if scope is not None:
            request_payload["scope"] = scope
        payload = self._request("POST", "/retrieve", json=request_payload)
        results = payload.get("results")
        if payload.get("query") != question or not isinstance(results, list) or len(results) > top_k:
            raise ServiceError("引用依据检索响应不符合契约。", "Invalid /retrieve envelope", "RAG_SCHEMA_INVALID")
        required = {"rank", "document_id", "page_number", "section_path", "similarity", "content", "chunk_id"}
        if any(not isinstance(item, dict) or not required.issubset(item) for item in results):
            raise ServiceError("引用依据结果字段不完整。", "Invalid retrieval hit", "RAG_SCHEMA_INVALID")
        return payload
