from __future__ import annotations

from pathlib import Path
import json

from fastapi.testclient import TestClient

from src.public_knowledge import PublicKnowledgeIndex
from src.public_server import create_app


CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"
QUESTION = "PaddleOCR 的 OCR 结果包含哪些字段？"


class StructuredGenerator:
    provider = "deepseek"
    model = "test-model"

    def __init__(
        self,
        evidence_id: str | list[str],
        *,
        relevant_sources: list[dict] | None = None,
        error: Exception | None = None,
    ):
        self.evidence_id = evidence_id
        self.evidence_ids = evidence_id if isinstance(evidence_id, list) else [evidence_id]
        self.relevant_sources = relevant_sources
        self.error = error
        self.calls = 0
        self.last_context = ""

    def generate(self, *, question: str, context: str) -> dict:
        self.calls += 1
        self.last_context = context
        if self.error:
            raise self.error
        return {
            "claims": [{"text": "请核对 OCR 结果及其对应字段。", "evidence_ids": self.evidence_ids}],
            "relevant_sources": self.relevant_sources or [],
        }

    def verify_claims_with_diagnostics(self, payload):
        # Protocol stub for citation plumbing tests, not a quality evaluation.
        return {"verdicts": [
            {"claim_index": i, "supported": True, "reason": "Unit-test verifier stub"}
            for i in range(len(payload))
        ]}, {}


def _index() -> PublicKnowledgeIndex:
    return PublicKnowledgeIndex(CORPUS)




def test_default_index_and_health_use_the_paddleocr_corpus(monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("RAG_PUBLIC_CORPUS_ROOT", "retired_corpus_should_not_load")
    monkeypatch.setenv(
        "RAG_PUBLIC_RETRIEVAL_CONFIG",
        "retired_corpus_should_not_load/public_retrieval_runtime.json",
    )
    with TestClient(create_app()) as client:
        workspace = client.get("/public/workspace").json()
        health = client.get("/health").json()

    assert workspace["workspace_id"] == "paddleocr"
    assert workspace["languages"] == ["zh"]
    assert workspace["source_count"] == len(json.loads((CORPUS.parent/'public_corpus_paddleocr/corpus_manifest.json').read_text(encoding='utf-8'))['sources'])
    assert workspace["current_version"] == "v3.0.0"
    assert workspace["rag_ready"] is True
    assert workspace['impact_evaluation'] is not None or workspace["retrieval_evaluation_status"] == "new_corpus_pending_rebenchmark"
    assert workspace.get('retrieval_evaluation') is None  # Historical v1 must remain hidden.
    assert health["rag_ready"] is True
    assert health["workspace"] == workspace["workspace"]
    assert workspace["workspace_id"] == "paddleocr"


def test_query_without_generation_returns_retrieved_evidence_without_fabricating_answer():
    index = _index()
    with TestClient(create_app(index=index)) as client:
        response = client.post("/public/query", json={"query": QUESTION, "language": "zh"})

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "GENERATION_NOT_CONFIGURED"
    assert body["answer"] == "N/A"
    assert body["evidence"]
    assert all(row["language"] == "zh" for row in body["evidence"])


def test_partial_supported_answer_preserves_specific_missing_materials():
    class PartialGenerator(StructuredGenerator):
        def generate(self, **kwargs):
            return {**super().generate(**kwargs), "evidence_gaps": ["未提供目标模型的精度和跟踪回归记录。"]}
    with TestClient(create_app(index=_index(), generator=PartialGenerator("E1"))) as client:
        result = client.post("/public/query", json={"query": QUESTION}).json()
    assert result["status"] == "OK"
    assert result["evidence_gaps"] == ["未提供目标模型的精度和跟踪回归记录。"]
    assert result["answer_completeness"] == "PARTIAL_SUPPORTED"
    assert result["sources"]


def test_model_abstention_keeps_specific_gap_without_publishing_claims():
    class GapGenerator:
        def generate(self, **kwargs):
            return {"claims": [], "relevant_sources": [], "evidence_gaps": ["当前来源未包含目标部署设备的实测结果。"]}
    with TestClient(create_app(index=_index(), generator=GapGenerator())) as client:
        result = client.post("/public/query", json={"query": QUESTION}).json()
    assert result["status"] == "ABSTAINED"
    assert result["sources"] == []
    assert result["evidence_gaps"] == ["当前来源未包含目标部署设备的实测结果。"]




def test_public_search_rejects_unrecognized_retrieval_policy():
    with TestClient(create_app(index=_index())) as client:
        response = client.post("/public/search", json={
            "query": QUESTION, "language": "zh", "retrieval_policy": "unbounded_custom_policy",
        })

    assert response.status_code == 422






def test_generation_resolves_short_evidence_aliases_and_derives_sources_from_claims():
    index = _index()
    hits = index.search(QUESTION, top_k=5, version="latest", language="zh")
    hit = hits[1]
    generator = StructuredGenerator(
        "E2",
        relevant_sources=[{"document_id": "untrusted-model-source", "page_number": 999}],
    )
    with TestClient(create_app(index=index, generator=generator)) as client:
        response = client.post("/public/query", json={
            "query": QUESTION, "top_k": 5, "version": "latest", "language": "zh",
        })

    accepted = response.json()
    assert response.status_code == 200
    assert accepted["status"] == "OK"
    assert accepted["sources"][0]["chunk_id"] == hit["chunk_id"]
    assert accepted["claims"][0]["evidence_ids"] == [hit["chunk_id"]]
    assert "E2" in generator.last_context

    invalid = StructuredGenerator(["E1", "not-in-current-retrieval"])
    with TestClient(create_app(index=index, generator=invalid)) as client:
        rejected = client.post("/public/query", json={
            "query": QUESTION, "top_k": 5, "version": "latest", "language": "zh",
        }).json()
    assert rejected["status"] == "ABSTAINED"
    assert rejected["evidence"]
    assert rejected["generation"]["failure_reason"] == "NO_VALID_EVIDENCE_CITATIONS"
    assert rejected["generation"]["claimed_citation_count"] == 2
    assert rejected["generation"]["valid_citation_count"] == 1


def test_no_positive_retrieval_match_skips_model_call_and_reports_evidence_gap():
    index = _index()
    generator = StructuredGenerator("unused")
    with TestClient(create_app(index=index, generator=generator)) as client:
        response = client.post("/public/query", json={"query": "qzxv-not-in-the-document-corpus", "language": "zh"})

    assert response.json()["status"] == "NO_EVIDENCE"
    assert response.json()["evidence"] == []
    assert response.json()["generation"]["failure_reason"] == "NO_POSITIVE_RETRIEVAL_EVIDENCE"
    assert generator.calls == 0


def test_provider_failure_keeps_retrieval_evidence_and_returns_actionable_status():
    index = _index()
    generator = StructuredGenerator("unused", error=ConnectionError("private provider detail"))
    with TestClient(create_app(index=index, generator=generator)) as client:
        response = client.post("/public/query", json={"query": QUESTION, "language": "zh"})

    body = response.json()
    assert body["status"] == "GENERATION_PROVIDER_UNAVAILABLE"
    assert body["evidence"]
    assert "private provider detail" not in str(body)


def test_public_generation_has_no_application_call_budget():
    index = _index()
    generator = StructuredGenerator("E1", relevant_sources=[])
    with TestClient(create_app(index=index, generator=generator)) as client:
        bodies = [client.post("/public/query", json={"query": QUESTION, "top_k": 1, "language": "zh"}).json()
                  for _ in range(5)]

    assert generator.calls == 5
    assert all(body["status"] == "OK" for body in bodies)


def test_private_company_request_is_stopped_before_retrieval_or_generation():
    index = _index()

    def unexpected_search(*args, **kwargs):
        raise AssertionError("private company requests must be stopped before retrieval")

    index.search = unexpected_search
    generator = StructuredGenerator("unused")
    with TestClient(create_app(index=index, generator=generator)) as client:
        response = client.post("/public/query", json={
            "query": "查询本公司的内部 API 与私有工单", "language": "zh",
        })

    assert response.json()["status"] == "OUT_OF_SCOPE"
    assert response.json()["evidence"] == []
    assert generator.calls == 0










def test_batch_search_accepts_top_k_and_applies_it_to_each_check():
    with TestClient(create_app(index=_index())) as client:
        response = client.post("/public/search-batch", json={
            "summary": QUESTION,
            "checks": [
                {"check_index": 0, "query": QUESTION},
                {"check_index": 1, "query": "OCR 配置"},
            ],
            "version": "current", "language": "zh", "top_k": 2,
        })

    assert response.status_code == 200
    assert all(len(check["results"]) <= 2 for check in response.json()["checks"])


def test_task_adaptive_policy_is_rejected_outside_pphuman():
    from types import SimpleNamespace
    import pytest
    from fastapi import HTTPException
    from src.public_api import SearchRequest, _validate_public_retrieval_policy

    index = SimpleNamespace(
        manifest={"workspace_id": "edge_ai_device"}, config={"allowed_policies": ["bm25"]},
    )
    payload = SearchRequest(query="设备配置", retrieval_policy="task_adaptive_rerank")
    with pytest.raises(HTTPException) as error:
        _validate_public_retrieval_policy(index, payload)
    assert error.value.status_code == 422


def test_api_accepts_the_same_eight_gap_contract_as_generator_decoder():
    from src.answer_generation import StructuredAnswerGenerator
    gaps=[f"Missing application evidence {i}" for i in range(6)]
    decoded=StructuredAnswerGenerator._decode({"claims":[],"relevant_sources":[],"evidence_gaps":gaps})
    class GappedGenerator(StructuredGenerator):
        def generate(self,**kwargs):
            return {**super().generate(**kwargs),"evidence_gaps":decoded['evidence_gaps']}
    with TestClient(create_app(index=_index(),generator=GappedGenerator('E1'))) as client:
        result=client.post('/public/query',json={'query':QUESTION}).json()
    assert result['status']=='OK'
    assert result['evidence_gaps']==gaps
