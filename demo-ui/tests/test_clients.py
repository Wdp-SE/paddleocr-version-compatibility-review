from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import requests

from config import DemoConfig
from services.agent_client import AgentClient
from services.rag_client import RAGClient, ServiceError
from services.public_knowledge_client import PublicKnowledgeClient


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError("failed")
            error.response = self
            raise error

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_rag_health_and_retrieve_contract():
    hit = {"rank": 1, "document_id": "safe", "page_number": 2,
           "section_path": ["项目", "目标"], "similarity": 0.82,
           "content": "项目目标：形成草稿。", "chunk_id": "safe:2"}
    session = Session([Response({"status": "healthy"}),
                       Response({"query": "项目目标", "results": [hit]})])
    client = RAGClient("http://localhost:8765", session=session)
    assert client.health()["status"] == "healthy"
    assert client.retrieve("项目目标", 3)["results"][0]["page_number"] == 2
    assert session.calls[1][3]["json"] == {"query": "项目目标", "top_k": 3}


def test_candidate_knowledge_calls_keep_selected_case_scope_and_session():
    hit = {
        "rank": 1, "document_id": "requirements", "version_id": "requirements-v2",
        "version_label": "V2.0", "version_status": "ACTIVE", "project_id": "PAYMENT",
        "document_type": "REQUIREMENT", "section_id": "REQ-023",
        "section_path": ["容量"], "page_number": 1, "chunk_id": "req-23",
        "text": "REQ-023 最大并发为 1000。", "similarity": 0.8,
    }
    session = Session([
        Response({"results": [hit]}),
        Response({"answer": "最大并发为 1000。", "status": "OK", "sources": [hit]}),
    ])
    client = RAGClient("http://localhost:8765", session=session, session_id="session_12345678")
    scope = {"project_ids": ["PAYMENT"], "active_only": True}

    assert client.search_candidate_versions("最大并发量是多少？", 5, scope)["results"][0]["document_id"] == "requirements"
    assert client.query_candidate_versions("最大并发量是多少？", scope)["status"] == "OK"
    assert [call[1] for call in session.calls] == [
        "http://localhost:8765/engineering/versions/search",
        "http://localhost:8765/engineering/versions/query",
    ]
    assert session.calls[0][3]["json"] == {"query": "最大并发量是多少？", "top_k": 5, "scope": scope}
    assert session.calls[1][3]["json"] == {"question": "最大并发量是多少？", "scope": scope}
    assert all(call[3]["headers"]["X-Demo-Session-ID"] == "session_12345678" for call in session.calls)


def test_public_query_budget_error_is_actionable():
    client = RAGClient("http://localhost:8765", session=Session([Response({}, status=429)]))
    with pytest.raises(ServiceError, match="仍可继续检索引用依据") as failure:
        client.query_candidate_versions("最大并发量是多少？", {"active_only": True})
    assert failure.value.code == "LLM_BUDGET_EXHAUSTED"


def test_rag_unavailable_and_malformed_handling():
    client = RAGClient("http://localhost:8765", session=Session([
        requests.ConnectionError("refused")
    ]))
    with pytest.raises(ServiceError) as failure:
        client.health()
    assert failure.value.code == "RAG_UNAVAILABLE"
    malformed = RAGClient("http://localhost:8765", session=Session([Response(ValueError())]))
    with pytest.raises(ServiceError, match="无法解析"):
        malformed.health()


def test_agent_safe_demo_invalid_docx_and_artifacts(tmp_path):
    config = DemoConfig(runtime_root=tmp_path).validate()
    client = AgentClient(config)
    assert client.status()["ready"] is True
    with pytest.raises(ValueError, match="有效"):
        client.save_upload("broken.docx", b"not-a-docx")
    record = client.load_demo_template()
    assert record["summary"]["section_count"] > 0
    assert record["summary"]["tasks"]
    result = client.run_workflow(record, use_demo_rag=True)
    assert result["workflow_status"] == "REVIEW_REQUIRED"
    assert result["requires_human_review"] is True
    assert result["missing_field_count"] > 0
    draft = Path(result["artifacts"]["draft"])
    assert zipfile.is_zipfile(draft)
    for key in ("evidence", "trace"):
        payload = json.loads(client.artifact_bytes(result["artifacts"][key]))
        assert isinstance(payload, dict)

def test_public_generation_timeout_is_never_retried():
    session = Session([requests.Timeout("late answer"), Response({"status": "OK"})])
    client = PublicKnowledgeClient(
        "http://localhost:8765", session=session, retry_limit=2, session_id="session_12345678",
    )
    with pytest.raises(ServiceError) as failure:
        client.query_official("parameter priority", version="wiki-1eadc6584f96", language="all")
    assert failure.value.code == "RAG_TIMEOUT"
    assert len(session.calls) == 1


def test_public_generation_forwards_selected_top_k():
    session = Session([Response({"status": "GENERATION_NOT_CONFIGURED"})])
    client = PublicKnowledgeClient("http://localhost:8765", session=session)

    client.query_official(
        "What is the API server health-check endpoint?",
        version="latest", language="all", top_k=8,
    )

    assert session.calls[0][3]["json"] == {
        "query": "What is the API server health-check endpoint?",
        "version": "latest",
        "language": "all",
        "top_k": 8,
    }


def test_public_knowledge_client_forwards_edge_device_scope_facets():
    session = Session([
        Response({"results": []}),
        Response({"evidence": []}),
        Response({"status": "GENERATION_NOT_CONFIGURED"}),
        Response({"status": "GENERATION_NOT_CONFIGURED"}),
    ])
    client = PublicKnowledgeClient("http://localhost:8765", session=session)
    scope = {
        "device_model": "reComputer Industrial J4012",
        "module_sku": "P3767-0000",
        "carrier_board": "recomputer-industrial-orin-j201",
        "software_baseline": "JetPack 7.2 (L4T 39.2.0)",
    }

    client.search("工业视觉", version="wiki-abc", language="zh", **scope)
    client.query_official("工业视觉", version="wiki-abc", language="zh", **scope)
    client.review_advice("升级视觉运行环境", ["evidence-1"], version="wiki-abc", **scope)
    client.review_advice_for_version("复核刷写影响", ["evidence-2"], version="wiki-abc", **scope)

    assert session.calls[0][3]["json"] == {
        "query": "工业视觉", "version": "wiki-abc", "language": "zh", "top_k": 5, **scope,
    }
    assert session.calls[1][3]["json"] == {
        "query": "工业视觉", "version": "wiki-abc", "language": "zh", "top_k": 5, **scope,
    }
    assert session.calls[2][3]["json"] == {
        "change_summary": "升级视觉运行环境", "evidence_chunk_ids": ["evidence-1"], "version": "wiki-abc", **scope,
    }
    assert session.calls[3][3]["json"] == {
        "change_summary": "复核刷写影响", "evidence_chunk_ids": ["evidence-2"], "version": "wiki-abc", **scope,
    }


def test_review_advice_posts_selected_evidence_once_without_retry():
    session = Session([Response({"status": "OK", "answer": "advice", "sources": []})])
    client = PublicKnowledgeClient(
        "http://localhost:8769", session=session, retry_limit=3, session_id="session_12345678",
    )

    result = client.review_advice("change summary", ["chunk-a", "chunk-b"])

    assert result["status"] == "OK"
    assert len(session.calls) == 1
    method, url, _, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "http://localhost:8769/public/review-advice"
    assert session.calls[0][2] == 60.0
    assert kwargs["json"] == {
        "change_summary": "change summary",
        "evidence_chunk_ids": ["chunk-a", "chunk-b"],
    }


def test_public_generation_budget_error_keeps_retrieval_available():
    session = Session([Response({}, status=429)])
    client = PublicKnowledgeClient("http://localhost:8765", session=session)
    with pytest.raises(ServiceError) as failure:
        client.query_official("parameter priority", version="wiki-1eadc6584f96", language="all")
    assert failure.value.code == "LLM_BUDGET_EXHAUSTED"
    assert "\u68c0\u7d22" in failure.value.public_message


def test_public_search_batch_posts_once_with_120_second_timeout_and_scope():
    session = Session([Response({"checks": [], "results": []})])
    client = PublicKnowledgeClient(
        "http://localhost:8765", session=session, retry_limit=3, timeout=17,
    )
    scope = {
        "device_model": "reComputer Industrial J4012",
        "module_sku": "P3767-0000",
    }

    client.search_batch(
        "切换行人跟踪模型", checks=[
            {"check_index": 0, "query": "推理配置"},
            {"check_index": 1, "query": "跟踪参数"},
        ], version="v2.9.0", language="zh", top_k=5,
        retrieval_policy="task_adaptive_rerank", **scope,
    )

    assert len(session.calls) == 1
    method, url, timeout, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "http://localhost:8765/public/search-batch"
    assert timeout == 120.0
    assert kwargs["json"] == {
        "summary": "切换行人跟踪模型",
        "checks": [
            {"check_index": 0, "query": "推理配置"},
            {"check_index": 1, "query": "跟踪参数"},
        ],
        "version": "v2.9.0", "language": "zh", "top_k": 5,
        "retrieval_policy": "task_adaptive_rerank", **scope,
    }


def test_public_query_uses_120_second_timeout_without_retry():
    session = Session([Response({"status": "GENERATION_NOT_CONFIGURED"})])
    client = PublicKnowledgeClient(
        "http://localhost:8765", session=session, retry_limit=3, timeout=17,
    )

    client.query_official("问题", version="v2.9.0")

    assert len(session.calls) == 1
    assert session.calls[0][2] == 120.0


def test_config_trace_is_read_only_bounded_and_keeps_selected_version():
    session = Session([Response({"status": "OK", "relations": []})])
    client = PublicKnowledgeClient("http://localhost:8765", session=session)
    client.config_trace(["v2.8.1:zh:deploy/pipeline/config/infer_cfg_pphuman"], version="v2.8.1")
    assert len(session.calls) == 1
    assert session.calls[0][1].endswith("/public/config-trace")
    assert session.calls[0][3]["json"] == {
        "document_ids": ["v2.8.1:zh:deploy/pipeline/config/infer_cfg_pphuman"],
        "version": "v2.8.1",
    }
