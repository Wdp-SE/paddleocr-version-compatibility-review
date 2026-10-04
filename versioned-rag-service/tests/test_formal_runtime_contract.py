from __future__ import annotations

import pytest

from main import build_parser
from src.answer_generation import (
    GenerationProviderError,
    GenerationResponseError,
    StructuredAnswerGenerator,
    default_generation_model,
    generation_api_key_env,
    validate_review_evidence_membership,
)
from src.context_expansion import SectionContextExpander
from src.rd_v2_runtime import (
    FINAL_DENSE_REPRESENTATION,
    FINAL_RETRIEVAL_POLICY,
    validate_citation_membership,
)
from src.trusted_qa import (
    TrustedQAMode,
    VersionResolutionStatus,
    build_answer_evidence_audit,
    collect_retrieval_signals,
    decide_post_answer_enforcement,
)


def test_formal_retrieval_policy_is_fixed() -> None:
    assert FINAL_RETRIEVAL_POLICY == "DENSE_ONLY"
    assert FINAL_DENSE_REPRESENTATION == "SECTION_PATH"


def test_structured_answer_schema_is_strict() -> None:
    value = StructuredAnswerGenerator._decode(
        '{"claims":[{"text":"应执行降级流程","evidence_ids":["chunk-1"]}],"relevant_sources":'
        '[{"document_id":"DESIGN-001","page_number":3}]}'
    )
    assert value["claims"] == [{"text": "应执行降级流程", "evidence_ids": ["chunk-1"]}]
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode(
            {
                "claims": [{"text": "未经约束的答案", "evidence_ids": []}],
                "relevant_sources": [],
            }
        )


def test_answer_schema_requires_each_claim_to_cite_a_non_empty_chunk_id() -> None:
    for claim in (
        {"text": "未引用的结论", "evidence_ids": []},
        {"text": "空 ID", "evidence_ids": [""]},
        {"text": "错误类型", "evidence_ids": [123]},
    ):
        with pytest.raises(ValueError):
            StructuredAnswerGenerator._decode({"claims": [claim], "relevant_sources": []})


def test_legacy_answer_schema_remains_compatible_without_gap_field() -> None:
    result = StructuredAnswerGenerator._decode({
        "claims": [{"text": "直接事实。", "evidence_ids": ["chunk-1"]}],
        "relevant_sources": [],
    })
    assert result["claims"] == [{"text": "直接事实。", "evidence_ids": ["chunk-1"]}]
    assert result["final_answer"] == "直接事实。"
    assert result["evidence_gaps"] == []


def test_structured_review_schema_requires_human_review_and_grounded_checks() -> None:
    value = StructuredAnswerGenerator._decode_review({
        "change_interpretation": "将启动参数的优先级提升。",
        "impact_candidates": [{
            "evidence_chunk_id": "chunk-1",
            "reason": "该章节定义参数优先级。",
            "suggested_action": "核对示例和运维说明是否同步。",
        }],
        "evidence_gaps": ["尚未检查对应英文说明。"],
        "version_ambiguities": [],
        "reviewer_actions": ["逐版本核对受影响说明。"],
        "review_status": "REQUIRES_HUMAN_REVIEW",
    })

    assert value["impact_candidates"][0]["evidence_chunk_id"] == "chunk-1"
    assert value["review_status"] == "REQUIRES_HUMAN_REVIEW"
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode_review({
            **value,
            "review_status": "APPROVED",
        })
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode_review({**value, "unexpected": True})
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode_review({key: item for key, item in value.items() if key != "reviewer_actions"})
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode_review('{"impact_candidates":')


def test_review_evidence_membership_rejects_model_invented_chunk_ids() -> None:
    review = {
        "impact_candidates": [{
            "evidence_chunk_id": "chunk-outside-request",
            "reason": "理由", "suggested_action": "核对",
        }]
    }

    with pytest.raises(ValueError, match="evidence"):
        validate_review_evidence_membership(review, {"chunk-1", "chunk-2"})


def test_deepseek_generator_uses_its_key_and_non_thinking_json_mode(monkeypatch) -> None:
    import json
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "claims": [{"text": "由检索证据支持的回答。", "evidence_ids": ["chunk-1"]}],
                            "relevant_sources": [
                                {"document_id": "chunk-1", "page_number": 2}
                            ],
                        }, ensure_ascii=False)
                    }
                }]
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization_present"] = bool(
            request.get_header("Authorization")
        )
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    generator = StructuredAnswerGenerator(
        provider="deepseek", model="deepseek-v4-flash"
    )
    result = generator.generate(question="问题", context="chunk-1: 内容")

    assert result["claims"] == [{"text": "由检索证据支持的回答。", "evidence_ids": ["chunk-1"]}]
    assert result["final_answer"] == "由检索证据支持的回答。"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization_present"] is True
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "默认 1 到 3 条" in captured["body"]["messages"][0]["content"]
    assert captured["body"]["max_tokens"] >= 256
    assert "test-secret-value" not in json.dumps(captured["body"])
    assert captured["timeout"] > 0


def test_deepseek_review_uses_structured_review_prompt_and_never_approves(monkeypatch) -> None:
    import json
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            review = {
                "change_interpretation": "提高参数优先级。",
                "impact_candidates": [{
                    "evidence_chunk_id": "chunk-1", "reason": "该段定义参数顺序。",
                    "suggested_action": "检查相关示例。",
                }],
                "evidence_gaps": [], "version_ambiguities": [],
                "reviewer_actions": ["人工确认。"],
                "review_status": "REQUIRES_HUMAN_REVIEW",
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(review, ensure_ascii=False)}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization_present"] = bool(request.get_header("Authorization"))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")
    result = generator.generate_review(change_summary="假设调整参数", context="chunk-1 | 官方片段")

    assert result["impact_candidates"][0]["evidence_chunk_id"] == "chunk-1"
    assert result["review_status"] == "REQUIRES_HUMAN_REVIEW"
    assert "研发资料变更审查助手" in captured["body"]["messages"][0]["content"]
    assert "假设调整参数" in captured["body"]["messages"][1]["content"]
    assert captured["authorization_present"] is True
    assert "test-secret-value" not in json.dumps(captured["body"])


def test_deepseek_network_failure_is_identified_without_exposing_request_details(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")

    def unreachable(_request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("proxy unavailable"))

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    generator = StructuredAnswerGenerator(
        provider="deepseek", model="deepseek-v4-flash"
    )

    with pytest.raises(ConnectionError, match="generation provider is unreachable") as exc:
        generator.generate(question="问题", context="chunk-1: 内容")
    assert "test-secret-value" not in str(exc.value)


def test_deepseek_diagnostics_report_returned_model_usage_and_finish_reason(monkeypatch) -> None:
    import json
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            answer = {
                "claims": [{"text": "证据支持的回答。", "evidence_ids": ["chunk-1"]}],
                "relevant_sources": [{"document_id": "chunk-1", "page_number": 1}],
            }
            return json.dumps({
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
            }).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")

    answer, diagnostics = generator.generate_with_diagnostics(question="问题", context="证据")

    assert answer["final_answer"] == "证据支持的回答。"
    assert diagnostics == {
        "provider": "deepseek", "requested_model": "deepseek-v4-flash",
        "returned_model": "deepseek-v4-flash", "finish_reason": "stop",
        "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
    }
    assert "test-secret-value" not in str(diagnostics)


@pytest.mark.parametrize("http_status, expected_code", [
    (401, "GENERATION_AUTH_FAILED"),
    (402, "GENERATION_BILLING_REQUIRED"),
    (429, "GENERATION_RATE_LIMITED"),
    (503, "GENERATION_PROVIDER_UNAVAILABLE"),
])
def test_deepseek_http_failures_have_safe_distinct_codes(monkeypatch, http_status, expected_code) -> None:
    import urllib.error
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")
    calls = []

    def rejected(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, http_status, "secret provider body", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", rejected)
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")

    with pytest.raises(GenerationProviderError) as exc:
        generator.generate(question="问题", context="证据")

    assert exc.value.code == expected_code
    assert "secret" not in str(exc.value)
    assert len(calls) == 1  # Never silently retry a potentially billable POST.


def test_deepseek_url_timeout_is_not_labeled_as_generic_network_failure(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")

    def timed_out(_request, timeout):
        raise urllib.error.URLError(TimeoutError("private proxy host"))

    monkeypatch.setattr(urllib.request, "urlopen", timed_out)
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")

    with pytest.raises(TimeoutError, match="timed out") as exc:
        generator.generate(question="问题", context="证据")
    assert "private proxy host" not in str(exc.value)


def test_dashscope_diagnostics_and_rate_limit_use_safe_response_fields(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-secret-value")
    answer = '{"claims":[{"text":"supported","evidence_ids":["chunk-1"]}],"relevant_sources":[]}'
    response = SimpleNamespace(
        status_code=200,
        model="qwen-turbo",
        usage={"input_tokens": 41, "output_tokens": 8, "total_tokens": 49},
        output={"choices": [{"finish_reason": "stop", "message": {"content": answer}}]},
    )
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(
        Generation=SimpleNamespace(call=lambda **_kwargs: response)
    ))
    generator = StructuredAnswerGenerator(provider="dashscope", model="qwen-turbo")

    generated, diagnostics = generator.generate_with_diagnostics(question="question", context="evidence")
    assert generated["final_answer"] == "supported"
    assert diagnostics["usage"] == {"input_tokens": 41, "output_tokens": 8, "total_tokens": 49}
    assert diagnostics["returned_model"] == "qwen-turbo"
    assert diagnostics["finish_reason"] == "stop"

    response.status_code = 429
    response.message = "secret provider body"
    with pytest.raises(GenerationProviderError) as exc:
        generator.generate(question="question", context="evidence")
    assert exc.value.code == "GENERATION_RATE_LIMITED"
    assert "secret" not in str(exc.value)


def test_truncated_json_reports_finish_reason_without_response_body(monkeypatch) -> None:
    import json
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-value")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 70, "completion_tokens": 1024, "total_tokens": 1094},
                "choices": [{"finish_reason": "length", "message": {"content": "{private incomplete"}}],
            }).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")

    with pytest.raises(GenerationResponseError) as exc:
        generator.generate_with_diagnostics(question="question", context="evidence")

    assert exc.value.code == "GENERATION_RESPONSE_TRUNCATED"
    assert exc.value.diagnostics["finish_reason"] == "length"
    assert exc.value.diagnostics["usage"]["total_tokens"] == 1094
    assert "private incomplete" not in str(exc.value)


def test_generation_provider_configuration_selects_matching_secret_and_model() -> None:
    assert generation_api_key_env("deepseek") == "DEEPSEEK_API_KEY"
    assert default_generation_model("deepseek") == "deepseek-v4-flash"
    assert generation_api_key_env("dashscope") == "DASHSCOPE_API_KEY"
    assert default_generation_model("dashscope") == "qwen-turbo"


def test_context_expansion_stays_inside_document_version_section() -> None:
    chunks = [
        {
            "chunk_id": "v1-1",
            "document_id": "DOC-1",
            "version_id": "V1",
            "section_id": "S1",
            "page_number": 1,
            "text": "章节开头",
            "position": 0,
        },
        {
            "chunk_id": "v1-2",
            "document_id": "DOC-1",
            "version_id": "V1",
            "section_id": "S1",
            "page_number": 2,
            "text": "章节正文",
            "position": 1,
        },
        {
            "chunk_id": "v2-1",
            "document_id": "DOC-1",
            "version_id": "V2",
            "section_id": "S1",
            "page_number": 1,
            "text": "新版本内容",
            "position": 0,
        },
    ]
    expander = SectionContextExpander.from_chunks(chunks, neighbor_children=1)
    output = expander.expand(
        [
            {
                **chunks[1],
                "distance": 0.82,
                "dense_score": 0.82,
                "retrieval_rank": 1,
                "bm25_score": 99,
                "rrf_score": 88,
                "relevance_score": 77,
            }
        ]
    )
    assert {item["chunk_id"] for item in output} == {"v1-1", "v1-2"}
    assert all(item["retrieval_sources"] == ["dense"] for item in output)
    assert all(
        not ({"bm25_score", "rrf_score", "relevance_score"} & set(item))
        for item in output
    )


def test_dense_signals_have_formal_version_governance() -> None:
    snapshot = collect_retrieval_signals(
        question_id="q-1",
        retrieval_results=[
            {
                "document_id": "DOC-1",
                "page_number": 2,
                "dense_score": 0.81,
                "relevance_score": 100,
            }
        ],
        version_governance_enabled=True,
        version_resolution_status=VersionResolutionStatus.RESOLVED,
        eligible_document_count=1,
    )
    assert snapshot.top1_score == 0.81
    assert snapshot.version_resolution_status == VersionResolutionStatus.RESOLVED
    assert "rerank_score" not in snapshot.model_fields


def test_forged_citation_is_removed_and_answer_fails_closed() -> None:
    evidence = [{"document_id": "DOC-1", "page_number": 2}]
    claimed = [{"document_id": "DOC-OTHER", "page_number": 99}]
    assert validate_citation_membership(claimed, evidence) == []
    audit = build_answer_evidence_audit(
        generation_performed=True,
        structured_output_valid=True,
        final_answer="一个没有有效引用的回答",
        claimed_citations=claimed,
        validated_citations=[],
        citation_membership_checked=True,
    )
    decision = decide_post_answer_enforcement(audit, mode=TrustedQAMode.ENFORCE)
    assert decision.enforced is True


def test_cli_exposes_only_formal_business_commands() -> None:
    parser = build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert commands == {
        "serve",
        "validate-artifacts",
        "ingest-version",
        "activate-version",
        "diff",
        "catalog",
    }
