from src.public_reranking import (
    bound_rerank_excerpts,
    build_candidate_pool,
    route_query,
)


def test_route_query_keeps_exact_configuration_queries_on_bm25():
    assert route_query(
        "ppyoloe_crn_l_36e_pphuman.yml 的 num_classes 是多少？",
        task_type="lookup",
    ) == "DIRECT_LOOKUP"


def test_build_candidate_pool_preserves_bm25_anchor_and_deduplicates():
    bm25 = [{"chunk_id": f"b{i}", "rank": i + 1} for i in range(8)]
    expanded = [
        {"chunk_id": "b5", "rank": 1, "content": "配置含有 num_classes 参数。"},
        {"chunk_id": "x1", "rank": 2, "content": "推理配置引用模型类别数。"},
    ]
    result = build_candidate_pool(bm25, expanded, limit=6)
    assert len({row["chunk_id"] for row in result}) == len(result)
    assert {row["chunk_id"] for row in bm25[:5]} <= {row["chunk_id"] for row in result}


def test_bound_rerank_excerpts_caps_each_candidate_and_total():
    candidates = [
        {"chunk_id": f"c{i}", "content": "参数配置与部署验证。" * 200}
        for i in range(40)
    ]
    bounded = bound_rerank_excerpts(candidates)
    excerpts = [row["excerpt"] for row in bounded]
    assert all(len(text) <= 600 for text in excerpts)
    assert sum(map(len, excerpts)) <= 12_000
    assert all(text for text in excerpts)


def test_validate_ranked_ids_rejects_missing_duplicate_and_foreign_ids():
    from src.public_reranking import validate_ranked_ids

    ids = ["a", "b"]
    assert validate_ranked_ids({"ranked_ids": ["b", "a"]}, ids) == ["b", "a"]
    assert validate_ranked_ids({"ranked_ids": ["a"]}, ids) is None
    assert validate_ranked_ids({"ranked_ids": ["a", "a"]}, ids) is None
    assert validate_ranked_ids({"ranked_ids": ["a", "x"]}, ids) is None
    assert validate_ranked_ids({"ranked_ids": ["a", "b"], "scores": [0.9, 0.8]}, ids) is None


def test_select_coverage_ranked_rotates_across_checks_before_filling():
    from src.public_reranking import select_coverage_ranked

    searches = [
        ({"check_index": 0}, [{"chunk_id": "a"}, {"chunk_id": "b"}]),
        ({"check_index": 1}, [{"chunk_id": "c"}, {"chunk_id": "d"}]),
    ]
    result = select_coverage_ranked(searches, ["a", "b", "c", "d"], max_items=3)
    assert [row["chunk_id"] for row in result] == ["a", "c", "b"]


def test_route_query_uses_task_and_distinguishes_conceptual_questions():
    assert route_query("变更影响审查", task_type="change_review") == "CHANGE_REVIEW"
    assert route_query(
        "ppyoloe_crn_l_36e_pphuman.yml 的 num_classes 为什么这样配置？",
        task_type="lookup",
    ) == "CONCEPTUAL_LOOKUP"
    assert route_query("hello", task_type="lookup") == "GENERAL_LOOKUP"


def test_pphuman_alias_query_maps_chinese_terms_to_corpus_vocabulary():
    from src.public_knowledge import pphuman_alias_query

    expanded = pphuman_alias_query("行人跟踪配置")
    assert "pedestrian" in expanded
    assert "tracking" in expanded
    assert "configuration" in expanded


def test_rerank_candidate_ids_returns_only_validated_ids(monkeypatch):
    from src.answer_generation import StructuredAnswerGenerator

    calls = []

    def fake_complete(self, messages):
        calls.append(messages)
        return '{"ranked_ids":["E2","E1"]}', {
            "provider": "deepseek", "requested_model": "deepseek-v4-flash",
            "finish_reason": "stop", "usage": {"input_tokens": 20},
        }

    monkeypatch.setattr(StructuredAnswerGenerator, "_complete_with_diagnostics", fake_complete)
    generator = object.__new__(StructuredAnswerGenerator)
    ranked, diagnostics = generator.rerank_candidate_ids(
        task="排序与配置切换有关的资料",
        candidates=[
            {"chunk_id": "c1", "content": "行人检测模型配置中的类别数参数。"},
            {"chunk_id": "c2", "content": "推理配置如何关联类别映射。"},
        ],
    )
    assert ranked == ["c2", "c1"]
    assert diagnostics["status"] == "OK"
    assert diagnostics["provider"] == "deepseek"
    assert diagnostics["usage"]["input_tokens"] == 20
    assert len(calls) == 1
    assert "不可信" in calls[0][0]["content"]
    payload = calls[0][1]["content"]
    import json
    assert [row["id"] for row in json.loads(payload)["candidates"]] == ["E1", "E2"]


def test_rerank_candidate_ids_rejects_incomplete_order(monkeypatch):
    import pytest
    from src.answer_generation import GenerationResponseError, StructuredAnswerGenerator

    calls = []
    def fake_complete(self, messages):
        calls.append(messages)
        return '{"ranked_ids":["E1"]}', {"finish_reason": "stop"}

    monkeypatch.setattr(StructuredAnswerGenerator, "_complete_with_diagnostics", fake_complete)
    generator = object.__new__(StructuredAnswerGenerator)
    with pytest.raises(GenerationResponseError):
        generator.rerank_candidate_ids(
            task="排序", candidates=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        )
    assert len(calls) == 1


def test_rerank_candidate_ids_does_not_retry_provider_timeout(monkeypatch):
    import pytest
    from src.answer_generation import GenerationProviderError, StructuredAnswerGenerator

    calls = []
    def fake_complete(self, messages):
        calls.append(messages)
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(StructuredAnswerGenerator, "_complete_with_diagnostics", fake_complete)
    generator = object.__new__(StructuredAnswerGenerator)
    with pytest.raises(GenerationProviderError) as failure:
        generator.rerank_candidate_ids(task="排序", candidates=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])
    assert failure.value.code == "GENERATION_PROVIDER_TIMEOUT"
    assert len(calls) == 1
