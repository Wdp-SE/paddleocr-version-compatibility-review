"""Output budgets and short aliases preserve the evidence generation boundary."""

import json

import pytest

from src.answer_generation import GenerationResponseError, StructuredAnswerGenerator


@pytest.mark.parametrize("operation,expected_minimum,maximum", [
    ("answer", 4096, 4096), ("review", 4096, 4096), ("rerank", 2048, 2048),
])
def test_http_output_budget_matches_bounded_structured_task(monkeypatch, operation, expected_minimum, maximum):
    import urllib.request

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    captured = {}
    answer = {"claims": [{"text": "Supported", "evidence_ids": ["long-id-1"]}], "relevant_sources": []}
    review = {
        "change_interpretation": "Change", "impact_candidates": [],
        "evidence_gaps": ["Need runtime evidence"], "version_ambiguities": [],
        "reviewer_actions": ["Review"], "review_status": "REQUIRES_HUMAN_REVIEW",
    }
    result = {"answer": answer, "review": review, "rerank": {"ranked_ids": ["E2", "E1"]}}[operation]

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}).encode()

    def complete(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", complete)
    generator = StructuredAnswerGenerator(provider="deepseek", model="deepseek-v4-flash")
    if operation == "answer":
        generator.generate(question="Question", context="long-id-1")
    elif operation == "review":
        generator.generate_review(change_summary="Change", context="long-id-1")
    else:
        ranked, _ = generator.rerank_candidate_ids(task="Reorder", candidates=[{"chunk_id": "long-id-1"}, {"chunk_id": "long-id-2"}])
        assert ranked == ["long-id-2", "long-id-1"]
    assert expected_minimum <= captured["max_tokens"] <= maximum
    assert "test-only-key" not in json.dumps(captured)


def test_rerank_uses_short_aliases_and_maps_long_ids_without_loss(monkeypatch):
    candidate_ids = [f"v2-9-0-very-long-repository-source-identifier-{index}-chunk-000001" for index in range(24)]

    def complete(self, messages):
        payload = json.loads(messages[1]["content"])
        assert [row["id"] for row in payload["candidates"]] == [f"E{index}" for index in range(1, 25)]
        assert not any(candidate_id in messages[1]["content"] for candidate_id in candidate_ids)
        return json.dumps({"ranked_ids": [f"E{index}" for index in range(24, 0, -1)]}), {"finish_reason": "stop"}

    monkeypatch.setattr(StructuredAnswerGenerator, "_complete_with_diagnostics", complete)
    ranked, _ = object.__new__(StructuredAnswerGenerator).rerank_candidate_ids(
        task="Query", candidates=[{"chunk_id": value, "content": "Evidence"} for value in candidate_ids],
    )
    assert ranked == list(reversed(candidate_ids))


@pytest.mark.parametrize("output", [
    {"ranked_ids": ["E1"]}, {"ranked_ids": ["E1", "E1"]},
    {"ranked_ids": ["E1", "E3"]}, {"ranked_ids": ["source-1", "source-2"]},
    {"ranked_ids": ["E1", "E2"], "score": 1.0},
])
def test_rerank_rejects_incomplete_duplicate_foreign_or_nonalias_permutations(monkeypatch, output):
    monkeypatch.setattr(StructuredAnswerGenerator, "_complete_with_diagnostics", lambda *_args: (json.dumps(output), {"finish_reason": "stop"}))
    with pytest.raises(GenerationResponseError):
        object.__new__(StructuredAnswerGenerator).rerank_candidate_ids(
            task="Query", candidates=[{"chunk_id": "source-1"}, {"chunk_id": "source-2"}],
        )


def test_partial_answer_retains_supported_claim_and_separate_bounded_gap():
    decoded = StructuredAnswerGenerator._decode({
        "claims": [{"text": "The indexed YAML specifies tracker type.", "evidence_ids": ["cfg-1"]}],
        "relevant_sources": [], "evidence_gaps": ["No indexed runtime measurement validates behavior after the switch."],
    })
    assert decoded["final_answer"] == "The indexed YAML specifies tracker type."
    assert decoded["evidence_gaps"] == ["No indexed runtime measurement validates behavior after the switch."]


@pytest.mark.parametrize("gaps", ["free text", [""], ["x" * 1001], ["x"] * 9])
def test_answer_rejects_unbounded_or_invalid_gap_payload(gaps):
    with pytest.raises(ValueError):
        StructuredAnswerGenerator._decode({"claims": [], "relevant_sources": [], "evidence_gaps": gaps})


def test_answer_schema_repair_reuses_context_once_and_records_failure(monkeypatch):
    calls=[]
    invalid={"claims":[{"text":"fact","evidence_ids":[]}],"relevant_sources":[]}
    valid={"claims":[{"text":"fact","evidence_ids":["E1"]}],"relevant_sources":[]}
    def complete(self,messages):
        calls.append(messages)
        return json.dumps(invalid if len(calls)==1 else valid),{"finish_reason":"stop","usage":{"total_tokens":10}}
    monkeypatch.setattr(StructuredAnswerGenerator,'_complete_with_diagnostics',complete)
    result,diag=object.__new__(StructuredAnswerGenerator).generate_with_diagnostics(question='Q',context='E1: evidence')
    assert result['claims'][0]['evidence_ids']==['E1']
    assert len(calls)==2 and 'E1: evidence' in calls[1][1]['content']
    assert diag['schema_repair_attempts']==1
    assert diag['usage']['total_tokens']==20


def test_schema_repair_never_loops_or_exposes_provider_content(monkeypatch):
    calls=[]
    def complete(self,messages):
        calls.append(messages);return 'private invalid text',{'finish_reason':'stop'}
    monkeypatch.setattr(StructuredAnswerGenerator,'_complete_with_diagnostics',complete)
    with pytest.raises(GenerationResponseError) as error:
        object.__new__(StructuredAnswerGenerator).generate_with_diagnostics(question='Q',context='E1')
    assert len(calls)==2
    assert error.value.diagnostics['validation_failure']=='INVALID_JSON'
    assert 'private' not in str(error.value.diagnostics)
