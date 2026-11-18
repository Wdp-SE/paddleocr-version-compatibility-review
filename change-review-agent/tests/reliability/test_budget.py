from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.reliability.budget import (
    BudgetAccountingError,
    BudgetDimension,
    BudgetExceededError,
    BudgetLedger,
    BudgetReservationStatus,
    ExecutionBudget,
)
from app.reliability.errors import ErrorCode
from app.reliability.trace import (
    TraceContext,
    TraceEvent,
    TraceEventType,
    TraceRecorder,
    TraceSpanKind,
)


class FakeClock:
    def __init__(self) -> None:
        self.elapsed = 100.0
        self.current = datetime(2026, 8, 29, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds
        self.current += timedelta(seconds=seconds)


def read_events(recorder: TraceRecorder, run_id: str) -> list[TraceEvent]:
    return [
        TraceEvent.model_validate_json(line)
        for line in recorder.path_for(run_id).read_text(encoding="utf-8").splitlines()
    ]


def test_reserve_commit_and_remaining_are_atomic() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=2))
    reservation = ledger.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="tool.call", attempt=1
    )
    reserved = ledger.snapshot()
    assert reservation.status is BudgetReservationStatus.RESERVED
    assert reserved.committed[BudgetDimension.TOOL_CALLS] == 0
    assert reserved.reserved[BudgetDimension.TOOL_CALLS] == 1
    assert reserved.remaining[BudgetDimension.TOOL_CALLS] == 1

    committed = ledger.commit(reservation)
    snapshot = ledger.snapshot()
    assert committed.status is BudgetReservationStatus.COMMITTED
    assert snapshot.committed[BudgetDimension.TOOL_CALLS] == 1
    assert snapshot.reserved[BudgetDimension.TOOL_CALLS] == 0
    assert snapshot.remaining[BudgetDimension.TOOL_CALLS] == 1


def test_release_restores_remaining_without_committing() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_downloads=1))
    reservation = ledger.reserve(
        {BudgetDimension.DOWNLOADS: 1}, operation="download", attempt=1
    )
    released = ledger.release(reservation)
    snapshot = ledger.snapshot()
    assert released.status is BudgetReservationStatus.RELEASED
    assert snapshot.committed[BudgetDimension.DOWNLOADS] == 0
    assert snapshot.reserved[BudgetDimension.DOWNLOADS] == 0
    assert snapshot.remaining[BudgetDimension.DOWNLOADS] == 1


@pytest.mark.parametrize("terminal", ["commit", "release"])
def test_double_terminal_transition_is_rejected(terminal: str) -> None:
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=1))
    reservation = ledger.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="tool.call", attempt=1
    )
    getattr(ledger, terminal)(reservation)
    with pytest.raises(BudgetAccountingError, match="cannot"):
        getattr(ledger, terminal)(reservation)


def test_commit_after_release_and_release_after_commit_are_rejected() -> None:
    first = BudgetLedger(ExecutionBudget(max_tool_calls=1))
    released = first.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="tool.call", attempt=1
    )
    first.release(released)
    with pytest.raises(BudgetAccountingError, match="cannot commit"):
        first.commit(released)

    second = BudgetLedger(ExecutionBudget(max_tool_calls=1))
    committed = second.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="tool.call", attempt=1
    )
    second.commit(committed)
    with pytest.raises(BudgetAccountingError, match="cannot release"):
        second.release(committed)


def test_multidimension_reserve_is_all_or_nothing() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(
        ExecutionBudget(max_tool_calls=2, max_search_calls=0),
        clock=clock,
    )
    before = ledger.snapshot()
    with pytest.raises(BudgetExceededError) as raised:
        ledger.reserve(
            {
                BudgetDimension.TOOL_CALLS: 1,
                BudgetDimension.SEARCH_CALLS: 1,
            },
            operation="search",
            attempt=1,
        )
    after = ledger.snapshot()
    assert raised.value.error.code is ErrorCode.BUDGET_EXCEEDED
    assert raised.value.error.details["dimension"] == "SEARCH_CALLS"
    assert after.committed == before.committed
    assert after.reserved == before.reserved


@pytest.mark.parametrize(
    ("field", "dimension"),
    [
        ("max_tool_calls", BudgetDimension.TOOL_CALLS),
        ("max_retry_attempts", BudgetDimension.RETRY_ATTEMPTS),
        ("max_search_calls", BudgetDimension.SEARCH_CALLS),
        ("max_downloads", BudgetDimension.DOWNLOADS),
        ("max_llm_calls", BudgetDimension.LLM_CALLS),
    ],
)
def test_call_dimensions_enforce_limits(field: str, dimension: BudgetDimension) -> None:
    ledger = BudgetLedger(ExecutionBudget(**{field: 1}))
    reservation = ledger.reserve({dimension: 1}, operation="fixture", attempt=1)
    ledger.commit(reservation)
    with pytest.raises(BudgetExceededError):
        ledger.reserve({dimension: 1}, operation="fixture", attempt=2)


def test_token_reservation_commits_actual_and_releases_unused() -> None:
    ledger = BudgetLedger(
        ExecutionBudget(
            max_prompt_tokens=100,
            max_completion_tokens=40,
            max_total_tokens=140,
        )
    )
    reservation = ledger.reserve(
        {
            BudgetDimension.PROMPT_TOKENS: 100,
            BudgetDimension.COMPLETION_TOKENS: 40,
            BudgetDimension.TOTAL_TOKENS: 140,
        },
        operation="llm",
        attempt=1,
    )
    ledger.commit(
        reservation,
        actual_usage={
            BudgetDimension.PROMPT_TOKENS: 60,
            BudgetDimension.COMPLETION_TOKENS: 20,
            BudgetDimension.TOTAL_TOKENS: 80,
        },
    )
    snapshot = ledger.snapshot()
    assert snapshot.committed[BudgetDimension.PROMPT_TOKENS] == 60
    assert snapshot.committed[BudgetDimension.COMPLETION_TOKENS] == 20
    assert snapshot.committed[BudgetDimension.TOTAL_TOKENS] == 80
    assert snapshot.remaining[BudgetDimension.TOTAL_TOKENS] == 60
    assert snapshot.reserved[BudgetDimension.TOTAL_TOKENS] == 0


def test_actual_usage_over_reservation_is_truthfully_committed_and_rejected() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_total_tokens=100))
    reservation = ledger.reserve(
        {BudgetDimension.TOTAL_TOKENS: 80}, operation="llm", attempt=1
    )
    with pytest.raises(BudgetExceededError) as raised:
        ledger.commit(
            reservation,
            actual_usage={BudgetDimension.TOTAL_TOKENS: 120},
        )
    snapshot = ledger.snapshot()
    assert raised.value.error.details["phase"] == "POSTFLIGHT"
    assert raised.value.error.details["actual_usage"] == 120
    assert snapshot.committed[BudgetDimension.TOTAL_TOKENS] == 120
    assert snapshot.remaining[BudgetDimension.TOTAL_TOKENS] == 0
    assert ledger.get_reservation(reservation.reservation_id).status is BudgetReservationStatus.COMMITTED


def test_strict_token_limit_rejects_missing_reliable_upper_bound() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_total_tokens=100))
    with pytest.raises(BudgetExceededError) as raised:
        ledger.reserve(
            {BudgetDimension.LLM_CALLS: 1},
            observational_dimensions={BudgetDimension.TOTAL_TOKENS},
            operation="llm",
            attempt=1,
        )
    assert raised.value.error.details["dimension"] == "TOTAL_TOKENS"
    assert "upper bound" in raised.value.error.message


def test_observational_token_accounting_is_allowed_without_hard_limit() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_llm_calls=1))
    reservation = ledger.reserve(
        {BudgetDimension.LLM_CALLS: 1},
        observational_dimensions={BudgetDimension.TOTAL_TOKENS},
        operation="llm",
        attempt=1,
    )
    ledger.commit(
        reservation,
        actual_usage={BudgetDimension.TOTAL_TOKENS: 123},
    )
    assert ledger.snapshot().committed[BudgetDimension.TOTAL_TOKENS] == 123


def test_actual_usage_for_unknown_dimension_is_rejected() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_llm_calls=1))
    reservation = ledger.reserve(
        {BudgetDimension.LLM_CALLS: 1}, operation="llm", attempt=1
    )
    with pytest.raises(BudgetAccountingError, match="unreserved"):
        ledger.commit(
            reservation,
            actual_usage={BudgetDimension.TOTAL_TOKENS: 1},
        )
    assert ledger.get_reservation(reservation.reservation_id).status is BudgetReservationStatus.RESERVED


def test_duration_budget_uses_injected_monotonic_clock() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(ExecutionBudget(max_duration_ms=1_000), clock=clock)
    clock.advance(0.999)
    assert ledger.preview({BudgetDimension.OPERATION_ATTEMPTS: 1}).allowed is True
    clock.advance(0.001)
    preview = ledger.preview({BudgetDimension.OPERATION_ATTEMPTS: 1})
    assert preview.allowed is False
    assert preview.stop_dimension is BudgetDimension.DURATION_MS
    with pytest.raises(BudgetExceededError) as raised:
        ledger.reserve(
            {BudgetDimension.OPERATION_ATTEMPTS: 1},
            operation="fixture",
            attempt=1,
        )
    assert raised.value.error.details["dimension"] == "DURATION_MS"


def test_preview_does_not_replace_atomic_reserve() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=1))
    assert ledger.preview({BudgetDimension.TOOL_CALLS: 1}).allowed is True
    winner = ledger.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="winner", attempt=1
    )
    with pytest.raises(BudgetExceededError):
        ledger.reserve(
            {BudgetDimension.TOOL_CALLS: 1}, operation="loser", attempt=1
        )
    ledger.commit(winner)


def test_disabled_budget_allows_and_accounts_usage() -> None:
    ledger = BudgetLedger(ExecutionBudget.disabled())
    reservation = ledger.reserve(
        {BudgetDimension.OPERATION_ATTEMPTS: 1}, operation="fixture", attempt=1
    )
    ledger.commit(reservation)
    snapshot = ledger.snapshot()
    assert snapshot.committed[BudgetDimension.OPERATION_ATTEMPTS] == 1
    assert snapshot.remaining[BudgetDimension.OPERATION_ATTEMPTS] is None


def test_snapshot_is_json_serializable_and_remaining_never_negative() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_total_tokens=10))
    reservation = ledger.reserve(
        {BudgetDimension.TOTAL_TOKENS: 10}, operation="llm", attempt=1
    )
    with pytest.raises(BudgetExceededError):
        ledger.commit(
            reservation,
            actual_usage={BudgetDimension.TOTAL_TOKENS: 12},
        )
    payload = ledger.snapshot().model_dump(mode="json")
    json.dumps(payload)
    assert all(value is None or value >= 0 for value in payload["remaining"].values())


def test_budget_consumed_and_exceeded_trace_events(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path)
    context = TraceContext("budget_trace", "fixture", recorder)
    span = context.start_span(
        kind=TraceSpanKind.OPERATION,
        operation="tool.call",
        stage="RUN",
    )
    # This test counts tool-call events, not elapsed-time events. A real clock
    # can add DURATION_MS consumption under slower disk/CI scheduling.
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=1),clock=FakeClock())
    reservation = ledger.reserve(
        {BudgetDimension.TOOL_CALLS: 1},
        operation="tool.call",
        attempt=1,
        trace_context=context,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        stage="RUN",
    )
    ledger.commit(
        reservation,
        trace_context=context,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        stage="RUN",
    )
    with pytest.raises(BudgetExceededError):
        ledger.reserve(
            {BudgetDimension.TOOL_CALLS: 1},
            operation="tool.call",
            attempt=2,
            trace_context=context,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            stage="RUN",
        )
    events = read_events(recorder, "budget_trace")
    assert [event.event_type for event in events] == [
        TraceEventType.OPERATION_STARTED,
        TraceEventType.BUDGET_CONSUMED,
        TraceEventType.BUDGET_EXCEEDED,
    ]
    assert events[1].output_summary["dimension"] == "TOOL_CALLS"
    assert events[2].error_code is ErrorCode.BUDGET_EXCEEDED


class BrokenTraceContext:
    def emit(self, **kwargs):
        raise OSError("trace projection unavailable")


def test_trace_failure_does_not_roll_back_committed_ledger_state() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=1))
    reservation = ledger.reserve(
        {BudgetDimension.TOOL_CALLS: 1}, operation="tool.call", attempt=1
    )
    ledger.commit(
        reservation,
        trace_context=BrokenTraceContext(),
        span_id="span_00000000000000000000",
    )
    assert ledger.snapshot().committed[BudgetDimension.TOOL_CALLS] == 1


def test_trace_failure_does_not_replace_budget_exceeded_error() -> None:
    ledger = BudgetLedger(ExecutionBudget(max_tool_calls=0))
    with pytest.raises(BudgetExceededError):
        ledger.reserve(
            {BudgetDimension.TOOL_CALLS: 1},
            operation="tool.call",
            attempt=1,
            trace_context=BrokenTraceContext(),
            span_id="span_00000000000000000000",
        )
