"""保存された段階別観測から導く指標。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bench_harness.analysis.output_phases import phase_metrics
from bench_harness.types import (
    ContentBlock,
    OutputPhases,
    PhaseObservation,
    StreamResult,
    StreamTiming,
    _OutputTokenCounts,
    _PhaseTokenCount,
)


def _result(phases: OutputPhases | None) -> StreamResult:
    return StreamResult(
        timing=StreamTiming(
            sent_at_utc=datetime(2026, 9, 29, tzinfo=UTC), sent_at_ns=1_000, end_ns=5_000_001_000
        ),
        output_phases=phases,
    )


def test_text_interval_starts_at_its_own_first_data() -> None:
    result = _result(
        OutputPhases(
            thinking=PhaseObservation(
                has_block=True, first_ns=1_000, last_ns=2_000_001_000, char_count=4
            ),
            text=PhaseObservation(
                has_block=True, first_ns=3_000_001_000, last_ns=5_000_001_000, char_count=6
            ),
        )
    )
    metrics = phase_metrics(result)

    assert metrics.status == "text"
    assert metrics.text_wait_s == pytest.approx(3.0)
    assert metrics.thinking_duration_s == pytest.approx(2.0)
    assert metrics.text_duration_s == pytest.approx(2.0)
    assert metrics.thinking_chars == 4
    assert metrics.text_chars == 6
    assert metrics.thinking_chars_per_s == pytest.approx(2.0)
    assert metrics.text_chars_per_s == pytest.approx(3.0)


def test_thinking_only_has_no_text_speed() -> None:
    metrics = phase_metrics(
        _result(
            OutputPhases(
                thinking=PhaseObservation(
                    has_block=True, first_ns=1_000, last_ns=1_000_001_000, char_count=2
                )
            )
        )
    )
    assert metrics.status == "thinking_only"
    assert metrics.text_chars == 0
    assert metrics.text_wait_s is None
    assert metrics.text_chars_per_s is None


def test_single_delta_has_zero_duration_and_unknown_speed() -> None:
    metrics = phase_metrics(
        _result(
            OutputPhases(
                text=PhaseObservation(has_block=True, first_ns=1_000, last_ns=1_000, char_count=12)
            )
        )
    )
    assert metrics.text_duration_s == 0
    assert metrics.text_chars == 12
    assert metrics.text_chars_per_s is None


def test_old_record_has_unknown_phase_metrics() -> None:
    metrics = phase_metrics(_result(None))
    assert metrics.status == "no_text"
    assert metrics.observed is False
    assert metrics.text_chars is None
    assert metrics.text_chars_per_s is None


def test_old_text_block_preserves_arrival_without_inventing_timing() -> None:
    old = _result(None).model_copy(update={"blocks": [ContentBlock(type="text", text="本文")]})
    metrics = phase_metrics(old)
    assert metrics.status == "text"
    assert metrics.observed is False
    assert metrics.text_wait_s is None
    assert metrics.text_chars_per_s is None


# --- 思考が出たかどうか (issue #130) -----------------------------------------


def test_thinking_observed_is_true_when_thinking_and_text_both_arrive() -> None:
    metrics = phase_metrics(
        _result(
            OutputPhases(
                thinking=PhaseObservation(
                    has_block=True, first_ns=1_000, last_ns=2_000_001_000, char_count=4
                ),
                text=PhaseObservation(
                    has_block=True, first_ns=3_000_001_000, last_ns=5_000_001_000, char_count=6
                ),
            )
        )
    )
    assert metrics.status == "text"
    assert metrics.thinking_observed is True


def test_thinking_observed_is_true_when_only_thinking_arrives() -> None:
    metrics = phase_metrics(
        _result(
            OutputPhases(
                thinking=PhaseObservation(
                    has_block=True, first_ns=1_000, last_ns=1_000_001_000, char_count=2
                )
            )
        )
    )
    assert metrics.status == "thinking_only"
    assert metrics.thinking_observed is True


def test_thinking_observed_is_false_when_only_text_arrives() -> None:
    metrics = phase_metrics(
        _result(
            OutputPhases(
                text=PhaseObservation(
                    has_block=True, first_ns=1_000, last_ns=1_000_001_000, char_count=6
                )
            )
        )
    )
    assert metrics.status == "text"
    assert metrics.thinking_observed is False


def test_old_record_thinking_observed_is_true_when_a_thinking_block_arrived() -> None:
    old = _result(None).model_copy(
        update={"blocks": [ContentBlock(type="thinking", text="考え中")]}
    )
    metrics = phase_metrics(old)
    assert metrics.status == "thinking_only"
    assert metrics.observed is False
    assert metrics.thinking_observed is True


def test_old_record_thinking_observed_is_false_when_only_a_text_block_arrived() -> None:
    old = _result(None).model_copy(update={"blocks": [ContentBlock(type="text", text="本文")]})
    metrics = phase_metrics(old)
    assert metrics.status == "text"
    assert metrics.observed is False
    assert metrics.thinking_observed is False


@pytest.mark.parametrize(
    "thinking_span,text_span,separable",
    [
        ((1, 4), (3, 5), False),
        ((3, 5), (1, 4), False),
        ((1, 5), (2, 4), False),
        ((2, 4), (1, 5), False),
        ((1, 5), (3, 3), False),
        ((3, 3), (1, 5), False),
        ((1, 3), (3, 5), True),
        ((3, 5), (1, 3), True),
        ((3, 3), (3, 3), True),
        ((1, 3), (3, 3), True),
    ],
)
def test_phase_interval_separation_also_controls_retokenized_speed(
    thinking_span: tuple[int, int], text_span: tuple[int, int], separable: bool
) -> None:
    phases = OutputPhases(
        thinking=PhaseObservation(
            has_block=True,
            first_ns=thinking_span[0] * 1_000_000_000,
            last_ns=thinking_span[1] * 1_000_000_000,
            char_count=4,
        ),
        text=PhaseObservation(
            has_block=True,
            first_ns=text_span[0] * 1_000_000_000,
            last_ns=text_span[1] * 1_000_000_000,
            char_count=6,
        ),
    )
    counts = _OutputTokenCounts(thinking=_PhaseTokenCount(count=2), text=_PhaseTokenCount(count=3))
    result = _result(phases).model_copy(update={"output_token_counts": counts})
    metrics = phase_metrics(result)
    assert metrics.status == "text"
    assert metrics.text_chars == 6 and metrics.thinking_chars == 4
    assert result.output_phases == phases
    assert result.output_token_counts == counts
    if not separable:
        assert metrics.duration_reason is not None
        assert metrics.text_duration_s is None and metrics.thinking_duration_s is None
        assert metrics.text_chars_per_s is None and metrics.thinking_chars_per_s is None
        assert metrics.text_retokenized_tps is None and metrics.thinking_retokenized_tps is None
    else:
        assert metrics.duration_reason is None
        assert metrics.thinking_duration_s == thinking_span[1] - thinking_span[0]
        assert metrics.text_duration_s == text_span[1] - text_span[0]
        for phase, duration, count in (
            ("thinking", metrics.thinking_duration_s, 2),
            ("text", metrics.text_duration_s, 3),
        ):
            speed = getattr(metrics, f"{phase}_retokenized_tps")
            assert speed == (count / duration if duration else None)
