"""保存済みのSSE観測から、思考と本文の数値指標を導く。"""

from __future__ import annotations

from dataclasses import dataclass

from bench_harness.types import StreamResult

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class PhaseMetrics:
    status: str
    observed: bool = True
    duration_reason: str | None = None
    text_wait_s: float | None = None
    thinking_duration_s: float | None = None
    text_duration_s: float | None = None
    thinking_chars: int | None = None
    text_chars: int | None = None
    thinking_chars_per_s: float | None = None
    text_chars_per_s: float | None = None
    thinking_retokenized_tps: float | None = None
    text_retokenized_tps: float | None = None


def _duration(first_ns: int | None, last_ns: int | None) -> float | None:
    if first_ns is None or last_ns is None:
        return None
    return (last_ns - first_ns) / _NS_PER_S


def _speed(chars: int, duration_s: float | None) -> float | None:
    if duration_s is None or duration_s <= 0:
        return None
    return chars / duration_s


def phase_metrics(result: StreamResult) -> PhaseMetrics:
    """旧記録の段階別時刻をblocksから補完しない。"""
    phases = result.output_phases
    if phases is None:
        if any(block.type == "text" and block.text for block in result.blocks):
            status = "text"
        elif any(block.type == "thinking" and block.text for block in result.blocks):
            status = "thinking_only"
        else:
            status = "no_text"
        return PhaseMetrics(status=status, observed=False)
    thinking = phases.thinking
    text = phases.text
    thinking_duration = _duration(thinking.first_ns, thinking.last_ns)
    text_duration = _duration(text.first_ns, text.last_ns)
    duration_reason = None
    if (
        thinking.first_ns is not None
        and thinking.last_ns is not None
        and text.first_ns is not None
        and text.last_ns is not None
        and thinking.first_ns < text.last_ns
        and text.first_ns < thinking.last_ns
    ):
        # 両端だけでは、交互出力や重複した区間を段階ごとに分離できない。
        thinking_duration = None
        text_duration = None
        duration_reason = "思考と本文の観測区間を分離できないため、段階別の継続時間・速度は不明"
    if text.char_count:
        status = "text"
    elif thinking.char_count:
        status = "thinking_only"
    else:
        status = "no_text"
    wait = (
        (text.first_ns - result.timing.sent_at_ns) / _NS_PER_S
        if text.first_ns is not None
        else None
    )
    counts = result.output_token_counts
    return PhaseMetrics(
        status=status,
        duration_reason=duration_reason,
        text_wait_s=wait,
        thinking_duration_s=thinking_duration,
        text_duration_s=text_duration,
        thinking_chars=thinking.char_count,
        text_chars=text.char_count,
        thinking_chars_per_s=_speed(thinking.char_count, thinking_duration),
        text_chars_per_s=_speed(text.char_count, text_duration),
        thinking_retokenized_tps=(
            _speed(counts.thinking.count, thinking_duration)
            if counts is not None and counts.thinking.count is not None
            else None
        ),
        text_retokenized_tps=(
            _speed(counts.text.count, text_duration)
            if counts is not None and counts.text.count is not None
            else None
        ),
    )
