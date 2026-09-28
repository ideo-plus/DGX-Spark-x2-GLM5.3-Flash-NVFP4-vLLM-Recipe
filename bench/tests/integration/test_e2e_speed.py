"""速さの計測を、偽のサーバーで端から端まで確かめる (task 5.2)。

3 つの流れを、どれもコマンドの入口 (`cli.main`) から通して確かめる。

1. **繰り返しと感度** (9.3、9.4): 同じ偽のサーバーに 2 回流して比べ、すべての
   条件が「収まる」になる。3 回目だけ、**1 つの条件のイベントの間の遅れ**を
   10% 増やすと、その条件の生成速度だけが「収まらない」になる
2. **試行の途中でサーバーが落ちる** (8.7、10.4、10.5): 要約が作れて、未完了と
   表示され、比較の**表より先に**警告が出る
3. **外に出てはいけないもの** (1.8、8.3): 要約、公開した場所、比較の結果、
   標準出力と標準エラーに、認証の情報の値も、送った内容も、応答の本文も
   含まれない

## 時刻を判定する試験の決まり (Implementation Notes 1.4)

- 絶対の時間の下限は 1 つも判定しない。判定するのは、**同じ偽のサーバーに対する
  2 つの計測ランの相対の差** (中央値どうしの比と、再標本化の判定) だけである
- 個々の値を狭い幅で全数判定しない。中央値で見る
- 条件ごとに慣らしの試行を 1 回入れて、接続の確立を最初のトークンまでの時間に
  混ぜない (`warmup_trials = 1`)
- この Mac での実測 (2026-09-20): `decode_tps` の変動係数は 0.1% 前後、
  `ttft_s` と `prefill_tps` は 0.3% 前後。注 2.4 の表によれば、許容の幅 2% に
  対して変動係数が 1% 以下なら、同じ分布どうしを「収まらない」と誤る確率は
  0% である。試行は 20 回にしてあるので、注 2.4 の「20 回以上なら判定が安定する」
  も満たす。この 3 つの試験を 25 回続けて流して、失敗 0 を確かめてある

## 遅れの作り方 (偽のサーバーの契約)

`SseEvent.delay_s` で、イベントごとの遅れを直に指定する (`Script.first_delay_s`
と `Script.gap_s` は 0 にする)。こうすると、

- `ttft_s` = 最初の `content_block_delta` までの遅れ
- `decode_tps` = `(出力のトークン数 − 1) ÷ (残りの delta の遅れの合計)`
  = `1 ÷ gap`

になり、**最初のトークンまでの時間と、生成の速さを、別々に動かせる**。3 番目の
計測ランでは、`decode/prose/ja` の条件だけ、delta の間の遅れを 10% 増やす。
最初のトークンまでの遅れは動かさないので、動くのは `decode/prose/ja` の
`decode_tps` の 1 行だけになる。

生データの置き場所と公開の場所は、必ず `tmp_path` の下にする (リポジトリの
`results/` と `docs/results/` を汚さない。8.5)。要求を送る先は 127.0.0.1 の
偽のサーバーだけである。
"""

from __future__ import annotations

import gzip
import statistics
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from bench_harness import cli
from bench_harness.analysis.compare import compare_runs
from bench_harness.analysis.summarize import trial_values
from bench_harness.runner import EXIT_ABORTED, EXIT_OK
from bench_harness.store import RunStore, list_run_dirs
from bench_harness.types import RunStatus, Summary, TrialRecord
from fake_server import (
    FakeServer,
    Script,
    SseEvent,
    UsageSpec,
    replace,
    text_events,
    thinking_events,
    tool_use_events,
)

TARGET_NAME: Final[str] = "fake"
PROFILE_NAME: Final[str] = "e2e"

API_KEY_ENV: Final[str] = "BENCH_TEST_E2E_KEY"
API_KEY_VALUE: Final[str] = "sk-bench-e2e-DO-NOT-LEAK-7c1f"
RESPONSE_SENTINEL: Final[str] = "RESPONSE-SENTINEL-bench-e2e-9a4d"

PREFLIGHT_MESSAGE_REQUESTS: Final[int] = 1
"""前提の確認 (2.7) が `/v1/messages` に送る要求の数。

「送った要求の数 == 残ったレコードの数」を突き合わせるときの下駄である
(前提の確認の要求には、試行のレコードがない)。完了した計測ランの側でも
同じ式を確かめるので、この数がずれたら、その試験が先に落ちる。
"""

# --- 速さの knob ------------------------------------------------------------

DECODE_OUTPUT_TOKENS: Final[int] = 16
"""生成速度の試行の出力のトークン数。

`suites/base.py` の `MIN_SPEED_OUTPUT_TOKENS` (16) 以上にする。下回ると
`TOO_FEW_OUTPUT_TOKENS` の印が付いて、`decode_tps` の集計から外れてしまう。
"""

DECODE_TTFT_S: Final[float] = 0.03
DECODE_GAP_S: Final[float] = 0.002
OTHER_TTFT_S: Final[float] = 0.03
OTHER_GAP_S: Final[float] = 0.0005
SLOWDOWN: Final[float] = 1.10
"""3 番目の計測ランで、1 つの条件の delta の間の遅れに掛ける率 (9.3 の「10% 増やす」)。"""

SLOW_CONDITION: Final[str] = "decode/prose/ja"
_SLOW_CONDITION_MARKER: Final[str] = "長く詳しい報告書"
"""`decode/prose/ja` の指示にだけ現れる文字列 (`suites/decode.py` の `_prose_instruction`)。"""

_DECODE_MARKERS: Final[tuple[str, ...]] = (
    "Write one complete, self-contained Python module",  # decode/code/en
    "Write one long, detailed report about",  # decode/prose/en
    "Python モジュールを書いてください",  # decode/code/ja
    _SLOW_CONDITION_MARKER,  # decode/prose/ja
    "JSON",  # decode/json/en, decode/json/ja
)

DECODE_CONDITIONS: Final[tuple[str, ...]] = (
    "decode/code/en",
    "decode/code/ja",
    "decode/prose/en",
    SLOW_CONDITION,
    "decode/json/en",
    "decode/json/ja",
)
PREFILL_TARGET_TOKENS: Final[int] = 1000
PREFILL_CONDITIONS: Final[tuple[str, ...]] = ("prefill/cold/1k", "prefill/warm/1k")


# --- 台本 --------------------------------------------------------------------


def speed_script(
    *, output_tokens: int, ttft_s: float, gap_s: float, last_chunk: str | None = None
) -> Script:
    """遅れをイベントごとに指定した、速さの試験用の台本。

    `ttft_s` は最初の `content_block_delta` までの遅れ、`gap_s` は残りの delta
    の間の遅れである。ほかのイベント (`message_start`、`content_block_start`、
    `content_block_stop`、`message_delta`、`message_stop`) の遅れは 0 にして
    あるので、`ttft_s` と `decode_tps` が別々に決まる。
    """
    chunks = ["tok "] * output_tokens
    if last_chunk is not None:
        chunks[-1] = last_chunk
    events: list[SseEvent] = []
    deltas = 0
    for event in text_events("".join(chunks), chunks=chunks):
        if event.type == "content_block_delta":
            delay = ttft_s if deltas == 0 else gap_s
            deltas += 1
            events.append(replace(event, delay_s=delay))
        else:
            events.append(replace(event, delay_s=0.0))
    return Script(
        events=tuple(events),
        first_delay_s=0.0,
        gap_s=0.0,
        stop_reason="max_tokens",
        usage=UsageSpec(output_tokens=output_tokens),
    )


class SpeedResponder:
    """要求の中身を見て、条件ごとの台本を返す (偽のサーバーの `set_response_factory`)。

    生成速度の 6 つの条件は、指示の文で見分けられる (`suites/decode.py` の型紙)。
    `slow_condition` を真にすると、`decode/prose/ja` の delta の間の遅れだけが
    `SLOWDOWN` 倍になる。ほかの条件と、最初のトークンまでの遅れは変わらない。
    """

    def __init__(self) -> None:
        self.slow_condition = False
        self._decode = speed_script(
            output_tokens=DECODE_OUTPUT_TOKENS, ttft_s=DECODE_TTFT_S, gap_s=DECODE_GAP_S
        )
        self._decode_slow = speed_script(
            output_tokens=DECODE_OUTPUT_TOKENS,
            ttft_s=DECODE_TTFT_S,
            gap_s=DECODE_GAP_S * SLOWDOWN,
        )
        self._other = speed_script(
            output_tokens=DECODE_OUTPUT_TOKENS, ttft_s=OTHER_TTFT_S, gap_s=OTHER_GAP_S
        )

    def __call__(self, body: dict[str, Any]) -> Script:
        text = user_text(body)
        if _SLOW_CONDITION_MARKER in text:
            return self._decode_slow if self.slow_condition else self._decode
        if any(marker in text for marker in _DECODE_MARKERS):
            return self._decode
        return self._other


def user_text(body: Mapping[str, Any]) -> str:
    """要求の本文から、ユーザーの発話の文字列だけを取り出してつなぐ。"""
    parts: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(str(block["text"]))
    return "\n".join(parts)


# --- 設定ファイルと置き場所の下ごしらえ --------------------------------------


@dataclass(frozen=True)
class Bed:
    """1 回のコマンドに要る、設定ファイルと生データの置き場所の一式。"""

    targets: Path
    profiles: Path
    results_root: Path

    def run_options(self) -> list[str]:
        return [
            "--targets",
            str(self.targets),
            "--profiles",
            str(self.profiles),
            "--results-root",
            str(self.results_root),
        ]


def write_bed(
    tmp_path: Path,
    base_url: str,
    *,
    api_key_env: str | None = None,
    trials: int = 20,
    concurrency_rounds: int = 2,
    max_consecutive_failures: int = 5,
    first_event_s: float = 5.0,
) -> Bed:
    """試験用の対象サーバーの定義と計測の設定を書く。

    `length_tolerance` を 0.15 にしてあるのは、偽のサーバーが入力のトークン数を
    「JSON の文字数 ÷ 4.0」で数える一方、合成の文章は設定の種類ごとの比で長さを
    決めるので、狙いから 1 割ほどずれるためである (この試験は長さの印を見ない)。
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    targets = tmp_path / "targets.toml"
    lines = [f"[targets.{TARGET_NAME}]", f'base_url = "{base_url}"', 'model = "fake-model"']
    if api_key_env is not None:
        lines.append(f'api_key_env = "{api_key_env}"')
    targets.write_text("\n".join(lines) + "\n", encoding="utf-8")

    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        f"""
[profiles.{PROFILE_NAME}]
seed = 11
min_successes = 5
max_consecutive_failures = {max_consecutive_failures}
length_tolerance = 0.15
compare_tolerance = 0.02
metrics_interval_s = 60.0

[profiles.{PROFILE_NAME}.sampling]
temperature = 0.0
thinking = "server_default"

[profiles.{PROFILE_NAME}.timeout]
connect_s = 5.0
first_event_s = {first_event_s}
idle_s = 5.0
total_s = 30.0

[profiles.{PROFILE_NAME}.decode]
trials = {trials}
warmup_trials = 1
max_tokens = 64

[profiles.{PROFILE_NAME}.prefill]
trials = {trials}
warmup_trials = 1
max_tokens = 16
target_input_tokens = [{PREFILL_TARGET_TOKENS}]

[profiles.{PROFILE_NAME}.concurrency]
levels = [2]
rounds = {concurrency_rounds}
max_tokens = 64
input_tokens = 400
""",
        encoding="utf-8",
    )
    return Bed(targets=targets, profiles=profiles, results_root=tmp_path / "results")


def bench(*argv: str) -> int:
    """`bench` を、その場 (同じプロセス) で 1 回走らせる。"""
    return cli.main(list(argv))


def run_bench(bed: Bed, *suites: str) -> tuple[int, Path]:
    """`bench run` を 1 回流して、終了の値と、増えた計測ランのディレクトリを返す。"""
    before = {path.name for path in _run_dirs(bed)}
    argv = ["run", "--target", TARGET_NAME, "--profile", PROFILE_NAME]
    for suite in suites:
        argv += ["--suite", suite]
    code = bench(*argv, *bed.run_options())
    added = [path for path in _run_dirs(bed) if path.name not in before]
    assert len(added) == 1, f"増えた計測ランが 1 つではない: {added}"
    return code, added[0]


def _run_dirs(bed: Bed) -> list[Path]:
    return list_run_dirs(bed.results_root) if bed.results_root.is_dir() else []


# --- 生データの読み出し ------------------------------------------------------


def values_of(run_dir: Path, condition: str, metric: str) -> list[float]:
    """要約と比較が使うのと同じ道 (`trial_values`) で、試行ごとの値を取り出す (注 4.1)。"""
    store = RunStore.open(run_dir)
    records, warnings = store.read_trials()
    assert warnings == [], warnings
    values = trial_values(records, store.manifest())
    return [item.value for item in values[(condition, metric)]]


def median_of(run_dir: Path, condition: str, metric: str) -> float:
    return statistics.median(values_of(run_dir, condition, metric))


def read_all_text(root: Path) -> dict[str, str]:
    """`root` の下のすべてのファイルを、文字列にして返す (`.gz` は展開する)。"""
    texts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        texts[path.relative_to(root).as_posix()] = data.decode("utf-8", errors="replace")
    return texts


def prompt_excerpt(store: RunStore, records: Sequence[TrialRecord]) -> str:
    """保存された要求の本文から、目印になる長さの、送った内容の一部を取り出す。"""
    body = store.get_body(records[0].request_body_ref)
    text = user_text(dict(body))
    longest = max(text.splitlines(), key=len)
    excerpt = longest[:80]
    assert len(excerpt) >= 60, f"目印にするには短すぎる: {excerpt!r}"
    return excerpt


# --- 流れ 1: 繰り返しと感度 (9.3、9.4) ---------------------------------------


def test_thinking_only_is_not_text_speed_success(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.set_response(
        Script(
            events=thinking_events("思考だけ", chunks=("思考", "だけ")),
            gap_s=0.001,
            stop_reason="max_tokens",
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")

    assert code == EXIT_OK
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    for condition in ("decode/json/en", "decode/json/ja"):
        all_output = next(
            row
            for row in summary.results
            if row.condition == condition and row.metric == "decode_tps"
        )
        text_speed = next(
            row
            for row in summary.results
            if row.condition == condition and row.metric == "text_chars_per_s"
        )
        assert all_output.continuous is not None and all_output.continuous.n == 10
        assert text_speed.continuous is None or text_speed.continuous.n == 0
        counts = summary.output_phases[condition]
        assert counts["requests"] == 10
        assert counts["request_successes"] == 10
        assert counts["request_failures"] == 0
        assert counts["text_not_reached"] == 10
        assert counts["thinking_only"] == 10
        assert counts["text_speed_n"] == 0
    assert "本文未到達" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "10/10" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "本文未到達" in (run_dir / "summary.json").read_text(encoding="utf-8")
    comparison = compare_runs(run_dir, run_dir).report
    assert any(
        "計測ラン A" in warning and "本文未到達" in warning for warning in comparison.warnings
    )
    assert any(
        "計測ラン B" in warning and "本文未到達" in warning for warning in comparison.warnings
    )


def test_phase_values_reach_summary_compare_and_publish_without_body_text(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    thought = "THINKING-SENTINEL-REDACT"
    answer = "TEXT-SENTINEL-REDACT"
    tool_argument = "TOOL-ARG-SENTINEL-REDACT"
    monkeypatch.setenv(API_KEY_ENV, API_KEY_VALUE)
    fake_server.set_response(
        Script(
            events=(
                *thinking_events(thought, chunks=(thought[:8], thought[8:])),
                *text_events(answer, index=1, chunks=(answer[:8], answer[8:])),
                *tool_use_events("tool", {"value": tool_argument}, index=2),
            ),
            gap_s=0.002,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, api_key_env=API_KEY_ENV, trials=10)
    code_a, run_a = run_bench(bed, "decode")
    code_b, run_b = run_bench(bed, "decode")
    assert (code_a, code_b) == (EXIT_OK, EXIT_OK)
    first_store = RunStore.open(run_a)
    first_records, _ = first_store.read_trials()
    request_text = prompt_excerpt(first_store, first_records)
    assert fake_server.requests_for("/v1/messages")[-1].headers["x-api-key"] == API_KEY_VALUE

    for run_dir in (run_a, run_b):
        assert values_of(run_dir, "decode/json/en", "text_chars_per_s")
        assert values_of(run_dir, "decode/json/en", "thinking_chars_per_s")
        waits = values_of(run_dir, "decode/json/en", "text_wait_s")
        durations = values_of(run_dir, "decode/json/en", "text_duration_s")
        assert waits and durations
        assert min(waits) > max(durations), "本文の区間に思考の待ち時間を含めない"
        summary_json = (run_dir / "summary.json").read_text(encoding="utf-8")
        summary_md = (run_dir / "summary.md").read_text(encoding="utf-8")
        assert "text_chars_per_s" in summary_json
        assert "text_chars_per_s" in summary_md
        legend = next(line for line in summary_md.splitlines() if line.startswith("- 値の意味:"))
        text_speed_unit = legend.split("`text_chars_per_s` = ", 1)[1].split(
            "、`thinking_retokenized_tps`", 1
        )[0]
        assert "文字/秒" in text_speed_unit
        assert "tok/s" not in text_speed_unit

    comparison = compare_runs(run_a, run_b)
    assert any(
        row.condition == "decode/json/en" and row.metric == "text_chars_per_s"
        for row in comparison.report.rows
    )
    publish_root = tmp_path / "published"
    assert bench("publish", str(run_a), "--docs-root", str(publish_root)) == EXIT_OK
    published = read_all_text(publish_root)
    assert sorted(published) == [f"{run_a.name}/summary.json", f"{run_a.name}/summary.md"]

    exposed = [
        (run_a / "summary.json").read_text(encoding="utf-8"),
        (run_a / "summary.md").read_text(encoding="utf-8"),
        comparison.to_json(),
        comparison.to_markdown(),
        *published.values(),
        capsys.readouterr().out,
    ]
    sentinels = (thought, answer, tool_argument, API_KEY_VALUE, request_text)
    assert all(sentinel not in item for sentinel in sentinels for item in exposed)


def test_json_text_arrival_does_not_claim_json_validity(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=text_events("not valid JSON", chunks=("not ", "valid JSON")),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    assert values_of(run_dir, "decode/json/en", "text_chars_per_s")
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert not any(
        row.condition.startswith("decode/json/") and "valid" in row.metric
        for row in summary.results
    )


def test_single_text_delta_has_no_finite_text_speed(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(events=text_events("一つのdeltaに複数語"), usage=UsageSpec(output_tokens=16))
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    speed = next(
        row
        for row in summary.results
        if row.condition == "decode/json/en" and row.metric == "text_chars_per_s"
    )
    assert speed.continuous is None or speed.continuous.n == 0
    assert "Infinity" not in summary.model_dump_json()
    assert "NaN" not in summary.model_dump_json()


def test_default_run_sends_no_output_retokenization_request(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(speed_script(output_tokens=16, ttft_s=0.001, gap_s=0.001))
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")

    assert code == EXIT_OK
    assert fake_server.call_count("/tokenize") == 0
    settings = RunStore.open(run_dir).manifest().output_retokenization
    assert settings is not None
    assert settings.enabled is False
    assert settings.method == "/tokenize"
    assert settings.add_special_tokens is False


def test_opt_in_retokenization_preserves_stream_end_and_records_count_source(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=(*thinking_events("思考"), *text_events("本文", index=1, chunks=("本", "文"))),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    fake_server.set_tokenize_response(
        200, {"count": 2, "tokens": [10, 11], "model": "fake-model"}, delay_s=0.02
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    before = set(_run_dirs(bed))
    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        "--retokenize-output",
        *bed.run_options(),
    )
    added = set(_run_dirs(bed)) - before
    assert code == EXIT_OK and len(added) == 1
    run_dir = added.pop()
    store = RunStore.open(run_dir)
    records, warnings = store.read_trials()
    assert warnings == []
    counted = fake_server.requests_for("/tokenize")
    assert counted
    assert records[0].result.timing.end_ns <= counted[0].arrived_at_ns
    settings = store.manifest().output_retokenization
    assert settings is not None
    assert settings.enabled is True
    assert settings.method == "/tokenize"
    assert settings.block_join == "concat"
    assert settings.add_special_tokens is False
    assert all(record.result.output_token_counts is not None for record in records)
    measured = [
        record for record in records if record.condition == "decode/json/en" and not record.warmup
    ]
    expected_speeds: list[float] = []
    for record in measured:
        counts = record.result.output_token_counts
        phases = record.result.output_phases
        assert counts is not None and phases is not None
        assert counts.text.count == 2
        assert counts.text.count_source == "retokenized"
        assert phases.text.first_ns is not None and phases.text.last_ns is not None
        expected_speeds.append(
            counts.text.count / ((phases.text.last_ns - phases.text.first_ns) / 1e9)
        )
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    speed = next(
        row
        for row in summary.results
        if row.condition == "decode/json/en" and row.metric == "text_retokenized_tps"
    )
    assert speed.continuous is not None
    assert speed.continuous.n == len(measured)
    assert speed.continuous.mean == pytest.approx(statistics.mean(expected_speeds))
    assert speed.count_source == "retokenized"

    before_second = set(_run_dirs(bed))
    second_code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        "--retokenize-output",
        *bed.run_options(),
    )
    added_second = set(_run_dirs(bed)) - before_second
    assert second_code == EXIT_OK and len(added_second) == 1
    second_run = added_second.pop()
    comparison = compare_runs(run_dir, second_run).report
    compared_speed = next(
        row
        for row in comparison.rows
        if row.condition == "decode/json/en" and row.metric == "text_retokenized_tps"
    )
    assert compared_speed.value_a is not None and compared_speed.value_b is not None
    assert compared_speed.count_source_a == "retokenized"
    assert compared_speed.count_source_b == "retokenized"
    assert compared_speed.verdict is not None


def test_different_retokenization_methods_are_visible_and_not_judged_together(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=(*thinking_events("思考"), *text_events("本文", index=1, chunks=("本", "文"))),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    fake_server.set_tokenize_response(200, {"count": 2, "tokens": [10, 11]})
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    before = set(_run_dirs(bed))
    assert (
        bench(
            "run",
            "--target",
            TARGET_NAME,
            "--profile",
            PROFILE_NAME,
            "--suite",
            "decode",
            "--retokenize-output",
            *bed.run_options(),
        )
        == EXIT_OK
    )
    enabled_run = (set(_run_dirs(bed)) - before).pop()
    default_code, default_run = run_bench(bed, "decode")
    assert default_code == EXIT_OK

    comparison = compare_runs(enabled_run, default_run)
    assert "count_source" in comparison.to_json()
    assert "再計数" in comparison.to_markdown()
    assert any(
        row.metric == "text_retokenized_tps" and row.verdict is None
        for row in comparison.report.rows
    )


def test_saved_phase_results_are_summarized_and_compared_after_server_shutdown(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=(*thinking_events("思考"), *text_events("本文", index=1, chunks=("本", "文"))),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    requests_before = len(fake_server.requests)
    fake_server.stop()

    assert bench("summarize", str(run_dir)) == EXIT_OK
    assert bench("compare", str(run_dir), str(run_dir)) == EXIT_OK
    assert len(fake_server.requests) == requests_before
    assert values_of(run_dir, "decode/json/en", "text_chars_per_s")


def test_interleaved_phase_missing_reason_survives_storage_summary_compare_and_publish(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=(
                *text_events("AB", chunks=("A", "B")),
                *thinking_events("AB", index=1, chunks=("A", "B")),
                *text_events("AB", index=2, chunks=("A", "B")),
            ),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    requests_before = len(fake_server.requests)
    fake_server.stop()

    assert bench("summarize", str(run_dir)) == EXIT_OK
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    condition = "decode/json/en"
    counts = summary.output_phases[condition]
    assert counts["request_successes"] == counts["text_arrived"] == 10
    assert counts["text_speed_n"] == counts["text_not_reached"] == counts["unknown"] == 0
    warning = next(w for w in summary.phase_warnings if w.startswith(f"{condition}:"))
    assert "観測区間を分離できない" in warning
    assert "10 件" in warning
    assert warning in (run_dir / "summary.md").read_text(encoding="utf-8")
    for metric in ("text_duration_s", "thinking_duration_s", "text_chars_per_s"):
        assert values_of(run_dir, condition, metric) == []
    assert values_of(run_dir, condition, "text_chars") == [4.0] * 10
    assert values_of(run_dir, condition, "thinking_chars") == [2.0] * 10
    assert values_of(run_dir, condition, "text_wait_s")
    assert values_of(run_dir, condition, "decode_tps")

    comparison = compare_runs(run_dir, run_dir)
    for label in ("A", "B"):
        assert any(
            f"計測ラン {label}" in item and warning in item for item in comparison.report.warnings
        )
    text_speed = next(
        row
        for row in comparison.report.rows
        if row.condition == condition and row.metric == "text_chars_per_s"
    )
    assert text_speed.value_a is None and text_speed.value_b is None
    assert warning in comparison.to_markdown()
    publish_root = tmp_path / "published"
    assert bench("publish", str(run_dir), "--docs-root", str(publish_root)) == EXIT_OK
    published = publish_root / run_dir.name
    assert (published / "summary.json").read_bytes() == (run_dir / "summary.json").read_bytes()
    assert (published / "summary.md").read_bytes() == (run_dir / "summary.md").read_bytes()
    assert len(fake_server.requests) == requests_before


def test_compare_does_not_drop_a_text_speed_metric_present_in_both_runs(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=text_events("本文", chunks=("本", "文")),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    _, run_a = run_bench(bed, "decode")
    _, run_b = run_bench(bed, "decode")
    assert values_of(run_a, "decode/json/en", "text_chars_per_s")
    assert values_of(run_b, "decode/json/en", "text_chars_per_s")

    report = compare_runs(run_a, run_b).report
    assert any(
        row.condition == "decode/json/en"
        and row.metric == "text_chars_per_s"
        and row.value_a is not None
        and row.value_b is not None
        for row in report.rows
    )


def test_summarize_never_requests_missing_output_counts_after_capture(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(
        Script(
            events=(*thinking_events("思考"), *text_events("本文", index=1, chunks=("本", "文"))),
            gap_s=0.001,
            usage=UsageSpec(output_tokens=256),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)
    code, run_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    request_count = len(fake_server.requests)
    fake_server.stop()

    assert bench("summarize", str(run_dir)) == EXIT_OK
    assert fake_server.call_count("/tokenize") == 0
    assert len(fake_server.requests) == request_count


def test_decode_cli_records_and_summarizes_all_six_conditions(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    responder = SpeedResponder()
    fake_server.set_response_factory(responder)
    bed = write_bed(tmp_path, fake_server.base_url, trials=10)

    code, run_dir = run_bench(bed, "decode")

    assert code == EXIT_OK
    store = RunStore.open(run_dir)
    records, warnings = store.read_trials()
    assert warnings == []
    assert {record.condition for record in records} == set(DECODE_CONDITIONS)
    assert all(
        sum(record.condition == condition and not record.warmup for record in records) == 10
        for condition in DECODE_CONDITIONS
    )
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    measured = {(row.condition, row.metric) for row in summary.results}
    assert {
        (condition, metric)
        for condition in DECODE_CONDITIONS
        for metric in ("decode_tps", "ttft_s")
    } <= measured
    assert {(condition, "text_chars_per_s") for condition in DECODE_CONDITIONS} <= measured
    for condition in DECODE_CONDITIONS:
        assert values_of(run_dir, condition, "decode_tps")
        assert values_of(run_dir, condition, "ttft_s")


def test_two_runs_against_the_same_fake_all_fall_within_and_a_ten_percent_change_does_not(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """9.3: 同じ偽のサーバーに 2 回流すとすべて収まり、1 つの条件を 10% 遅くすると収まらない。

    確かめる順序は、まず**生データで遅れが本当に変わったこと**、次に判定である
    (仕掛けが効いていない状態で判定だけ見ても、何も守れない)。
    """
    responder = SpeedResponder()
    fake_server.set_response_factory(responder)
    bed = write_bed(tmp_path, fake_server.base_url)

    code_a, run_a = run_bench(bed, "decode", "prefill")
    code_b, run_b = run_bench(bed, "decode", "prefill")
    responder.slow_condition = True
    code_c, run_c = run_bench(bed, "decode", "prefill")
    capsys.readouterr()  # 計測ランの出力は、比較の判定に混ぜない

    assert (code_a, code_b, code_c) == (EXIT_OK, EXIT_OK, EXIT_OK)
    assert fake_server.errors == []

    # --- 裏取り: 3 番目の計測ランは、本当に 1 つの条件だけが遅くなっている ---
    expected_drop = 1.0 - 1.0 / SLOWDOWN  # 遅れが 1.1 倍 → 速さは 1/1.1 倍
    measured_drop = 1.0 - median_of(run_c, SLOW_CONDITION, "decode_tps") / median_of(
        run_a, SLOW_CONDITION, "decode_tps"
    )
    assert abs(measured_drop - expected_drop) < 0.02, measured_drop
    for condition in DECODE_CONDITIONS:
        if condition == SLOW_CONDITION:
            continue
        ratio = median_of(run_c, condition, "decode_tps") / median_of(
            run_a, condition, "decode_tps"
        )
        assert abs(ratio - 1.0) < 0.02, (condition, ratio)
    # 最初のトークンまでの遅れは動かしていない
    ttft_ratio = median_of(run_c, SLOW_CONDITION, "ttft_s") / median_of(
        run_a, SLOW_CONDITION, "ttft_s"
    )
    assert abs(ttft_ratio - 1.0) < 0.05, ttft_ratio

    # --- 2 回流して比べる: すべて収まる、対応のある比較になっている ---
    same = compare_runs(run_a, run_b)
    assert same.report.repeatability is not None
    legacy_metrics = {"decode_tps", "ttft_s", "prefill_tps"}
    legacy_rows = [row for row in same.report.rows if row.metric in legacy_metrics]
    assert legacy_rows
    assert all(row.verdict is not None and row.verdict.paired for row in legacy_rows)
    assert all(row.verdict is not None and row.verdict.verdict == "within" for row in legacy_rows)
    assert not any(
        f"{row.condition} / {row.metric}" in same.report.repeatability.outside
        for row in legacy_rows
    )
    assert [warning for warning in same.report.warnings if "対応のない比較" in warning] == []
    assert [warning for warning in same.report.warnings if "未完了" in warning] == []
    measured = {(row.condition, row.metric) for row in same.report.rows}
    assert {
        *(
            (condition, metric)
            for condition in DECODE_CONDITIONS
            for metric in ("decode_tps", "ttft_s")
        ),
        *(
            (condition, metric)
            for condition in PREFILL_CONDITIONS
            for metric in ("ttft_s", "prefill_tps")
        ),
    } <= measured
    assert {(condition, "text_chars_per_s") for condition in DECODE_CONDITIONS} <= measured

    # --- 入口の出力にも、収まったという結論が出る ---
    assert bench("compare", str(run_a), str(run_b)) == EXIT_OK
    out = capsys.readouterr().out
    assert "decode_tps" in out
    assert "| 対応あり |" in out
    assert "| 対応なし |" not in out  # 表の升目としての「対応なし」は 1 つもない

    # --- 10% 遅くした計測ランとの比較: その 1 行だけが収まらない ---
    changed = compare_runs(run_a, run_c)
    assert changed.report.repeatability is not None
    assert changed.report.repeatability.all_within is False
    assert f"{SLOW_CONDITION} / decode_tps" in changed.report.repeatability.outside
    outside = {
        (row.condition, row.metric)
        for row in changed.report.rows
        if row.verdict is not None and row.verdict.verdict == "outside"
    }
    assert (SLOW_CONDITION, "decode_tps") in outside
    assert {condition for condition, _ in outside} == {SLOW_CONDITION}
    assert all(
        row.verdict is not None and row.verdict.paired
        for row in changed.report.rows
        if row.metric in legacy_metrics
    )
    # 判定は結果であって失敗ではないので、入口は 0 で終わり、結論を標準出力に出す
    capsys.readouterr()
    assert bench("compare", str(run_a), str(run_c)) == EXIT_OK
    changed_out = capsys.readouterr().out
    assert changed_out == changed.to_markdown()
    assert "収まらなかった" in changed_out
    assert f"{SLOW_CONDITION} / decode_tps" in changed_out


# --- 流れ 2: 試行の途中でサーバーが落ちる (8.7、10.4、10.5) ------------------


def kill_after_requests(server: FakeServer, count: int) -> threading.Thread:
    """`/v1/messages` が `count` 件届いたら、偽のサーバーを止める見張りを起こす。

    偽のサーバー自身の口 (`wait_for_requests` と `stop`) だけを使う。`stop()` は
    止まったストリームがあっても待たされない (偽のサーバーの契約)。
    """

    def watch() -> None:
        try:
            server.wait_for_requests(count, path="/v1/messages", timeout_s=30.0)
        except TimeoutError:
            return
        server.stop()

    thread = threading.Thread(target=watch, name="fake-killer", daemon=True)
    thread.start()
    return thread


def test_a_server_that_dies_mid_run_still_summarizes_and_warns_before_any_table(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """8.7、10.4、10.5: 途中で落ちても、それまでが残り、未完了と出て、比較の先頭で警告する。"""
    fake_server.set_response(
        speed_script(output_tokens=DECODE_OUTPUT_TOKENS, ttft_s=0.0, gap_s=0.0)
    )
    # 止まった瞬間に流れていた要求は、最初のイベントの時間切れになる。成功する試行は
    # すぐに応答するので、制限時間を短くしても結果は変わらず、待ちだけが縮む
    bed = write_bed(
        tmp_path,
        fake_server.base_url,
        trials=10,
        max_consecutive_failures=2,
        first_event_s=2.0,
    )

    code, complete_dir = run_bench(bed, "decode")
    assert code == EXIT_OK
    complete_records, _ = RunStore.open(complete_dir).read_trials()
    # 完了した計測ランでの突き合わせ (注 3.5)。前提の確認の 1 件だけが、レコードを持たない
    assert fake_server.call_count("/v1/messages") == PREFLIGHT_MESSAGE_REQUESTS + len(
        complete_records
    )

    fake_server.reset()
    killer = kill_after_requests(fake_server, PREFLIGHT_MESSAGE_REQUESTS + 7)
    code, aborted_dir = run_bench(bed, "decode")
    killer.join(timeout=30.0)
    assert not killer.is_alive()
    capsys.readouterr()

    assert code == EXIT_ABORTED
    store = RunStore.open(aborted_dir)
    assert store.manifest().status is RunStatus.ABORTED
    records, warnings = store.read_trials()
    assert warnings == []
    assert records, "止まるまでの試行が 1 つも残っていない"
    assert any(record.result.error is None for record in records), (
        "止まるまでに成功した試行が、1 つも残っていない (8.7)"
    )

    # --- 送った要求の数 == 残ったレコードの数 == 保存した本文の数 (注 3.5) ---
    failed = [record for record in records if record.result.error is not None]
    assert len(failed) >= 2, "連続の失敗で止まったはずなのに、失敗したレコードが足りない"
    # サーバーが落ちたあとの要求は、接続そのものができない (= サーバーに届いていない)。
    # 落ちた瞬間に流れていた要求は、届いたうえで途中で切れる (`protocol`) ので、届いた側に数える
    unreached = sum(
        1
        for record in failed
        if record.result.error is not None and record.result.error.kind == "connect"
    )
    assert fake_server.call_count("/v1/messages") == (
        PREFLIGHT_MESSAGE_REQUESTS + len(records) - unreached
    )
    refs = [record.request_body_ref for record in records]
    assert all(refs), "本文の参照を持たないレコードがある"
    stored = sorted(
        path.name.removesuffix(".json.gz") for path in (aborted_dir / "bodies").iterdir()
    )
    assert sorted(set(refs)) == stored
    for ref in sorted(set(refs)):
        assert store.get_body(ref), ref

    # --- 要約ができて、未完了と出る (10.4) ---
    summary_json = aborted_dir / "summary.json"
    summary_md = aborted_dir / "summary.md"
    assert summary_json.is_file()
    assert summary_md.is_file()
    summary = Summary.model_validate_json(summary_json.read_text(encoding="utf-8"))
    assert summary.incomplete is True
    assert summary.conditions.status is RunStatus.ABORTED
    head = summary_md.read_text(encoding="utf-8").splitlines()[:5]
    assert any("未完了" in line for line in head), head

    # --- 比較の、どの表よりも先に警告が出る (10.5) ---
    assert bench("compare", str(complete_dir), str(aborted_dir)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    incomplete_at = [index for index, line in enumerate(lines) if "未完了" in line]
    table_at = [index for index, line in enumerate(lines) if line.startswith("| 条件 |")]
    assert incomplete_at, "未完了の警告が出ていない"
    assert table_at, "比較の表が 1 つも出ていない"
    assert min(incomplete_at) < min(table_at)
    report = compare_runs(complete_dir, aborted_dir).report
    assert [
        warning
        for warning in report.warnings
        if "未完了" in warning and aborted_dir.name in warning
    ]


# --- 流れ 3: 外に出てはいけないもの (1.8、8.3) -------------------------------


def test_no_secret_no_prompt_and_no_response_text_leaves_the_raw_data(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """1.8、8.3: 認証の情報の値、送った内容、応答の本文が、要約と公開の場所に出ない。"""
    monkeypatch.setenv(API_KEY_ENV, API_KEY_VALUE)
    fake_server.set_response(
        speed_script(
            output_tokens=DECODE_OUTPUT_TOKENS,
            ttft_s=0.0,
            gap_s=0.0,
            last_chunk=RESPONSE_SENTINEL,
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url, api_key_env=API_KEY_ENV)

    code, run_dir = run_bench(bed, "concurrency")
    assert code == EXIT_OK
    # 仕掛けの裏取り: 認証の情報は、確かにヘッダーに付いている (1.8)
    assert fake_server.requests_for("/v1/messages")[-1].headers["x-api-key"] == API_KEY_VALUE
    # 包みの計測 (issue #9) の count_tokens の要求にも、同じ鍵が付く
    counted = fake_server.requests_for("/v1/messages/count_tokens")[-1]
    assert counted.headers["x-api-key"] == API_KEY_VALUE

    store = RunStore.open(run_dir)
    records, _ = store.read_trials()
    prompt = prompt_excerpt(store, records)
    raw = read_all_text(bed.results_root)
    # 裏取り: 目印は、生データには**確かに入っている** (入っていなければ、何も守れていない)。
    # どのファイルに入っているかまで確かめる (要約に漏れたものを「生データにある」と
    # 読み違えないため。生データは 8.1 で本文を残すので、ここにあるのが正しい)
    trials = f"{run_dir.name}/trials.jsonl"
    bodies = [name for name in raw if name.startswith(f"{run_dir.name}/bodies/")]
    assert RESPONSE_SENTINEL in raw[trials], "応答の目印が trials.jsonl にない"
    assert [name for name in bodies if prompt in raw[name]], "送った内容の目印が bodies/ にない"
    # 認証の情報の値は、生データを含めて、どこにも出ない (1.8)
    assert [name for name, text in raw.items() if API_KEY_VALUE in text] == []

    publish_root = tmp_path / "published"
    comparison_dir = tmp_path / "comparison"
    assert bench("publish", str(run_dir), "--docs-root", str(publish_root)) == EXIT_OK
    assert bench("compare", str(run_dir), str(run_dir), "--out", str(comparison_dir)) == EXIT_OK
    captured = capsys.readouterr()

    published = read_all_text(publish_root)
    assert sorted(published) == [
        f"{run_dir.name}/summary.json",
        f"{run_dir.name}/summary.md",
    ]
    # 公開した場所に、生データの置き場所を指すものが 1 つもない (記号リンクも含めて実体で見る)
    assert not any(
        path.resolve().is_relative_to(bed.results_root.resolve())
        for path in publish_root.rglob("*")
    )
    # 要約に中身がある (空の要約なら、下の「含まれない」の判定は何も守らない)
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary.results, "要約に結果の行が 1 つもない"

    checked: dict[str, str] = {
        "summary.json": (run_dir / "summary.json").read_text(encoding="utf-8"),
        "summary.md": (run_dir / "summary.md").read_text(encoding="utf-8"),
        "comparison.md": (comparison_dir / "comparison.md").read_text(encoding="utf-8"),
        "comparison.json": compare_runs(run_dir, run_dir).to_json(),
        "stdout": captured.out,
        "stderr": captured.err,
        **{f"公開: {name}": text for name, text in published.items()},
    }
    for label, text in checked.items():
        assert API_KEY_VALUE not in text, label
        assert RESPONSE_SENTINEL not in text, label
        assert prompt not in text, label
