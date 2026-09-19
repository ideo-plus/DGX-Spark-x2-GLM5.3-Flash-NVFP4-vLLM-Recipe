"""2 つの計測ランの比較 (`analysis/compare.py`, task 4.3) の試験。

生データは、`RunStore.create` と手で組み立てた `TrialRecord` で作る (置き場所は
`tmp_path` の下で、git のリポジトリの外)。速さの値は、狙った生成速度から時刻を
逆算して作るので、条件ごとに狙った差 (10% など) をそのまま作れる。失敗の試行
にも、もっともらしい時刻とトークン数を持たせてある (Implementation Notes 4.1:
持たせないと、失敗を混ぜる壊れ方を検出できない)。

割合の行 (task 4.2 が足す `quality/*` と `agent/stage/*`) は、`Summary` と
`MetricResult(proportion=...)` を直に組み立てて、`compare_summaries` に渡す
(4.2 の出来上がりを待たずに、比較の側の決まりを確かめるため)。

確かめること (tasks.md 4.3 の完了の状態):

- 同じ生データの例どうしの比較で、すべての条件が「収まる」になる
- 片方の速さを 10% 変えた例で、その条件が「収まらない」になる
- 設定の違う例で、警告が先頭に出る

そのほか、要件 9.1〜9.6、10.5 と、Implementation Notes 2.4 (差と差の割合は
中央値どうしから出し、再標本化の差の中央値とは別の欄に出す)、4.1
(`trial_values` を使い、速さの式を書き写さない) に沿った振る舞いを確かめる。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from bench_harness.analysis import compare as compare_module
from bench_harness.analysis.compare import (
    compare_runs,
    compare_summaries,
    render_comparison_json,
    render_comparison_markdown,
    write_comparison,
)
from bench_harness.analysis.stats import binomial_interval
from bench_harness.store.rawstore import RunStore, StoreError
from bench_harness.types import (
    ComparisonReport,
    ComparisonRow,
    ContentBlock,
    DecodeSettings,
    MetricResult,
    Profile,
    ProportionStat,
    RequestError,
    RunManifest,
    RunStatus,
    Sampling,
    StreamResult,
    StreamTiming,
    SuiteName,
    Summary,
    TargetDef,
    Tier,
    TrialRecord,
    Usage,
)

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
AT_END = datetime(2026, 9, 19, 12, 30, 0, tzinfo=UTC)
RUN_A = "20260919T120000Z-fake-aaaaaa"
RUN_B = "20260919T130000Z-fake-bbbbbb"

CONDITION = "decode/code/en"
TPS = (20.0, 21.0, 22.0, 23.0)
"""A の生成速度。中央値は 21.5。"""
TPS_FASTER = (22.0, 23.1, 24.2, 25.3)
"""A のちょうど 1.1 倍。中央値は 23.65 で、差の割合は 10%。"""
TTFT = (0.50, 0.55, 0.60, 0.65)

OUTPUT_TOKENS = 101
"""生成速度の分子が 100 になるので、時刻の逆算が単純になる。"""


# --- 生データの組み立て -----------------------------------------------------


def _manifest(
    run_id: str,
    *,
    status: RunStatus = RunStatus.RUNNING,
    target_name: str = "fake",
    **overrides: Any,
) -> RunManifest:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "target": TargetDef(
            name=target_name,
            base_url=HttpUrl("http://127.0.0.1:8000"),
            model="glm-5.3-flash",
        ),
        "server_model": "glm-5.3-flash-server",
        "server_version": "0.0.0-test",
        "started_at": AT,
        "suites": [SuiteName.DECODE],
        "profile_name": "test",
        "profile": Profile(name="test", min_successes=1),
        "harness_version": "0.1.0+gtest",
        "context_limit": 32000,
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)


def _ns(seconds: float) -> int:
    return round(seconds * 1_000_000_000)


def _speed_trial(
    run_id: str,
    *,
    condition: str = CONDITION,
    trial_index: int = 0,
    tps: float = 20.0,
    ttft: float = 0.5,
    tier: Tier = "primary",
    failed: bool = False,
    warmup: bool = False,
) -> TrialRecord:
    """狙った生成速度と最初のトークンまでの時間から、時刻を逆算した試行を作る。

    失敗の試行にも、途中まで進んだもっともらしい時刻とトークン数を持たせる
    (Implementation Notes 4.1)。
    """
    sent = trial_index * 100.0
    first = sent + ttft
    last = first + (OUTPUT_TOKENS - 1) / tps
    timing = StreamTiming(
        sent_at_utc=AT,
        sent_at_ns=_ns(sent),
        first_token_ns=_ns(first),
        last_token_ns=_ns(last),
        end_ns=_ns(last + 0.01),
    )
    return TrialRecord(
        run_id=run_id,
        suite=SuiteName.DECODE,
        condition=condition,
        tier=tier,
        trial_index=trial_index,
        warmup=warmup,
        request_body_ref="0" * 64,
        result=StreamResult(
            timing=timing,
            usage=Usage(input_tokens=11 + trial_index, output_tokens=OUTPUT_TOKENS),
            stop_reason=None if failed else "max_tokens",
            blocks=[ContentBlock(type="text", text="本文")],
            error=RequestError(kind="stream_error", message="途中で切れた") if failed else None,
        ),
    )


def _speed_records(
    run_id: str,
    tps: Sequence[float] = TPS,
    ttft: Sequence[float] = TTFT,
    *,
    condition: str = CONDITION,
    tier: Tier = "primary",
    order: Iterable[int] | None = None,
    failed: Sequence[int] = (),
) -> list[TrialRecord]:
    """`order` を渡すと、書き足す順だけを入れ替える (値と番号の対応は変えない)。"""
    indexes = list(range(len(tps))) if order is None else list(order)
    return [
        _speed_trial(
            run_id,
            condition=condition,
            trial_index=index,
            tps=tps[index],
            ttft=ttft[index],
            tier=tier,
            failed=index in failed,
        )
        for index in indexes
    ]


def _write_run(
    tmp_path: Path,
    dir_name: str,
    run_id: str,
    records: Sequence[TrialRecord],
    *,
    finish: bool = True,
    **manifest_overrides: Any,
) -> Path:
    store = RunStore.create(tmp_path / dir_name, _manifest(run_id, **manifest_overrides))
    for record in records:
        store.append_trial(record)
    if finish:
        store.set_status(RunStatus.COMPLETED, AT_END)
    return store.run_dir


def _pair(
    tmp_path: Path,
    *,
    tps_a: Sequence[float] = TPS,
    tps_b: Sequence[float] = TPS,
    manifest_a: dict[str, Any] | None = None,
    manifest_b: dict[str, Any] | None = None,
    finish_b: bool = True,
) -> tuple[Path, Path]:
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A, tps_a), **(manifest_a or {}))
    dir_b = _write_run(
        tmp_path,
        "b",
        RUN_B,
        _speed_records(RUN_B, tps_b),
        finish=finish_b,
        **(manifest_b or {}),
    )
    return dir_a, dir_b


def _row(report: ComparisonReport, condition: str, metric: str) -> ComparisonRow:
    matched = [r for r in report.rows if r.condition == condition and r.metric == metric]
    assert len(matched) == 1, f"{condition} の {metric} の行が {len(matched)} 個ある"
    return matched[0]


# --- 同じ生データどうし (tasks.md 4.3 の完了の状態 1) ------------------------


def test_identical_raw_data_is_within_everywhere_and_paired(tmp_path: Path) -> None:
    """同じ生データの例どうしの比較で、すべての条件が「収まる」になる。"""
    dir_a, dir_b = _pair(tmp_path)
    report = compare_runs(dir_a, dir_b).report

    assert (report.run_a, report.run_b) == (RUN_A, RUN_B)
    assert {(row.condition, row.metric) for row in report.rows} == {
        (CONDITION, "decode_tps"),
        (CONDITION, "ttft_s"),
    }
    for row in report.rows:
        assert row.verdict is not None, f"{row.metric} が判定できていない"
        assert row.verdict.verdict == "within", f"{row.metric} が収まらない"
        assert row.verdict.paired is True, f"{row.metric} が対応のある比較になっていない"
        assert row.diff == pytest.approx(0.0, abs=1e-9)
    assert report.warnings == [], "設定が同じなのに警告が出ている"
    assert report.excluded == []
    assert report.repeatability is not None
    assert report.repeatability.all_within is True
    assert report.repeatability.outside == []


def test_comparing_a_run_with_itself_is_within(tmp_path: Path) -> None:
    """同じ計測ランどうしの比較は、必ず「収まる」で、対応のある比較になる。"""
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A))
    report = compare_runs(dir_a, dir_a).report

    assert report.rows
    for row in report.rows:
        assert row.verdict is not None
        assert (row.verdict.verdict, row.verdict.paired) == ("within", True)
    assert report.repeatability is not None
    assert report.repeatability.all_within is True


# --- 10% 変えた例 (tasks.md 4.3 の完了の状態 2) -------------------------------


def test_a_ten_percent_change_is_outside_and_listed_in_the_conclusion(tmp_path: Path) -> None:
    """片方の速さを 10% 変えた例で、その条件が「収まらない」になる (9.2、9.3)。"""
    dir_a, dir_b = _pair(tmp_path, tps_b=TPS_FASTER)
    report = compare_runs(dir_a, dir_b).report

    decode = _row(report, CONDITION, "decode_tps")
    assert decode.verdict is not None
    assert decode.verdict.verdict == "outside"
    assert decode.relative_diff == pytest.approx(0.1, rel=1e-6)

    ttft = _row(report, CONDITION, "ttft_s")
    assert ttft.verdict is not None
    assert ttft.verdict.verdict == "within", "変えていない値まで収まらなくなっている"

    assert report.repeatability is not None
    assert report.repeatability.all_within is False
    assert report.repeatability.outside == [f"{CONDITION} / decode_tps"]


def test_diff_and_relative_diff_come_from_the_medians_not_the_paired_statistic(
    tmp_path: Path,
) -> None:
    """`diff` と `relative_diff` は中央値どうし、`median_diff` は試行ごとの差の中央値。

    3 つがすべて違う値になる生データで確かめる (Implementation Notes 2.4: 2 つを
    同じ欄に混ぜない)。`relative_diff` を B の値で割る壊れ方も、ここで落ちる。
    """
    tps_a = (10.0, 20.0, 30.0, 40.0)  # 中央値 25.0
    tps_b = (12.0, 19.0, 35.0, 60.0)  # 中央値 27.0、試行ごとの差は 2, -1, 5, 20
    dir_a, dir_b = _pair(tmp_path, tps_a=tps_a, tps_b=tps_b)
    row = _row(compare_runs(dir_a, dir_b).report, CONDITION, "decode_tps")

    assert row.value_a == pytest.approx(25.0)
    assert row.value_b == pytest.approx(27.0)
    assert row.diff == pytest.approx(2.0)
    assert row.relative_diff == pytest.approx(2.0 / 25.0)
    assert row.relative_diff != pytest.approx(2.0 / 27.0), "B の値で割っている"
    assert row.verdict is not None
    assert row.verdict.paired is True
    assert row.verdict.median_diff == pytest.approx(3.5), "試行ごとの差の中央値になっていない"
    assert row.verdict.median_diff != pytest.approx(row.diff)


# --- 対応のある比較にする条件 (9.2、design.md analysis/stats) -----------------


def test_pairing_falls_back_to_unpaired_when_the_seed_differs(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path, manifest_b={"profile": Profile(name="test", seed=7)})
    row = _row(compare_runs(dir_a, dir_b).report, CONDITION, "decode_tps")

    assert row.verdict is not None
    assert row.verdict.paired is False, "種が違うのに対応のある比較にしている"


def test_pairing_falls_back_to_unpaired_when_the_generator_version_differs(
    tmp_path: Path,
) -> None:
    dir_a, dir_b = _pair(tmp_path, manifest_b={"generator_version": 2})
    row = _row(compare_runs(dir_a, dir_b).report, CONDITION, "decode_tps")

    assert row.verdict is not None
    assert row.verdict.paired is False, "生成器の版が違うのに対応のある比較にしている"


def test_pairing_falls_back_to_unpaired_when_the_trial_keys_differ(tmp_path: Path) -> None:
    """片方に失敗した試行があると、番号がそろわないので対応のない比較にする。"""
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A))
    dir_b = _write_run(tmp_path, "b", RUN_B, _speed_records(RUN_B, failed=(3,)))
    row = _row(compare_runs(dir_a, dir_b).report, CONDITION, "decode_tps")

    assert row.verdict is not None
    assert row.verdict.paired is False, "番号がそろわないのに対応のある比較にしている"
    assert row.verdict.n == 3, "少ないほうの組の大きさを使っていない"


def test_pairing_orders_both_sides_by_the_trial_key(tmp_path: Path) -> None:
    """書き足した順が違っても、番号で突き合わせるので判定は変わらない。

    並べ替えを外すと、試行ごとの差が取り違えられて `median_diff` が変わる。
    """
    tps_a = (10.0, 20.0, 30.0, 40.0, 50.0)
    tps_b = (11.0, 25.0, 31.0, 60.0, 52.0)
    ttft = (0.50, 0.55, 0.60, 0.65, 0.70)
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A, tps_a, ttft))
    in_order = _write_run(tmp_path, "b1", RUN_B, _speed_records(RUN_B, tps_b, ttft))
    shuffled = _write_run(
        tmp_path,
        "b2",
        RUN_B,  # 同じ識別子なので、再標本化の種も同じになる
        _speed_records(RUN_B, tps_b, ttft, order=(2, 4, 0, 3, 1)),
    )

    expected = compare_runs(dir_a, in_order).report
    actual = compare_runs(dir_a, shuffled).report
    assert [row.verdict for row in actual.rows] == [row.verdict for row in expected.rows]
    decode = _row(expected, CONDITION, "decode_tps")
    assert decode.verdict is not None
    assert decode.verdict.median_diff == pytest.approx(2.0), "試行ごとの差の中央値が違う"


# --- 値が足りない行 (9.2、10.2) ----------------------------------------------


def test_a_row_with_fewer_than_two_values_is_not_judged_and_warns(tmp_path: Path) -> None:
    """値が 2 つに満たない行は判定できない。「収まった」と読ませない (9.3)。"""
    records_a = _speed_records(RUN_A, (20.0, 21.0), (0.5, 0.55), failed=(1,))
    records_b = _speed_records(RUN_B, (20.0, 21.0), (0.5, 0.55), failed=(1,))
    dir_a = _write_run(tmp_path, "a", RUN_A, records_a)
    dir_b = _write_run(tmp_path, "b", RUN_B, records_b)
    report = compare_runs(dir_a, dir_b).report

    decode = _row(report, CONDITION, "decode_tps")
    assert decode.verdict is None
    assert decode.diff == pytest.approx(0.0), "値そのものは並べる (9.1)"

    assert any(
        CONDITION in warning and "decode_tps" in warning and "判定できない" in warning
        for warning in report.warnings
    ), f"判定できない行の警告がない: {report.warnings}"

    assert report.repeatability is not None
    assert report.repeatability.all_within is False, "判定できない行を「収まった」にしている"
    assert any("decode_tps" in item for item in report.repeatability.outside)


# --- 片方にしかない条件 (9.5) -------------------------------------------------


def test_conditions_present_in_only_one_run_are_excluded_and_listed(tmp_path: Path) -> None:
    dir_a = _write_run(
        tmp_path,
        "a",
        RUN_A,
        [*_speed_records(RUN_A), *_speed_records(RUN_A, condition="decode/prose/ja")],
    )
    dir_b = _write_run(
        tmp_path,
        "b",
        RUN_B,
        [*_speed_records(RUN_B), *_speed_records(RUN_B, condition="decode/code/ja")],
    )
    report = compare_runs(dir_a, dir_b).report

    assert {(row.condition, row.metric) for row in report.rows} == {
        (CONDITION, "decode_tps"),
        (CONDITION, "ttft_s"),
    }
    assert report.excluded == [
        "decode/code/ja / decode_tps (B のみ)",
        "decode/code/ja / ttft_s (B のみ)",
        "decode/prose/ja / decode_tps (A のみ)",
        "decode/prose/ja / ttft_s (A のみ)",
    ]
    assert report.excluded == sorted(report.excluded)


def test_no_common_condition_leaves_the_rows_empty(tmp_path: Path) -> None:
    """共通する条件がなければ、行は空で、結論は「収まった」にしない。"""
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A, condition="decode/code/en"))
    dir_b = _write_run(tmp_path, "b", RUN_B, _speed_records(RUN_B, condition="decode/prose/ja"))
    report = compare_runs(dir_a, dir_b).report

    assert report.rows == []
    assert len(report.excluded) == 4
    assert report.repeatability is not None
    assert report.repeatability.all_within is False
    assert report.repeatability.outside, "理由を書かずに「収まらない」とだけ言っている"


# --- 主な結果と参考 (4.4、9.3) -----------------------------------------------


def _tiered_pair(tmp_path: Path) -> tuple[Path, Path]:
    """主な結果は同じ、参考だけ 10% 違う 2 つの計測ラン。"""
    dir_a = _write_run(
        tmp_path,
        "a",
        RUN_A,
        [
            *_speed_records(RUN_A),
            *_speed_records(RUN_A, condition="concurrency/c8", tier="reference"),
        ],
    )
    dir_b = _write_run(
        tmp_path,
        "b",
        RUN_B,
        [
            *_speed_records(RUN_B),
            *_speed_records(RUN_B, TPS_FASTER, condition="concurrency/c8", tier="reference"),
        ],
    )
    return dir_a, dir_b


def test_reference_tier_rows_do_not_decide_the_conclusion(tmp_path: Path) -> None:
    """参考の条件が収まらなくても、結論 (9.3) は主な結果だけで決める (4.4)。"""
    dir_a, dir_b = _tiered_pair(tmp_path)
    report = compare_runs(dir_a, dir_b).report

    reference = _row(report, "concurrency/c8", "decode_tps")
    assert reference.tier == "reference"
    assert reference.verdict is not None
    assert reference.verdict.verdict == "outside"

    assert report.repeatability is not None
    assert report.repeatability.all_within is True, "参考の条件を結論に数えている"
    assert report.repeatability.outside == []


def test_repeatability_is_none_when_the_target_names_differ(tmp_path: Path) -> None:
    """対象サーバーの定義が違う比較では、繰り返しの結論を出さない (9.3)。"""
    dir_a, dir_b = _pair(tmp_path, manifest_b={"target_name": "candidate-d"})
    report = compare_runs(dir_a, dir_b).report

    assert report.repeatability is None
    assert any("candidate-d" in warning and "fake" in warning for warning in report.warnings)


# --- 警告 (9.4、10.5) --------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides_b", "needles"),
    [
        pytest.param(
            {"profile": Profile(name="test", sampling=Sampling(temperature=0.7))},
            ("temperature", "0.0", "0.7"),
            id="sampling",
        ),
        pytest.param(
            {"profile": Profile(name="test", decode=DecodeSettings(max_tokens=512))},
            ("decode.max_tokens", "1024", "512"),
            id="max_tokens",
        ),
        pytest.param(
            {"profile": Profile(name="test", decode=DecodeSettings(trials=12))},
            ("decode.trials", "10", "12"),
            id="trials",
        ),
        pytest.param(
            {"profile": Profile(name="test", decode=DecodeSettings(warmup_trials=5))},
            ("decode.warmup_trials", "2", "5"),
            id="warmup_trials",
        ),
        pytest.param(
            {"harness_version": "0.1.0+gother"},
            ("道具の版", "0.1.0+gtest", "0.1.0+gother"),
            id="harness_version",
        ),
        pytest.param(
            {"generator_version": 2},
            ("生成器の版", "1", "2"),
            id="generator_version",
        ),
        pytest.param(
            {"profile_name": "full"},
            ("設定の名前", "test", "full"),
            id="profile_name",
        ),
        pytest.param(
            {"profile": Profile(name="test", seed=7)},
            ("seed", "0", "7"),
            id="seed",
        ),
    ],
)
def test_setting_differences_warn_naming_both_values(
    tmp_path: Path, overrides_b: dict[str, Any], needles: tuple[str, ...]
) -> None:
    """設定の違う例で、警告が先頭に出る (tasks.md 4.3 の完了の状態 3)。"""
    dir_a, dir_b = _pair(tmp_path, manifest_b=overrides_b)
    report = compare_runs(dir_a, dir_b).report

    matched = [
        warning for warning in report.warnings if all(needle in warning for needle in needles)
    ]
    assert len(matched) == 1, f"{needles} の警告がない、または重なっている: {report.warnings}"


def test_an_incomplete_run_warns(tmp_path: Path) -> None:
    """未完了の計測ランを比較に使うと、先頭で警告する (10.5)。"""
    dir_a, dir_b = _pair(tmp_path, finish_b=False)
    report = compare_runs(dir_a, dir_b).report

    matched = [
        warning
        for warning in report.warnings
        if "未完了" in warning and RUN_B in warning and "running" in warning
    ]
    assert len(matched) == 1, f"未完了の警告がない: {report.warnings}"
    assert not any(RUN_A in warning and "未完了" in warning for warning in report.warnings)


def test_a_dirty_harness_version_warns_even_when_both_are_equal(tmp_path: Path) -> None:
    """未コミットの変更で測った計測ランは、版が同じでも注意を促す (3.5 の申し送り)。"""
    dirty = {"harness_version": "0.1.0+gtest.dirty"}
    dir_a, dir_b = _pair(tmp_path, manifest_a=dirty, manifest_b=dirty)
    report = compare_runs(dir_a, dir_b).report

    assert [warning for warning in report.warnings if ".dirty" in warning]


def test_an_unknown_harness_version_warns(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path, manifest_b={"harness_version": "0.1.0+unknown"})
    report = compare_runs(dir_a, dir_b).report

    assert [warning for warning in report.warnings if "unknown" in warning]


def test_server_and_target_differences_are_reported_as_information(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(
        tmp_path,
        manifest_b={"server_model": "glm-5.3-flash-w8a8", "server_version": "0.0.1-test"},
    )
    report = compare_runs(dir_a, dir_b).report

    assert any("glm-5.3-flash-w8a8" in warning for warning in report.warnings)
    assert any("0.0.1-test" in warning for warning in report.warnings)


def test_read_warnings_from_either_run_are_reported(tmp_path: Path) -> None:
    """途中で切れた行を捨てたことを、比較の警告にも出す (Implementation Notes 2.3)。"""
    dir_a, dir_b = _pair(tmp_path)
    trials = dir_b / "trials.jsonl"
    data = trials.read_bytes()
    trials.write_bytes(data[: -len(data) // 8])  # 最後の行を途中で切る

    result = compare_runs(dir_a, dir_b)
    assert result.read_warnings_b, "B の読み取りの警告が返っていない"
    assert result.read_warnings_a == []
    assert any(
        RUN_B in warning and any(part in warning for part in result.read_warnings_b)
        for warning in result.report.warnings
    ), f"読み取りの警告が比較の警告に出ていない: {result.report.warnings}"


def test_a_different_compare_tolerance_warns_and_uses_run_a(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(
        tmp_path,
        tps_b=TPS_FASTER,
        manifest_b={"profile": Profile(name="test", compare_tolerance=0.5)},
    )
    report = compare_runs(dir_a, dir_b).report

    assert any("許容の幅" in warning and "0.02" in warning for warning in report.warnings)
    row = _row(report, CONDITION, "decode_tps")
    assert row.verdict is not None
    assert row.verdict.tolerance == pytest.approx(0.02), "A の許容の幅を使っていない"
    assert row.verdict.verdict == "outside"


def test_an_explicit_tolerance_overrides_the_profile(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path, tps_b=TPS_FASTER)
    report = compare_runs(dir_a, dir_b, tolerance=0.5).report

    row = _row(report, CONDITION, "decode_tps")
    assert row.verdict is not None
    assert row.verdict.tolerance == pytest.approx(0.5)
    assert row.verdict.verdict == "within", "10% の差が、許容の幅 50% に収まっていない"


# --- 割合の行 (9.6) ----------------------------------------------------------


def _proportion_summary(
    run_id: str,
    stat: ProportionStat | None,
    *,
    condition: str = "quality/toolcall",
    metric: str = "accuracy",
) -> Summary:
    """割合の行だけを持つ要約を、直に組み立てる (4.2 の出来上がりを待たない)。"""
    return Summary(
        conditions=_manifest(run_id, status=RunStatus.COMPLETED, finished_at=AT_END),
        incomplete=False,
        results=[MetricResult(condition=condition, metric=metric, tier="primary", proportion=stat)],
    )


def test_disjoint_proportion_intervals_are_different() -> None:
    report = compare_summaries(
        _proportion_summary(RUN_A, binomial_interval(0, 50)),
        _proportion_summary(RUN_B, binomial_interval(30, 50)),
    )
    row = _row(report, "quality/toolcall", "accuracy")

    assert row.proportion_verdict == "different"
    assert row.verdict is None, "割合の行に、連続の値の判定を付けている"
    assert row.value_a == pytest.approx(0.0)
    assert row.value_b == pytest.approx(0.6)
    assert row.diff == pytest.approx(0.6)
    assert row.relative_diff is None, "0 で割っている"
    assert report.repeatability is not None
    assert report.repeatability.outside == ["quality/toolcall / accuracy"]


def test_overlapping_proportion_intervals_are_not_distinguishable() -> None:
    report = compare_summaries(
        _proportion_summary(RUN_A, binomial_interval(25, 50)),
        _proportion_summary(RUN_B, binomial_interval(27, 50)),
    )
    row = _row(report, "quality/toolcall", "accuracy")

    assert row.proportion_verdict == "not_distinguishable"
    assert row.relative_diff == pytest.approx((27 - 25) / 25)
    assert report.repeatability is not None
    assert report.repeatability.all_within is True


def test_a_missing_proportion_on_one_side_is_not_judged() -> None:
    report = compare_summaries(
        _proportion_summary(RUN_A, binomial_interval(25, 50)),
        _proportion_summary(RUN_B, None),
    )
    row = _row(report, "quality/toolcall", "accuracy")

    assert row.proportion_verdict is None
    assert row.verdict is None
    assert row.value_b is None
    assert row.diff is None
    assert report.repeatability is not None
    assert report.repeatability.all_within is False


def test_continuous_rows_without_values_are_not_judged() -> None:
    """要約だけを渡した比較では、試行ごとの値がないので判定できない。"""
    from bench_harness.analysis.stats import describe

    def summary(run_id: str, values: Sequence[float]) -> Summary:
        return Summary(
            conditions=_manifest(run_id, status=RunStatus.COMPLETED, finished_at=AT_END),
            incomplete=False,
            results=[
                MetricResult(
                    condition=CONDITION,
                    metric="decode_tps",
                    tier="primary",
                    continuous=describe(values),
                )
            ],
        )

    report = compare_summaries(summary(RUN_A, [10.0, 20.0]), summary(RUN_B, [11.0, 21.0]))
    row = _row(report, CONDITION, "decode_tps")

    assert row.verdict is None
    assert row.value_a == pytest.approx(15.0)
    assert row.diff == pytest.approx(1.0)


# --- 何度実行しても同じ結果 (design.md analysis/compare) ----------------------


def test_the_rendered_output_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path, tps_b=TPS_FASTER)
    first = compare_runs(dir_a, dir_b)
    second = compare_runs(dir_a, dir_b)

    assert render_comparison_markdown(
        first.report, proportions=first.proportions
    ) == render_comparison_markdown(second.report, proportions=second.proportions)
    assert render_comparison_json(first.report) == render_comparison_json(second.report)
    assert first.report == second.report


_HASHSEED_SCRIPT = """
import sys
from pathlib import Path

from bench_harness.analysis.compare import (
    compare_runs,
    render_comparison_json,
    render_comparison_markdown,
)

result = compare_runs(Path(sys.argv[1]), Path(sys.argv[2]))
sys.stdout.write(render_comparison_markdown(result.report, proportions=result.proportions))
sys.stdout.write(render_comparison_json(result.report))
"""


def _run_script(dir_a: Path, dir_b: Path, hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, "-c", _HASHSEED_SCRIPT, str(dir_a), str(dir_b)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout


def test_the_output_does_not_depend_on_pythonhashseed(tmp_path: Path) -> None:
    """辞書や集合の巡り方に依存していないことを、別のプロセスで確かめる。"""
    dir_a, dir_b = _pair(tmp_path, tps_b=TPS_FASTER)

    assert _run_script(dir_a, dir_b, "0") == _run_script(dir_a, dir_b, "12345")


def test_the_output_has_no_nan(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path, tps_b=TPS_FASTER)
    result = compare_runs(dir_a, dir_b)
    text = render_comparison_markdown(result.report, proportions=result.proportions)

    assert "nan" not in text.lower()
    assert "inf" not in text.lower()
    for row in result.report.rows:
        for value in (row.value_a, row.value_b, row.diff, row.relative_diff):
            assert value is None or (value == value and abs(value) != float("inf"))


# --- 人が読む比較の結果 ------------------------------------------------------


def test_markdown_puts_the_warnings_before_any_table(tmp_path: Path) -> None:
    """警告は、どの表よりも先に出す (9.4、10.5)。"""
    dir_a, dir_b = _pair(tmp_path, finish_b=False, manifest_b={"generator_version": 2})
    result = compare_runs(dir_a, dir_b)
    lines = render_comparison_markdown(result.report, proportions=result.proportions).splitlines()

    warning_lines = [i for i, line in enumerate(lines) if "未完了" in line]
    table_lines = [i for i, line in enumerate(lines) if line.startswith("|")]
    assert warning_lines, "未完了の警告が出ていない"
    assert table_lines, "表が出ていない"
    assert max(warning_lines) < min(table_lines), "警告が表より後ろに出ている"


def test_markdown_separates_primary_from_reference(tmp_path: Path) -> None:
    dir_a, dir_b = _tiered_pair(tmp_path)
    result = compare_runs(dir_a, dir_b)
    text = render_comparison_markdown(result.report, proportions=result.proportions)

    primary_at = text.index("## 主な結果")
    reference_at = text.index("## 参考")
    assert primary_at < reference_at
    assert primary_at < text.index(CONDITION) < reference_at
    assert reference_at < text.index("concurrency/c8")


def test_markdown_shows_the_judgement_the_pairing_and_the_bootstrap_columns(
    tmp_path: Path,
) -> None:
    dir_a, dir_b = _tiered_pair(tmp_path)
    result = compare_runs(dir_a, dir_b)
    text = render_comparison_markdown(result.report, proportions=result.proportions)

    assert "収まる" in text
    assert "収まらない" in text
    assert "対応あり" in text
    assert "再標本化" in text, "再標本化の統計量の欄がない"
    assert "95% 区間" in text
    assert "許容の幅" in text
    assert "読み方" in text
    assert "20 回以上" in text, "試行の回数を増やす助言がない"


def test_markdown_explains_how_to_read_the_comparison(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path)
    result = compare_runs(dir_a, dir_b)
    text = render_comparison_markdown(result.report, proportions=result.proportions)
    reading = text[text.index("## 読み方") :]

    assert "対応" in reading
    assert "許容の幅" in reading
    assert "区間" in reading
    assert "保守的" in reading, "割合の判定が保守的であることを書いていない"


def test_markdown_shows_the_proportion_rows_with_their_counts() -> None:
    report = compare_summaries(
        _proportion_summary(RUN_A, binomial_interval(10, 50)),
        _proportion_summary(RUN_B, binomial_interval(40, 50)),
    )
    text = render_comparison_markdown(
        report,
        proportions={
            ("quality/toolcall", "accuracy"): (
                binomial_interval(10, 50),
                binomial_interval(40, 50),
            )
        },
    )

    assert "差が意味あり" in text
    assert "10/50" in text
    assert "40/50" in text


def test_markdown_states_the_repeatability_conclusion(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path)
    result = compare_runs(dir_a, dir_b)
    text = render_comparison_markdown(result.report, proportions=result.proportions)

    assert "すべて" in text and "収まった" in text


def test_write_comparison_writes_the_markdown(tmp_path: Path) -> None:
    dir_a, dir_b = _pair(tmp_path)
    result = compare_runs(dir_a, dir_b)
    out_path = write_comparison(result.report, tmp_path / "compare.md")

    assert out_path.read_text(encoding="utf-8") == render_comparison_markdown(result.report)
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".compare")] == []


# --- 壊れた入力 --------------------------------------------------------------


def test_a_directory_that_is_not_a_run_raises_store_error(tmp_path: Path) -> None:
    dir_a = _write_run(tmp_path, "a", RUN_A, _speed_records(RUN_A))
    not_a_run = tmp_path / "empty"
    not_a_run.mkdir()

    with pytest.raises(StoreError):
        compare_runs(dir_a, not_a_run)
    with pytest.raises(StoreError):
        compare_runs(not_a_run, dir_a)


# --- 依存の向き (design.md) -------------------------------------------------


def test_compare_imports_nothing_outside_the_allowed_set() -> None:
    """`analysis` は `types`、`store`、`analysis.stats`、`analysis.summarize` だけ。"""
    source = Path(compare_module.__file__ or "")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対 import は使わない (依存の向きを見えなくする)"
            assert node.module is not None
            modules.add(node.module)

    allowed_internal = {
        "bench_harness.types",
        "bench_harness.store",
        "bench_harness.store.rawstore",
        "bench_harness.analysis.stats",
        "bench_harness.analysis.summarize",
    }
    for module in sorted(modules):
        root = module.split(".")[0]
        if root == "bench_harness":
            assert module in allowed_internal, f"読み込んではいけない module: {module}"
        else:
            assert root in sys.stdlib_module_names or root == "pydantic", (
                f"標準ライブラリと pydantic 以外を読み込んでいる: {module}"
            )
