"""思考のみのdecode試行を本文速度の成功と区別する。"""

from __future__ import annotations

from pathlib import Path

from bench_harness import cli
from bench_harness.store import list_run_dirs
from bench_harness.types import Summary
from fake_server import (
    FakeServer,
    Script,
    UsageSpec,
    text_response,
    thinking_events,
    thinking_response,
)


def run_decode(base_url: str, tmp_path: Path, *, thinking: str = "server_default") -> Path:
    """偽のサーバーに対して decode を 10 試行流し、できた計測ランのディレクトリを返す。"""
    targets = tmp_path / "targets.toml"
    targets.write_text(
        f'[targets.fake]\nbase_url = "{base_url}"\nmodel = "fake-model"\n',
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        "[profiles.test]\nmin_successes = 1\n"
        "[profiles.test.sampling]\n"
        f'thinking = "{thinking}"\n'
        "[profiles.test.decode]\ntrials = 10\nwarmup_trials = 0\nmax_tokens = 256\n",
        encoding="utf-8",
    )
    root = tmp_path / "results"
    code = cli.main(
        [
            "run",
            "--target",
            "fake",
            "--profile",
            "test",
            "--suite",
            "decode",
            "--targets",
            str(targets),
            "--profiles",
            str(profiles),
            "--results-root",
            str(root),
        ]
    )
    assert code == 0
    runs = list_run_dirs(root)
    assert len(runs) == 1
    return runs[0]


def test_thinking_only_is_not_text_speed_success(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.set_response(
        Script(
            events=thinking_events("思考だけ", chunks=("思考", "だけ")),
            gap_s=0.001,
            stop_reason="max_tokens",
            usage=UsageSpec(output_tokens=256),
        )
    )
    run_dir = run_decode(fake_server.base_url, tmp_path)
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    for condition in ("decode/json/en", "decode/json/ja"):
        total = next(
            row
            for row in summary.results
            if row.condition == condition and row.metric == "decode_tps"
        )
        text_speed = next(
            row
            for row in summary.results
            if row.condition == condition and row.metric == "text_chars_per_s"
        )
        assert total.continuous is not None and total.continuous.n == 10
        assert text_speed.continuous is None
        assert summary.output_phases[condition]["request_successes"] == 10
        assert summary.output_phases[condition]["thinking_only"] == 10
        assert summary.output_phases[condition]["thinking_observed"] == 10
        assert summary.output_phases[condition]["text_speed_n"] == 0
    assert "本文未到達" in (run_dir / "summary.json").read_text(encoding="utf-8")
    assert "本文未到達" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "思考あり" in (run_dir / "summary.md").read_text(encoding="utf-8")


def test_thinking_with_an_answer_is_counted_as_thinking_observed_not_thinking_only(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """思考のあとに本文が届いた試行は、本文への到達として数え、二重に「思考のみ」へ入れない

    (issue #130)。
    """
    fake_server.set_response(thinking_response("考え中", answer="答え"))
    run_dir = run_decode(fake_server.base_url, tmp_path)
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    for condition in ("decode/json/en", "decode/json/ja"):
        counts = summary.output_phases[condition]
        assert counts["text_arrived"] == 10
        assert counts["thinking_observed"] == 10
        assert counts["thinking_only"] == 0


# --- issue #131: thinking = "off" なのに思考が出た試行の警告 -----------------


def test_off_thinking_with_thinking_in_the_response_warns_with_a_count(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """`thinking = "off"` を指定したのに思考が出た試行の件数が、条件ごとに警告として
    `summary.json` と `summary.md` に出る (issue #131 のコメント。#130 の
    `thinking_observed` を使う)。"""
    fake_server.set_response(thinking_response("考え中", answer="答え"))
    run_dir = run_decode(fake_server.base_url, tmp_path, thinking="off")
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))

    md = (run_dir / "summary.md").read_text(encoding="utf-8")
    for condition in ("decode/json/en", "decode/json/ja"):
        assert summary.output_phases[condition]["thinking_observed"] == 10
        warnings = [w for w in summary.phase_warnings if w.startswith(f"{condition}:")]
        assert warnings, f"{condition} に警告が付いていない"
        assert all("本文未到達" not in w and "段階別計測情報が不明" not in w for w in warnings), (
            warnings
        )
        assert any("10" in w for w in warnings), warnings
        for warning in warnings:
            assert warning in md


def test_off_thinking_with_a_text_only_response_has_no_thinking_warning(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """`thinking = "off"` でも、実際に思考が出なかった試行には、思考の警告が付かない。"""
    fake_server.set_response(text_response("ok"))
    run_dir = run_decode(fake_server.base_url, tmp_path, thinking="off")
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))

    for condition in ("decode/json/en", "decode/json/ja"):
        assert summary.output_phases[condition]["thinking_observed"] == 0
    assert summary.phase_warnings == []


def test_server_default_thinking_with_thinking_in_the_response_has_no_off_warning(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """`server_default` では、思考が出ても「off なのに思考が出た」警告は付かない
    (`thinking_observed` の数そのものは、既存の別の試験がすでに固定している)。"""
    fake_server.set_response(thinking_response("考え中", answer="答え"))
    run_dir = run_decode(fake_server.base_url, tmp_path, thinking="server_default")
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))

    for condition in ("decode/json/en", "decode/json/ja"):
        assert summary.output_phases[condition]["thinking_observed"] == 10
    assert summary.phase_warnings == []
