"""思考のみのdecode試行を本文速度の成功と区別する。"""

from __future__ import annotations

from pathlib import Path

from bench_harness import cli
from bench_harness.store import list_run_dirs
from bench_harness.types import Summary
from fake_server import FakeServer, Script, UsageSpec, thinking_events


def test_thinking_only_is_not_text_speed_success(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.set_response(
        Script(
            events=thinking_events("思考だけ", chunks=("思考", "だけ")),
            gap_s=0.001,
            stop_reason="max_tokens",
            usage=UsageSpec(output_tokens=256),
        )
    )
    targets = tmp_path / "targets.toml"
    targets.write_text(
        f'[targets.fake]\nbase_url = "{fake_server.base_url}"\nmodel = "fake-model"\n',
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        "[profiles.test]\nmin_successes = 1\n"
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
    summary = Summary.model_validate_json((runs[0] / "summary.json").read_text(encoding="utf-8"))
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
        assert summary.output_phases[condition]["text_speed_n"] == 0
    assert "本文未到達" in (runs[0] / "summary.json").read_text(encoding="utf-8")
    assert "本文未到達" in (runs[0] / "summary.md").read_text(encoding="utf-8")
