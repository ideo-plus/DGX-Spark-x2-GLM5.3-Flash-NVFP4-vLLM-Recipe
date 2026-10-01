"""`config.py` (対象サーバーの定義と計測の設定の読み込み) の試験。

確かめること (タスク 1.3 の完了の状態):

- 知らない対象サーバーの名前が、名前つきのエラーになる (使える名前の一覧つき)
- 下限を割った値 (`decode.trials` など) が、項目の名前つき (ドット区切りの経路) のエラーになる
- 知らない項目の名前が、その項目の経路つきのエラーになる
- 認証の情報の値が、`repr`、`str`、書き出した JSON、エラーのメッセージのどこにも現れない (1.8)
- 環境変数がない、または空のときは、変数の名前つきのエラーになる
- 生データの置き場所が git の管理の対象 (追跡されている、無視されていない) だと失敗する (8.5)
- `rev-parse --is-inside-work-tree` が「リポジトリの外」以外の理由で失敗しても、
  管理の対象の場所を黙って許可しない (レビュー指摘 1)
- git の実行ファイルが見つからないときも `ConfigError` になる (レビュー指摘 2)
- 上書き (`apply_overrides`) は検証をやり直す。下限を割る上書きは弾かれる
- 実物の `targets.toml`、`profiles.toml` が読み込め、`candidate-d`、`quick`、`full`、
  `probe`、`fast` を含む (issue #39)
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from bench_harness import config as c
from bench_harness import types as t

# --- 対象サーバーの名前 ---------------------------------------------------


def test_unknown_target_name_lists_available_names(tmp_path: Path) -> None:
    targets_path = tmp_path / "targets.toml"
    targets_path.write_text(
        '[targets.alpha]\nbase_url = "http://localhost:8000"\nmodel = "m"\n\n'
        '[targets.beta]\nbase_url = "http://localhost:8001"\nmodel = "m"\n',
        encoding="utf-8",
    )

    with pytest.raises(c.ConfigError) as exc_info:
        c.select_target("gamma", targets_path)

    message = str(exc_info.value)
    assert "gamma" in message
    assert "alpha" in message
    assert "beta" in message


def test_target_name_comes_from_the_table_key(tmp_path: Path) -> None:
    targets_path = tmp_path / "targets.toml"
    targets_path.write_text(
        '[targets.alpha]\nbase_url = "http://localhost:8000"\nmodel = "m"\n',
        encoding="utf-8",
    )

    target = c.select_target("alpha", targets_path)

    assert target.name == "alpha"


# --- 下限違反と知らない項目 -------------------------------------------------


def test_profile_below_minimum_decode_trials_names_the_dotted_path(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "[profiles.tiny]\n\n[profiles.tiny.decode]\ntrials = 5\n",
        encoding="utf-8",
    )

    with pytest.raises(c.ConfigError) as exc_info:
        c.load_profiles(profiles_path)

    assert "profiles.tiny.decode.trials" in str(exc_info.value)


def test_unknown_target_key_is_named_in_the_error(tmp_path: Path) -> None:
    targets_path = tmp_path / "targets.toml"
    targets_path.write_text(
        '[targets.foo]\nbase_url = "http://localhost:8000"\nmodel = "m"\nbogus_field = 1\n',
        encoding="utf-8",
    )

    with pytest.raises(c.ConfigError) as exc_info:
        c.load_targets(targets_path)

    assert "targets.foo.bogus_field" in str(exc_info.value)


def test_unknown_profile_key_is_named_in_the_error(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "[profiles.tiny]\n\n[profiles.tiny.decode]\nbogus_field = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(c.ConfigError) as exc_info:
        c.load_profiles(profiles_path)

    assert "profiles.tiny.decode.bogus_field" in str(exc_info.value)


def test_unknown_profile_name_lists_available_names(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text("[profiles.quick]\n\n[profiles.full]\n", encoding="utf-8")

    with pytest.raises(c.ConfigError) as exc_info:
        c.select_profile("turbo", profiles_path)

    message = str(exc_info.value)
    assert "turbo" in message
    assert "quick" in message
    assert "full" in message


# --- 認証の情報 (1.8) -------------------------------------------------------

SENTINEL = "sentinel-9f3c7a1e-do-not-leak"  # 探しやすい、実在しない値


def _target_with_env(api_key_env: str) -> t.TargetDef:
    return t.TargetDef(
        name="alpha",
        base_url=HttpUrl("http://localhost:8000"),
        model="m",
        api_key_env=api_key_env,
    )


def test_resolve_api_key_returns_none_without_api_key_env() -> None:
    target = t.TargetDef(name="alpha", base_url=HttpUrl("http://localhost:8000"), model="m")

    assert c.resolve_api_key(target) is None


def test_resolve_api_key_hides_the_value_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_TEST_API_KEY", SENTINEL)
    target = _target_with_env("BENCH_TEST_API_KEY")

    secret = c.resolve_api_key(target)

    assert secret is not None
    assert secret.get_secret_value() == SENTINEL
    assert SENTINEL not in repr(secret)
    assert SENTINEL not in str(secret)
    assert SENTINEL not in target.model_dump_json()
    assert SENTINEL not in repr(target)


def test_resolve_api_key_missing_env_var_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCH_TEST_MISSING_KEY", raising=False)
    target = _target_with_env("BENCH_TEST_MISSING_KEY")

    with pytest.raises(c.ConfigError) as exc_info:
        c.resolve_api_key(target)

    assert "BENCH_TEST_MISSING_KEY" in str(exc_info.value)
    assert SENTINEL not in str(exc_info.value)


def test_resolve_api_key_empty_env_var_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCH_TEST_EMPTY_KEY", "")
    target = _target_with_env("BENCH_TEST_EMPTY_KEY")

    with pytest.raises(c.ConfigError):
        c.resolve_api_key(target)


# --- 生データの置き場所 (8.5) ------------------------------------------------


def _init_git_repo(root: Path, *, gitignore: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if gitignore is not None:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")


def test_results_root_outside_any_repo_is_accepted(tmp_path: Path) -> None:
    outside = tmp_path / "no-git-here" / "results"

    resolved = c.resolve_results_root(outside)

    assert resolved == outside
    assert not outside.exists()  # ディレクトリは作らない


def test_results_root_ignored_inside_a_repo_is_accepted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="results/\n")

    resolved = c.resolve_results_root(repo / "results")

    assert resolved == repo / "results"


def test_results_root_tracked_inside_a_repo_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="")

    with pytest.raises(c.ConfigError) as exc_info:
        c.resolve_results_root(repo / "results")

    assert "results" in str(exc_info.value)


def test_results_root_never_creates_the_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="results/\n")
    target = repo / "results"

    c.resolve_results_root(target)

    assert not target.exists()


# --- レビュー指摘 1: git の判定そのものの失敗を「リポジトリの外」と混同しない ----


def _write_fake_git(bin_dir: Path, real_git: str, guard_body: str) -> None:
    """`guard_body` に当てはまるときだけ壊れた応答を返し、それ以外は本物の git に
    委譲する、偽の `git` 実行ファイルを作る (PATH の先頭に置いて使う)。"""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_git = bin_dir / "git"
    fake_git.write_text(f'#!/bin/sh\n{guard_body}\nexec {real_git!r} "$@"\n', encoding="utf-8")
    mode = fake_git.stat().st_mode
    fake_git.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


REV_PARSE_GUARD = (
    'if [ "$1" = "rev-parse" ] && [ "$2" = "--is-inside-work-tree" ]; then\n'
    '  echo "fatal: index file corrupt (simulated failure)" >&2\n'
    "  exit 128\n"
    "fi"
)


def test_results_root_raises_when_the_work_tree_check_fails_for_an_unrelated_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rev-parse --is-inside-work-tree` が「リポジトリの外」以外の理由 (壊れた .git、
    権限など) で失敗しても、実際にコミットされている場所を黙って許可してはならない。"""
    real_git = shutil.which("git")
    assert real_git is not None, "この試験には本物の git が要る"

    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="")
    tracked_dir = repo / "results"
    tracked_dir.mkdir()
    (tracked_dir / "dummy.txt").write_text("dummy\n", encoding="utf-8")
    subprocess.run(["git", "add", "results/dummy.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-q",
            "-m",
            "add dummy",
        ],
        cwd=repo,
        check=True,
    )

    fake_bin = tmp_path / "fake-bin"
    _write_fake_git(fake_bin, real_git, REV_PARSE_GUARD)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(c.ConfigError):
        c.resolve_results_root(tracked_dir)


def test_results_root_outside_any_repo_still_accepted_when_git_check_is_hardened(
    tmp_path: Path,
) -> None:
    """指摘 1 の直しのあとも、本物にリポジトリの外にあるパスは引き続き許可される。"""
    outside = tmp_path / "still-outside" / "results"

    resolved = c.resolve_results_root(outside)

    assert resolved == outside


# --- レビュー指摘 2: git そのものが見つからないとき ---------------------------


def test_results_root_raises_config_error_when_git_is_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(c.ConfigError) as exc_info:
        c.resolve_results_root(tmp_path / "somewhere" / "results")

    assert not isinstance(exc_info.value, FileNotFoundError)


def test_default_results_root_uses_the_git_toplevel_of_this_repo() -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = Path(proc.stdout.strip()) / "results"

    assert c.default_results_root() == expected


def test_default_results_root_is_not_tracked() -> None:
    # このリポジトリの .gitignore に results/ が入っている前提の確認 (8.5)
    resolved = c.resolve_results_root()

    assert resolved == c.default_results_root()


# --- 上書きの検証し直し ------------------------------------------------------


def test_apply_overrides_revalidates_and_rejects_too_few_decode_trials() -> None:
    profile = t.Profile(name="quick")

    with pytest.raises(c.ConfigError) as exc_info:
        c.apply_overrides(profile, {"decode.trials": 3})

    assert "profiles.quick.decode.trials" in str(exc_info.value)


def test_apply_overrides_accepts_a_valid_change() -> None:
    profile = t.Profile(name="quick")

    updated = c.apply_overrides(profile, {"decode.trials": 15})

    assert updated.decode.trials == 15
    assert profile.decode.trials == 10  # 元の値は変わらない (凍結)


def test_apply_overrides_with_no_changes_returns_an_equal_profile() -> None:
    profile = t.Profile(name="quick")

    updated = c.apply_overrides(profile, {})

    assert updated == profile


def test_apply_overrides_rejects_an_unknown_path() -> None:
    profile = t.Profile(name="quick")

    with pytest.raises(c.ConfigError):
        c.apply_overrides(profile, {"decode.not_a_real_field": 1})


# --- 実物の設定ファイル -----------------------------------------------------


def test_default_paths_point_at_existing_files() -> None:
    assert c.default_targets_path().name == "targets.toml"
    assert c.default_profiles_path().name == "profiles.toml"
    assert c.default_targets_path().is_file()
    assert c.default_profiles_path().is_file()


def test_real_targets_file_loads_and_has_the_comparison_baseline() -> None:
    targets = c.load_targets()

    assert "candidate-d" in targets
    candidate = targets["candidate-d"]
    assert str(candidate.base_url) == "http://10.0.1.60:8001/"
    assert candidate.model == "glm-5.3-flash"
    assert candidate.max_context_tokens == 1_000_000


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_real_targets_include_every_mtp_variant(n: int) -> None:
    """MTP の対象が、起動側と同じ名前で、full と同じ宛先・モデル・文脈長で実在する (C7)。"""
    targets = c.load_targets()

    name = f"p2-nope-tp2-mtp{n}"
    assert name in targets
    target = targets[name]
    assert str(target.base_url) == "http://10.0.1.60:8000/"
    assert target.model == "glm-5-3-flash"
    assert target.max_context_tokens == 163840


def test_real_profiles_file_loads_quick_and_full() -> None:
    profiles = c.load_profiles()

    assert {"quick", "full"} <= set(profiles)
    quick, full = profiles["quick"], profiles["full"]

    assert quick.decode.trials == 10
    assert full.decode.trials == 20
    assert quick.agent.trials_per_stage == 50
    assert full.agent.trials_per_stage == 300
    assert quick.quality.code_problem_limit == 40
    assert full.quality.code_problem_limit is None


@pytest.mark.parametrize("value", ["on", "off", "enabled"])
def test_thinking_can_only_be_the_server_default(tmp_path: Path, value: str) -> None:
    """thinking は、対象サーバーの既定しか選べない (task 8.4)。

    2026-09-20 に実機で、`/v1/messages` からは切り替えられないと確かめた。効かない選択肢を
    受け付けると、「設定では切り替えたのに、実際には変わっていない」計測が黙って残る。
    """
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        f'[profiles.tiny]\n\n[profiles.tiny.sampling]\nthinking = "{value}"\n', encoding="utf-8"
    )

    with pytest.raises(c.ConfigError) as exc_info:
        c.load_profiles(profiles_path)

    assert "profiles.tiny.sampling.thinking" in str(exc_info.value)


def test_the_shipped_profiles_use_the_server_default_thinking() -> None:
    profiles = c.load_profiles()
    assert {profile.sampling.thinking for profile in profiles.values()} == {"server_default"}


# --- 計測の 3 段 (issue #39) -------------------------------------------------


def test_real_profiles_include_probe_derived_from_quick() -> None:
    """probe は quick を元に decode と quality だけを縮めた探りの設定 (issue #39)。"""
    profiles = c.load_profiles()
    probe, quick = profiles["probe"], profiles["quick"]

    assert c.select_profile("probe").name == "probe"
    assert probe.decode.model_dump() == {"trials": 10, "warmup_trials": 1, "max_tokens": 128}
    assert probe.quality.model_dump() == {
        "toolcall_tasks": 5,
        "needle_lengths": [8000],
        "needle_depths": [50],
        "trials_per_cell": 1,
        "code_max_tokens": 1024,
        "code_problem_limit": 1,
    }
    excluded = {"name", "decode", "quality"}
    assert probe.model_dump(exclude=excluded) == quick.model_dump(exclude=excluded)


def test_real_profiles_include_fast_derived_from_quick() -> None:
    """fast は quick を元に decode、concurrency、quality、prefill を変えた確認の設定

    (issue #39、#130)。
    """
    profiles = c.load_profiles()
    fast, quick = profiles["fast"], profiles["quick"]

    assert c.select_profile("fast").name == "fast"
    assert fast.decode.model_dump() == {"trials": 10, "warmup_trials": 1, "max_tokens": 256}
    assert fast.concurrency.model_dump() == {
        "levels": [1, 2, 4],
        "rounds": 2,
        "max_tokens": 128,
        "input_tokens": 2000,
    }
    assert fast.quality.model_dump() == {
        "toolcall_tasks": 10,
        "needle_lengths": [8000, 32000],
        "needle_depths": [50],
        "trials_per_cell": 1,
        "code_max_tokens": 2048,
        "code_problem_limit": 10,
    }
    assert fast.prefill.model_dump() == {
        "trials": 5,
        "warmup_trials": 1,
        "max_tokens": 16,
        "target_input_tokens": [32000, 64000, 128000],
    }
    excluded = {"name", "decode", "concurrency", "quality", "prefill"}
    assert fast.model_dump(exclude=excluded) == quick.model_dump(exclude=excluded)


_EXPECTED_QUICK: dict[str, Any] = {
    "name": "quick",
    "seed": 0,
    "sampling": {
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "thinking": "server_default",
    },
    "chars_per_token": {"prose_en": 4.831, "prose_ja": 1.794, "code": 3.505, "log": 2.316},
    "timeout": {"connect_s": 10.0, "first_event_s": 120.0, "idle_s": 60.0, "total_s": 900.0},
    "min_successes": 5,
    "max_consecutive_failures": 5,
    "length_tolerance": 0.05,
    "compare_tolerance": 0.02,
    "metrics_interval_s": 1.0,
    "output_sanity": {"repeat_min_chars": 12, "repeat_min_count": 8},
    "sandbox": {
        "runtime": "auto",
        "image": "bench-sandbox:py3.13-numpy2.5.3",
        "image_digest": "sha256:f75a1d43009306248ecbe68c0cc992b700a472037ba6ea336e519404ec1d1de6",
        "timeout_s": 20.0,
        "memory_mb": 512,
        "cpus": 1.0,
        "pids_limit": 64,
    },
    "decode": {"trials": 10, "warmup_trials": 2, "max_tokens": 1024},
    "prefill": {
        "trials": 5,
        "warmup_trials": 1,
        "max_tokens": 16,
        "target_input_tokens": [8000, 32000, 128000],
    },
    "concurrency": {"levels": [1, 2, 4, 8], "rounds": 5, "max_tokens": 256, "input_tokens": 2000},
    "quality": {
        "toolcall_tasks": 50,
        "needle_lengths": [8000, 32000, 128000],
        "needle_depths": [0, 25, 50, 75, 100],
        "trials_per_cell": 2,
        "code_max_tokens": 1024,
        "code_problem_limit": 40,
    },
    "agent": {
        "start_tokens": 20000,
        "end_tokens": 120000,
        "step_tokens": 20000,
        "trials_per_stage": 50,
        "conversations_per_stage": 5,
        "threshold": 0.01,
        "max_tokens": 256,
    },
}

_EXPECTED_FULL: dict[str, Any] = {
    "name": "full",
    "seed": 0,
    "sampling": {
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "thinking": "server_default",
    },
    "chars_per_token": {"prose_en": 4.831, "prose_ja": 1.794, "code": 3.505, "log": 2.316},
    "timeout": {"connect_s": 10.0, "first_event_s": 120.0, "idle_s": 60.0, "total_s": 900.0},
    "min_successes": 10,
    "max_consecutive_failures": 5,
    "length_tolerance": 0.05,
    "compare_tolerance": 0.02,
    "metrics_interval_s": 1.0,
    "output_sanity": {"repeat_min_chars": 12, "repeat_min_count": 8},
    "sandbox": {
        "runtime": "auto",
        "image": "bench-sandbox:py3.13-numpy2.5.3",
        "image_digest": "sha256:f75a1d43009306248ecbe68c0cc992b700a472037ba6ea336e519404ec1d1de6",
        "timeout_s": 20.0,
        "memory_mb": 512,
        "cpus": 1.0,
        "pids_limit": 64,
    },
    "decode": {"trials": 20, "warmup_trials": 2, "max_tokens": 1024},
    "prefill": {
        "trials": 10,
        "warmup_trials": 1,
        "max_tokens": 16,
        "target_input_tokens": [8000, 32000, 128000],
    },
    "concurrency": {"levels": [1, 2, 4, 8], "rounds": 20, "max_tokens": 256, "input_tokens": 2000},
    "quality": {
        "toolcall_tasks": 200,
        "needle_lengths": [8000, 32000, 128000],
        "needle_depths": [0, 25, 50, 75, 100],
        "trials_per_cell": 4,
        "code_max_tokens": 1024,
        "code_problem_limit": None,
    },
    "agent": {
        "start_tokens": 20000,
        "end_tokens": 120000,
        "step_tokens": 20000,
        "trials_per_stage": 300,
        "conversations_per_stage": 5,
        "threshold": 0.01,
        "max_tokens": 256,
    },
}


def test_real_profiles_quick_and_full_values_are_unchanged() -> None:
    """probe と fast を足しても quick と full は 1 項目も変わらない (issue #39)。"""
    profiles = c.load_profiles()

    assert profiles["quick"].model_dump(mode="json") == _EXPECTED_QUICK
    assert profiles["full"].model_dump(mode="json") == _EXPECTED_FULL
