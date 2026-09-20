"""コマンドの入口の試験 (task 5.1: cli)。

`main(argv)` をその場で呼ぶ (プロセスは起こさない)。要求が絡む試験は、実際の
ソケット越しに偽のサーバーを相手に行う。時刻の値を判定する試験は 1 つもない。
生データの置き場所は、必ず `tmp_path` の下にする (リポジトリの `results/` を
汚さない。8.5)。

計測を実際に流す試験は、試行の回数を設定が許す最小にしてある
(`decode.trials` の下限は 10、同時処理は 1 回ぶん 2 本 × 2 回)。
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from bench_harness import cli
from bench_harness.analysis.compare import compare_runs
from bench_harness.corpus import humaneval
from bench_harness.runner import (
    EXIT_ABORTED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PRECONDITION,
    PreconditionError,
    default_suite_registry,
)
from bench_harness.store import RunStore, list_run_dirs
from bench_harness.types import RunRequest, SuiteName
from fake_server import FakeServer, http_error_response, text_response

TARGET_NAME = "fake"
PROFILE_NAME = "test"
SECRET_VALUE = "s3cret-cli-value"
SECRET_ENV = "BENCH_TEST_CLI_KEY"
UNREACHABLE_URL = "http://127.0.0.1:1"


# --- 設定ファイルと置き場所の下ごしらえ -------------------------------------


@dataclass(frozen=True)
class Bed:
    """1 回のコマンドに要る、設定ファイルと生データの置き場所の一式。"""

    targets: Path
    profiles: Path
    results_root: Path

    def run_options(self) -> list[str]:
        """`bench run` に渡す、設定ファイルと置き場所の指定。"""
        return [
            "--targets",
            str(self.targets),
            "--profiles",
            str(self.profiles),
            "--results-root",
            str(self.results_root),
        ]

    def config_options(self) -> list[str]:
        """`bench calibrate` に渡す、設定ファイルの指定。"""
        return ["--targets", str(self.targets), "--profiles", str(self.profiles)]


def write_targets(tmp_path: Path, base_url: str, *, api_key_env: str | None = None) -> Path:
    lines = [f"[targets.{TARGET_NAME}]", f'base_url = "{base_url}"', 'model = "fake-model"']
    if api_key_env is not None:
        lines.append(f'api_key_env = "{api_key_env}"')
    path = tmp_path / "targets.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_profiles(tmp_path: Path, *, max_consecutive_failures: int = 5) -> Path:
    """試験用の計測の設定。制限時間は短く、出力の上限は小さくしてある。"""
    text = f"""
[profiles.{PROFILE_NAME}]
seed = 3
min_successes = 1
max_consecutive_failures = {max_consecutive_failures}
length_tolerance = 0.05
compare_tolerance = 0.02
metrics_interval_s = 60.0

[profiles.{PROFILE_NAME}.sampling]
temperature = 0.0
thinking = "server_default"

[profiles.{PROFILE_NAME}.timeout]
connect_s = 5.0
first_event_s = 5.0
idle_s = 5.0
total_s = 20.0

[profiles.{PROFILE_NAME}.decode]
trials = 10
warmup_trials = 0
max_tokens = 32

[profiles.{PROFILE_NAME}.concurrency]
levels = [2]
rounds = 2
max_tokens = 16
input_tokens = 400
"""
    path = tmp_path / "profiles.toml"
    path.write_text(text, encoding="utf-8")
    return path


def make_bed(
    tmp_path: Path,
    server: FakeServer | None = None,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_consecutive_failures: int = 5,
) -> Bed:
    if base_url is None:
        assert server is not None, "make_bed: server か base_url のどちらかが要る"
        base_url = server.base_url
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Bed(
        targets=write_targets(tmp_path, base_url, api_key_env=api_key_env),
        profiles=write_profiles(tmp_path, max_consecutive_failures=max_consecutive_failures),
        results_root=tmp_path / "results",
    )


def bench(*argv: str) -> int:
    """`bench` を、その場 (同じプロセス) で 1 回走らせる。"""
    return cli.main(list(argv))


def only_run_dir(bed: Bed) -> Path:
    dirs = list_run_dirs(bed.results_root)
    assert len(dirs) == 1, f"計測ランのディレクトリが 1 つではない: {dirs}"
    return dirs[0]


def complete_run(bed: Bed, capsys: pytest.CaptureFixture[str]) -> Path:
    """同時処理のまとまりを 1 回流して、計測ランのディレクトリを返す。"""
    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "concurrency",
        *bed.run_options(),
    )
    capsys.readouterr()  # ここまでの出力は、呼び出し側の判定に混ぜない
    assert code == EXIT_OK
    return only_run_dir(bed)


# --- 計測の実行 (完了の状態、1.1、1.2、8.7、10.4) ----------------------------


def test_run_prints_the_run_id_and_the_summary_location_and_exits_zero(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """5.1 の完了の状態: 偽のサーバーに対して、識別子と要約の場所を出して 0 で終わる。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "concurrency",
        *bed.run_options(),
    )

    assert code == EXIT_OK
    run_dir = only_run_dir(bed)
    assert run_dir.parent == bed.results_root
    out = capsys.readouterr().out
    # 識別子は、経路の一部としてではなく、それだけの行としても出す
    # (計測者が、次のコマンドにそのまま打ち込めるように)
    assert f"計測ラン: {run_dir.name}" in out
    assert str(run_dir / "summary.md") in out
    assert str(run_dir / "summary.json") in out
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.md").is_file()


def test_run_against_an_unreachable_target_prints_the_reason_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """5.1 の完了の状態: 接続できない対象サーバーでは、理由を出して 1 で終わる。"""
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "unreachable" in err  # どの前提が満たされていないか (1.4)
    assert "Traceback" not in err
    assert not bed.results_root.exists()  # 計測ランのディレクトリを作らない


def test_run_executes_only_the_selected_suites(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """1.2: 選んだまとまりだけを実行する。

    選ばなければ走るはずのまとまり (decode) を、片方の計測ランだけで走らせて
    比べる。選ぶ仕掛けを外すと、2 つ目の計測ランにも decode が現れる。
    """
    fake_server.set_response(text_response("ok"))
    both = make_bed(tmp_path / "both", fake_server)
    one = make_bed(tmp_path / "one", fake_server)

    assert (
        bench(
            "run",
            "--target",
            TARGET_NAME,
            "--profile",
            PROFILE_NAME,
            "--suite",
            "decode",
            "--suite",
            "concurrency",
            *both.run_options(),
        )
        == EXIT_OK
    )
    assert (
        bench(
            "run",
            "--target",
            TARGET_NAME,
            "--profile",
            PROFILE_NAME,
            "--suite",
            "concurrency",
            *one.run_options(),
        )
        == EXIT_OK
    )
    capsys.readouterr()

    assert suites_in(only_run_dir(both)) == {SuiteName.DECODE, SuiteName.CONCURRENCY}
    assert suites_in(only_run_dir(one)) == {SuiteName.CONCURRENCY}


def suites_in(run_dir: Path) -> set[SuiteName]:
    records, warnings = RunStore.open(run_dir).read_trials()
    assert warnings == []
    return {record.suite for record in records}


def test_run_without_a_suite_option_selects_the_three_speed_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--suite` を省いたときの既定 (task 8.1 で決めた: 速さの 3 つだけ)。

    8.1 で品質の検査と長い会話の検査を対応表に足したので、「実行できるまとまり
    すべて」を既定にすると、`quick` でも何時間もかかる計測ランが、うっかり
    始まってしまう (注 5.1)。5 つ全部を流すときは `--suite all` と書く。
    """
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    seen = spy_on_execute_run(monkeypatch)

    code = bench("run", "--target", TARGET_NAME, "--profile", PROFILE_NAME, *bed.run_options())

    assert code == EXIT_PRECONDITION  # 覗き見の側で止めている
    assert seen[0].suites == [SuiteName.DECODE, SuiteName.PREFILL, SuiteName.CONCURRENCY]
    capsys.readouterr()


def test_run_with_suite_all_selects_every_registered_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--suite all` は、対応表に載っているまとまりを、名前の定義順にすべて選ぶ (1.2)。"""
    # まとまりの名前と重なると、`all` という名前のまとまりを選べなくなる
    assert cli.ALL_SUITES not in {name.value for name in SuiteName}
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    seen = spy_on_execute_run(monkeypatch)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "all",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    assert seen[0].suites == [
        SuiteName.DECODE,
        SuiteName.PREFILL,
        SuiteName.CONCURRENCY,
        SuiteName.QUALITY,
        SuiteName.AGENT,
    ]
    capsys.readouterr()


def test_run_keeps_the_order_of_the_suite_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--suite` は繰り返せて、書いた順に実行する (1.2)。"""
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    seen = spy_on_execute_run(monkeypatch)

    bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "agent",
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert seen[0].suites == [SuiteName.AGENT, SuiteName.DECODE]
    capsys.readouterr()


def test_the_same_suite_twice_is_a_configuration_error(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """同じまとまりを 2 回選ぶと、設定の誤りになる (判定は runner が持つ)。"""
    bed = make_bed(tmp_path, fake_server)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "decode" in err
    assert not bed.results_root.exists()


def test_suite_all_cannot_be_mixed_with_other_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`all` は「すべて」なので、ほかの名前と混ぜて書けない。"""
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "all",
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "all" in err
    assert "Traceback" not in err


def spy_on_execute_run(monkeypatch: pytest.MonkeyPatch) -> list[RunRequest]:
    """`execute_run` を、指示を覚えてすぐ止める偽物に差し替える。"""
    seen: list[RunRequest] = []

    async def fake_execute_run(req: RunRequest, progress: Any, **kwargs: Any) -> Any:
        seen.append(req)
        raise PreconditionError("試験のために、計測を始める前に止めた")

    monkeypatch.setattr(cli, "execute_run", fake_execute_run)
    return seen


def spy_on_the_registry(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """`default_suite_registry` に渡された引数を覚えておく。

    入口は、使える名前を並べるためにも対応表を作る (引数なし)。計測ランのため
    に作った 1 回だけを見分けられるよう、引数をそのまま覚えておく。
    """
    seen: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> dict[SuiteName, Any]:
        seen.append(dict(kwargs))
        return dict(default_suite_registry(**kwargs))

    monkeypatch.setattr(cli, "default_suite_registry", spy)
    return seen


def loaders_passed(seen: Sequence[dict[str, Any]]) -> list[Any]:
    """対応表に渡された、公開の課題の読み方 (計測ランごとに 1 つ)。"""
    return [call["problems_loader"] for call in seen if "problems_loader" in call]


def test_run_builds_the_quality_suite_with_the_given_data_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--data-cache` と `--no-download` が、公開の課題の読み方に届く (6.5)。

    読み方そのものは呼ばない (取得もしない)。入口が組み立てた読み方を、ここで
    1 回だけ呼んで、どの置き場所を、取得を許さずに読もうとするかを確かめる。
    """
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    cache_dir = tmp_path / "data-cache"
    seen = spy_on_the_registry(monkeypatch)
    spy_on_execute_run(monkeypatch)
    calls: list[tuple[Path | None, bool | None]] = []

    def spy_load(cache: Path | None = None, **kwargs: Any) -> tuple[Any, list[Any]]:
        calls.append((cache, kwargs.get("allow_download")))
        raise humaneval.DatasetUnavailable("試験のために取得しない")

    monkeypatch.setattr(humaneval, "load_humaneval_plus", spy_load)

    bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "quality",
        "--data-cache",
        str(cache_dir),
        "--no-download",
        *bed.run_options(),
    )

    loaders = loaders_passed(seen)
    assert len(loaders) == 1
    loader = loaders[0]
    assert loader is not None
    with pytest.raises(humaneval.DatasetUnavailable):
        loader()
    assert calls == [(cache_dir, False)]
    capsys.readouterr()


def test_run_without_the_data_cache_options_leaves_the_default_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """既定では、まとまり自身の読み方 (git の管理の対象でない既定の置き場所) を使う。"""
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    seen = spy_on_the_registry(monkeypatch)
    spy_on_execute_run(monkeypatch)

    bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "quality",
        *bed.run_options(),
    )

    assert loaders_passed(seen) == [None]
    capsys.readouterr()


def test_run_passes_the_trial_count_overrides_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)
    seen = spy_on_execute_run(monkeypatch)

    bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        "--set",
        "decode.trials=12",
        "--set",
        "concurrency.rounds=3",
        *bed.run_options(),
    )

    assert seen[0].trials_override == {"decode.trials": 12, "concurrency.rounds": 3}
    capsys.readouterr()


def test_run_with_an_override_below_the_minimum_fails_with_the_field_name(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, fake_server)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        "--set",
        "decode.trials=5",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "decode.trials" in err
    assert "Traceback" not in err
    assert not bed.results_root.exists()


def test_run_with_a_non_integer_override_fails_with_the_value(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, fake_server)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--set",
        "decode.trials=たくさん",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "decode.trials" in err
    assert "たくさん" in err
    assert fake_server.call_count("/v1/messages") == 0


def test_run_with_an_unknown_target_lists_the_valid_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "run",
        "--target",
        "いない子",
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "いない子" in err
    assert TARGET_NAME in err  # 使える名前を示す
    assert not bed.results_root.exists()


def test_run_with_an_unknown_profile_lists_the_valid_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        "いない設定",
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "いない設定" in err
    assert PROFILE_NAME in err


def test_run_with_an_unknown_suite_lists_the_valid_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "はやさ",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "はやさ" in err
    for name in ("decode", "prefill", "concurrency", "quality", "agent"):
        assert name in err
    assert not bed.results_root.exists()


def test_run_of_an_aborted_run_still_writes_the_summaries_and_exits_two(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """10.3、10.4: 連続の失敗で止まっても、未完了の要約を作って 2 で終わる。"""
    fake_server.set_response_sequence([text_response("ok"), http_error_response(500)])
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=1)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_ABORTED
    run_dir = only_run_dir(bed)
    captured = capsys.readouterr()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.md").is_file()
    assert "未完了" in captured.err
    assert "Traceback" not in captured.err
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "未完了" in summary


def test_run_does_not_publish_anything(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """8.4: 公開は、計測者がそう指示したときだけ。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    calls: list[tuple[Any, ...]] = []

    def fake_publish(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise AssertionError("bench run が公開してはならない (8.4)")

    monkeypatch.setattr(cli, "publish_run", fake_publish)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "concurrency",
        *bed.run_options(),
    )

    assert code == EXIT_OK
    assert calls == []
    capsys.readouterr()


# --- 認証の情報 (1.8) --------------------------------------------------------


def test_the_api_key_value_never_appears_in_the_output(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """成功する計測ランでも、認証の情報の値は 1 文字も出さない (1.8)。"""
    monkeypatch.setenv(SECRET_ENV, SECRET_VALUE)
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server, api_key_env=SECRET_ENV)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "concurrency",
        *bed.run_options(),
    )

    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert SECRET_VALUE not in captured.out
    assert SECRET_VALUE not in captured.err
    # 仕掛けが効いていることの裏取り: 値は確かにヘッダーに付いている
    assert fake_server.requests_for("/v1/messages")[-1].headers["x-api-key"] == SECRET_VALUE


def test_the_api_key_value_never_appears_on_the_error_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """接続できない対象サーバーでも、認証の情報の値は出さない (1.8)。"""
    monkeypatch.setenv(SECRET_ENV, SECRET_VALUE)
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL, api_key_env=SECRET_ENV)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    captured = capsys.readouterr()
    assert SECRET_VALUE not in captured.out
    assert SECRET_VALUE not in captured.err


def test_a_missing_api_key_env_var_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(SECRET_ENV, raising=False)
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL, api_key_env=SECRET_ENV)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    assert SECRET_ENV in capsys.readouterr().err


# --- 要約の作り直し (8.2) ----------------------------------------------------


def test_summarize_reproduces_byte_identical_summaries(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    before_json = (run_dir / "summary.json").read_bytes()
    before_md = (run_dir / "summary.md").read_bytes()

    code = bench("summarize", str(run_dir))

    assert code == EXIT_OK
    assert (run_dir / "summary.json").read_bytes() == before_json
    assert (run_dir / "summary.md").read_bytes() == before_md
    assert str(run_dir / "summary.md") in capsys.readouterr().out


def test_summarize_accepts_a_run_id_under_the_results_root(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    (run_dir / "summary.md").unlink()

    code = bench("summarize", run_dir.name, "--results-root", str(bed.results_root))

    assert code == EXIT_OK
    assert (run_dir / "summary.md").is_file()


def test_summarize_shows_the_read_warnings_on_stderr(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """注 4.1: 生データの最後の行が途中で切れていたら、捨てて読み、警告を標準エラーに出す。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    trials = run_dir / "trials.jsonl"
    data = trials.read_bytes()
    last_line = data.rstrip(b"\n").rsplit(b"\n", 1)[-1]
    trials.write_bytes(data + last_line[: len(last_line) // 2])

    code = bench("summarize", str(run_dir))

    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "警告" in captured.err
    assert "trials.jsonl" in captured.err
    assert "警告" not in captured.out


def test_a_run_id_with_a_trailing_separator_is_still_a_run_id(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    (run_dir / "summary.md").unlink()

    code = bench("summarize", run_dir.name + "/", "--results-root", str(bed.results_root))

    assert code == EXIT_OK
    assert (run_dir / "summary.md").is_file()


def test_the_results_root_wins_over_a_same_named_directory_in_the_cwd(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """いまいる場所に同じ名前のディレクトリがあっても、置き場所の下の計測ランを読む。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    (run_dir / "summary.md").unlink()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / run_dir.name).mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    code = bench("summarize", run_dir.name, "--results-root", str(bed.results_root))

    assert code == EXIT_OK
    assert (run_dir / "summary.md").is_file()


def test_a_dot_is_a_path_and_not_a_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`.` は経路として扱い、置き場所の下を探さない (計測ランでなければ `StoreError`)。"""
    monkeypatch.chdir(tmp_path)

    code = bench("summarize", ".", "--results-root", str(tmp_path / "no-such-root"))

    assert code == EXIT_ABORTED
    assert "manifest.json" in capsys.readouterr().err


def test_summarize_with_an_unknown_run_id_lists_the_known_ones(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)

    code = bench("summarize", "そんな計測ランはない", "--results-root", str(bed.results_root))

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "そんな計測ランはない" in err
    assert run_dir.name in err


def test_summarize_of_a_directory_that_is_not_a_run_exits_two_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """計測ランでない場所は `StoreError` (注 3.5、4.4: 入口が終了の値 2 に直す)。"""
    not_a_run = tmp_path / "not-a-run"
    not_a_run.mkdir()

    code = bench("summarize", str(not_a_run))

    assert code == EXIT_ABORTED
    err = capsys.readouterr().err
    assert "manifest.json" in err
    assert "Traceback" not in err


# --- 比較 (9.1〜9.6) ---------------------------------------------------------


def test_compare_of_a_run_with_itself_prints_the_conclusion_and_exits_zero(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)

    code = bench("compare", str(run_dir), str(run_dir))

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "# 計測ランの比較" in out
    assert "## 繰り返しの結論" in out
    assert "収まった" in out


def test_compare_prints_what_the_comparison_result_renders(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """注 4.3: 件数を渡し忘れようがない `ComparisonResult.to_markdown()` の出力を、そのまま出す。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    out_dir = tmp_path / "out"

    code = bench("compare", str(run_dir), str(run_dir), "--out", str(out_dir))

    assert code == EXIT_OK
    expected = compare_runs(run_dir, run_dir).to_markdown()
    assert capsys.readouterr().out == expected
    written = [path for path in out_dir.iterdir() if path.is_file()]
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == expected


def test_compare_writes_the_report_when_out_is_given(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    out_dir = tmp_path / "compare-out"

    code = bench("compare", str(run_dir), str(run_dir), "--out", str(out_dir))

    assert code == EXIT_OK
    written = sorted(path.name for path in out_dir.iterdir())
    assert written == ["comparison.md"]
    assert (out_dir / "comparison.md").read_text(encoding="utf-8") == capsys.readouterr().out


def test_compare_of_a_directory_that_is_not_a_run_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_run = tmp_path / "not-a-run"
    not_a_run.mkdir()

    code = bench("compare", str(not_a_run), str(not_a_run))

    assert code == EXIT_ABORTED
    assert "Traceback" not in capsys.readouterr().err


# --- 公開 (8.4) --------------------------------------------------------------


def test_publish_copies_exactly_two_files(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    docs_root = tmp_path / "docs-results"

    code = bench("publish", str(run_dir), "--docs-root", str(docs_root))

    assert code == EXIT_OK
    dest = docs_root / run_dir.name
    assert sorted(path.name for path in dest.iterdir()) == ["summary.json", "summary.md"]
    captured = capsys.readouterr()
    assert str(dest) in captured.out
    assert "summary.md" in captured.err  # 公開の前に目で見ることを促す (注 4.4)


def test_publish_without_summaries_exits_two_with_the_reason(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)
    (run_dir / "summary.json").unlink()
    (run_dir / "summary.md").unlink()
    docs_root = tmp_path / "docs-results"

    code = bench("publish", str(run_dir), "--docs-root", str(docs_root))

    assert code == EXIT_ABORTED
    err = capsys.readouterr().err
    assert "bench summarize" in err
    assert "Traceback" not in err
    assert not docs_root.exists()


# --- 文字数とトークン数の比の測定 (2.5、2.7) --------------------------------


def test_calibrate_prints_a_toml_fragment_with_one_ratio_per_content_kind(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, fake_server)

    code = bench(
        "calibrate", "--target", TARGET_NAME, "--profile", PROFILE_NAME, *bed.config_options()
    )

    assert code == EXIT_OK
    out = capsys.readouterr().out
    parsed = tomllib.loads(out)
    ratios = parsed["profiles"][PROFILE_NAME]["chars_per_token"]
    assert sorted(ratios) == ["code", "log", "prose_en", "prose_ja"]
    for kind, value in ratios.items():
        assert isinstance(value, float), kind
        assert value > 0.0, kind
    assert fake_server.call_count("/v1/messages/count_tokens") == len(ratios)


def test_calibrate_against_an_unreachable_target_explains_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """数える口がなく、代わりの数え方もできないときは、理由を出して 1 で終わる (注 2.7)。"""
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    code = bench(
        "calibrate", "--target", TARGET_NAME, "--profile", PROFILE_NAME, *bed.config_options()
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "count_tokens" in err
    assert "Traceback" not in err


# --- 中断と、使い方 ----------------------------------------------------------


def test_a_keyboard_interrupt_during_a_run_exits_130_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bed = make_bed(tmp_path, base_url=UNREACHABLE_URL)

    async def interrupt(req: RunRequest, progress: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_run", interrupt)

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "decode",
        *bed.run_options(),
    )

    assert code == EXIT_INTERRUPTED
    assert "Traceback" not in capsys.readouterr().err


def test_a_keyboard_interrupt_during_summarize_exits_130(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    run_dir = complete_run(bed, capsys)

    def interrupt(run_dir_arg: Path) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "summarize_run", interrupt)

    code = bench("summarize", str(run_dir))

    assert code == EXIT_INTERRUPTED
    assert "Traceback" not in capsys.readouterr().err


def test_no_subcommand_prints_the_help_and_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = bench()

    assert code == EXIT_PRECONDITION
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()
    for name in ("run", "summarize", "compare", "publish", "calibrate"):
        assert name in captured.err


def test_every_subcommand_has_help(capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("run", "summarize", "compare", "publish", "calibrate"):
        with pytest.raises(SystemExit) as exc_info:
            bench(name, "--help")
        assert exc_info.value.code == 0
        assert "usage" in capsys.readouterr().out.lower()


def test_the_parser_keeps_version_and_the_five_commands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["summarize", "some-run"])

    assert args.command == "summarize"
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0


def test_run_help_documents_the_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        bench("run", "--help")

    out = capsys.readouterr().out
    for code in ("0", "1", "2", "130"):
        assert code in out


def test_run_help_documents_the_suites_and_the_dataset_cache(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """5 つのまとまり、既定の 3 つ、`all`、課題の置き場所の指定が、使い方に出る (8.1)。"""
    with pytest.raises(SystemExit):
        bench("run", "--help")

    out = capsys.readouterr().out
    for name in ("decode", "prefill", "concurrency", "quality", "agent", "all"):
        assert name in out
    assert "--data-cache" in out
    assert "--no-download" in out
    # 既定が「速さの 3 つ」であることが、使い方から分かる
    assert "既定" in out


def test_a_data_cache_inside_the_work_tree_is_a_configuration_error_before_anything_runs(
    fake_server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """git の管理の対象になり得る置き場所は、計測を始める前に、設定の誤りとして示す。

    あとで気付くと、先に流したまとまりの試行だけが残り、計測ランが `running` のまま
    要約なしで止まる (実機では、何時間ぶんもの結果が宙に浮く)。
    """
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    tracked_dir = Path(__file__).resolve().parents[2] / "src"

    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        "--suite",
        "all",
        "--data-cache",
        str(tracked_dir),
        "--no-download",
        *bed.run_options(),
    )

    assert code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "--data-cache" in err
    assert "Traceback" not in err
    assert list_run_dirs(bed.results_root) == []
    assert fake_server.call_count("/v1/messages") == 0
