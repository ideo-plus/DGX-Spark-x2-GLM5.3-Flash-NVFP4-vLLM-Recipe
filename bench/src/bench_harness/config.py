"""対象サーバーの定義と計測の設定 (`quick`、`full`) を TOML から読み込む。

読み込んだ値は `bench_harness.types` の凍結の型で検証する。認証の情報は
`TargetDef.api_key_env` に環境変数の名前だけを持ち、値は `resolve_api_key`
が実行時に読む (1.8)。生データの置き場所は `resolve_results_root` が git の
管理の対象でないことを確かめる (8.5)。

依存するのは標準ライブラリと pydantic、`bench_harness.types` だけ。
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import SecretStr, ValidationError

from bench_harness.types import Profile, TargetDef

__all__ = [
    "ConfigError",
    "default_targets_path",
    "default_profiles_path",
    "load_targets",
    "select_target",
    "load_profiles",
    "select_profile",
    "apply_overrides",
    "resolve_api_key",
    "default_results_root",
    "resolve_results_root",
]


class ConfigError(Exception):
    """設定の読み込みまたは検証で見つかった誤り。原因を項目の名前つきで示す。"""


# --- TOML の読み込みと既定の場所 -------------------------------------------


def _config_dir() -> Path:
    """`bench/config/` の場所。この module (`bench/src/bench_harness/config.py`) から辿る。"""
    return Path(__file__).resolve().parents[2] / "config"


def default_targets_path() -> Path:
    """既定の対象サーバーの定義ファイルの場所。"""
    return _config_dir() / "targets.toml"


def default_profiles_path() -> Path:
    """既定の計測の設定ファイルの場所。"""
    return _config_dir() / "profiles.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"設定ファイルが見つからない: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} の TOML を読み取れない: {exc}") from exc


def _format_validation_error(prefix: str, exc: ValidationError) -> str:
    """検証のエラーを、項目の名前つきの 1 行ずつにまとめる (例: `profiles.quick.decode.trials`)。"""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        path = f"{prefix}.{loc}" if loc else prefix
        lines.append(f"{path}: {err['msg']}")
    return "\n".join(lines)


# --- 対象サーバーの定義 -----------------------------------------------------


def load_targets(path: Path | None = None) -> dict[str, TargetDef]:
    """`targets.toml` の `[targets.<name>]` をすべて読み込む。`name` はテーブルの鍵から取る。"""
    data = _load_toml(path if path is not None else default_targets_path())
    table = data.get("targets", {})
    if not isinstance(table, dict):
        raise ConfigError("targets.toml の 'targets' はテーブルの集まりである必要がある")
    result: dict[str, TargetDef] = {}
    for name, fields in table.items():
        if not isinstance(fields, dict):
            raise ConfigError(f"targets.{name} はテーブルである必要がある")
        try:
            result[name] = TargetDef.model_validate({**fields, "name": name})
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(f"targets.{name}", exc)) from exc
    return result


def select_target(name: str, path: Path | None = None) -> TargetDef:
    """名前で対象サーバーの定義を 1 つ選ぶ。知らない名前は、使える名前つきの `ConfigError`。"""
    targets = load_targets(path)
    try:
        return targets[name]
    except KeyError:
        available = ", ".join(sorted(targets)) if targets else "(なし)"
        raise ConfigError(
            f"対象サーバーの定義 '{name}' が見つからない。使える名前: {available}"
        ) from None


# --- 計測の設定 -------------------------------------------------------------


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    """`profiles.toml` の `[profiles.<name>]` をすべて読み込む。`name` はテーブルの鍵から取る。"""
    data = _load_toml(path if path is not None else default_profiles_path())
    table = data.get("profiles", {})
    if not isinstance(table, dict):
        raise ConfigError("profiles.toml の 'profiles' はテーブルの集まりである必要がある")
    result: dict[str, Profile] = {}
    for name, fields in table.items():
        if not isinstance(fields, dict):
            raise ConfigError(f"profiles.{name} はテーブルである必要がある")
        try:
            result[name] = Profile.model_validate({**fields, "name": name})
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(f"profiles.{name}", exc)) from exc
    return result


def select_profile(name: str, path: Path | None = None) -> Profile:
    """名前で計測の設定を 1 つ選ぶ。知らない名前は、使える名前つきの `ConfigError`。"""
    profiles = load_profiles(path)
    try:
        return profiles[name]
    except KeyError:
        available = ", ".join(sorted(profiles)) if profiles else "(なし)"
        raise ConfigError(f"計測の設定 '{name}' が見つからない。使える名前: {available}") from None


def apply_overrides(profile: Profile, overrides: Mapping[str, int]) -> Profile:
    """試行の回数などを、ドット区切りの経路 (例: `decode.trials`) で上書きし、検証し直す。

    `Profile` は凍結 (frozen) なので、`model_copy(update=...)` では検証を通らない
    (tasks.md の Implementation Notes 1.2)。ここでは `model_dump()` した辞書を書き換えて
    から `model_validate()` に通すことで、下限などの検証を必ず経由させる。
    """
    if not overrides:
        return profile
    data = profile.model_dump()
    for dotted_key, value in overrides.items():
        _set_nested(data, dotted_key, value)
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(f"profiles.{profile.name}", exc)) from exc


def _set_nested(data: dict[str, Any], dotted_key: str, value: int) -> None:
    keys = dotted_key.split(".")
    target: Any = data
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise ConfigError(f"上書きの経路 '{dotted_key}' が Profile の項目に対応しない")
        target = target[key]
    last = keys[-1]
    if not isinstance(target, dict) or last not in target:
        raise ConfigError(f"上書きの経路 '{dotted_key}' が Profile の項目に対応しない")
    target[last] = value


# --- 認証の情報 (1.8) -------------------------------------------------------


def resolve_api_key(target: TargetDef) -> SecretStr | None:
    """認証の情報の値を、実行時に環境変数から読む。値そのものは設定ファイルに書かない。

    `api_key_env` が設定されていなければ `None`。設定されているのに環境変数が
    ない、または空なら、変数の名前つきの `ConfigError` (値そのものはエラーに含めない)。
    """
    if target.api_key_env is None:
        return None
    value = os.environ.get(target.api_key_env)
    if not value:
        raise ConfigError(
            f"環境変数 '{target.api_key_env}' が設定されていない"
            f" (対象サーバー '{target.name}' の認証の情報)"
        )
    return SecretStr(value)


# --- 生データの置き場所 (8.5) ------------------------------------------------


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """引数の列で git を呼ぶ (シェルは使わない)。

    `LC_ALL=C` を渡し `LANGUAGE` を外して、stderr のメッセージをロケールに
    左右されない英語に固定する (根拠の要る判定に使うため)。git そのものが
    見つからない、または実行できないときは `ConfigError` にする (ほかの失敗と
    揃える)。
    """
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env.pop("LANGUAGE", None)
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as exc:
        raise ConfigError(f"git を実行できない (見つからないか、実行の権限がない): {exc}") from exc


def _is_confirmed_outside_repo(proc: subprocess.CompletedProcess[str]) -> bool:
    """`rev-parse` の失敗が、git のリポジトリの外にいることによるものだと、
    `LC_ALL=C` で固定した既知の英語のメッセージから、根拠を持って判定できる場合だけ真。
    それ以外の失敗 (壊れた `.git`、権限など) は、ここでは判定できないとして扱う。
    """
    return proc.returncode != 0 and "not a git repository" in proc.stderr


def _git_toplevel(cwd: Path) -> Path | None:
    """`cwd` を含む git の作業木の最上位。確かにリポジトリの外だと分かる場合だけ `None`。"""
    proc = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if proc.returncode == 0:
        return Path(proc.stdout.strip())
    if _is_confirmed_outside_repo(proc):
        return None
    raise ConfigError(
        f"'{cwd}' が git のリポジトリの中かどうかを確かめられない"
        f" (git の応答: {proc.stderr.strip() or f'終了コード {proc.returncode}'})"
    )


def _nearest_existing_dir(path: Path) -> Path:
    """`path` かその先祖のうち、実在する最も近いディレクトリを返す (git を呼ぶための cwd)。"""
    current = path if path.is_absolute() else path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def default_results_root() -> Path:
    """既定の生データの置き場所 (git のリポジトリの直下の `results/`)。"""
    top = _git_toplevel(Path(__file__).resolve().parent)
    if top is None:
        raise ConfigError(
            "results_root の既定値を決められない (git のリポジトリの外にいる)。明示のパスを渡すこと"
        )
    return top / "results"


def resolve_results_root(explicit: Path | None = None) -> Path:
    """生データの置き場所を決め、git の管理の対象でないことを確かめる (8.5)。

    管理の対象の場所 (追跡されている、または無視されていない) を指定すると
    `ConfigError` になる。この関数はディレクトリを作らない。
    """
    root = explicit if explicit is not None else default_results_root()
    _ensure_not_tracked(root)
    return root


def _ensure_not_tracked(path: Path) -> None:
    resolved = path if path.is_absolute() else path.resolve()
    probe_dir = _nearest_existing_dir(resolved)

    inside = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=probe_dir)
    if inside.returncode == 0:
        if inside.stdout.strip() != "true":
            return  # 作業木の外 (bare リポジトリの中など) → 許可
        # 作業木の中 → 下の check-ignore で確かめる
    elif _is_confirmed_outside_repo(inside):
        return  # git のリポジトリの外だと、はっきりした根拠 (stderr) で確かめられた → 許可
    else:
        # 「リポジトリの外」だと確かめられない失敗 (壊れた .git、権限など) は、
        # 安全側に倒して許可しない (8.5: 黙って管理の対象の場所へ書き出さない)
        raise ConfigError(
            f"'{probe_dir}' が git のリポジトリの中かどうかを確かめられない"
            f" (git の応答: {inside.stderr.strip() or f'終了コード {inside.returncode}'})。"
            "安全のため、生データの置き場所として許可しない (8.5)"
        )

    # 末尾に `/` を付け、まだ存在しないディレクトリでも「ディレクトリのつもり」だと
    # git に伝える (`results/` のようなディレクトリ限定のパターンは、末尾の `/` か
    # 実在の確認がないと、存在しない経路には一致しない)
    ignored = _run_git(["check-ignore", "-q", f"{resolved}/"], cwd=probe_dir)
    if ignored.returncode == 0:
        return  # .gitignore などで無視されている → 許可

    raise ConfigError(
        f"生データの置き場所 '{resolved}' は git の管理の対象になり得る"
        " (追跡されている、または無視されていない)。.gitignore に加えるか、"
        "管理の対象外の場所を指定すること (8.5)"
    )
