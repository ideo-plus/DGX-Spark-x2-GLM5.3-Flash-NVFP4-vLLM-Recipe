"""構成とノードの定義を TOML から読み込み、Spark に触る前に検査する。

design.md 「types / config」の Service Interface を実装する。読み込みの入口は 3 つだけで、
どの誤りも `ConfigError` 1 つに寄せる (`bench_harness.config` と同じ流儀)。

- `load_configs(path, repo_root)`: `[configs.<名前>]` をすべて読み、検査 1〜6、8〜10 と、
  `kind` ごとのノードの数、`schema_version` を確かめる
- `load_nodes(path, repo_root)`: `[nodes.head]` と `[nodes.worker]` を読む
- `select_config(configs, name, nodes)`: 名前で 1 つ選び、検査 7 (直結の値) だけを、
  選んだ構成について行う

守る決まり:

- **すべて並べて示す** (design.md 「Error Handling」の「早く断る」)。誤りは、
  `configs.p1-nvfp4-tp2.args.max-num-seqs: 根拠がない` の形の 1 行にして、1 回の
  読み込みで見つかったものを、すべて 1 つの `ConfigError` に入れる
- そのために、入れ子の設定 (`Setting`) を 1 つずつ検証して誤りを集め、落ちた設定は
  当たり障りのない代わりの値に置き換えてから `ConfigDef` を検証する。入れ子が落ちたまま
  `ConfigDef` を検証すると、その構成の after 検証 (ノードの重複、名乗るモデルの名前) が
  走らず、誤りを取りこぼす (tasks.md の Implementation Notes 1.2)
- pydantic の英語の文は、そのまま見せない。項目の名前つきの日本語の文に直す
- 依存するのは、標準ライブラリと pydantic と `serving_kit.types` だけ (依存の向き)
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, HttpUrl, ValidationError

from serving_kit.types import (
    CONFIG_SCHEMA_VERSION,
    ConfigDef,
    ImageRef,
    NodeDef,
    NodeRole,
    Provenance,
    Setting,
    WeightsRef,
)

__all__ = ["ConfigError", "load_configs", "load_nodes", "select_config"]


class ConfigError(Exception):
    """構成またはノードの定義の、読み込みと検査で見つかった誤り。

    見つかった誤りを、項目の名前つきの日本語の文で、すべて並べて持つ。Spark に触る前に
    投げるので、この誤りで終わるときの終了コードは 1 (前提の不足) である
    (design.md 「Error Handling」)。
    """


# --- 検査の一覧 (design.md 「types / config」の検査 1〜10) ---------------

_NODE_ROLES: Final[tuple[NodeRole, ...]] = ("head", "worker")
"""ノードの定義に書ける役割。`types.NodeRole` と同じ 2 つ (試験で固定する)。"""

_NODE_COUNTS: Final[Mapping[str, int]] = {
    "serve": 2,
    "job": 2,
    "fetch": 2,
    "probe": 1,
    "inspect": 1,
}
"""`kind` ごとのノードの数 (design.md 「types / config」の `ConfigDef.nodes` の注記)。

型では断っていないので、ここで検査する (tasks.md の Implementation Notes 1.2)。
"""

_FABRIC_KINDS: Final[frozenset[str]] = frozenset({"serve", "job"})
"""検査 7 (直結の値) が掛かる `kind`。`fetch`、`inspect`、`probe` には掛からない。"""

_SETTING_SECTIONS: Final[tuple[str, ...]] = ("docker", "args", "env")

_SPECULATIVE_FLAGS: Final[frozenset[str]] = frozenset(
    {"--speculative-config", "--spec-method", "--spec-model", "--spec-tokens"}
)
"""検査 5: 投機的デコードの指定。最初の起動では使わない (requirements 6.7)。"""

_ALLOWED_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {
        "{node.fabric_addr}",
        "{node.fabric_ifname}",
        "{node.rank}",
        "{head.fabric_addr}",
        "{head.lan_addr}",
        "{weights.mount_at}",
        "{remote_root}",
    }
)
"""検査 6: 値に書ける置き換えの印。ほかの `{…}` は、JSON の値でなければ誤り。"""

_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\{[^{}]*\}")

_FORBIDDEN_DOCKER_FLAGS: Final[Mapping[str, str]] = {
    "--privileged": "特権でコンテナを動かす",
    "--pid": "ホストのプロセスの名前空間を共有する",
    "--userns": "ユーザーの名前空間の切り離しを外す",
    "--security-opt": "セキュリティの指定を上書きする",
    "--volume": "元の場所を検査できない (--mount だけを使う)",
    "-v": "元の場所を検査できない (--mount だけを使う)",
    "--rm": "終了したコンテナが消えて、記録が読めなくなる",
}
"""検査 8: 禁じる docker のフラグと、その理由。"""

_REMOTE_ROOT_MARKER: Final[str] = "{remote_root}"

_MOUNT_SOURCE_KEYS: Final[frozenset[str]] = frozenset({"source", "src"})

_ALLOWED_DEVICES: Final[frozenset[str]] = frozenset({"/dev/infiniband"})
"""検査 10: `--device` に書ける名前。一覧を広げるのは、設計の変更として扱う。"""

_ALLOWED_CAPABILITIES: Final[frozenset[str]] = frozenset({"SYS_NICE", "IPC_LOCK"})
"""検査 10: `--cap-add` に書ける名前。根拠があっても、一覧にないものは断る。"""

_WEIGHTS_DIR: Final[tuple[str, ...]] = ("serving", "weights")
"""マニフェストの置き場所 (リポジトリの最上位から見た道筋)。"""


# --- pydantic の誤りの、日本語への言い換え -------------------------------

_MESSAGES: Final[Mapping[str, str]] = {
    "missing": "項目がない",
    "extra_forbidden": "知らない項目がある (綴りの誤りの疑い)",
    "model_type": "テーブル (鍵と値の組) で書く",
    "dict_type": "テーブル (鍵と値の組) で書く",
    "list_type": "配列で書く",
    "tuple_type": "配列で書く",
    "string_type": "文字列で書く",
    "string_too_short": "空の文字列は書けない",
    "int_type": "整数で書く",
    "int_parsing": "整数で書く",
    "int_from_float": "小数ではなく整数で書く",
    "float_type": "数で書く",
    "float_parsing": "数で書く",
    "bool_type": "true か false で書く",
    "bool_parsing": "true か false で書く",
    "greater_than": "0 より大きい数にする",
    "greater_than_equal": "0 以上の数にする",
    "too_short": "少なくとも 1 つ書く",
    "url_type": "URL を文字列で書く",
    "url_parsing": "URL の形で書く",
    "url_scheme": "http か https の URL で書く",
    "ip_v4_address": "IPv4 のアドレスで書く",
    "datetime_type": "日時で書く",
    "datetime_from_date_parsing": "日時で書く",
}
"""pydantic の誤りの種類ごとの、日本語の文。"""

_PATTERN_MESSAGES: Final[Mapping[str, str]] = {
    "ref": "イメージの参照は <名前>@sha256:<64 桁の 16 進> の形にする (タグだけの参照は、"
    "あとから中身が変わるので使えない)",
    "revision": "版は 40 桁の 16 進の commit にする",
    "sha256": "sha256 は 64 桁の 16 進にする",
    "container_name": "コンテナの名前は、英数字とハイフンだけにする",
}
"""形の決まりに合わなかったときの、項目ごとの日本語の文 (検査 2、3)。"""

_LITERAL_CHOICES: Final[Mapping[str, tuple[str, ...]]] = {
    "kind": tuple(_NODE_COUNTS),
    "role": _NODE_ROLES,
    "only_on": _NODE_ROLES,
    "nodes": _NODE_ROLES,
}
"""決まった値のどれかにする項目の、使える値 (pydantic の英語の文を使わずに示す)。"""


def _field_name(loc: tuple[object, ...]) -> str:
    """誤りの場所の中で、いちばん内側の項目の名前 (配列の添字は飛ばす)。"""
    for part in reversed(loc):
        if isinstance(part, str):
            return part
    return ""


def _describe(err: Mapping[str, Any]) -> str:
    """pydantic の 1 つの誤りを、日本語の文に直す。"""
    kind = str(err.get("type", ""))
    if kind == "value_error":
        # 自分たちの検証が投げた日本語の文 ("Value error, " が前に付く)
        return str(err.get("msg", "")).removeprefix("Value error, ")
    field = _field_name(tuple(err.get("loc", ())))
    if kind == "string_pattern_mismatch":
        return _PATTERN_MESSAGES.get(field, "決まった形に合わない")
    if kind == "literal_error":
        choices = _LITERAL_CHOICES.get(field)
        if choices is not None:
            return f"決まった値のどれかにする (使えるのは {', '.join(choices)})"
        return "決まった値のどれかにする"
    return _MESSAGES.get(kind, f"値が決まりに合わない (検証の種類: {kind})")


def _error_lines(prefix: str, exc: ValidationError) -> list[str]:
    """検証の誤りを、項目の名前つきの 1 行ずつに直す。"""
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        name = f"{prefix}.{loc}" if loc else prefix
        lines.append(f"{name}: {_describe(err)}")
    return lines


def _build[M: BaseModel](model: type[M], raw: Any, item: str, errors: list[str]) -> M | None:
    """1 つの型を検証する。落ちたら、日本語の文を `errors` に足して `None` を返す。"""
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        errors.extend(_error_lines(item, exc))
        return None


# 検証の落ちた入れ子の代わりに置く値。これで `ConfigDef` の after 検証まで進めて、
# 構成そのものの誤りも同じ読み込みで集める (Implementation Notes 1.2)。
_SUBSTITUTE_SOURCE: Final[HttpUrl] = HttpUrl("https://example.invalid/")
_SUBSTITUTE_SETTING: Final[Setting] = Setting(
    flag="--placeholder",
    value="placeholder",
    why="検証の落ちた設定の代わり",
    source=_SUBSTITUTE_SOURCE,
    quote="placeholder",
)
_SUBSTITUTE_IMAGE: Final[ImageRef] = ImageRef(
    ref=f"placeholder@sha256:{'0' * 64}",
    seen_as="placeholder",
    size_bytes=1,
    source=_SUBSTITUTE_SOURCE,
    quote="placeholder",
)
_SUBSTITUTE_WEIGHTS: Final[WeightsRef] = WeightsRef(
    repo="placeholder",
    revision="0" * 40,
    manifest="placeholder.manifest.json",
    mount_at="/placeholder",
    source=_SUBSTITUTE_SOURCE,
    quote="placeholder",
)


# --- TOML の読み込み ----------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"定義のファイルが見つからない: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"定義のファイルを読めない: {path} ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} の TOML を読み取れない: {exc}") from exc


def _refuse(path: Path, errors: list[str]) -> ConfigError:
    """集めた誤りを、1 つの `ConfigError` にまとめる (すべて並べて示す)。"""
    body = "\n".join(errors)
    return ConfigError(f"{path} の定義に誤りがある (見つかったものをすべて示す):\n{body}")


def _check_top_level(data: Mapping[str, Any], allowed: tuple[str, ...]) -> list[str]:
    """知らない最上位の項目を断る (綴りの誤りを、黙って無視しない)。"""
    unknown = [key for key in data if key not in allowed]
    if not unknown:
        return []
    return [f"{key}: 知らない項目がある (使えるのは {', '.join(allowed)})" for key in unknown]


def _check_schema_version(data: Mapping[str, Any], *, required: bool) -> list[str]:
    """書式の版を確かめる (design.md Data Models の `schema_version`)。

    構成の定義では必須。ノードの定義では、見本が持たないので、書いてあるときだけ確かめる。
    """
    if "schema_version" not in data:
        if required:
            return [f"schema_version: 項目がない (いまの書式の版は {CONFIG_SCHEMA_VERSION})"]
        return []
    value = data["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value != CONFIG_SCHEMA_VERSION:
        return [
            f"schema_version: この道具が読めるのは {CONFIG_SCHEMA_VERSION} だけ"
            f" (書かれているのは {value!r})"
        ]
    return []


def _table(data: Mapping[str, Any], key: str, label: str, errors: list[str]) -> dict[str, Any]:
    """最上位のテーブル (`configs` または `nodes`) を取り出す。"""
    if key not in data:
        errors.append(f"{key}: {label}がない ([{key}.<名前>] を 1 つ以上書く)")
        return {}
    value = data[key]
    if not isinstance(value, dict):
        errors.append(f"{key}: テーブルの集まり ([{key}.<名前>]) で書く")
        return {}
    if not value:
        errors.append(f"{key}: {label}が 1 つも書かれていない")
        return {}
    return value


# --- 根拠と、実在するファイルの検査 (検査 1、3) -------------------------


def _check_measured(item: str, evidence: Provenance, repo_root: Path, errors: list[str]) -> None:
    """実測の記録が、実在するファイルを指しているかを確かめる (検査 1)。

    `docs/` の下の相対のパスであることは、型が確かめている。ここでは実在だけを見る。
    これで「あとで測る」という値のない設定が、構成に紛れ込まない。
    """
    if evidence.measured is None:
        return
    if not (repo_root / evidence.measured).is_file():
        errors.append(f"{item}.measured: 実測の記録のファイルがない: {evidence.measured}")


def _check_weights(item: str, weights: WeightsRef, repo_root: Path, errors: list[str]) -> None:
    """重みの参照を確かめる (検査 3。版の 40 桁は型が見ている)。"""
    _check_measured(item, weights, repo_root, errors)
    manifest = weights.manifest
    if manifest != Path(manifest).name:
        errors.append(
            f"{item}.manifest: {'/'.join(_WEIGHTS_DIR)}/ の下のファイルの名前だけを書く: {manifest}"
        )
        return
    if not repo_root.joinpath(*_WEIGHTS_DIR, manifest).is_file():
        errors.append(
            f"{item}.manifest: 重みのマニフェストのファイルがない:"
            f" {'/'.join(_WEIGHTS_DIR)}/{manifest}"
        )


# --- 設定 1 つの検査 (検査 5、6、8、9、10) ------------------------------


def _shown(value: str | None) -> str:
    """誤りの文の中で、値を示すための言い方 (値のないフラグも、そう分かるようにする)。"""
    return value if value else "(値がない)"


def _flag_and_value(setting: Setting) -> tuple[str, str | None]:
    """設定が渡すフラグと値を取り出す (`--flag=値` と、位置の引数の書き方も見る)。

    `flag` を書かない設定は位置の引数なので、値そのものをフラグとして見る。これで、
    フラグを位置の引数として書く抜け道も、同じ検査に掛かる。
    """
    text = setting.flag if setting.flag is not None else (setting.value or "")
    name, separator, inline = text.partition("=")
    # 前後の空白の変種 (" --privileged") を、早く断る (大小の違いまでは見ない)
    name = name.strip()
    if separator:
        return name, inline
    return name, setting.value if setting.flag is not None else None


def _is_json_braces(braces: str) -> bool:
    """波かっこの組が、置き換えの印ではなく JSON の値かどうか。

    design.md 「probe」の、層の数を減らす `--hf-overrides` の値は JSON になるので、中身
    (前後の空白を除く) が空か `"` で始まる組は、文字のまま扱う。入れ子の JSON は、内側の
    組だけがここに来る (`_PLACEHOLDER_RE` は、中に波かっこを含む組を拾わない)。
    """
    inner = braces[1:-1].strip()
    return not inner or inner.startswith('"')


def _check_placeholders(item: str, setting: Setting, errors: list[str]) -> None:
    """検査 6: 知らない置き換えの印を断る (JSON の値は、印として見ない)。"""
    for text, where in ((setting.flag, "flag"), (setting.value, "value")):
        if text is None:
            continue
        unknown = sorted(
            {
                braces
                for braces in _PLACEHOLDER_RE.findall(text)
                if braces not in _ALLOWED_PLACEHOLDERS and not _is_json_braces(braces)
            }
        )
        if unknown:
            allowed = ", ".join(sorted(_ALLOWED_PLACEHOLDERS))
            errors.append(
                f"{item}.{where}: 知らない置き換えの印がある: {', '.join(unknown)}"
                f" (使えるのは {allowed})"
            )


def _is_inside_remote_root(source: str) -> bool:
    """`--mount` の元が、Spark の置き場所の下にあるかどうか (検査 9)。

    `{remote_root}` そのものか、`{remote_root}/` で始まり、残りの道筋に `..` を含まない
    ものだけを通す。`{remote_root}` を実際の道筋に埋めるのは `plan` で、そこには元の道筋の
    検査がないので、ここが唯一の歯止めになる (requirements 2.3、2.4)。`..` の見方は、
    `types.py` の `_check_docs_relative_path` に揃える。
    """
    if source == _REMOTE_ROOT_MARKER:
        return True
    prefix = f"{_REMOTE_ROOT_MARKER}/"
    if not source.startswith(prefix):
        return False
    return ".." not in Path(source[len(prefix) :]).parts


def _check_mount(item: str, value: str | None, errors: list[str]) -> None:
    """検査 9: `--mount` は `type=bind` だけ、元は `{remote_root}` の下だけ。"""
    if not value:
        errors.append(
            f"{item}: --mount には値が要る (type=bind,source={_REMOTE_ROOT_MARKER}/…,target=… の形)"
        )
        return
    parts: dict[str, list[str]] = {}
    for piece in value.split(","):
        # `readonly` のような、値を取らない指定もある (鍵だけを覚えておく)
        key, _, item_value = piece.partition("=")
        parts.setdefault(key.strip(), []).append(item_value)
    if parts.get("type") != ["bind"]:
        errors.append(f"{item}: --mount は type=bind だけを使う: {value}")
    sources = [source for key in _MOUNT_SOURCE_KEYS for source in parts.get(key, [])]
    if not sources:
        errors.append(f"{item}: --mount に source がない: {value}")
    for source in sources:
        if not _is_inside_remote_root(source):
            errors.append(
                f"{item}: --mount の source は {_REMOTE_ROOT_MARKER} の下だけにする"
                f" (Spark のほかの場所を、コンテナに見せない): {source}"
            )


def _check_docker_setting(item: str, setting: Setting, errors: list[str]) -> None:
    """docker の設定を確かめる (検査 8、9、10)。"""
    flag, value = _flag_and_value(setting)
    reason = _FORBIDDEN_DOCKER_FLAGS.get(flag)
    if reason is not None:
        errors.append(f"{item}: {flag} は使えない ({reason})")
        return
    if flag == "--restart":
        if value != "no":
            errors.append(
                f"{item}: --restart は no だけを使う (自動で起こし直さない): {_shown(value)}"
            )
        return
    if flag == "--mount":
        _check_mount(item, value, errors)
        return
    if flag == "--device" and value not in _ALLOWED_DEVICES:
        errors.append(
            f"{item}: --device に書けるのは {', '.join(sorted(_ALLOWED_DEVICES))} だけ"
            f" (根拠があっても、一覧にないものは断る): {_shown(value)}"
        )
        return
    if flag == "--cap-add" and value not in _ALLOWED_CAPABILITIES:
        errors.append(
            f"{item}: --cap-add に書けるのは {', '.join(sorted(_ALLOWED_CAPABILITIES))} だけ"
            f" (根拠があっても、一覧にないものは断る): {_shown(value)}"
        )


def _check_setting(
    item: str, section: str, setting: Setting, repo_root: Path, errors: list[str]
) -> None:
    """設定 1 つを確かめる (検査 1 の実在、5、6、と docker の 8〜10)。

    docker の検査 (8、9、10) は `docker` の節にだけ掛ける (`args` と `env` は、イメージの
    参照のあとに続く引数と環境変数で、docker が読まない)。投機的デコードの検査 (5) だけは、
    どの節に書いても構成が投機的デコードを持つことに変わりがないので、3 つの節すべてに掛ける。
    """
    _check_measured(item, setting, repo_root, errors)
    _check_placeholders(item, setting, errors)
    flag, _ = _flag_and_value(setting)
    if flag in _SPECULATIVE_FLAGS:
        errors.append(f"{item}: 投機的デコードの指定は、最初の起動では使わない: {flag}")
    if section == "docker":
        _check_docker_setting(item, setting, errors)


def _check_node_count(prefix: str, kind: Any, nodes: Any, errors: list[str]) -> None:
    """`kind` ごとのノードの数を確かめる (tasks.md の Implementation Notes 1.2)。

    構成の全体の検証 (`ConfigDef`) が落ちても、この誤りを取りこぼさないように、TOML に
    書かれた値から見る。`kind` そのものや `nodes` の書き方の誤りは、型の段で示しているので、
    ここでは黙って返る (同じことを 2 度言わない)。
    """
    expected = _NODE_COUNTS.get(kind) if isinstance(kind, str) else None
    if expected is None or not isinstance(nodes, list):
        return
    if len(nodes) != expected:
        errors.append(
            f"{prefix}.nodes: kind が {kind} の構成は、ノードを {expected} つ書く"
            f" (書かれているのは {len(nodes)} つ)"
        )


# --- 構成 1 つの組み立て ------------------------------------------------


def _build_config(
    name: str, fields: Mapping[str, Any], repo_root: Path
) -> tuple[ConfigDef | None, list[str]]:
    """構成 1 つを検証する。入れ子を 1 つずつ見てから、構成の全体を見る。"""
    prefix = f"configs.{name}"
    errors: list[str] = []
    candidate: dict[str, Any] = dict(fields)

    written_name = candidate.get("name")
    if written_name is not None and written_name != name:
        errors.append(
            f"{prefix}.name: 名前は鍵から取るので書かない (書かれた名前: {written_name!r})"
        )
    candidate["name"] = name

    if "image" in candidate:
        image = _build(ImageRef, candidate["image"], f"{prefix}.image", errors)
        if image is None:
            candidate["image"] = _SUBSTITUTE_IMAGE
        else:
            candidate["image"] = image
            _check_measured(f"{prefix}.image", image, repo_root, errors)

    if candidate.get("weights") is not None:
        weights = _build(WeightsRef, candidate["weights"], f"{prefix}.weights", errors)
        if weights is None:
            candidate["weights"] = _SUBSTITUTE_WEIGHTS
        else:
            candidate["weights"] = weights
            _check_weights(f"{prefix}.weights", weights, repo_root, errors)

    for section in _SETTING_SECTIONS:
        raw_section = candidate.get(section)
        if not isinstance(raw_section, dict):
            continue
        built: dict[str, Setting] = {}
        for key, raw_setting in raw_section.items():
            item = f"{prefix}.{section}.{key}"
            setting = _build(Setting, raw_setting, item, errors)
            if setting is None:
                built[key] = _SUBSTITUTE_SETTING
                continue
            built[key] = setting
            _check_setting(item, section, setting, repo_root, errors)
        candidate[section] = built

    _check_node_count(prefix, candidate.get("kind"), candidate.get("nodes"), errors)
    return _build(ConfigDef, candidate, prefix, errors), errors


# --- 読み込みの入口 ----------------------------------------------------


def load_configs(path: Path, repo_root: Path) -> dict[str, ConfigDef]:
    """`configs.toml` の `[configs.<名前>]` をすべて読み、検査する。

    `name` はテーブルの鍵から取る。誤りは、項目の名前つきの日本語の文で、すべて並べた
    1 つの `ConfigError` にする。検査 7 (直結の値) は、ここではなく `select_config` で
    行う (取得と読み取りと縮小の確認は、直結の値を実測する前に流すため)。
    """
    data = _load_toml(path)
    errors: list[str] = []
    errors.extend(_check_schema_version(data, required=True))
    errors.extend(_check_top_level(data, ("schema_version", "configs")))

    result: dict[str, ConfigDef] = {}
    for name, fields in _table(data, "configs", "構成の定義", errors).items():
        if not isinstance(fields, dict):
            errors.append(f"configs.{name}: テーブル ([configs.{name}]) で書く")
            continue
        config, config_errors = _build_config(name, fields, repo_root)
        errors.extend(config_errors)
        if config is not None:
            result[name] = config

    if errors:
        raise _refuse(path, errors)
    return result


def load_nodes(path: Path, repo_root: Path) -> dict[NodeRole, NodeDef]:
    """`nodes.toml` の `[nodes.head]` と `[nodes.worker]` を読む。

    `role` はテーブルの鍵から取る。直結の側の 3 つ (`fabric_addr`、`fabric_ifname`、
    `fabric_measured`) は、実測するまで空のままでよい。空のままでも、取得、イメージの中の
    読み取り、縮小の確認の構成は選べる (検査 7 は `select_config` で行う)。
    """
    data = _load_toml(path)
    errors: list[str] = []
    errors.extend(_check_schema_version(data, required=False))
    errors.extend(_check_top_level(data, ("schema_version", "nodes")))

    table = _table(data, "nodes", "ノードの定義", errors)
    for key in table:
        if key not in _NODE_ROLES:
            errors.append(f"nodes.{key}: 知らない役割 (使えるのは {', '.join(_NODE_ROLES)})")

    result: dict[NodeRole, NodeDef] = {}
    for role in _NODE_ROLES:
        fields = table.get(role)
        if fields is None:
            continue
        item = f"nodes.{role}"
        if not isinstance(fields, dict):
            errors.append(f"{item}: テーブル ([{item}]) で書く")
            continue
        candidate = dict(fields)
        written_role = candidate.get("role")
        if written_role is not None and written_role != role:
            errors.append(
                f"{item}.role: 役割は鍵から取るので書かない (書かれた役割: {written_role!r})"
            )
        candidate["role"] = role
        node = _build(NodeDef, candidate, item, errors)
        if node is None:
            continue
        if node.fabric_measured is not None and not (repo_root / node.fabric_measured).is_file():
            errors.append(
                f"{item}.fabric_measured: 実測の記録のファイルがない: {node.fabric_measured}"
            )
        result[role] = node

    if errors:
        raise _refuse(path, errors)
    return result


def select_config(
    configs: Mapping[str, ConfigDef], name: str, nodes: Mapping[NodeRole, NodeDef]
) -> ConfigDef:
    """名前で構成を 1 つ選び、検査 7 (直結の値) を、選んだ構成について行う。

    知らない名前は、使える名前を添えて断る。`kind` が `serve` か `job` で、ノードが 2 つの
    構成は、2 台の直結の値 (`fabric_addr`、`fabric_ifname`、`fabric_measured`) が揃って
    いなければ断る (requirements 4.7)。`fetch`、`inspect`、`probe` には掛からない。
    """
    config = configs.get(name)
    if config is None:
        available = ", ".join(sorted(configs)) if configs else "(なし)"
        raise ConfigError(f"構成の定義 '{name}' が見つからない。使える名前: {available}")

    errors: list[str] = []
    needs_fabric = config.kind in _FABRIC_KINDS and len(config.nodes) == 2
    for role in config.nodes:
        node = nodes.get(role)
        if node is None:
            errors.append(f"nodes.{role}: ノードの定義がない (構成 '{name}' が使う役割)")
            continue
        if not needs_fabric:
            continue
        for field, value in (
            ("fabric_addr", node.fabric_addr),
            ("fabric_ifname", node.fabric_ifname),
            ("fabric_measured", node.fabric_measured),
        ):
            if value is None:
                errors.append(
                    f"nodes.{role}.{field}: 直結の値がない"
                    f" (kind が {config.kind} の 2 台の構成に要る。"
                    "`serve netcheck links` の実測で埋める)"
                )
    if errors:
        body = "\n".join(errors)
        raise ConfigError(f"構成 '{name}' は選べない (見つかったものをすべて示す):\n{body}")
    return config
