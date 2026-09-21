"""構成とノードから、`docker run` の引数の列を組み立てる (design.md 「組み立てと読み取り › plan」)。

**入出力のない、純粋な関数だけを置く**。ファイルも、環境変数 (`os.environ`) も、時刻も、
ネットワークも読まない。`subprocess` を import しない (依存の向きにより、読み込むのは
`serving_kit.types` と `serving_kit.config` だけ。`remote` は読み込まない)。同じ入力からは、
いつでも同じ列ができる。

守る決まり:

- **1 つの起動の仕組み** (design.md 「Architecture Integration」): この道具が起こす
  コンテナは、5 つの `kind` (`serve`、`probe`、`job`、`fetch`、`inspect`) のどれも、この
  1 つの関数を通る。どれも `docker run -d` で切り離して起こし、名前とラベルを必ず付ける。
  前面で動かす指定 (`-it`、`-a`、`--attach`) と、終了時に自動で消す指定 (`--rm`) を作る
  経路がなく、`docker build` の列を作る経路もない (requirements 8.6)
- **引数の順序** (design.md 「plan」): `docker run` → `-d` → `--pull never` → `--name` →
  ラベル (鍵の順) → `--restart no` → 構成の `docker` の設定 → 環境変数 → イメージの参照 →
  構成の `args` (TOML に書いた順)。`flag` を書かない設定は、値だけの位置の引数になる
- **道具が必ず付けるもの**: `-d`、`--pull never`、`--name`、`--label`、`--restart no`。
  根拠は `TOOL_SETTINGS` に、構成と同じ `Setting` (`Provenance`) の形で持つ。構成の
  `docker` に同じフラグが書いてあれば、二重に付けずに断る (長い形、短い形、`=` つなぎ、
  位置の引数、前後の空白つき、短い形の組み合わせのどれで書いても断る)
- **環境変数**: 構成の `env` に名前と値の組で書いたものだけを `-e NAME=value` で渡す。
  Mac の側の環境は引き継がない (`-e NAME` の値なしの形は、docker が動く側の環境から値を
  取るので使わない)。名前が秘密らしいものは断る (requirements 2.6)。`docker` の節からの
  `-e` / `--env` / `--env-file` は、この検査を通らないので断る
- **置き換えの印**: 7 つの印を、`str.format` ではなく決まった文字列の置換で埋める
  (`--hf-overrides` の JSON の波かっこを壊さない)。埋める値がない印は、黙って空の文字列に
  せず、項目の名前と理由を示して断る
- **誤りは、すべて並べて示す** (design.md 「Error Handling」の「早く断る」)。誤りは
  `PlanError` (`ConfigError` の一種) で、終了コードは 1 (前提の不足) になる

`config-sha256` の取り方 (design.md 「plan」):

**2 台ぶん (構成の `nodes` のすべて) の、組み立てた引数の列 (置き換えの印を埋めたあと) から、
`--label` の引数 (フラグとその値) を除いたものの sha256**。2 台のどちらのコンテナにも、同じ
値が付く。`description`、`why`、根拠、`ready_timeout_s` を変えても変わらず、コンテナの中身に
効く変更 (フラグ、値、イメージ、重み、ノードの値) だけで変わる。

直列化は、**1 つ 1 つの語に、UTF-8 のバイト数を前に付けて連ねる** (`<バイト数>:<語>`)。
役割の名前と、語の数も同じ形で挟む。長さが前に付くので、区切りの曖昧さがなく、`["a b"]` と
`["a", "b"]` が同じ材料にならない (前者は `3:a b`、後者は `1:a1:b` で、語の数も違う)。

設計の文からの、意図した違いと、受け入れたこと (`remote.py` と同じ流儀で並べる):

1. **誤りの型は `PlanError` で、`config.ConfigError` の一種にした**。ここで見つかるのは
   Spark に触る前に分かる前提の不足なので、`config` の検査と同じ終了コード 1 に乗る
   (design.md 「Error Handling」の表)。そのため、この module は `config` を読み込む
   (design.md の依存の向きが許す範囲)
2. **`--name` は、`config-sha256` の材料に残す** (design が除くと書いているのは `--label`
   だけ)。受け入れたこと: A/B の回の番号は名前に入るので、同じ腕の 1 回目と 2 回目で
   `config-sha256` が変わる。`already_running` の判定 (2 台の名前とダイジェストと
   `config-sha256` の一致) は、A/B の回を跨がない
3. **秘密らしい名前は、部分一致で断る** (design の文のとおり)。受け入れたこと:
   `TOKENIZERS_PARALLELISM` のような、秘密でない名前も断られる。断る側に寄せる
4. **秘密らしい語に `CREDENTIAL` を足した** (design が挙げるのは `TOKEN`、`KEY`、
   `SECRET`、`PASSWORD` の 4 つ)
5. **使わない役割を指す `only_on` を断る** (design にない検査)。黙って落とすと、設定が
   消えたことに気づけないため
6. **構成の `docker` に書けないフラグは、書き方を問わずに断る** (design は長い形だけを
   挙げる)。短い形 (`-l`、`-e`)、`=` つなぎ、位置の引数、前後の空白、短い形の組み合わせ
   (`-itd`) を、1 文字ずつ見て断る。受け入れたこと: 値を詰めて書いた短いフラグ
   (`-w/data` のような形) で、値の側に `d`、`l`、`e`、`i`、`t`、`a` が入っていると、
   これも断る (この例は `d` で断る)。構成に書ける正当な docker の設定は、すべて長い形で
   書けるので、厳しい側に寄せた
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from pydantic import HttpUrl

from serving_kit.config import ConfigError
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    NodeDef,
    NodeRole,
    Setting,
)

__all__ = [
    "CONTAINER_NAME_PREFIX",
    "LABEL_CONFIG",
    "LABEL_CONFIG_SHA256",
    "LABEL_IMAGE",
    "LABEL_KIND",
    "LABEL_NAMESPACE",
    "LABEL_OWNER",
    "LABEL_ROLE",
    "LABEL_RUN",
    "LABEL_STARTED_AT",
    "LABEL_WEIGHTS",
    "OWNER",
    "OWNER_FILTER",
    "TOOL_SETTINGS",
    "PlanError",
    "build_plans",
]


class PlanError(ConfigError):
    """構成とノードから引数の列を組み立てられなかったときの誤り。

    `ConfigError` の一種にしてある。ここで見つかるのは、構成とノードの定義の食い違い
    (埋める値のない置き換えの印、道具が必ず付ける引数の重複、秘密らしい環境変数の名前) で、
    どれも Spark に触る前に分かる前提の不足なので、`config` の検査と同じ終了コード 1 に
    乗せるのが正しい (design.md 「Error Handling」の表)。
    """


# --- ラベル (design.md 「plan」の共通の 7 つ) ---------------------------

LABEL_NAMESPACE: Final[str] = "vllm-baseline"
"""ラベルの鍵の前置き。変えたら、古いラベルのコンテナが管理の外に出る (Revalidation Triggers)。"""

OWNER: Final[str] = "serving-kit"
"""`vllm-baseline.owner` の値。この値のコンテナだけが、この道具のものである。"""

LABEL_OWNER: Final[str] = f"{LABEL_NAMESPACE}.owner"
LABEL_CONFIG: Final[str] = f"{LABEL_NAMESPACE}.config"
LABEL_KIND: Final[str] = f"{LABEL_NAMESPACE}.kind"
LABEL_ROLE: Final[str] = f"{LABEL_NAMESPACE}.role"
LABEL_IMAGE: Final[str] = f"{LABEL_NAMESPACE}.image"
LABEL_STARTED_AT: Final[str] = f"{LABEL_NAMESPACE}.started-at"
LABEL_CONFIG_SHA256: Final[str] = f"{LABEL_NAMESPACE}.config-sha256"
LABEL_WEIGHTS: Final[str] = f"{LABEL_NAMESPACE}.weights"
"""重みの参照を持つ構成だけに付く 8 つめのラベル (`<repo>@<revision>`)。"""

LABEL_RUN: Final[str] = f"{LABEL_NAMESPACE}.run"
"""通信の確認の A/B の回だけに付くラベル (`<腕>-<回>`。design.md 「netcheck」)。"""

OWNER_FILTER: Final[str] = f"{LABEL_OWNER}={OWNER}"
"""`docker ps --filter label=…` に渡す文字列。自分のものだけを選ぶ、唯一の仕組み。"""

CONTAINER_NAME_PREFIX: Final[str] = "vb"
"""コンテナの名前の前置き (`vb-<構成>-<役割>`)。"""

_STARTED_AT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
"""ラベルに書く UTC の日時の形 (秒まで。ラベルは人も読む)。"""


# --- 道具が必ず付ける引数と、その根拠 -----------------------------------

_DOCKER_RUN_REFERENCE: Final[HttpUrl] = HttpUrl(
    "https://docs.docker.com/reference/cli/docker/container/run/"
)
"""出典は、Docker の公式の `docker container run` の参照 (research.md §c が引いた URL)。"""

_TOOL_DETACH: Final[Setting] = Setting(
    flag="-d",
    why="ssh のセッションを抜けても動き続ける。5 つの kind のどれも、切り離して起こす"
    " (前面で動かす経路を作らない)",
    source=_DOCKER_RUN_REFERENCE,
    quote="The `--detach` (or `-d`) flag starts a container as a background process"
    " that doesn't occupy your terminal window.",
)

_TOOL_PULL: Final[Setting] = Setting(
    flag="--pull",
    value="never",
    why="固定したダイジェストのイメージが手元にないときに、黙って取得しない (requirements 3.3)",
    source=_DOCKER_RUN_REFERENCE,
    quote="Do not pull the image, even if it's missing, and produce an error"
    " if the image does not exist in the image cache.",
)

_TOOL_NAME: Final[Setting] = Setting(
    flag="--name",
    value=f"{CONTAINER_NAME_PREFIX}-<構成>-<役割>",
    why="人が読める識別と、起動の前に巻き戻すコマンドまで書けるようにするため"
    " (design.md 「remote」の計画の組)",
    source=_DOCKER_RUN_REFERENCE,
    quote="Assign a name to the container",
)

_TOOL_LABEL: Final[Setting] = Setting(
    flag="--label",
    value="<鍵>=<値>",
    why="この道具が起こしたコンテナだけを選ぶ、唯一の仕組み (requirements 2.3)。"
    "名前での絞り込みは部分一致なので使わない",
    source=_DOCKER_RUN_REFERENCE,
    quote="A label is a `key=value` pair that applies metadata to a container.",
)

_TOOL_RESTART: Final[Setting] = Setting(
    flag="--restart",
    value="no",
    why="P1 では、落ちたら落ちたままにして、落ちた事実を記録する (自動の起こし直しは P6)",
    source=_DOCKER_RUN_REFERENCE,
    quote="--restart | no | Restart policy to apply when a container exits",
)

TOOL_SETTINGS: Final[Mapping[str, Setting]] = {
    "detach": _TOOL_DETACH,
    "pull": _TOOL_PULL,
    "name": _TOOL_NAME,
    "label": _TOOL_LABEL,
    "restart": _TOOL_RESTART,
}
"""道具が必ず付ける 5 つの引数と、その根拠 (design.md 「plan」: 構成には書かない)。

`--name` と `--label` の `value` は、形を示すための見本で、組み立てのときに入れ替える。
"""

_WHY_LABEL: Final[str] = (
    "道具が必ず付ける (共通の 7 つ、または 8 つ)。docker は同じ鍵のラベルを後勝ちで畳むので、"
    "構成から書けると、所有のラベルの絞り込みから外れる (requirements 2.3)"
)
_WHY_ENV: Final[str] = (
    "環境変数は、構成の env の節に、名前と値の組で書く。docker の節から渡すと、秘密らしい"
    "名前の検査を通らず、値なしの形は、docker が動く側 (Spark) の環境から値を取る"
    " (requirements 2.6)"
)
_WHY_FOREGROUND: Final[str] = (
    "切り離して起こすので、前面で動かす指定は使えない (記録が消え、片付けの筋道が 2 つになる)"
)

_TOOL_OWNED_FLAGS: Final[Mapping[str, str]] = {
    "-d": "道具が必ず付ける (切り離して起こす)",
    "--detach": "道具が必ず付ける (切り離して起こす)",
    "--pull": "道具が必ず付ける (--pull never)",
    "--name": f"道具が必ず付ける ({CONTAINER_NAME_PREFIX}-<構成>-<役割>)",
    "-l": _WHY_LABEL,
    "--label": _WHY_LABEL,
    "--label-file": f"ファイルからラベルを足すと、ラベルでの選択が壊れる。{_WHY_LABEL}",
    "--restart": "道具が必ず付ける (--restart no)",
}
"""構成の `docker` に書いてあれば断るフラグ (二重に付けない)。短い形も並べる。"""

_ENV_FLAGS: Final[Mapping[str, str]] = {
    "-e": _WHY_ENV,
    "--env": _WHY_ENV,
    "--env-file": f"ファイルから環境変数を足すと、名前の検査を通らない。{_WHY_ENV}",
}
"""環境変数を渡すフラグ (構成の `docker` からは渡せない。requirements 2.6)。"""

_FOREGROUND_FLAGS: Final[Mapping[str, str]] = {
    "-i": _WHY_FOREGROUND,
    "--interactive": _WHY_FOREGROUND,
    "-t": _WHY_FOREGROUND,
    "--tty": _WHY_FOREGROUND,
    "-a": _WHY_FOREGROUND,
    "--attach": _WHY_FOREGROUND,
    "--rm": "終了したコンテナが消えて、記録が読めなくなる",
}
"""前面で動かす指定と、終了時に自動で消す指定 (構成にあれば断る。requirements 8.6)。

`--rm` は `config` の検査 8 でも断るが、引数の列を組み立てるのはここだけなので、ここが
最後の歯止めになる。
"""

_REFUSED_DOCKER_FLAGS: Final[Mapping[str, str]] = {
    **_TOOL_OWNED_FLAGS,
    **_ENV_FLAGS,
    **_FOREGROUND_FLAGS,
}
"""構成の `docker` の節に書けないフラグと、その理由 (3 つの一覧を合わせたもの)。"""

_REFUSED_SHORT_FLAG_CHARS: Final[Mapping[str, str]] = {
    flag[1]: flag
    for flag in _REFUSED_DOCKER_FLAGS
    if len(flag) == 2 and flag.startswith("-") and not flag.startswith("--")
}
"""短い形の 1 文字と、それが表すフラグ (`d`、`l`、`e`、`i`、`t`、`a`)。

`-itd` のような短い形の組み合わせと、`-eNAME=v` のように値を詰めて書く形を、1 文字ずつ
見て断るために使う。一覧は `_REFUSED_DOCKER_FLAGS` から作るので、出どころは 1 つである。
"""

_ENV_FLAG: Final[str] = "-e"
"""環境変数を渡すフラグ。`NAME=value` の形だけを使う (道具だけが付ける)。"""

_SECRET_WORDS: Final[tuple[str, ...]] = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")
"""秘密らしい環境変数の名前 (requirements 2.6)。

design.md 「plan」が挙げる 4 つに `CREDENTIAL` を足した (断る側に広げた)。
"""

_ENV_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CONFIG_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
_ARM_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_SECTIONS: Final[tuple[str, ...]] = ("docker", "args", "env")


# --- 置き換えの印 -------------------------------------------------------

_MARKER_NODE_FABRIC_ADDR: Final[str] = "{node.fabric_addr}"
_MARKER_NODE_FABRIC_IFNAME: Final[str] = "{node.fabric_ifname}"
_MARKER_NODE_RANK: Final[str] = "{node.rank}"
_MARKER_HEAD_FABRIC_ADDR: Final[str] = "{head.fabric_addr}"
_MARKER_HEAD_LAN_ADDR: Final[str] = "{head.lan_addr}"
_MARKER_WEIGHTS_MOUNT_AT: Final[str] = "{weights.mount_at}"
_MARKER_REMOTE_ROOT: Final[str] = "{remote_root}"


def _substitutions(
    config: ConfigDef, nodes: Mapping[NodeRole, NodeDef], node: NodeDef, rank: int
) -> Mapping[str, tuple[str | None, str]]:
    """7 つの印と、その値 (埋められないときは `None` と、その理由)。

    `{node.…}` と `{remote_root}` は、いま組み立てている 1 台の値。`{head.…}` は、2 台の
    どちらのコンテナでも head の値になる (分散の初期化の宛先)。`{node.rank}` は、構成の
    `nodes` に書かれた順 (head が 0、worker が 1)。
    """
    head = nodes.get("head")
    weights = config.weights
    return {
        _MARKER_NODE_FABRIC_ADDR: (
            None if node.fabric_addr is None else str(node.fabric_addr),
            f"nodes.{node.role}.fabric_addr が空 (`serve netcheck links` の実測で埋める)",
        ),
        _MARKER_NODE_FABRIC_IFNAME: (
            node.fabric_ifname,
            f"nodes.{node.role}.fabric_ifname が空 (`serve netcheck links` の実測で埋める)",
        ),
        _MARKER_NODE_RANK: (str(rank), ""),
        _MARKER_HEAD_FABRIC_ADDR: (
            None if head is None or head.fabric_addr is None else str(head.fabric_addr),
            "nodes.head.fabric_addr が空 (`serve netcheck links` の実測で埋める)",
        ),
        _MARKER_HEAD_LAN_ADDR: (
            None if head is None else str(head.lan_addr),
            "nodes.head の定義がない",
        ),
        _MARKER_WEIGHTS_MOUNT_AT: (
            None if weights is None else weights.mount_at,
            f"構成 '{config.name}' が weights を持たない",
        ),
        _MARKER_REMOTE_ROOT: (node.remote_root, ""),
    }


def _filled(
    text: str, subs: Mapping[str, tuple[str | None, str]], item: str, errors: list[str]
) -> str:
    """決まった 7 つの印を、文字列の置換で埋める。

    `str.format` を使わないのは、`--hf-overrides` の値が JSON で、波かっこを含むため
    (tasks.md の Implementation Notes 1.3)。知らない `{…}` は、`config` の検査 6 が
    読み込みのときに断っているので、ここでは触らずに、そのまま通す。
    """
    for marker, (value, reason) in subs.items():
        if marker not in text:
            continue
        if value is None:
            errors.append(f"{item}: {marker} を埋める値がない ({reason})")
            continue
        text = text.replace(marker, value)
    return text


# --- 設定 1 つを、引数の列に直す ----------------------------------------


def _argv_of(setting: Setting) -> tuple[str, ...]:
    """道具が必ず付ける設定を、引数の列に直す (印を持たないので、置換をしない)。"""
    if setting.flag is None:
        return (setting.value or "",)
    if setting.value is None:
        return (setting.flag,)
    return (setting.flag, setting.value)


def _rendered(
    setting: Setting,
    subs: Mapping[str, tuple[str | None, str]],
    item: str,
    errors: list[str],
) -> tuple[str, ...]:
    """構成の設定 1 つを、印を埋めて引数の列に直す。

    `flag` のない設定は位置の引数 (値だけ)、`value` のない設定はフラグだけになる。
    """
    flag = None if setting.flag is None else _filled(setting.flag, subs, f"{item}.flag", errors)
    value = None if setting.value is None else _filled(setting.value, subs, f"{item}.value", errors)
    if flag is None:
        if value is None:  # 型が断っているので、ここには来ない (念のための歯止め)
            errors.append(f"{item}: flag のない設定は位置の引数なので、value が要る")
            return ()
        return (value,)
    if value is None:
        return (flag,)
    return (flag, value)


def _env_argv(
    setting: Setting,
    subs: Mapping[str, tuple[str | None, str]],
    item: str,
    errors: list[str],
) -> tuple[str, ...]:
    """環境変数 1 つを `-e NAME=value` に直す (requirements 2.6)。"""
    name = setting.flag
    if name is None:
        errors.append(f"{item}: 環境変数の設定には、変数の名前を flag に書く")
        return ()
    if setting.value is None:
        errors.append(
            f"{item}: 環境変数には値が要る (-e {name} の値なしの形は、docker が Mac の側の"
            "環境から値を取るので使わない)"
        )
        return ()
    _check_env_name(name, item, errors)
    return (_ENV_FLAG, f"{name}={_filled(setting.value, subs, f'{item}.value', errors)}")


def _check_env_name(name: str, item: str, errors: list[str]) -> None:
    """環境変数の名前を確かめる (秘密らしい名前を断る。requirements 2.6)。"""
    if not _ENV_NAME_RE.fullmatch(name):
        errors.append(f"{item}: 環境変数の名前は、英数字と下線だけにする: {name}")
        return
    found = [word for word in _SECRET_WORDS if word in name.upper()]
    if found:
        errors.append(
            f"{item}: 秘密らしい名前の環境変数は、Spark に渡さない"
            f" ({', '.join(found)} を含む): {name}"
        )


# --- 構成そのものの検査 -------------------------------------------------


def _flag_of(setting: Setting) -> str:
    """設定が渡すフラグの名前 (`--flag=値` と、位置の引数の書き方も見る)。

    `config._flag_and_value` と同じ見方にして (`=` で切り、前後の空白を落とす)、フラグを
    位置の引数として書く抜け道 (`flag` を書かずに `value = "-l"`) も、同じ検査に掛ける。
    """
    text = setting.flag if setting.flag is not None else (setting.value or "")
    name, _, _ = text.partition("=")
    return name.strip()


def _refused_flag(written: str) -> tuple[str, str] | None:
    """断るフラグなら、(その短い形か長い形, 理由) を返す。ほかは `None`。

    長い形と短い形は、そのまま一覧で引く。短い形は、組み合わせ (`-itd`) と、値を詰めて
    書く形 (`-eNAME=v`) があるので、`-` で始まり `--` で始まらない語は、1 文字ずつ見て、
    断る短い形を 1 つでも含めば断る。docker の短いフラグは 1 文字なので、この見方で、
    組み合わせの途中に置いた `d`、`l`、`e`、`i`、`t`、`a` も捕まる。

    値を詰めて書いた短いフラグ (`-w/data` のような形) で、値の側にこの 6 文字が入って
    いると、これも断る。構成に書ける正当な docker の設定は、すべて長い形で書けるので、
    厳しい側に寄せた (module の docstring の「意図した違い」の 6)。
    """
    reason = _REFUSED_DOCKER_FLAGS.get(written)
    if reason is not None:
        return written, reason
    if not written.startswith("-") or written.startswith("--"):
        return None
    for char in written[1:]:
        short = _REFUSED_SHORT_FLAG_CHARS.get(char)
        if short is not None:
            return short, _REFUSED_DOCKER_FLAGS[short]
    return None


def _check_docker_flags(config: ConfigDef, prefix: str, errors: list[str]) -> None:
    """構成の `docker` の節に書けないフラグを断る。

    道具が必ず付ける引数 (`-d`、`--pull`、`--name`、`--label`、`--restart`)、環境変数の
    フラグ (`-e`、`--env`、`--env-file`)、前面で動かす指定と `--rm` を、長い形、短い形、
    `=` つなぎ、位置の引数、前後の空白つき、短い形の組み合わせのどれで書いても断る。
    """
    for key, setting in config.docker.items():
        written = _flag_of(setting)
        found = _refused_flag(written)
        if found is None:
            continue
        flag, reason = found
        shown = flag if flag == written else f"{written} の {flag}"
        errors.append(f"{prefix}.docker.{key}: {shown} は構成に書けない ({reason})")


def _check_only_on(config: ConfigDef, prefix: str, errors: list[str]) -> None:
    """使わない役割を指す `only_on` を断る (黙って落とすと、設定が消えたことに気づけない)。"""
    for section in _SECTIONS:
        settings: Mapping[str, Setting] = getattr(config, section)
        for key, setting in settings.items():
            if setting.only_on is not None and setting.only_on not in config.nodes:
                errors.append(
                    f"{prefix}.{section}.{key}: only_on が、この構成が使わない役割を指している"
                    f" ({setting.only_on}。使うのは {', '.join(config.nodes)})"
                )


def _applies(setting: Setting, role: NodeRole) -> bool:
    """この設定が、この役割のノードに付くかどうか (2 台の差は `only_on` と印だけ)。"""
    return setting.only_on is None or setting.only_on == role


# --- 名前、時刻、A/B の回 -----------------------------------------------


def _check_config_name(name: str, errors: list[str]) -> None:
    """構成の名前が、コンテナの名前にできるかを確かめる。"""
    if not _CONFIG_NAME_RE.fullmatch(name):
        errors.append(
            f"configs.{name}: この名前は、コンテナの名前"
            f" ({CONTAINER_NAME_PREFIX}-<構成>-<役割>) にできない"
            " (英数字とハイフンだけにする)"
        )


def _container_name(config_name: str, role: NodeRole, repeat_index: int | None) -> str:
    """`vb-<構成>-<役割>` (A/B の回は `-r<回>` が付く。design.md 「netcheck」)。"""
    suffix = "" if repeat_index is None else f"-r{repeat_index}"
    return f"{CONTAINER_NAME_PREFIX}-{config_name}-{role}{suffix}"


def _started_at_label(started_at: datetime, errors: list[str]) -> str:
    """起こした時刻のラベル (UTC)。素の日時は受けない。"""
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        errors.append("started_at: 時差の付いた日時にする (素の日時は、UTC か不明かが決まらない)")
        return ""
    return started_at.astimezone(UTC).strftime(_STARTED_AT_FORMAT)


def _run_label(arm: str | None, repeat_index: int | None, errors: list[str]) -> str | None:
    """A/B の回のラベル `<腕>-<回>` (A/B でなければ `None`)。"""
    if arm is None and repeat_index is None:
        return None
    if arm is None or repeat_index is None:
        errors.append("arm と repeat_index は、A/B の回として両方を渡す (片方だけは受けない)")
        return None
    if not _ARM_RE.fullmatch(arm):
        errors.append(f"arm: 腕の名前は、英数字とハイフンと下線と点だけにする: {arm}")
        return None
    if repeat_index < 1:
        errors.append(f"repeat_index: 回の番号は 1 から始める: {repeat_index}")
        return None
    return f"{arm}-{repeat_index}"


def _checked_extra_env(extra_env: Mapping[str, str] | None, errors: list[str]) -> Mapping[str, str]:
    """A/B の腕が足す環境変数を確かめる (構成の `env` と同じ決まりを掛ける)。

    ここに書く値は、置き換えの印を埋めない (計測者が、実測した値をそのまま渡す口である)。
    """
    if extra_env is None:
        return {}
    for name in extra_env:
        _check_env_name(name, f"extra_env.{name}", errors)
    return dict(extra_env)


# --- ラベルと、config-sha256 --------------------------------------------


def _labels(
    config: ConfigDef, role: NodeRole, started_at: str, config_sha256: str, run: str | None
) -> dict[str, str]:
    """共通の 7 つ (+ 重み + A/B の回) のラベル (design.md 「plan」)。"""
    labels = {
        LABEL_OWNER: OWNER,
        LABEL_CONFIG: config.name,
        LABEL_KIND: config.kind,
        LABEL_ROLE: role,
        LABEL_IMAGE: config.image.ref,
        LABEL_STARTED_AT: started_at,
        LABEL_CONFIG_SHA256: config_sha256,
    }
    if config.weights is not None:
        labels[LABEL_WEIGHTS] = f"{config.weights.repo}@{config.weights.revision}"
    if run is not None:
        labels[LABEL_RUN] = run
    return labels


def _label_argv(labels: Mapping[str, str]) -> tuple[str, ...]:
    """ラベルを `--label <鍵>=<値>` の列にする (鍵の順に並べて、列を決まった形にする)。"""
    label_flag = TOOL_SETTINGS["label"]
    argv: list[str] = []
    for key in sorted(labels):
        argv.extend(_argv_of(label_flag.model_copy(update={"value": f"{key}={labels[key]}"})))
    return tuple(argv)


def _without_labels(argv: Sequence[str]) -> list[str]:
    """`--label` の引数 (フラグとその値) を除く (`config-sha256` の材料から外す)。

    いまの組み立てでは、材料に渡すのはラベルを挟む前の列なので、ここで落ちるものはない。
    ラベルが列に入る場所が増えても材料が変わらないようにするための歯止めとして置く
    (design.md 「plan」の「`--label` の引数を除いたもの」を、そのまま形にした)。
    """
    kept: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == TOOL_SETTINGS["label"].flag:
            skip = True
            continue
        kept.append(item)
    return kept


def _framed(text: str) -> str:
    """語に、UTF-8 のバイト数を前に付ける (区切りの曖昧さをなくす。module の docstring)。"""
    return f"{len(text.encode('utf-8'))}:{text}"


def _config_sha256(bodies: Sequence[tuple[NodeRole, Sequence[str]]]) -> str:
    """2 台ぶんの引数の列 (ラベルを除く) から、`config-sha256` を取る。"""
    frames: list[str] = []
    for role, argv in bodies:
        stripped = _without_labels(argv)
        frames.append(_framed(role))
        frames.append(_framed(str(len(stripped))))
        frames.extend(_framed(item) for item in stripped)
    return hashlib.sha256("".join(frames).encode("utf-8")).hexdigest()


# --- 組み立て -----------------------------------------------------------


def _front(container_name: str) -> tuple[str, ...]:
    """`docker run` と、ラベルより前に来る、道具が必ず付ける引数。"""
    return (
        "docker",
        "run",
        *_argv_of(TOOL_SETTINGS["detach"]),
        *_argv_of(TOOL_SETTINGS["pull"]),
        *_argv_of(TOOL_SETTINGS["name"].model_copy(update={"value": container_name})),
    )


def _back(
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    node: NodeDef,
    rank: int,
    extra_env: Mapping[str, str],
    prefix: str,
    errors: list[str],
) -> tuple[str, ...]:
    """ラベルより後ろの引数 (docker の設定 → 環境変数 → イメージの参照 → `args`)。"""
    subs = _substitutions(config, nodes, node, rank)
    argv: list[str] = list(_argv_of(TOOL_SETTINGS["restart"]))
    for key, setting in config.docker.items():
        if _applies(setting, node.role):
            argv.extend(_rendered(setting, subs, f"{prefix}.docker.{key}", errors))
    for key, setting in config.env.items():
        if _applies(setting, node.role):
            argv.extend(_env_argv(setting, subs, f"{prefix}.env.{key}", errors))
    for name, value in extra_env.items():
        argv.extend((_ENV_FLAG, f"{name}={value}"))
    argv.append(config.image.ref)
    for key, setting in config.args.items():
        if _applies(setting, node.role):
            argv.extend(_rendered(setting, subs, f"{prefix}.args.{key}", errors))
    return tuple(argv)


def _refuse(config_name: str, errors: Sequence[str]) -> PlanError:
    """集めた誤りを、1 つの `PlanError` にまとめる (すべて並べて示す)。"""
    seen: dict[str, None] = dict.fromkeys(errors)
    body = "\n".join(seen)
    return PlanError(
        f"構成 '{config_name}' から、起動の引数の列を組み立てられない"
        f" (見つかったものをすべて示す):\n{body}"
    )


def build_plans(
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    arm: str | None = None,
    repeat_index: int | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[ContainerPlan, ...]:
    """構成とノードから、構成が使うノードのぶんだけ、コンテナの計画を組み立てる。

    引数:
        config: 選んだ構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。`{head.…}` の印は、ここの `head` から埋める。
        started_at: 起こす時刻。時差の付いた日時だけを受け、ラベルには UTC で書く。
        arm: 通信の確認の A/B の腕の名前 (A/B のときだけ。design.md 「netcheck」)。
        repeat_index: A/B の回の番号 (1 から)。コンテナの名前に `-r<回>` が付く。
        extra_env: A/B の腕ごとに足す環境変数。構成の `env` と同じ決まりを掛ける。

    返り値:
        `config.nodes` に書かれた順の `ContainerPlan`。2 台のどちらにも、同じ
        `config-sha256` のラベルが付く。

    例外:
        PlanError: 埋める値のない置き換えの印、道具が必ず付ける引数の重複、秘密らしい
            環境変数の名前、ノードの定義の不足など。見つかったものを、項目の名前つきの
            日本語の文で、すべて並べて示す。
    """
    prefix = f"configs.{config.name}"
    errors: list[str] = []

    started = _started_at_label(started_at, errors)
    run = _run_label(arm, repeat_index, errors)
    added_env = _checked_extra_env(extra_env, errors)
    _check_config_name(config.name, errors)
    _check_docker_flags(config, prefix, errors)
    _check_only_on(config, prefix, errors)

    drafts: list[tuple[NodeRole, str, tuple[str, ...], tuple[str, ...]]] = []
    for rank, role in enumerate(config.nodes):
        node = nodes.get(role)
        if node is None:
            errors.append(f"nodes.{role}: ノードの定義がない (構成 '{config.name}' が使う役割)")
            continue
        name = _container_name(config.name, role, repeat_index)
        drafts.append(
            (
                role,
                name,
                _front(name),
                _back(config, nodes, node, rank, added_env, prefix, errors),
            )
        )

    if errors:
        raise _refuse(config.name, errors)

    config_sha256 = _config_sha256([(role, (*front, *back)) for role, _, front, back in drafts])
    plans: list[ContainerPlan] = []
    for role, name, front, back in drafts:
        labels = _labels(config, role, started, config_sha256, run)
        plans.append(
            ContainerPlan(
                node=role,
                container_name=name,
                labels=labels,
                argv=(*front, *_label_argv(labels), *back),
            )
        )
    return tuple(plans)
