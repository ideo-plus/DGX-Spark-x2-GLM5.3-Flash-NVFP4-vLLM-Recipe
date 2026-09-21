"""起動の前の関門と、了承の仕組み (design.md 「関門 › guards」)。

**状態を変える前に、読み取りだけで、進めてよいかを判定する**。ここから出る遠隔の呼び出しは、
どれも `mutating=False` の読み取りで、`docker run` / `stop` / `rm` / `pull`、`mkdir`、配布を
1 つも出さない (試験で固定する)。判定の結果は `types.GateResult` の列で返し、断る理由は、
見つけたものと、要る量と空いている量を添えた日本語の文にする。

守る決まり:

- **クリーンルーム** (requirements 2.4): コンテナの一覧は、つねに
  `--filter label=vllm-baseline.owner=serving-kit` で絞って取る。ラベルのないコンテナに、
  `docker inspect` も `docker logs` も `docker top` も、どの操作も向けない。よそのものに
  ついて読むのは、`nvidia-smi` が返すプロセスの名前とメモリの量だけである
- **コンテナを対象にする操作の不変条件** (requirements 2.3): 対象にできるのは、(a) ラベルで
  絞った一覧から来た識別子か、(b) 了承済みの計画が自分で起こす名前だけ。巻き戻すコマンドは
  `build_approved_plan` が `ContainerPlan` の列から自分で作るので、**ほかの名前を巻き戻しに
  入れる引数がない** (`types.ApprovedPlan` の検証と二重に守る)
- **了承** (requirements 2.1): 前に進むコマンドと、巻き戻すコマンドと、対象の機械を見せて、
  `yes` の入力を待つ。`--yes` のときは、見せてから通す。端末でなく `--yes` もなければ、
  `ApprovalError` (終了コード 1)。了承を得た計画は `request_approval` が実行役に渡す
- 関門の順序は `GATE_ORDER` (design.md 「関門」の表と、System Flows の起動)

依存の向きにより、この module が読み込む `serving_kit` は `types`、`remote`、`plan` だけ
である (`lifecycle` 以降は読み込まない)。

設計の文からの、意図した違いと、受け入れたこと (`remote.py`、`plan.py` と同じ流儀で並べる):

1. **「`models/<slug>/` の一覧と大きさ」は、照合の結果の記録の `file_count` と
   `total_bytes` で見る**。遠隔で流せるコマンドの許可の一覧 (design.md 「remote」) に、
   ファイルを並べるもの (`ls`、`find`、`stat`、`du`) がないためである。記録は
   `serve verify` が実際のファイルを見て書くので、版と一覧と大きさの確かめは、記録を
   通して行う (`types.VerificationRecord` が、この 2 つの項目を持つのはこのため)。
   **限界**: これは記録の読み直しなので、照合のあとに `models/<slug>/` のファイルが
   消された・書き換えられた場合は、この関門では分からない (失敗は、vLLM がロードする
   ときまで遅れる)。疑わしいときは、`serve verify` で 2 台を照合し直す
2. **1 つの関門が落ちても、残りの関門を流して並べる** (design.md 「cli」の `serve check`:
   「すべての関門を流して、結果を並べる」)。ただし `gate_reachable` が落ちた台は、何も
   読めないので、その台の残りを飛ばす
3. **`gate_own_state` だけは、実行役を取らない純粋な関数にした**。一覧の取得は
   `list_own_containers` が受け持つ。`already_running` の判定 (`match_running`) と同じ
   1 回の読み取りを使えるようにして、2 度読みで食い違うことをなくすためである
4. **巻き戻すコマンドの形 (`docker stop -t 90 <名前>` と `docker rm <名前>`) を、
   `stop_argv` と `remove_argv` で出す**。`remote.CallGuard` は、了承を得た計画と引数の列が
   **完全に一致**するかを見るので、あとで実際に止める側 (lifecycle) が、同じ関数で同じ列を
   作れなければならない
5. **重みの置き場所の名前 (`<slug>`) を `weights_slug` で決める**。design の
   `state/<slug>.verified.json` と `models/<slug>/` の `<slug>` は、`serving/weights/` の
   マニフェストのファイルの名前 (`RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`) と同じ
   作り方 (`/` を `__` に置き換える) にした。取得と照合 (task 3.3) も、この関数を使う
6. **照合の結果の記録は、範囲 (`scope`) ごとに、別のファイルに置く** (計測者の判断、
   2026-09-21)。全体の照合 (`scope = "all"`。`serve verify`) は design のとおり
   `<remote_root>/state/<slug>.verified.json`、縮小の確認用 (`scope = "probe_files"`。
   `serve fetch --probe-files`) は `<remote_root>/state/<slug>.probe.verified.json` に
   する。1 つの名前を共有すると、`serve verify` と `serve fetch --probe-files` が、互いの
   記録を上書きし、もう一方の構成の関門が (中身は正しいのに) 断られる。道筋は
   `verification_record_path(node, repo, scope)` が決めるので、**記録を書く側 (task 3.3)
   も、この関数を使って道筋を決めること**
7. **構成の `kind` で、要る範囲を決めて、その道筋だけを読む**。縮小の確認 (`probe`) は
   `probe_files`、ほかの重みを持つ構成は `all` である。読んだ記録の中の `scope` が、その
   道筋の範囲と違えば断る (置き間違いを、黙って通さない)。`scope = "all"` の記録は、
   `models/<slug>/` を見たものなので、`probe/<slug>/` に設定とトークナイザがある証拠に
   ならない (どちらも、断る側に寄せた)
8. **公開の `gate_*` は、どれも、例外を外に出さない**。読み取りが届かなかったこと
   (`RemoteError`) も、読めなかった出力 (`ValueError`) も、自分で `passed=False` の
   `GateResult` に変える。design の image の節が、`serve pull-image` (task 3.1) に
   `gate_reachable` と `gate_disk_space` を直に呼ばせるので、`run_gates` を通らない
   呼び方でも、結果の形が同じである必要がある。`run_gates` の `_safely` は、二重の
   歯止めとして残す。純粋な関数の `gate_own_state` だけは、読み取りをしないので例外がない
9. **絞り込んだ一覧の行が、本当に自分のものかを、こちらでも確かめる**。`docker` の
   `--filter label=` に任せきりにせず、`list_own_containers` (一覧を得る、ただ 1 つの
   入口) が、1 行ずつ所有のラベルを見る。違う行が 1 つでもあれば、絞り込みが効いていない
   という異常なので、その行を捨てずに断る (捨てて進むと、よそのコンテナを自分のものとして
   扱いかねない)。誤りの文に、その行の名前も識別子も出さない (requirements 2.4)。
   `Labels` の 1 つの文字列に、同じ鍵が 2 度あるときも、後ろで上書きせずに断る
   (値に `,鍵=値` を書けば、鍵を足したように見せられるため)
10. **`docker stop` / `docker rm` の対象は、位置で読む**。値を取るフラグ (`-t`、
   `--time`、`-s`、`--signal`。`=` つなぎも) の値だけを飛ばし、`-` で始まらない語は、
   数字だけであっても対象として扱う (`docker stop -t 90 12345` の `12345` は、コンテナの
   名前か短い識別子でありうる)。読み方の分からないフラグを含む列は、対象を読み違えるので
   断る。`build_approved_plan` は、渡された `ContainerPlan` の引数の列そのもの
   (`docker run` で始まる、`--name` がその計画の名前、`--label` に所有のラベル) も
   確かめる (手で組み立てた計画で、起こす名前と巻き戻す名前をずらせないようにする)。
   **コンテナを起こす列 (`docker run`、`docker create`。2 語の形も) は、前に進むコマンドに
   直に書けない** (起こす道を 1 つにして、この検査を必ず通す)。2 語の形の
   `docker container stop` / `rm` は、`remote.CallGuard` が、許可の一覧にないサブコマンド
   として断る (層の受け持ち。どちらも試験で固定)
11. **巻き戻しを実際に流す前に、もう一度、一覧で相手を確かめる** (`rollback_commands`。
   task 3.1 のレビューで足した)。了承を得た計画の巻き戻しは、起動の**前**に、計画が起こす
   名前で書く。ところが `docker stop <名前>` / `docker rm <名前>` は、その名前のコンテナが
   **誰のものでも**止めて消すので、`docker run` が名前の衝突で失敗した場合 (その名前は、
   よそのコンテナのもの) に、計画のとおり流すと、よそのものを止めてしまう。`docker run` の
   標準エラーの文面で衝突を見分けるのは、版で変わりうるので歯止めにならない。そこで、流す
   直前に `list_own_containers` を読み、**その台の一覧に、その名前の行があるときだけ**流す
   (名前は 1 台の中で一意なので、自分の一覧にあれば、それは自分のコンテナである)。3.1 の
   `image` と、3.4 以降の `lifecycle`、4.x は、この関数で巻き戻しを決める
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol, TextIO

from pydantic import ValidationError

from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_KIND,
    LABEL_OWNER,
    OWNER,
    OWNER_FILTER,
)
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    ApprovedPlan,
    ConfigDef,
    ContainerPlan,
    GateResult,
    GpuApp,
    ManifestFile,
    NodeDef,
    NodeRole,
    PlannedCommand,
    PlannedPush,
    PlannedRun,
    Setting,
    VerificationRecord,
    VerificationScope,
    WeightsManifest,
)

__all__ = [
    "GATE_DISK_SPACE",
    "GATE_GPU_IDLE",
    "GATE_IMAGE_DIGEST",
    "GATE_LAYOUT",
    "GATE_ORDER",
    "GATE_OWN_STATE",
    "GATE_PORTS_FREE",
    "GATE_REACHABLE",
    "GATE_WEIGHTS_VERIFIED",
    "READ_TIMEOUT_S",
    "SPACE_MARGIN_PERCENT",
    "STOP_TIMEOUT_S",
    "ApprovalError",
    "AssumeYesConfirmer",
    "Confirmer",
    "OwnContainer",
    "RunningMatch",
    "TerminalConfirmer",
    "build_approved_plan",
    "config_ports",
    "format_plan",
    "gate_disk_space",
    "gate_gpu_idle",
    "gate_image_digest",
    "gate_layout",
    "gate_own_state",
    "gate_ports_free",
    "gate_reachable",
    "gate_weights_verified",
    "list_own_containers",
    "make_confirmer",
    "match_running",
    "mount_sources",
    "parse_gpu_apps",
    "parse_listening_ports",
    "parse_own_containers",
    "remove_argv",
    "request_approval",
    "required_free_bytes",
    "rollback_commands",
    "run_gates",
    "stop_argv",
    "verification_record_path",
    "weights_slug",
]


# --- 関門の名前と、決まった値 -------------------------------------------

GATE_REACHABLE: Final[str] = "reachable"
GATE_OWN_STATE: Final[str] = "own_state"
GATE_GPU_IDLE: Final[str] = "gpu_idle"
GATE_LAYOUT: Final[str] = "layout"
GATE_IMAGE_DIGEST: Final[str] = "image_digest"
GATE_WEIGHTS_VERIFIED: Final[str] = "weights_verified"
GATE_DISK_SPACE: Final[str] = "disk_space"
GATE_PORTS_FREE: Final[str] = "ports_free"

GATE_ORDER: Final[tuple[str, ...]] = (
    GATE_REACHABLE,
    GATE_OWN_STATE,
    GATE_GPU_IDLE,
    GATE_LAYOUT,
    GATE_IMAGE_DIGEST,
    GATE_WEIGHTS_VERIFIED,
    GATE_DISK_SPACE,
    GATE_PORTS_FREE,
)
"""関門を流す順序 (design.md 「関門」の表と、System Flows の起動)。

`gate_gpu_idle` が `gate_own_state` のあとに来るので、GPU の関門の条件は「使っている
プロセスが 1 つでもあれば断る」だけでよい (その時点で、自分のコンテナは 1 つも動いて
いない)。
"""

READ_TIMEOUT_S: Final[float] = 30.0
"""1 つの読み取りの時間切れ。どれも、すぐに返るコマンドである。"""

STOP_TIMEOUT_S: Final[int] = 90
"""巻き戻すときの `docker stop -t`。`--shutdown-timeout 60` の drain を待たせる
(design.md 「System Flows」の起動)。"""

SPACE_MARGIN_PERCENT: Final[int] = 10
"""ディスクの空きに見る余裕 (design.md 「関門」: 要る量 + 10%)。"""

_STATE_RUNNING: Final[str] = "running"
"""`docker ps --format json` の `State` が、動いていることを表す値。"""

_KIND_FETCH: Final[str] = "fetch"
_SCOPE_PROBE_FILES: Final[VerificationScope] = "probe_files"
_SCOPE_ALL: Final[VerificationScope] = "all"
_KIND_PROBE: Final[str] = "probe"

_RECORD_SUFFIX: Final[Mapping[VerificationScope, str]] = {
    _SCOPE_ALL: "verified.json",
    _SCOPE_PROBE_FILES: "probe.verified.json",
}
"""照合の記録のファイルの名前の、範囲ごとの後ろ (module の docstring の「意図した違い」の 6)。

全体の照合 (`serve verify`) と、縮小の確認用の照合 (`serve fetch --probe-files`) を、別の
ファイルに置く。同じ名前だと、あとから書いたほうが、もう一方の記録を上書きしてしまう。
"""

_IDENTITY_LABELS: Final[tuple[str, ...]] = (
    LABEL_OWNER,
    LABEL_CONFIG,
    LABEL_IMAGE,
    LABEL_CONFIG_SHA256,
)
"""`already_running` の判定に使うラベル (design.md 「lifecycle」の State Management)。

構成の名前、イメージのダイジェスト、`config-sha256` が、コンテナの名前と一緒に、2 台とも
一致するときだけ、「すでに動いている」として扱う。所有のラベルも見るのは、`match_running`
に、一覧を通らずに組み立てたものが渡ったときの、二重の歯止めである
(一覧の入口では、`list_own_containers` が断る)。
"""

_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._-]+")
_MOUNT_FLAG: Final[str] = "--mount"
_MOUNT_SOURCE_KEYS: Final[frozenset[str]] = frozenset({"source", "src"})
_SECTIONS: Final[tuple[str, ...]] = ("docker", "args", "env")
_YES: Final[str] = "yes"
_PROMPT: Final[str] = f"この計画を実行してよければ '{_YES}' と入力する: "


class ApprovalError(Exception):
    """了承を得られなかった、または、了承を尋ねられなかった。

    design.md 「Error Handling」の終了コード 1 (前提の不足と断り) に乗る。
    """


@dataclass(frozen=True)
class OwnContainer:
    """自分のラベルで絞った一覧の 1 件 (`docker ps -a --format json` の 1 行)。

    この道具が触ってよいコンテナは、ここに出たものだけである。`labels` は、`Labels` の
    `鍵=値,鍵=値` の 1 つの文字列を解いたもの。
    """

    id: str
    name: str
    state: str
    image: str
    labels: Mapping[str, str]

    @property
    def kind(self) -> str | None:
        """ラベルから読んだ構成の種類 (`serve`、`probe`、`job`、`fetch`、`inspect`)。"""
        return self.labels.get(LABEL_KIND)

    @property
    def running(self) -> bool:
        """いま動いているかどうか。"""
        return self.state == _STATE_RUNNING


@dataclass(frozen=True)
class RunningMatch:
    """いま動いている自分のコンテナと、選んだ構成の突き合わせ (design.md 「lifecycle」)。

    - `already_running`: 名前、構成の名前、イメージ、`config-sha256` が、構成の使う
      ノードのすべてで一致した (呼ぶ側は、いまの状態を示して終了コード 0 で終わる)
    - `differences`: 名前が同じコンテナが動いているのに、中身が違う項目 (呼ぶ側は、
      これを示して終了コード 1 で断り、`serve stop` を促す)。コンテナがそもそも無いことは、
      違いとして数えない (ふつうの起動の前の状態である)
    """

    already_running: bool
    differences: tuple[str, ...]


# --- 小さな読み取りの助け -----------------------------------------------


def _result(gate: str, node: NodeRole | None, *, passed: bool, detail: str) -> GateResult:
    return GateResult(gate=gate, node=node, passed=passed, detail=detail)


def _bytes_text(value: int) -> str:
    """バイト数を、人が読める形と、そのままの数の両方で示す。"""
    for name, unit in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= unit:
            return f"{value / unit:.1f} {name} ({value:,} バイト)"
    return f"{value:,} バイト"


def _text_field(raw: Mapping[str, object], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"コンテナの一覧の {line_number} 行目に、項目 {key} (文字列) がない")
    return value


def _optional_int(text: str) -> int | None:
    """数として読めなければ空にする (`[N/A]` の列があるため)。"""
    stripped = text.strip()
    return int(stripped) if stripped.isdecimal() else None


def _settings_of(config: ConfigDef, section: str) -> Mapping[str, Setting]:
    settings: Mapping[str, Setting] = getattr(config, section)
    return settings


# --- 読み取りの出力を読む (入出力のない関数) ----------------------------


def weights_slug(repo: str) -> str:
    """重みの置き場所と、照合の記録の名前に使う短い名前 (`<所有者>__<名前>`)。

    `serving/weights/<slug>.manifest.json` の名前の作り方に合わせる (module の docstring の
    「意図した違い」の 5)。道筋の一部になるので、使える文字を絞る。
    """
    slug = repo.replace("/", "__")
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"重みの入手先の名前を、置き場所の名前にできない: {repo}")
    return slug


def verification_record_path(node: NodeDef, repo: str, scope: VerificationScope) -> str:
    """照合の結果の記録の置き場所 (**範囲ごとに、別のファイル**)。

    - `all` (全体の照合。`serve verify`): `<remote_root>/state/<slug>.verified.json`
    - `probe_files` (設定とトークナイザだけ。`serve fetch --probe-files`):
      `<remote_root>/state/<slug>.probe.verified.json`

    分ける理由は、module の docstring の「意図した違い」の 6 にある (互いの記録を上書き
    しないため)。**記録を書く側 (task 3.3) も、この関数で道筋を決めること。**
    """
    return f"{node.remote_root}/state/{weights_slug(repo)}.{_RECORD_SUFFIX[scope]}"


def required_free_bytes(config: ConfigDef, manifest: WeightsManifest | None) -> int:
    """この構成を置くのに要るディスクの量 (余裕を足す前)。

    重みを持つ構成はマニフェストの合計、持たない構成はイメージの大きさである
    (design.md 「関門」の `gate_disk_space`)。縮小の確認の構成では、マニフェストのうち
    safetensors を除いたファイル (設定とトークナイザ。数十 MiB) だけを数える。
    """
    if config.weights is not None and manifest is not None:
        return sum(entry.size for entry in _scoped_files(config, manifest))
    return config.image.size_bytes


def config_ports(config: ConfigDef, role: NodeRole) -> tuple[int, ...]:
    """この構成が、この台で使う待ち受けのポート (`is_port = True` の設定の値)。

    ポートを、構成とは別の一覧で二重に持たない (design.md 「types / config」)。
    """
    ports: set[int] = set()
    for section in _SECTIONS:
        for setting in _settings_of(config, section).values():
            applies = setting.only_on is None or setting.only_on == role
            if setting.is_port and applies and setting.value is not None:
                ports.add(int(setting.value))
    return tuple(sorted(ports))


def mount_sources(argv: Sequence[str]) -> tuple[str, ...]:
    """組み立てた引数の列から、`--mount` の元 (Spark の側の置き場所) を取り出す。

    置き換えの印は `plan` が埋めたあとなので、そのまま `test -d` に掛けられる。
    """
    sources: list[str] = []
    pending = False
    for item in argv:
        if pending:
            pending = False
            source = _mount_source(item)
            if source is not None:
                sources.append(source)
            continue
        if item == _MOUNT_FLAG:
            pending = True
        elif item.startswith(f"{_MOUNT_FLAG}="):
            source = _mount_source(item[len(_MOUNT_FLAG) + 1 :])
            if source is not None:
                sources.append(source)
    return tuple(dict.fromkeys(sources))


def _mount_source(value: str) -> str | None:
    """`type=bind,source=…,target=…` から、元の道筋を取り出す。"""
    for part in value.split(","):
        key, _, found = part.partition("=")
        if key.strip() in _MOUNT_SOURCE_KEYS:
            return found.strip()
    return None


def parse_own_containers(text: str) -> tuple[OwnContainer, ...]:
    """`docker ps -a --format json` の出力 (1 行に 1 つの JSON) を読む。

    空の出力は、自分のコンテナが 1 つもない、ふつうの状態である。
    """
    containers: list[OwnContainer] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"コンテナの一覧の {line_number} 行目を読めない: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"コンテナの一覧の {line_number} 行目が、JSON の object でない")
        containers.append(
            OwnContainer(
                id=_text_field(raw, "ID", line_number),
                name=_text_field(raw, "Names", line_number),
                state=_text_field(raw, "State", line_number),
                image=_text_field(raw, "Image", line_number),
                labels=_parse_labels(_text_field(raw, "Labels", line_number)),
            )
        )
    return tuple(containers)


def _parse_labels(text: str) -> dict[str, str]:
    """`Labels` の `鍵=値,鍵=値` の 1 つの文字列を解く。

    この道具が付けるラベルの値に `,` は入らない (構成の名前、種類、役割、ダイジェスト、
    UTC の日時、16 進の sha256)。docker は、この形でしか出さないので、値に `,鍵=値` を
    書いたラベルを持つコンテナは、鍵を足したように見せられる。**同じ鍵が 2 度現れたら、
    後ろで上書きせずに断る** (所有や `config-sha256` の鍵を、あとから差し替えられない)。
    値そのものは、誤りの文に出さない。
    """
    labels: dict[str, str] = {}
    for part in text.split(","):
        if not part:
            continue
        key, separator, value = part.partition("=")
        if not separator:
            continue
        name = key.strip()
        if name in labels:
            raise ValueError(f"コンテナのラベルに、同じ鍵が 2 度ある: {name}")
        labels[name] = value
    return labels


def parse_gpu_apps(text: str) -> tuple[GpuApp, ...]:
    """`nvidia-smi --query-compute-apps=pid,process_name,used_memory` の出力を読む。

    GB10 のユニファイドメモリのために `[N/A]` になる列があっても、落ちずに読む
    (読めなかった値は空にする)。空の出力は、GPU を使っているプロセスがない状態である。
    """
    apps: list[GpuApp] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        columns = [column.strip() for column in stripped.split(",")]
        if len(columns) < 3:
            raise ValueError(
                f"GPU のプロセスの一覧の {line_number} 行目に、3 つの列がない: {stripped}"
            )
        memory = columns[-1].split()
        apps.append(
            GpuApp(
                pid=_optional_int(columns[0]),
                process_name=",".join(columns[1:-1]),
                used_memory_mib=_optional_int(memory[0]) if memory else None,
            )
        )
    return tuple(apps)


def parse_listening_ports(text: str) -> frozenset[int]:
    """`ss -ltnH` の出力から、待ち受けているポートの番号を読む。

    局所のアドレスの列は、IPv4 (`0.0.0.0:8080`)、IPv6 (`[::]:8080`)、インターフェースの
    付いた形 (`127.0.0.53%lo:53`) のどれでも、最後の `:` の後ろがポートである。番号は、
    そのまま突き合わせる (前方一致で見ると、`80800` が `8080` に当たってしまう)。

    読めない行があれば、その行を黙って飛ばさずに断る。飛ばすと、出力の形が変わったときに
    「何も待ち受けていない」と読めてしまい、関門が素通しになる。
    """
    ports: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        port = fields[3].rsplit(":", 1)[-1] if len(fields) >= 4 else ""
        if fields[0] != "LISTEN" or not port.isdecimal():
            raise ValueError(f"待ち受けの一覧の {line_number} 行目を読めない: {line.strip()}")
        ports.add(int(port))
    return frozenset(ports)


def _scoped_files(config: ConfigDef, manifest: WeightsManifest) -> tuple[ManifestFile, ...]:
    """この構成が見るマニフェストのファイル (縮小の確認は、safetensors を除いたもの)。"""
    if config.kind == _KIND_PROBE:
        return manifest.probe_files
    return manifest.files


def _required_scope(config: ConfigDef) -> VerificationScope:
    """この構成に要る照合の対象 (design.md 「関門」の `kind = "probe"` の扱い)。"""
    return _SCOPE_PROBE_FILES if config.kind == _KIND_PROBE else _SCOPE_ALL


# --- 1 つ 1 つの関門 ----------------------------------------------------


def list_own_containers(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> tuple[OwnContainer, ...]:
    """自分のラベルで絞ったコンテナの一覧を取る (絞らない一覧は、取らない、見ない)。

    **この道具が、コンテナの一覧を得る、ただ 1 つの入口である**。絞り込みは docker に
    任せきりにせず、返ってきた 1 行ずつについて、所有のラベルが自分の値かを、こちらでも
    確かめる。違う行が 1 つでもあれば、絞り込みが効いていないという異常なので、その行を
    捨てずに断る (捨てて進むと、よそのコンテナを「自分のもの」として扱いかねない)。
    誤りの文に、その行の名前も識別子も出さない (よそのコンテナを、読まない、見せない)。
    """
    argv = ("docker", "ps", "-a", "--filter", f"label={OWNER_FILTER}", "--format", "json")
    result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        raise ValueError(f"コンテナの一覧を取れなかった: {result.stderr.strip() or result.stdout}")
    containers = parse_own_containers(result.stdout)
    foreign = sum(1 for container in containers if container.labels.get(LABEL_OWNER) != OWNER)
    if foreign:
        raise ValueError(
            f"所有のラベル ({LABEL_OWNER}={OWNER}) のない行が、絞り込んだ一覧に"
            f" {foreign} 件ある (絞り込みが効いていないので、この一覧は使わない)"
        )
    return containers


def gate_reachable(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> GateResult:
    """ssh で入れるか (design.md 「関門」: `uname -n`)。"""
    return _safely(GATE_REACHABLE, node.role, partial(_reachable, runner, node, timeout_s))


def _reachable(runner: RemoteRunner, node: NodeDef, timeout_s: float) -> GateResult:
    """gate_reachable の中身。読めなかったことは、`gate_reachable` が断りに変える。"""
    try:
        result = runner.run(node, ("uname", "-n"), timeout_s=timeout_s, mutating=False)
    except RemoteError as exc:
        return _result(GATE_REACHABLE, node.role, passed=False, detail=f"ssh で入れない: {exc}")
    if result.exit_code != 0:
        return _result(
            GATE_REACHABLE,
            node.role,
            passed=False,
            detail=f"ssh で入れたが、ホストの名前を読めなかった: {result.stderr.strip()}",
        )
    return _result(
        GATE_REACHABLE,
        node.role,
        passed=True,
        detail=f"{node.ssh_host} ({result.stdout.strip()}) に入れる",
    )


def gate_own_state(node: NodeRole, containers: Sequence[OwnContainer]) -> GateResult:
    """自分のコンテナが 1 つも残っていないか (design.md 「関門」)。

    一覧の取得は `list_own_containers` が受け持つので、ここは純粋な関数である
    (module の docstring の「意図した違い」の 3)。同時に動かす自分のコンテナは、1 つの
    構成ぶんだけなので、種類を問わず 1 つでも動いていたら断る。終了したものが残っている
    ときも、名前の衝突になるので断る。
    """
    running = [container for container in containers if container.running]
    left = [container for container in containers if not container.running]
    reasons: list[str] = []
    if running:
        if any(container.kind == _KIND_FETCH for container in running):
            reasons.append(
                f"重みの取得のコンテナが動いている ({_shown_containers(running)})。"
                "取得の終わりを待ってから起こす (ディスクとページキャッシュを使うため)"
            )
        else:
            reasons.append(
                f"自分のコンテナが動いている ({_shown_containers(running)})。"
                "先に `serve stop` で止める"
            )
    if left:
        reasons.append(
            f"終了した自分のコンテナが残っている ({_shown_containers(left)})。"
            "`serve logs` で記録を回収してから `serve stop` で消す (残すと名前が衝突する)"
        )
    if reasons:
        return _result(GATE_OWN_STATE, node, passed=False, detail=" / ".join(reasons))
    return _result(GATE_OWN_STATE, node, passed=True, detail="自分のコンテナは 1 つもない")


def _shown_containers(containers: Iterable[OwnContainer]) -> str:
    return ", ".join(
        f"{container.name} [{container.kind or '種類のラベルがない'}] {container.state}"
        for container in containers
    )


def gate_gpu_idle(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> GateResult:
    """GPU を使っているプロセスがないか (requirements 2.2)。

    よそのものについて読むのは、この 3 つの列 (pid、名前、メモリの量) だけである
    (requirements 2.4)。GPU を使わない種類 (`fetch`、`inspect`) でも、同じ関門を流す。
    """
    return _safely(GATE_GPU_IDLE, node.role, partial(_gpu_idle, runner, node, timeout_s))


def _gpu_idle(runner: RemoteRunner, node: NodeDef, timeout_s: float) -> GateResult:
    """gate_gpu_idle の中身。読めなかったことは、`gate_gpu_idle` が断りに変える。"""
    argv = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    )
    result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _result(
            GATE_GPU_IDLE,
            node.role,
            passed=False,
            detail=f"GPU のプロセスを読めなかった: {result.stderr.strip() or result.stdout}",
        )
    apps = parse_gpu_apps(result.stdout)
    if apps:
        return _result(
            GATE_GPU_IDLE,
            node.role,
            passed=False,
            detail=(
                f"GPU を使っているプロセスがある ({_shown_apps(apps)})。"
                "この道具は、よそのプロセスを止めない。空いてから起こす"
            ),
        )
    return _result(GATE_GPU_IDLE, node.role, passed=True, detail="GPU を使っているプロセスはない")


def _shown_apps(apps: Iterable[GpuApp]) -> str:
    shown: list[str] = []
    for app in apps:
        pid = "pid 不明" if app.pid is None else f"pid {app.pid}"
        memory = (
            "メモリの量は読めない ([N/A])"
            if app.used_memory_mib is None
            else f"{app.used_memory_mib} MiB"
        )
        shown.append(f"{app.process_name} ({pid}, {memory})")
    return ", ".join(shown)


def gate_layout(
    runner: RemoteRunner,
    node: NodeDef,
    plan: ContainerPlan,
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> GateResult:
    """構成が結び付ける置き場所が、この台にあるか (design.md 「関門」: `test -d`)。

    `--mount` は、元のディレクトリを自動では作らないので、これがないと最初の `docker run`
    が失敗する。
    """
    return _safely(GATE_LAYOUT, node.role, partial(_layout, runner, node, plan, timeout_s))


def _layout(
    runner: RemoteRunner, node: NodeDef, plan: ContainerPlan, timeout_s: float
) -> GateResult:
    """gate_layout の中身。読めなかったことは、`gate_layout` が断りに変える。"""
    sources = mount_sources(plan.argv)
    if not sources:
        return _result(
            GATE_LAYOUT, node.role, passed=True, detail="この構成は、置き場所を結び付けない"
        )
    missing = [
        source
        for source in sources
        if runner.run(node, ("test", "-d", source), timeout_s=timeout_s, mutating=False).exit_code
        != 0
    ]
    if missing:
        return _result(
            GATE_LAYOUT,
            node.role,
            passed=False,
            detail=(
                f"構成が結び付ける置き場所がない ({', '.join(missing)})。"
                "`serve push` で、2 台に置き場所を作ってから起こす"
            ),
        )
    return _result(
        GATE_LAYOUT, node.role, passed=True, detail=f"置き場所がある ({', '.join(sources)})"
    )


def gate_image_digest(
    runner: RemoteRunner,
    node: NodeDef,
    config: ConfigDef,
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> GateResult:
    """手元のイメージが、構成のダイジェストと合うか (requirements 3.2、3.3)。

    `RepoDigests` の項目と、構成の `ref` (`<名前>@sha256:<64 桁>`) を、そのまま突き合わせる。
    `docker.io/` の接頭辞が付いた形は、いまは「合わない」として断る (**実物の出力の形は、
    7.1 でイメージを取得したときに確かめて、見本を足す**)。
    """
    return _safely(
        GATE_IMAGE_DIGEST, node.role, partial(_image_digest, runner, node, config, timeout_s)
    )


def _image_digest(
    runner: RemoteRunner, node: NodeDef, config: ConfigDef, timeout_s: float
) -> GateResult:
    """gate_image_digest の中身。読めなかったことは、`gate_image_digest` が断りに変える。"""
    reference = config.image.ref
    argv = ("docker", "image", "inspect", "--format", "{{json .RepoDigests}}", reference)
    result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _result(
            GATE_IMAGE_DIGEST,
            node.role,
            passed=False,
            detail=(
                f"固定したイメージが手元にない ({reference})。"
                f"`serve pull-image` で取得する: {result.stderr.strip()}"
            ),
        )
    digests = _parse_repo_digests(result.stdout)
    if reference in digests:
        return _result(
            GATE_IMAGE_DIGEST,
            node.role,
            passed=True,
            detail=f"ダイジェストが一覧にある ({reference})",
        )
    return _result(
        GATE_IMAGE_DIGEST,
        node.role,
        passed=False,
        detail=(
            f"手元のイメージが、構成のダイジェストと合わない。構成: {reference}、"
            f"手元の一覧: {', '.join(digests) if digests else '(空)'}"
        ),
    )


def _parse_repo_digests(text: str) -> tuple[str, ...]:
    """`{{json .RepoDigests}}` の出力 (JSON の配列。空のときは `null`) を読む。"""
    stripped = text.strip()
    if not stripped:
        return ()
    raw = json.loads(stripped)
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"RepoDigests が、文字列の配列でない: {stripped}")
    return tuple(str(item) for item in raw)


def gate_weights_verified(
    runner: RemoteRunner,
    node: NodeDef,
    config: ConfigDef,
    manifest: WeightsManifest | None,
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> GateResult:
    """重みの照合の記録が、マニフェストと合うか (requirements 3.5)。

    起動のたびに 184 GiB の sha256 を計算し直さない。全体の照合は `serve fetch` の終わりと
    `serve verify` で行い、その結果の記録 (版、ファイルの数、大きさの合計、合わなかった
    ファイル) を、ここでマニフェストと突き合わせる (module の docstring の「意図した違い」
    の 1)。
    """
    return _safely(
        GATE_WEIGHTS_VERIFIED,
        node.role,
        partial(_weights_verified, runner, node, config, manifest, timeout_s),
    )


def _weights_verified(
    runner: RemoteRunner,
    node: NodeDef,
    config: ConfigDef,
    manifest: WeightsManifest | None,
    timeout_s: float,
) -> GateResult:
    """gate_weights_verified の中身。読めなかったことは、`gate_weights_verified` が断りに変える。"""
    weights = config.weights
    if weights is None:
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=True,
            detail="この構成は、重みを使わない",
        )
    if manifest is None:
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=False,
            detail=(
                f"構成 '{config.name}' は重みを使うのに、マニフェスト"
                f" (serving/weights/{weights.manifest}) が渡されていない"
            ),
        )
    if manifest.repo != weights.repo or manifest.revision != weights.revision:
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=False,
            detail=(
                "渡されたマニフェストが、構成の重みと違う。マニフェスト:"
                f" {manifest.repo}@{manifest.revision}、構成: {weights.repo}@{weights.revision}"
            ),
        )
    scope = _required_scope(config)
    path = verification_record_path(node, weights.repo, scope)
    result = runner.run(node, ("cat", path), timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        how_to_verify = (
            "`serve fetch --probe-files` で、設定とトークナイザを取得して照合する"
            if scope == _SCOPE_PROBE_FILES
            else "`serve fetch` で取得してから、`serve verify` で 2 台を照合する"
        )
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=False,
            detail=(
                f"この構成に要る、重みの照合の記録がない (対象 {scope}、{path})。{how_to_verify}"
            ),
        )
    try:
        record = VerificationRecord.model_validate_json(result.stdout)
    except ValidationError as exc:
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=False,
            detail=f"重みの照合の記録を読めない ({path}): {exc.error_count()} 件の誤り",
        )
    problems = _record_problems(record, config, manifest, node.role)
    if problems:
        return _result(
            GATE_WEIGHTS_VERIFIED,
            node.role,
            passed=False,
            detail=f"重みの照合の記録が合わない ({path}): " + " / ".join(problems),
        )
    return _result(
        GATE_WEIGHTS_VERIFIED,
        node.role,
        passed=True,
        detail=(
            f"{record.repo}@{record.revision} を照合済み (対象 {record.scope}、"
            f"{record.file_count} ファイル、{_bytes_text(record.total_bytes)}、"
            f"{record.verified_at.isoformat()})"
        ),
    )


def _record_problems(
    record: VerificationRecord,
    config: ConfigDef,
    manifest: WeightsManifest,
    role: NodeRole,
) -> tuple[str, ...]:
    """照合の記録と、マニフェストの食い違いを、すべて並べる。"""
    scope = _required_scope(config)
    files = _scoped_files(config, manifest)
    expected_bytes = sum(entry.size for entry in files)
    problems: list[str] = []
    if record.repo != manifest.repo:
        problems.append(f"入手先が違う (記録: {record.repo}、マニフェスト: {manifest.repo})")
    if record.revision != manifest.revision:
        problems.append(f"版が違う (記録: {record.revision}、マニフェスト: {manifest.revision})")
    if record.node != role:
        problems.append(f"ほかの台の記録である (記録: {record.node}、この台: {role})")
    if record.scope != scope:
        problems.append(
            f"記録の中の対象が、この道筋の範囲と違う (記録: {record.scope}、"
            f"この道筋と、この構成に要るもの: {scope})"
        )
    if record.mismatched:
        problems.append(f"合わないファイルがある: {', '.join(record.mismatched)}")
    if record.file_count != len(files):
        problems.append(
            f"ファイルの数が合わない (記録: {record.file_count}、マニフェスト: {len(files)})"
        )
    if record.total_bytes != expected_bytes:
        problems.append(
            f"大きさの合計が合わない (記録: {record.total_bytes:,} バイト、"
            f"マニフェスト: {expected_bytes:,} バイト)"
        )
    return tuple(problems)


def gate_disk_space(
    runner: RemoteRunner,
    node: NodeDef,
    required_bytes: int,
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> GateResult:
    """ディスクの空きが、要る量 + 10% に足りるか (requirements 2.5)。"""
    return _safely(
        GATE_DISK_SPACE, node.role, partial(_disk_space, runner, node, required_bytes, timeout_s)
    )


def _disk_space(
    runner: RemoteRunner, node: NodeDef, required_bytes: int, timeout_s: float
) -> GateResult:
    """gate_disk_space の中身。読めなかったことは、`gate_disk_space` が断りに変える。"""
    argv = ("df", "-B1", "--output=avail", node.remote_root)
    result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _result(
            GATE_DISK_SPACE,
            node.role,
            passed=False,
            detail=(
                f"ディスクの空きを読めなかった ({node.remote_root}):"
                f" {result.stderr.strip() or result.stdout}"
                " (Spark の置き場所がまだなければ、`serve push` で作る)"
            ),
        )
    available = _parse_available_bytes(result.stdout)
    needed = required_bytes + required_bytes * SPACE_MARGIN_PERCENT // 100
    if available < needed:
        return _result(
            GATE_DISK_SPACE,
            node.role,
            passed=False,
            detail=(
                f"ディスクの空きが足りない ({node.remote_root})。"
                f"要る量: {_bytes_text(needed)} (中身 {_bytes_text(required_bytes)} +"
                f" 余裕 {SPACE_MARGIN_PERCENT}%)、空いている量: {_bytes_text(available)}"
            ),
        )
    return _result(
        GATE_DISK_SPACE,
        node.role,
        passed=True,
        detail=(
            f"ディスクの空きは足りる。要る量: {_bytes_text(needed)}"
            f" (中身 {_bytes_text(required_bytes)} + 余裕 {SPACE_MARGIN_PERCENT}%)、"
            f"空いている量: {_bytes_text(available)}"
        ),
    )


def _parse_available_bytes(text: str) -> int:
    """`df -B1 --output=avail` の出力 (見出しの行と、バイト数の行) を読む。"""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.isdecimal():
            return int(stripped)
    raise ValueError(f"ディスクの空きの行が読めない: {text.strip()!r}")


def gate_ports_free(
    runner: RemoteRunner,
    node: NodeDef,
    config: ConfigDef,
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> GateResult:
    """構成が使う待ち受けのポートが空いているか (design.md 「関門」: `ss -ltnH`)。

    `--network host` では、コンテナの外のプロセスとも衝突する。
    """
    return _safely(
        GATE_PORTS_FREE, node.role, partial(_ports_free, runner, node, config, timeout_s)
    )


def _ports_free(
    runner: RemoteRunner, node: NodeDef, config: ConfigDef, timeout_s: float
) -> GateResult:
    """gate_ports_free の中身。読めなかったことは、`gate_ports_free` が断りに変える。"""
    ports = config_ports(config, node.role)
    if not ports:
        return _result(
            GATE_PORTS_FREE,
            node.role,
            passed=True,
            detail="この構成は、待ち受けのポートを持たない",
        )
    result = runner.run(node, ("ss", "-ltnH"), timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _result(
            GATE_PORTS_FREE,
            node.role,
            passed=False,
            detail=f"待ち受けの一覧を読めなかった: {result.stderr.strip() or result.stdout}",
        )
    listening = parse_listening_ports(result.stdout)
    busy = [port for port in ports if port in listening]
    if busy:
        return _result(
            GATE_PORTS_FREE,
            node.role,
            passed=False,
            detail=(
                f"構成が使う待ち受けのポートを、すでに何かが使っている"
                f" ({', '.join(str(port) for port in busy)})。"
                "`--network host` では、コンテナの外のプロセスとも衝突する"
            ),
        )
    return _result(
        GATE_PORTS_FREE,
        node.role,
        passed=True,
        detail=f"使うポートは空いている ({', '.join(str(port) for port in ports)})",
    )


# --- 関門の全体 ---------------------------------------------------------


def _safely(gate: str, role: NodeRole, read: Callable[[], GateResult]) -> GateResult:
    """1 つの関門を流し、届かなかったこと・読めなかったことを、断りとして返す。"""
    try:
        return read()
    except RemoteError as exc:
        return _result(gate, role, passed=False, detail=f"読み取りが届かなかった: {exc}")
    except ValueError as exc:
        return _result(gate, role, passed=False, detail=f"読み取りの出力を読めなかった: {exc}")


def run_gates(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    *,
    manifest: WeightsManifest | None = None,
    containers: Mapping[NodeRole, Sequence[OwnContainer]] | None = None,
    timeout_s: float = READ_TIMEOUT_S,
) -> tuple[GateResult, ...]:
    """構成が使う台のそれぞれで、関門を `GATE_ORDER` の順に流す。

    引数:
        runner: 遠隔の実行役。ここからは、読み取りしか出さない。
        config: 選んだ構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        plans: `plan.build_plans` が組み立てたコンテナの計画 (置き場所を、印を埋めた
            あとの `--mount` から読むために使う)。
        manifest: 重みのマニフェスト (重みを使う構成では要る)。
        containers: すでに取ってある、自分のコンテナの一覧 (`already_running` の判定と、
            同じ 1 回の読み取りを使いたいときに渡す)。渡さなければ、ここで取る。
        timeout_s: 1 つの読み取りの時間切れ。

    返り値:
        台ごとに、`GATE_ORDER` の順に並べた結果。1 つ落ちても残りを流して並べる
        (`serve check` が、結果を一覧にするため)。ただし、入れなかった台は、何も
        読めないので、その台の残りの関門を飛ばす。

    例外:
        ValueError: 構成が使う役割のノードの定義か、その計画がないとき。
    """
    plans_by_role = {plan.node: plan for plan in plans}
    required_bytes = required_free_bytes(config, manifest)
    results: list[GateResult] = []
    for role in config.nodes:
        node = nodes.get(role)
        if node is None:
            raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
        plan = plans_by_role.get(role)
        if plan is None:
            raise ValueError(f"構成 '{config.name}' の {role} のコンテナの計画がない")
        reachable = gate_reachable(runner, node, timeout_s=timeout_s)
        results.append(reachable)
        if not reachable.passed:
            continue
        known = None if containers is None else containers.get(role)
        rest: tuple[tuple[str, Callable[[], GateResult]], ...] = (
            (
                GATE_OWN_STATE,
                partial(_own_state, runner, node, known, timeout_s=timeout_s),
            ),
            (GATE_GPU_IDLE, partial(gate_gpu_idle, runner, node, timeout_s=timeout_s)),
            (GATE_LAYOUT, partial(gate_layout, runner, node, plan, timeout_s=timeout_s)),
            (
                GATE_IMAGE_DIGEST,
                partial(gate_image_digest, runner, node, config, timeout_s=timeout_s),
            ),
            (
                GATE_WEIGHTS_VERIFIED,
                partial(gate_weights_verified, runner, node, config, manifest, timeout_s=timeout_s),
            ),
            (
                GATE_DISK_SPACE,
                partial(gate_disk_space, runner, node, required_bytes, timeout_s=timeout_s),
            ),
            (GATE_PORTS_FREE, partial(gate_ports_free, runner, node, config, timeout_s=timeout_s)),
        )
        results.extend(_safely(gate, role, read) for gate, read in rest)
    return tuple(results)


def _own_state(
    runner: RemoteRunner,
    node: NodeDef,
    known: Sequence[OwnContainer] | None,
    *,
    timeout_s: float,
) -> GateResult:
    """自分のコンテナの一覧を (まだ取っていなければ) 取って、関門に掛ける。"""
    found = list_own_containers(runner, node, timeout_s=timeout_s) if known is None else known
    return gate_own_state(node.role, found)


def match_running(
    plans: Sequence[ContainerPlan], containers: Mapping[NodeRole, Sequence[OwnContainer]]
) -> RunningMatch:
    """いま動いている自分のコンテナが、選んだ構成そのものかを見る (3.4 が使う口)。

    design.md 「lifecycle」の State Management: 名前と、構成の名前、イメージのダイジェスト、
    `config-sha256` のラベルが、構成の使う台のすべてで一致するときだけ、「すでに動いている」
    として扱う (関門より前に判定する)。名前が同じで中身が違うときは、違う項目を返すので、
    呼ぶ側は、それを示して断れる。
    """
    differences: list[str] = []
    matched = 0
    for plan in plans:
        found = next(
            (
                container
                for container in containers.get(plan.node, ())
                if container.name == plan.container_name
            ),
            None,
        )
        if found is None:
            continue
        if not found.running:
            differences.append(
                f"{plan.node}: {plan.container_name} は動いていない (状態は {found.state})"
            )
            continue
        node_differences = [
            f"{plan.node}: ラベル {key} が違う"
            f" (動いているもの: {found.labels.get(key, '(ない)')}、選んだ構成: {plan.labels[key]})"
            for key in _IDENTITY_LABELS
            if found.labels.get(key) != plan.labels.get(key)
        ]
        if node_differences:
            differences.extend(node_differences)
            continue
        matched += 1
    return RunningMatch(
        already_running=bool(plans) and matched == len(plans),
        differences=tuple(differences),
    )


# --- 了承を得る計画 ------------------------------------------------------


def stop_argv(name: str) -> tuple[str, ...]:
    """自分のコンテナを止める引数の列 (実際に止める側も、この関数で作る)。"""
    return ("docker", "stop", "-t", str(STOP_TIMEOUT_S), name)


def remove_argv(name: str) -> tuple[str, ...]:
    """自分のコンテナを消す引数の列。"""
    return ("docker", "rm", name)


def build_approved_plan(
    plans: Sequence[ContainerPlan], *, extra_forward: Sequence[PlannedCommand] = ()
) -> ApprovedPlan:
    """前に進むコマンドと、それを巻き戻すコマンドの組を作る (design.md 「remote」)。

    **巻き戻すコマンドは、この関数が `ContainerPlan` の列から自分で作る**。引数として
    自由に渡せないので、この計画が起こす名前以外のコンテナを、巻き戻しに入れる道がない
    (`types.ApprovedPlan` の検証と、二重に守る)。

    引数:
        plans: これから起こすコンテナの計画。
        extra_forward: コンテナを起こす前に流す、状態を変える呼び出し (起動の記録の配布、
            イメージの取得など)。ここにも、この計画が起こさないコンテナを対象にする
            `docker stop` / `docker rm` は書けない。

    例外:
        ValueError: 計画の引数の列が、その計画の名前と所有のラベルで起こすものでないとき。
            `extra_forward` が、この計画の外のコンテナを止めようとしたときも同じ。
    """
    names = tuple(plan.container_name for plan in plans)
    _check_plans(plans)
    _check_extra_forward(extra_forward, names)
    forward: list[PlannedCommand] = list(extra_forward)
    forward.extend(
        PlannedRun(
            node=plan.node,
            argv=plan.argv,
            container=plan.container_name,
            purpose=f"{plan.node} で {plan.container_name} を起こす",
        )
        for plan in plans
    )
    rollback = [
        PlannedRun(
            node=plan.node,
            argv=stop_argv(plan.container_name),
            container=plan.container_name,
            purpose=f"{plan.node} の {plan.container_name} を止める (片付け)",
        )
        for plan in plans
    ]
    rollback.extend(
        PlannedRun(
            node=plan.node,
            argv=remove_argv(plan.container_name),
            container=plan.container_name,
            purpose=f"{plan.node} の {plan.container_name} を消す (片付け)",
        )
        for plan in plans
    )
    return ApprovedPlan(forward=tuple(forward), rollback=tuple(rollback), own_container_names=names)


def _check_plans(plans: Sequence[ContainerPlan]) -> None:
    """計画の引数の列が、その計画の名前と、所有のラベルで起こすものかを確かめる。

    `plan.build_plans` を通った列は、必ずこの形になる。手で組み立てた `ContainerPlan` が
    渡ると、「前に進むコマンドはよその名前を起こし、巻き戻しは自分の名前を止める」という
    食い違いを作れてしまうので、ここで断つ (requirements 2.3)。
    """
    owner_label = f"{LABEL_OWNER}={OWNER}"
    for plan in plans:
        argv = plan.argv
        if tuple(argv[:2]) != ("docker", "run"):
            raise ValueError(
                f"コンテナの計画の引数の列が docker run で始まっていない: {plan.container_name}"
            )
        pairs = tuple(zip(argv, argv[1:], strict=False))
        started = [value for flag, value in pairs if flag == "--name"]
        if started != [plan.container_name]:
            raise ValueError(
                "コンテナの計画の --name が、その計画の名前と合わない"
                f" (列: {', '.join(started) or '(ない)'}、計画: {plan.container_name})"
            )
        if owner_label not in [value for flag, value in pairs if flag == "--label"]:
            raise ValueError(
                f"コンテナの計画に、所有のラベル ({owner_label}) がない"
                f" (あとから自分のものとして選べない): {plan.container_name}"
            )


_CONTAINER_TARGETING: Final[frozenset[str]] = frozenset({"stop", "rm"})
"""前に進むコマンドとして書かれても、対象を見張る docker のサブコマンド。"""

_STARTING_SUBCOMMANDS: Final[frozenset[str]] = frozenset({"run", "create"})
"""コンテナを起こす docker のサブコマンド (前に進むコマンドに、直に書けない)。"""

_VALUE_FLAGS: Final[Mapping[str, frozenset[str]]] = {
    "stop": frozenset({"-t", "--time", "-s", "--signal"}),
    "rm": frozenset(),
}
"""値を取るフラグ (`docker stop -t 90 <名前>` の 90 を、対象と読み違えないため)。"""

_BOOLEAN_FLAGS: Final[Mapping[str, frozenset[str]]] = {
    "stop": frozenset(),
    "rm": frozenset({"-f", "--force", "-l", "--link", "-v", "--volumes"}),
}
"""値を取らないフラグ。ここにないフラグを含む列は、対象を読み違えるので断る。"""


def _targets_of(subcommand: str, rest: Sequence[str]) -> tuple[str, ...]:
    """`docker stop` / `docker rm` の引数から、**位置で**、対象の語だけを取り出す。

    値を取るフラグの次の語だけを飛ばす。`-` で始まらない語は、数字だけであっても対象と
    して扱う (コンテナの名前にも、短い識別子にも、数字だけのものがありうる)。読み方の
    分からないフラグ (一覧にない、`-` で始まる語) があれば、どれが対象かを決められない
    ので、断る。
    """
    value_flags = _VALUE_FLAGS[subcommand]
    boolean_flags = _BOOLEAN_FLAGS[subcommand]
    targets: list[str] = []
    skip = False
    for word in rest:
        if skip:
            skip = False
            continue
        if not word.startswith("-"):
            targets.append(word)
            continue
        name, separator, _ = word.partition("=")
        if name in value_flags:
            skip = not separator  # `--time=90` は、次の語を取らない
            continue
        if name in boolean_flags and not separator:
            continue
        raise ValueError(
            f"docker {subcommand} に、読み方の分からないフラグがある"
            f" (どれが対象かを決められない): {word}"
        )
    return tuple(targets)


def _check_extra_forward(commands: Sequence[PlannedCommand], names: Sequence[str]) -> None:
    """前に進むコマンドに、この計画の外のコンテナへの操作が紛れないことを見る。

    コンテナを起こす列 (`docker run`、`docker create`。2 語の形も) は、ここには書けない。
    起こす道を `plans` (`ContainerPlan`) の 1 つだけにして、名前と所有のラベルの検査
    (`_check_plans`) を必ず通すためである。

    **層の受け持ち**: ここで対象を読むのは、1 語の形 (`docker stop` / `docker rm`) だけで
    ある。2 語の形 (`docker container stop` / `rm`) は、`remote.CallGuard` が、許可の一覧に
    ないサブコマンドとして断る (試験で固定)。
    """
    for command in commands:
        if not isinstance(command, PlannedRun):
            continue
        argv = command.argv
        if argv[0] != "docker" or len(argv) < 2:
            continue
        if argv[1] in _STARTING_SUBCOMMANDS or (
            argv[1] == "container" and len(argv) > 2 and argv[2] in _STARTING_SUBCOMMANDS
        ):
            raise ValueError(
                "コンテナを起こす列は、前に進むコマンドに直に書けない"
                " (起こす道は、計画 (ContainerPlan) の 1 つだけにする):"
                f" {shlex.join(argv)}"
            )
        if argv[1] not in _CONTAINER_TARGETING:
            continue
        outside = [word for word in _targets_of(argv[1], argv[2:]) if word not in names]
        if command.container is not None and command.container not in names:
            outside.append(command.container)
        if outside:
            raise ValueError(
                "この計画が起こさないコンテナを、前に進むコマンドに入れられない"
                f" (自分のものでないコンテナを、名前で止めないため): {', '.join(outside)}"
            )


def rollback_commands(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    *,
    timeout_s: float = READ_TIMEOUT_S,
) -> tuple[tuple[PlannedRun, ...], tuple[str, ...]]:
    """**実際に流してよい巻き戻しだけ**を、自分のラベルで絞った一覧から決める。

    `build_approved_plan` は、起動の**前**に、計画が起こす名前の `docker stop` / `docker rm` を
    書く (そうしないと、片付けが了承のない呼び出しとして断られる)。しかし、**`docker stop
    <名前>` と `docker rm <名前>` は、その名前のコンテナが誰のものでも、止めて消す**。
    `docker run` が名前の衝突で失敗した場合、その名前は、この道具のものではないコンテナを
    指しているので、計画に書いてあるからといって、そのまま流してはならない
    (requirements 2.3、design.md 「remote」の不変条件)。

    衝突かどうかを、`docker run` の標準エラーの文面で見分けるのは、歯止めにならない (文面は
    docker の版で変わりうるし、実機で確かめていない)。そこで、**流す直前に
    `list_own_containers` (自分のラベルで絞り、行の所有のラベルも確かめる、ただ 1 つの入口)
    を読み、その台の一覧に、その計画の名前の行があるときだけ流す**。docker の名前は、1 台の
    中で一意なので、自分の一覧にその名前があれば、その名前はこの道具のコンテナを指している。

    一覧にその名前がなければ、その台には止めるものがない (コンテナができなかった、または、
    その名前はよそのもの) ので、`stop` / `rm` を 1 つも返さない。一覧そのものを読めなかった台
    (入れない、所有のラベルのない行が紛れた、出力が読めない) でも、確かめられないので流さず、
    理由を返す (呼ぶ側は、その文を見せて終わる)。理由には、よそのコンテナの名前も識別子も
    入らない (`list_own_containers` が、そもそも出さない)。

    **これは読み取りだけの関数である**。返した列を実際に流すのは、呼ぶ側 (`image`、あとで
    `lifecycle`) で、列は `stop_argv` / `remove_argv` で作るので、了承を得た計画の巻き戻しと
    完全に一致する (`remote.CallGuard` は、完全な一致で見る)。並び (2 台ぶんの `stop` の
    あとに `rm`) も `build_approved_plan` と同じにしてあり、試験が、その一致を固定する。

    引数:
        runner: 遠隔の実行役。ここからは、コンテナの一覧の読み取りしか出さない。
        nodes: 役割ごとのノードの定義。
        plans: 起こした (起こそうとした) コンテナの計画。
        timeout_s: 一覧の読み取りの時間切れ。

    返り値:
        流してよい巻き戻しの列と、確かめられなかった台の理由の列。

    例外:
        ValueError: `plans` が使う役割のノードの定義が `nodes` にないとき。
    """
    present: list[ContainerPlan] = []
    unchecked: list[str] = []
    for plan in plans:
        node = nodes.get(plan.node)
        if node is None:
            raise ValueError(f"nodes.{plan.node} の定義がない ({plan.container_name} の片付け)")
        try:
            containers = list_own_containers(runner, node, timeout_s=timeout_s)
        except (RemoteError, ValueError) as exc:
            unchecked.append(
                f"{plan.node}: 自分のコンテナの一覧を読めないので、{plan.container_name} の"
                f" 片付けに行かない (名前だけで止めると、よそのコンテナを止めうる): {exc}。"
                "コンテナが残っているかもしれないので、入れるようになってから `serve stop` で"
                "片付ける"
            )
            continue
        if any(container.name == plan.container_name for container in containers):
            present.append(plan)
    commands = [
        PlannedRun(
            node=plan.node,
            argv=stop_argv(plan.container_name),
            container=plan.container_name,
            purpose=f"{plan.node} の {plan.container_name} を止める (片付け)",
        )
        for plan in present
    ]
    commands.extend(
        PlannedRun(
            node=plan.node,
            argv=remove_argv(plan.container_name),
            container=plan.container_name,
            purpose=f"{plan.node} の {plan.container_name} を消す (片付け)",
        )
        for plan in present
    )
    return tuple(commands), tuple(unchecked)


def format_plan(plan: ApprovedPlan, nodes: Mapping[NodeRole, NodeDef]) -> str:
    """計画の全体 (前に進むコマンド、巻き戻すコマンド、対象の機械) を、見せる文にする。"""
    used = tuple(dict.fromkeys(command.node for command in (*plan.forward, *plan.rollback)))
    lines = ["Spark の状態を変える操作を行う。対象の機械:"]
    lines.extend(f"  {role}: {_shown_node(nodes, role)}" for role in used)
    lines.append("")
    lines.append(f"前に進むコマンド ({len(plan.forward)} 件):")
    lines.extend(
        f"  {number}. {_shown_node(nodes, command.node)}: {_shown_command(command)}"
        for number, command in enumerate(plan.forward, start=1)
    )
    lines.append("")
    lines.append(
        f"巻き戻すコマンド ({len(plan.rollback)} 件。"
        "失敗と中断のときに流す。対象は、この計画が起こすコンテナだけ):"
    )
    lines.extend(
        f"  {number}. {_shown_node(nodes, command.node)}: {_shown_command(command)}"
        for number, command in enumerate(plan.rollback, start=1)
    )
    return "\n".join(lines)


def _shown_node(nodes: Mapping[NodeRole, NodeDef], role: NodeRole) -> str:
    node = nodes.get(role)
    return role if node is None else f"{role} ({node.ssh_host})"


def _shown_command(command: PlannedCommand) -> str:
    if isinstance(command, PlannedPush):
        deletes = "あり" if command.delete else "なし"
        return f"配布 {command.local_dir}/ → {command.remote_subdir}/ (--delete {deletes})"
    return shlex.join(command.argv)


# --- 了承 ---------------------------------------------------------------


class Confirmer(Protocol):
    """計画を見せて、了承を得る口 (design.md 「関門」)。"""

    def confirm(self, plan_text: str) -> bool: ...


def _show(stream: TextIO, plan_text: str) -> None:
    """計画を、進捗と同じ stderr に見せる (標準出力は、後の処理が読む)。"""
    stream.write(f"{plan_text}\n")
    stream.flush()


class TerminalConfirmer:
    """端末で `yes` の入力を待つ了承 (design.md 「関門」の 1 つめの実装)。

    端末でなければ (`sys.stdin.isatty()` が偽)、尋ねられないので `ApprovalError` にする
    (終了コード 1)。試験のために、入力と出力と、端末かどうかを差し替えられる。
    """

    def __init__(
        self,
        *,
        stdin: TextIO | None = None,
        stderr: TextIO | None = None,
        isatty: Callable[[], bool] | None = None,
    ) -> None:
        self._stdin: TextIO = sys.stdin if stdin is None else stdin
        self._stderr: TextIO = sys.stderr if stderr is None else stderr
        self._isatty: Callable[[], bool] = self._stdin.isatty if isatty is None else isatty

    def confirm(self, plan_text: str) -> bool:
        """計画を見せて、`yes` の入力だけを了承として受ける。"""
        _show(self._stderr, plan_text)
        if not self._isatty():
            raise ApprovalError(
                "端末ではないので、了承を尋ねられない (会話で了承を得てから --yes を付けて打ち直す)"
            )
        self._stderr.write(_PROMPT)
        self._stderr.flush()
        # 落とすのは、末尾の改行だけ。前後の空白まで落とすと、打ち間違い (" yes"、"yes ")
        # が了承になる。了承は、`yes` との完全な一致だけで通す
        answer = self._stdin.readline().removesuffix("\n").removesuffix("\r")
        return answer == _YES


class AssumeYesConfirmer:
    """`--yes` が付いたときの了承 (design.md 「関門」の 2 つめの実装)。

    計画を見せてから通す。Claude が計測者の代わりに打つときは、会話の中で了承を得てから
    `--yes` を付ける (README に書く)。
    """

    def __init__(self, *, stderr: TextIO | None = None) -> None:
        self._stderr: TextIO = sys.stderr if stderr is None else stderr

    def confirm(self, plan_text: str) -> bool:
        """計画を見せて、通す。"""
        _show(self._stderr, plan_text)
        self._stderr.write("--yes が付いているので、了承されたものとして進む\n")
        self._stderr.flush()
        return True


def make_confirmer(
    *,
    assume_yes: bool,
    stdin: TextIO | None = None,
    stderr: TextIO | None = None,
    isatty: Callable[[], bool] | None = None,
) -> Confirmer:
    """`--yes` の有無で、2 つの実装を選ぶ。"""
    if assume_yes:
        return AssumeYesConfirmer(stderr=stderr)
    return TerminalConfirmer(stdin=stdin, stderr=stderr, isatty=isatty)


def request_approval(
    confirmer: Confirmer,
    runner: RemoteRunner,
    plan: ApprovedPlan,
    nodes: Mapping[NodeRole, NodeDef],
) -> None:
    """計画を見せて了承を得て、実行役に渡す (requirements 2.1)。

    断られたら `ApprovalError` にして、実行役には何も渡さない。状態を変える呼び出しは、
    了承を得た計画に含まれていなければ `remote` が断るので、これが唯一の入口になる。
    """
    if not confirmer.confirm(format_plan(plan, nodes)):
        raise ApprovalError("計測者が了承しなかったので、状態を変える操作は 1 つも行わない")
    runner.approve(plan)
