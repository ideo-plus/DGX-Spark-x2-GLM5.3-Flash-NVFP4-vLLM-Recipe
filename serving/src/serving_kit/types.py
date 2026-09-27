"""全体で共有する型。

依存の向きの出発点であり、この module はほかの `serving_kit` の module を読み込まない
(design.md 「Architecture Pattern & Boundary Map」の Dependency direction:
`types → config → remote → plan, observe → guards → …`)。標準ライブラリと pydantic だけに
依存する。後続の部品は、関数と手続きの実装だけを受け持ち、このファイルを変更しない
(tasks.md 1.2)。

守る決まり:

- すべての型は `frozen` かつ `extra="forbid"`。知らない項目は、綴りの誤りとして断る
- 根拠 (`Provenance`) は、「出典 (`source`) と原文 (`quote`) の組」か「実測の記録の場所
  (`measured`)」の、どちらか一方だけを受ける (requirements 3.6、design.md 「types /
  config」の検査 1)。`measured` が指すファイルが実在するかの検査は、リポジトリの位置を
  知っている `config` の仕事なので、ここでは行わない
- 認証の情報の値は、どの型にも入れない。環境変数は、構成の `env` に名前と値の組として
  書いたものだけを渡す (requirements 2.6。秘密らしい名前の断りは `plan` が行う)
- 確かめの結果の型 (`SmokeOutcome`、`ProbeOutcome`、`ThinkingOutcome`、`WatchOutcome`) は、
  送った内容と応答の本文を入れる項目を持たない (requirements 10.5)
- 記録の時刻は UTC の `datetime`。Spark の上の道筋は文字列、Mac の上の道筋は `Path`

この module に置くのは、データの型だけである。`RemoteRunner` と `Confirmer` の
`Protocol`、`ConfigError` や `RemoteError` などの例外は、設計がそれぞれの部品
(`remote`、`config`、`guards`) に置いているので、ここには置かない。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StrictBool,
    field_validator,
    model_validator,
)

CONFIG_SCHEMA_VERSION: Final[int] = 1
"""構成の定義とノードの定義の書式の版 (design.md Data Models の `schema_version`)。

書式を変えたら上げる。上げたときは、すべての構成を検査し直す (Revalidation Triggers)。
"""

MANIFEST_EXCLUDED_PATHS: Final[frozenset[str]] = frozenset({"README.md", ".gitattributes"})
"""重みのマニフェストに載せないファイル (design.md Data Models のマニフェスト)。

モデルカード (`README.md`) は、第三者のレシピに当たりうるので取得も記載もしない
(requirements 8.8、11.3)。
"""

_ROLLBACK_SUBCOMMANDS: Final[frozenset[str]] = frozenset({"stop", "rm"})
"""巻き戻しに置ける docker のサブコマンド (design.md 「remote」)。"""

_MAX_FAILURE_EXCERPT_LINES: Final[int] = 40
"""誤りの前後の行の上限 (design.md 「observe」)。"""

_MAX_PORT: Final[int] = 65535


class _Frozen(BaseModel):
    """この module の型に共通の設定。

    書いたあとに変えず (更新は `model_copy(update=...)` で行う)、知らない項目は、綴りの
    誤りとして弾く (design.md 「types / config」)。`bench_harness.types` は、凍結しない
    型も要るので基底を 2 段にしているが、ここでは凍結しない型がないので 1 段にする。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- 印と分類 -----------------------------------------------------------

NodeRole = Literal["head", "worker"]
"""2 台の Spark の役割 (design.md 「types / config」の `NodeDef.role`)。"""

ConfigKind = Literal["serve", "probe", "job", "fetch", "inspect"]
"""構成の種類。推論サーバー、縮小の確認、通信の確認のジョブ、重みの取得、イメージの
中の読み取りの 5 つで、この道具が起こすコンテナは、どれも同じ仕組みを通る
(design.md 「Architecture Integration」の「1 つの起動の仕組み」)。
"""

ContainerState = Literal["absent", "running", "exited"]
"""自分のコンテナの状態 (design.md 「lifecycle」の `NodeStatus.container_state`)。"""

Lang = Literal["en", "ja"]
"""短い要求の言語。英語と日本語を 1 つずつ送る (requirements 6.5)。"""

VerificationScope = Literal["all", "probe_files"]
"""照合の対象。`probe_files` は、マニフェストのうち safetensors を除いたファイル
(設定とトークナイザ) だけを見る (design.md 「guards」の `kind = "probe"` の扱い)。
"""

StartStatus = Literal["ready", "already_running", "refused", "failed"]
"""起動の結末 (design.md 「lifecycle」の `start`)。"""

StopStatus = Literal["stopped", "already_stopped", "gpu_not_released"]
"""停止の結末 (design.md 「lifecycle」の `stop`)。"""

ProbeStatus = Literal["ready", "failed", "inconclusive"]
"""1 台の縮小の確認の結末 (design.md 「probe」の「結果は 3 つ」)。"""

WatchFinding = Literal["unresponsive", "stalled"]
"""見張りの判定 (design.md 「watch」)。"""

ThermalFinding = Literal["thermal"]
"""熱の判定。`WatchFinding` とは別に持つ: `WatchFinding` (`unresponsive`/`stalled`) は
推論サーバー側の判定で記録の回収を伴うが、`thermal` は熱の判定で、推論サーバーは壊れて
いないため記録の回収を伴わない。"""

CpuCluster = Literal["x925", "a725"]
"""GB10 の 2 種類の CPU コア群 (X925 は周波数の上限を掛ける対象、A725 は上限を掛けない対象。
issue #10、docs/results/2026-09-23-thermal-source.md)。"""

ThinkingVariant = Literal[
    "none",
    "output_config_low",
    "chat_template_low",
    "output_config_medium",
    "clear_thinking",
]
"""thinking の深さの渡し方の 5 通り (design.md 「thinking」の (1)〜(5))。

`output_config_medium` は、効かないはずの値である。
"""


class KnownFailure(StrEnum):
    """知っている起動の失敗の種類 (design.md 「observe」の `KnownFailure` の表)。

    どれにも当てはまらない終了は `UNCLASSIFIED` にして、誤りの文面をそのまま残す。
    """

    PE_DIM_ASSERT = "pe_dim_assert"
    NO_ATTENTION_BACKEND = "no_attention_backend"
    NO_KERNEL_IMAGE = "no_kernel_image"
    KPOOL_BLOCK_SIZE = "kpool_block_size"
    STARTUP_MEMORY_CHECK = "startup_memory_check"
    DEEP_GEMM_MISSING = "deep_gemm_missing"
    UNCLASSIFIED = "unclassified"


# --- 根拠と、構成の定義 -------------------------------------------------


def _check_docs_relative_path(value: str, field: str) -> None:
    """実測の記録の道筋が、リポジトリの `docs/` の下の相対のパスであることを確かめる。

    design.md 「types / config」: `measured` は、リポジトリの `docs/` の下の、実在する
    ファイルへの相対のパス。実在するかどうかは `config` が確かめる。
    """
    if not value.startswith("docs/") or value == "docs/":
        raise ValueError(f"{field} は、リポジトリの docs/ の下の相対のパスにする: {value}")
    if value.endswith("/") or ".." in Path(value).parts:
        raise ValueError(f"{field} の道筋が正しくない: {value}")


class Provenance(_Frozen):
    """設定の根拠 (requirements 3.6)。

    「出典 (`source`) と原文 (`quote`) の組」か「実測の記録の場所 (`measured`)」の、
    どちらか一方だけを受ける。両方ある、どちらもない、片方だけ (出典だけで原文がない、
    原文だけで出典がない) は、ここで断る (design.md 「types / config」の検査 1)。
    """

    source: HttpUrl | None = None
    quote: str | None = Field(default=None, min_length=1)
    measured: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_evidence(self) -> Self:
        if self.source is not None and self.quote is None:
            raise ValueError("根拠に source があるのに quote (原文の抜粋) がない")
        if self.quote is not None and self.source is None:
            raise ValueError("根拠に quote があるのに source (出典の URL) がない")
        cited = self.source is not None
        if cited and self.measured is not None:
            raise ValueError("根拠は、source と quote の組か measured の、どちらか一方だけにする")
        if not cited and self.measured is None:
            raise ValueError("根拠がない (source と quote の組、または measured が要る)")
        if self.measured is not None:
            _check_docs_relative_path(self.measured, "measured")
        return self


class Setting(Provenance):
    """構成の中の 1 つの設定 (requirements 3.6、design.md 「types / config」)。

    TOML の鍵は、フラグではなく設定の識別子にするので、同じフラグを 2 度書ける
    (`--ulimit`、`--mount`)。`flag` を書かないと位置の引数になり、`value` を書かないと
    値のないフラグになる (design.md Data Models の例)。
    """

    flag: str | None = Field(default=None, min_length=1)
    value: str | None = None
    why: str = Field(min_length=1)
    only_on: NodeRole | None = None
    is_port: bool = False

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.flag is None and self.value is None:
            raise ValueError("flag のない設定は位置の引数なので、value が要る")
        if self.is_port:
            if self.value is None:
                raise ValueError("is_port の設定には、待ち受けのポートの番号を value に書く")
            if not self.value.isdecimal() or not 0 < int(self.value) <= _MAX_PORT:
                raise ValueError(f"is_port の value がポートの番号ではない: {self.value}")
        return self


class ImageRef(Provenance):
    """使うイメージの参照 (requirements 3.2)。

    `ref` は `<名前>@sha256:<64 桁の 16 進>` またはローカルの完全なイメージ ID
    `sha256:<64 桁の 16 進>` を受ける。タグや短縮 ID は中身を固定できないため断る。
    """

    ref: str = Field(pattern=r"^(?:[a-z0-9][a-z0-9._/-]*@)?sha256:[0-9a-f]{64}$")
    seen_as: str = Field(min_length=1)
    size_bytes: PositiveInt


class WeightsRef(Provenance):
    """使う重みの参照 (requirements 3.4)。

    版は 40 桁の commit で固定する。`manifest` は `serving/weights/` の下のファイルの
    名前で、実在するかどうかは `config` が確かめる (design.md の検査 3)。
    """

    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    manifest: str = Field(min_length=1)
    mount_at: str = Field(min_length=1)

    @property
    def identity(self) -> str:
        """Hub の重みの同一性 (入手先と版。現行の `repo@revision` と同じ文字列)。"""
        return f"{self.repo}@{self.revision}"


class WeightsOrigin(_Frozen):
    """元の重みの同一性 (派生の根拠。requirements 3.4、issue #57 やること 1)。

    「手元で変換した重み」の根拠は、出典 URL ではなく、どの重みを、どの版から取ったかで
    ある。それを `repo` と 40 桁の `revision` の組で持つ。
    """

    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class OriginWeightsRef(WeightsOrigin):
    """構成に書く、元の重みの参照 (issue #57 やること 1)。

    `manifest` は元の重みのマニフェストの名前で、`WeightsRef` と同じく
    `serving/weights/` の下のファイルの名前である。実在するかどうかは `config` が確かめる。
    """

    manifest: str = Field(min_length=1)


class ConversionSpec(_Frozen):
    """手元で重みを変換したときの条件 (issue #57 やること 1)。

    `tool` はリポジトリの中の相対の道筋、`commit` はその道具の版 (40 桁の 16 進)、
    `args` は渡した引数、`target_pattern` は変換の対象を選ぶ正規表現である。`tool` の
    実在は確かめない (コミットで固定した道具は、HEAD に同じ道筋で残るとは限らない)。
    """

    tool: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    args: tuple[str, ...] = ()
    target_pattern: str = Field(min_length=1)

    @field_validator("tool")
    @classmethod
    def _check_tool(cls, value: str) -> str:
        if value.startswith("/") or ".." in Path(value).parts:
            raise ValueError(f"変換の道具の道筋は、リポジトリの中の相対のパスにする: {value}")
        return value

    @field_validator("target_pattern")
    @classmethod
    def _check_target_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"対象の正規表現をコンパイルできない: {value} ({exc})") from exc
        return value


class Derivation(_Frozen):
    """派生の重みの同一性 (参照・マニフェスト・照合の記録が共有する。issue #57)。

    名前・元の重み・変換の条件の組で決まる。どれか 1 つでも違えば、別の重みである。
    """

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]*$")
    origin: WeightsOrigin
    conversion: ConversionSpec

    @property
    def identity(self) -> str:
        return f"derived:{self.name}:{self.origin.repo}@{self.origin.revision}"


class DerivedWeightsRef(_Frozen):
    """手元で変換した重みの参照 (issue #57 やること 1)。

    Hub の `WeightsRef` は入手先と版の組で重みを指すが、手元で作った重みにはそれがない。
    代わりに、元の重みの参照 (`origin`)、変換の条件 (`conversion`)、変換の結果の
    マニフェストの名前 (`manifest`)、置き場所 (`mount_at`) を持つ。根拠は元の参照と
    変換の条件そのものなので、`Provenance` は継承しない。`kind = "derived"` を必須の
    明示の項目にして、TOML と JSON を自己記述にする (値の有無からは推測しない)。
    """

    kind: Literal["derived"]
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]*$")
    origin: OriginWeightsRef
    conversion: ConversionSpec
    manifest: str = Field(min_length=1)
    mount_at: str = Field(min_length=1)

    @property
    def derivation(self) -> Derivation:
        """この参照の同一性 (元の参照から `WeightsOrigin` を作って組にする)。"""
        return Derivation(
            name=self.name,
            origin=WeightsOrigin(repo=self.origin.repo, revision=self.origin.revision),
            conversion=self.conversion,
        )

    @property
    def identity(self) -> str:
        return self.derivation.identity


AnyWeightsRef = WeightsRef | DerivedWeightsRef
"""構成が指せる重みの参照 (Hub か、手元で変換したものか)。"""


class ConfigDef(_Frozen):
    """名前の付いた構成の定義 (requirements 3.1)。

    `docker`、`args`、`env` の並ぶ順序は、TOML に書いた順である (`tomllib` は鍵の順序を
    保つ)。コンテナの中で動かすプログラムは、専用の項目ではなく docker の設定
    (`--entrypoint`) で表す。`nodes` の数は、`serve`、`job`、`fetch` が 2 つ、`probe` と
    `inspect` が 1 つになる (design.md 「types / config」)。
    """

    name: str = Field(min_length=1)
    kind: ConfigKind
    description: str = Field(min_length=1)
    nodes: tuple[NodeRole, ...] = Field(min_length=1)
    image: ImageRef
    weights: AnyWeightsRef | None = None
    docker: dict[str, Setting]
    args: dict[str, Setting]
    env: dict[str, Setting] = Field(default_factory=dict)
    ready_timeout_s: PositiveInt
    served_model_name: str | None = None
    allow_speculative: StrictBool = False
    """投機的デコードの指定 (検査 5 の 4 つのフラグ) を許す構成だけ `true` にする。

    起動後は、許す構成では `/metrics` に投機の指標が出ることを正常とし、出ないことを異常と
    する (design.md 6.7、ADR 0006 K1)。`StrictBool` にするのは、TOML の文字列 `"true"` を
    真偽値として読み替えないためである。
    """

    @model_validator(mode="after")
    def _check_nodes_and_model_name(self) -> Self:
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError(f"nodes に同じ役割が 2 度ある: {', '.join(self.nodes)}")
        needs_name = self.kind in ("serve", "probe")
        if needs_name and self.served_model_name is None:
            raise ValueError(f"kind が {self.kind} の構成には served_model_name が要る")
        if not needs_name and self.served_model_name is not None:
            raise ValueError(f"kind が {self.kind} の構成は served_model_name を持たない")
        name = self.served_model_name
        if name is not None and (not name or any(ch in name for ch in " \t,/")):
            raise ValueError(f"served_model_name は、/ や空白を含まない 1 つの名前にする: {name}")
        if self.kind == "probe" and isinstance(self.weights, DerivedWeightsRef):
            raise ValueError(
                "kind が probe の構成は、手元で変換した派生の重みを使えない (派生の重みには"
                " serve fetch --probe-files の道がなく、縮小の確認用の置き場所 probe/<名前>/ に"
                "設定とトークナイザを置けない)"
            )
        return self


class NodeDef(_Frozen):
    """1 台の Spark の定義 (design.md Data Models のノードの定義)。

    直結の側の 3 つ (`fabric_addr`、`fabric_ifname`、`fabric_measured`) は、実測するまで
    空のままにする。空のままでも、取得、イメージの中の読み取り、縮小の確認の構成は選べる
    (2 台の `serve` と `job` の構成だけが、`config` の検査 7 で断られる)。
    """

    role: NodeRole
    ssh_host: str = Field(min_length=1)
    lan_addr: IPv4Address
    remote_root: str = Field(min_length=1)
    fabric_addr: IPv4Address | None = None
    fabric_ifname: str | None = Field(default=None, min_length=1)
    fabric_measured: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_paths(self) -> Self:
        if not self.remote_root.startswith("/") or self.remote_root.endswith("/"):
            raise ValueError(
                f"remote_root は、末尾に / を付けない絶対のパスにする: {self.remote_root}"
            )
        if self.fabric_measured is not None:
            _check_docs_relative_path(self.fabric_measured, "fabric_measured")
        return self


# --- 遠隔の実行 ---------------------------------------------------------


class CommandResult(_Frozen):
    """遠隔で 1 つのコマンドを流した結果 (design.md 「remote」)。"""

    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


class PlannedRun(_Frozen):
    """計画の中の、遠隔のコマンドの 1 つ (design.md 「remote」の計画の組)。

    `container` は、この呼び出しが対象にするコンテナの名前である。巻き戻すコマンドは、
    その計画が起こす名前のコンテナだけを対象にできるので、名前を型で持つ。
    """

    action: Literal["run"] = "run"
    node: NodeRole
    argv: tuple[str, ...] = Field(min_length=1)
    container: str | None = Field(default=None, min_length=1)
    purpose: str = ""


class PlannedPush(_Frozen):
    """計画の中の、配布の 1 つ (design.md 「remote」)。

    配布は、つねに状態を変える呼び出しとして扱うので、前に進むコマンドとして計画に
    入れる。宛先を `remote_root` の下の決まった部分に限る検査は、`remote` が行う。
    """

    action: Literal["push"] = "push"
    node: NodeRole
    local_dir: Path
    remote_subdir: str = Field(min_length=1)
    delete: bool = False
    purpose: str = ""


PlannedCommand = Annotated[PlannedRun | PlannedPush, Field(discriminator="action")]
"""計画の中の 1 つの呼び出し。`action` で 2 つを読み分ける。"""


class ApprovedPlan(_Frozen):
    """了承を得た計画 (design.md 「remote」。`guards` が作り、`SshRunner` が受け取る)。

    前に進むコマンドと、それを巻き戻すコマンドの組で持つ。巻き戻すコマンドは、この計画が
    起こす名前 (`own_container_names`) のコンテナの `docker stop` と `docker rm` だけで、
    ほかの名前を入れられない。これで、失敗や中断のあとの片付けが、了承のない呼び出しとして
    断られず、自分のものでないコンテナを名前で止めることもなくなる。
    """

    forward: tuple[PlannedCommand, ...] = Field(min_length=1)
    rollback: tuple[PlannedRun, ...] = ()
    own_container_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_rollback(self) -> Self:
        for command in self.rollback:
            if command.container is None:
                raise ValueError("巻き戻すコマンドは、対象の container の名前を持つ")
            if command.container not in self.own_container_names:
                raise ValueError(
                    "巻き戻せるのは、この計画が起こすコンテナだけ "
                    f"(own_container_names にない: {command.container})"
                )
            if (
                command.argv[0] != "docker"
                or len(command.argv) < 2
                or command.argv[1] not in _ROLLBACK_SUBCOMMANDS
            ):
                raise ValueError(f"rollback に置けるのは docker stop と rm だけ: {command.argv}")
        return self


# --- 組み立て -----------------------------------------------------------


class ContainerPlan(_Frozen):
    """組み立てた、1 台ぶんのコンテナの計画 (design.md 「plan」)。

    この道具が起こすコンテナは、5 つの `kind` のどれもここを通る。名前は
    `vb-<構成>-<役割>` で、英数字とハイフンだけからなる。
    """

    node: NodeRole
    container_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]*$")
    labels: dict[str, str]
    argv: tuple[str, ...] = Field(min_length=1)


# --- 記録からの読み取り -------------------------------------------------


class LaunchObservation(_Frozen):
    """起動の記録から読み取った事実 (design.md 「observe」)。

    読めなかった項目は、断らずに空のまま返す。文字列は vLLM の版で変わりうるので、
    イメージを替えたら確かめ直す (Revalidation Triggers)。
    """

    vllm_version: str | None = None
    attention_backend: str | None = None
    attention_candidates: tuple[str, ...] = ()
    moe_backend: str | None = None
    kv_cache_tokens: int | None = None
    kv_cache_gib: float | None = None
    max_concurrency_note: str | None = None
    model_loading_gib: float | None = None
    model_loading_s: float | None = None
    engine_init_s: float | None = None
    speculative_config_seen: bool = False
    known_failure: KnownFailure | None = None
    failure_excerpt: tuple[str, ...] = Field(default=(), max_length=_MAX_FAILURE_EXCERPT_LINES)


class NcclObservation(_Frozen):
    """通信の記録から読み取った事実 (design.md 「observe」)。

    `gdrdma_seen` は、Spark では偽が正常である。
    """

    nccl_version: str | None = None
    network: Literal["IB", "Socket"] | None = None
    ib_no_device: bool = False
    ib_devices_line: str | None = None
    merged_nic: bool = False
    coll_channels: int | None = None
    gdrdma_seen: bool = False
    socket_channel_seen: bool = False


# --- 関門 ---------------------------------------------------------------


class GateResult(_Frozen):
    """1 つの関門の結果 (design.md 「guards」)。

    断るときは、`detail` に、見つけたものと、要る量と空いている量を書く。
    """

    gate: str = Field(min_length=1)
    node: NodeRole | None
    passed: bool
    detail: str


# --- 重み ---------------------------------------------------------------


class ManifestFile(_Frozen):
    """マニフェストの 1 行 (design.md Data Models のマニフェスト)。"""

    path: str = Field(min_length=1)
    size: NonNegativeInt
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestFiles(_Frozen):
    """マニフェストのファイルの一覧と合計 (Hub と派生で共有する。issue #57)。

    `files` は `path` の順に並べ、モデルカード (`README.md`) と `.gitattributes` は
    載せない。並びと合計の検証は、派生のマニフェストにも同じように掛かる。

    コミットするファイルのバイト列 (`canonical_bytes`) と、その SHA-256 (`content_sha256`) の
    所有者も、この型である (`weights.to_json_bytes` は、ここへ委譲する)。照合の記録は、
    `content_sha256` で、どのマニフェストの中身を照合したかを結び付ける。
    """

    total_bytes: NonNegativeInt
    files: tuple[ManifestFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_files(self) -> Self:
        paths = [entry.path for entry in self.files]
        excluded = sorted(set(paths) & MANIFEST_EXCLUDED_PATHS)
        if excluded:
            raise ValueError(f"files に載せないファイルがある: {', '.join(excluded)}")
        for path in paths:
            if path.startswith("/") or ".." in Path(path).parts:
                raise ValueError(f"files の path は、重みの置き場所の下の相対のパスにする: {path}")
        if len(set(paths)) != len(paths):
            raise ValueError("files に同じ path が 2 度ある")
        if paths != sorted(paths):
            raise ValueError("files は path の順に並べる")
        total = sum(entry.size for entry in self.files)
        if total != self.total_bytes:
            raise ValueError(f"total_bytes ({self.total_bytes}) が files の合計 ({total}) と違う")
        return self

    def canonical_bytes(self) -> bytes:
        """マニフェストを、同じ入力なら同じバイト列になる形で書き出す。

        鍵の順 (アルファベット順)、2 字の字下げ、末尾の改行を固定する (`logs.py` の
        `collect.json` と同じ流儀)。`weights.write_manifest` が書くバイト列は、これである。
        """
        payload = self.model_dump(mode="json")
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        return text.encode("utf-8")

    @property
    def content_sha256(self) -> str:
        """`canonical_bytes` の SHA-256 (小文字の 16 進 64 桁)。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def probe_files(self) -> tuple[ManifestFile, ...]:
        """縮小の確認が使う、safetensors を除いたファイル (設定とトークナイザ)。

        design.md 「guards」: `kind = "probe"` の構成では、重みの照合と空きの関門が、この
        ファイルだけを `probe/<slug>/` について見る。
        """
        return tuple(entry for entry in self.files if not entry.path.endswith(".safetensors"))


class WeightsManifest(ManifestFiles):
    """Hub の重みのマニフェスト (requirements 3.4、design.md Data Models)。

    Mac で作ってコミットしたものが、照合の正解である。
    """

    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at: datetime

    @property
    def identity(self) -> str:
        return f"{self.repo}@{self.revision}"


class DerivedWeightsManifest(ManifestFiles):
    """手元で変換した重みのマニフェスト (issue #57 やること 1・2)。

    #56 の道具が書いたものを Mac に写し、`serving/weights/` の下にコミットしたものが、
    照合の正解である。`kind = "derived"` と、元の重みと変換の条件 (`derivation`) を持つ。
    """

    kind: Literal["derived"]
    derivation: Derivation
    generated_at: datetime

    @property
    def identity(self) -> str:
        return self.derivation.identity


AnyWeightsManifest = WeightsManifest | DerivedWeightsManifest
"""読み込んだ重みのマニフェスト (Hub か、手元で変換したものか)。"""


class VerificationRecord(_Frozen):
    """2 台での照合の結果の記録 (design.md 「weights」「guards」)。

    `serve verify` が `state/<slug>.verified.json` に書き、`gate_weights_verified` が、
    マニフェストの版と、ファイルの一覧と大きさと一緒に確かめる。合わないファイルは、
    `mismatched` に名前を並べる (黙って取り直さない)。
    """

    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    scope: VerificationScope
    node: NodeRole
    verified_at: datetime
    file_count: NonNegativeInt
    total_bytes: NonNegativeInt
    mismatched: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """合わないファイルが 1 つもなかったかどうか。"""
        return not self.mismatched

    @property
    def identity(self) -> str:
        return f"{self.repo}@{self.revision}"


class DerivedVerificationRecord(_Frozen):
    """手元で変換した重みの、2 台での照合の結果の記録 (issue #57 やること 2)。

    `serve verify` が `state/<名前>.derived.verified.json` に書く。Hub の記録とは
    別の道筋・別の型にして、別の種類の記録を読まないことを構造で保証する。

    `manifest_sha256` は、照合の正解にしたマニフェストの中身の SHA-256 (`content_sha256`) で
    ある。必須の項目にして、マニフェストを差し替えたあとに、前のマニフェストで作った記録が
    通らないようにする (`guards` が、いまのマニフェストの値と突き合わせる)。
    """

    kind: Literal["derived"]
    derivation: Derivation
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: VerificationScope
    node: NodeRole
    verified_at: datetime
    file_count: NonNegativeInt
    total_bytes: NonNegativeInt
    mismatched: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """合わないファイルが 1 つもなかったかどうか。"""
        return not self.mismatched

    @property
    def identity(self) -> str:
        return self.derivation.identity


AnyVerificationRecord = VerificationRecord | DerivedVerificationRecord
"""照合の結果の記録 (Hub か、手元で変換したものか)。"""


# --- 運転 ---------------------------------------------------------------


class GpuApp(_Frozen):
    """GPU を使っているプロセス (design.md 「guards」の `gate_gpu_idle`)。

    よそのものについて知るのは、`nvidia-smi` が返すこの 3 つの列だけである
    (requirements 2.2、2.4)。GB10 のユニファイドメモリのために、メモリの量が読めない
    (`[N/A]`) ことがあるので、`used_memory_mib` は空を許す。
    """

    pid: int | None = None
    process_name: str
    used_memory_mib: int | None = None


class NodeStatus(_Frozen):
    """1 台の状態 (design.md 「lifecycle」)。

    状態の出どころは、ラベルの付いた自分のコンテナだけである。`config_name`、`kind`、
    `image_digest`、`config_sha256`、`started_at` は、そのコンテナのラベルから読む
    (`config_sha256` と `kind` は、`already_running` の判定と、動いている取得の待ちに要る)。
    """

    node: NodeRole
    container_state: ContainerState
    exit_code: int | None = None
    config_name: str | None = None
    kind: ConfigKind | None = None
    image_digest: str | None = None
    config_sha256: str | None = None
    started_at: datetime | None = None
    gpu_apps: tuple[GpuApp, ...] = ()
    fabric_link_up: bool | None = None


class ServiceStatus(_Frozen):
    """2 台にまたがる 1 つの推論サーバーの状態 (design.md 「lifecycle」)。

    `max_model_len` は、起動の記録ではなく `/v1/models` から読む (requirements 6.3)。
    """

    nodes: tuple[NodeStatus, ...] = Field(min_length=1)
    health_ok: bool | None = None
    served_model: str | None = None
    max_model_len: int | None = None
    running_requests: int | None = None
    waiting_requests: int | None = None


class LaunchRecord(_Frozen):
    """起動の記録 (requirements 3.10、design.md 「lifecycle」の State Management)。

    起動の直前に `state/<構成>.launch.json` に書き、`serve logs` が記録と一緒に回収する。
    記録であって状態ではないので、判定には使わない。
    """

    config_name: str = Field(min_length=1)
    image_digest: str = Field(min_length=1)
    weights: str | None = None
    started_at: datetime
    plans: tuple[ContainerPlan, ...] = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo_commit: str = Field(min_length=1)
    repo_dirty: bool
    gates: tuple[GateResult, ...]
    """起動の直前に通った関門の結果 (issue #84)。`memory_free` の `detail` に、起動前の
    `MemFree`/`MemAvailable`/`Cached` が残るので、instanttensor の起動が落ちた回の、直前の
    空きを後から読める。"""


class StartOutcome(_Frozen):
    """起動の結末 (design.md 「lifecycle」の `start`、System Flows の起動)。

    `refused` は、関門の不通過、了承されなかったこと、名前が同じで中身が違う構成への
    起動 (終了コード 1)。`failed` は、時間切れとコンテナの終了 (終了コード 2) で、
    `log_tails` に 2 台の記録の末尾を持つ。
    """

    status: StartStatus
    detail: str = ""
    gates: tuple[GateResult, ...] = ()
    service: ServiceStatus | None = None
    observation: LaunchObservation | None = None
    log_tails: dict[NodeRole, str] = Field(default_factory=dict)


class StopOutcome(_Frozen):
    """停止の結末 (design.md 「lifecycle」の `stop`)。

    `gpu_not_released` のときは、残っているプロセスの名前を示す (止めには行かない)。
    """

    status: StopStatus
    detail: str = ""
    remaining_gpu_apps: tuple[GpuApp, ...] = ()


class SmokeReply(_Frozen):
    """短い要求 1 つの結果 (requirements 6.5、10.5)。

    応答の本文は画面に出すだけで、この型にも、どのファイルにも残さない。残すのは、
    HTTP の状態、終わりの理由、トークンの数、置き換え文字 (U+FFFD) の有無だけである。
    """

    lang: Lang
    http_status: int | None = None
    stop_reason: str | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    replacement_char: bool = False


class SmokeOutcome(_Frozen):
    """短い要求の確かめの結末 (design.md 「lifecycle」の `smoke`)。

    `serve smoke` は英語と日本語を 1 つずつ、`probe` は 1 つだけ送る。
    """

    replies: tuple[SmokeReply, ...] = Field(min_length=1)
    detail: str = ""


class ProbeOutcome(_Frozen):
    """1 台の縮小の確認の結末 (requirements 5.1〜5.5、design.md 「probe」)。

    `failed` のときは、`observation.known_failure` に失敗の種類が入る。`inconclusive` は、
    縮めた形を推論サーバーが受け付けないなど、モデルの懸念とは別の理由である
    (requirements 5.5)。どの結末でも、記録を回収してから、必ず止めて消す。
    """

    status: ProbeStatus
    config_name: str = Field(min_length=1)
    observation: LaunchObservation | None = None
    reply: SmokeReply | None = None
    detail: str = ""


# --- 通信の確認 ---------------------------------------------------------


class InterfaceLink(_Frozen):
    """1 つのインターフェースの読み取り (requirements 4.1、design.md 「netcheck」)。

    `addrs` は `ip -br addr` が示す形 (`192.168.100.1/24`)。`roce_device` は、入っていれば
    `ibdev2netdev` から読む。
    """

    name: str = Field(min_length=1)
    state: str = Field(min_length=1)
    mtu: int | None = None
    speed_mbps: int | None = None
    addrs: tuple[str, ...] = ()
    roce_device: str | None = None


class LinkReport(_Frozen):
    """1 台ぶんのリンクの報告 (requirements 4.1、4.2)。

    `cable_count` は、つながっているインターフェースの数から判断したケーブルの本数。
    入っていない道具は、入れずに `tools_missing` に記録する。
    """

    node: NodeRole
    interfaces: tuple[InterfaceLink, ...] = ()
    cable_count: int | None = None
    tools_missing: tuple[str, ...] = ()
    detail: str = ""


class BandwidthSample(_Frozen):
    """メッセージの大きさ 1 つぶんの帯域 (design.md 「netcheck」)。

    `algbw = S / t`、`busbw = algbw × 2(n−1)/n` (NCCL の公式の Performance の文書の定義。
    n=2 では係数は 1)。単位は 10 進の Gbps。
    """

    size_bytes: PositiveInt
    algbw_gbps: NonNegativeFloat
    busbw_gbps: NonNegativeFloat


class BandwidthRun(_Frozen):
    """帯域の計測の 1 回 (design.md 「netcheck」の A/B)。

    A/B の各回は、どの腕 (`arm`) の何回目 (`repeat_index`) かをラベル `run` と、回収した
    記録の置き場所 (`log_dir`) で区別する。
    """

    arm: str = Field(min_length=1)
    repeat_index: PositiveInt = 1
    samples: tuple[BandwidthSample, ...] = ()
    nccl: NcclObservation | None = None
    log_dir: Path | None = None


class AbOutcome(_Frozen):
    """A/B の結果 (requirements 4.5、design.md 「netcheck」)。

    採用できるのは、足した側の最小が、最小の設定の側の最大を上回ったときだけである
    (大きなメッセージの `busbw` で比べる)。
    """

    added_env: dict[str, str] = Field(default_factory=dict)
    compared_size_bytes: PositiveInt
    baseline_runs: tuple[BandwidthRun, ...] = Field(min_length=1)
    candidate_runs: tuple[BandwidthRun, ...] = Field(min_length=1)
    adopt: bool
    detail: str = ""


# --- 見張り -------------------------------------------------------------


class WatchSample(_Frozen):
    """見張りの 1 回の観察 (requirements 7.7、design.md 「watch」)。

    `gpu_utilization_pct` が読めない (GB10 で `utilization.gpu` が出ない) ときは、値を空に
    して、判定を生成のトークンの数だけで行う。

    `thermal_zones_c` は熱区域の番号 (`int`) から℃への対応、`hwmon_temps_c` は
    `"hwmon<N>/temp<M>"` の識別子から℃への対応 (どちらも、見張りの開始時に発見した一覧の
    ぶんだけ持つ)。`cpu_utilization_pct` はコア番号からコアごとの使用率 (%) への対応で、
    初回や直前の観察が読めなかった回は空になる。
    """

    taken_at_utc: datetime
    health_ok: bool
    generation_tokens_total: float | None = None
    running_requests: int | None = None
    waiting_requests: int | None = None
    gpu_utilization_pct: dict[NodeRole, int | None] = Field(default_factory=dict)
    gpu_temperature_c: dict[NodeRole, int | None] = Field(default_factory=dict)
    gpu_sm_clock_mhz: dict[NodeRole, int | None] = Field(default_factory=dict)
    gpu_power_w: dict[NodeRole, float | None] = Field(default_factory=dict)
    thermal_zones_c: dict[NodeRole, dict[int, float | None]] = Field(default_factory=dict)
    hwmon_temps_c: dict[NodeRole, dict[str, float | None]] = Field(default_factory=dict)
    cpu_utilization_pct: dict[NodeRole, dict[int, float]] = Field(default_factory=dict)
    cpu_scaling_max_freq_khz: dict[NodeRole, dict[int, int | None]] = Field(default_factory=dict)
    """見張りの開始時に発見した cpufreq のコアごとの周波数の上限 (kHz。issue #10)。行数が
    発見した数と合わない回や、値を整数に変えられない行は空になる。"""


class WatchEvent(_Frozen):
    """見張りが見つけたこと (design.md 「watch」の判定)。

    `unresponsive` は応答の確認が 3 回続けて失敗したとき、`stalled` は処理中の要求が
    あるのに生成のトークンの数が 5 分増えないときである。`thermal` は、いずれかの熱区域が
    しきい値以上になったとき (記録の回収を伴わない)。時刻と、その前後の観察を残す。
    """

    finding: WatchFinding | ThermalFinding
    at_utc: datetime
    context: tuple[WatchSample, ...] = ()
    detail: str = ""


class WatchOutcome(_Frozen):
    """見張りの要約 (requirements 7.7)。

    見張りは、何も止めず、起こし直さず、決めた時間まで続ける。台ごと・項目ごとの最大値
    (`*_max_*`) は、読めた値だけから計算し、1 度も読めなければ空にする。
    """

    config_name: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    samples_path: Path
    sample_count: NonNegativeInt
    events: tuple[WatchEvent, ...] = ()
    gpu_utilization_available: bool = True
    detail: str = ""
    thermal_threshold_c: float = 90.0
    thermal_zones_found: dict[NodeRole, tuple[int, ...]] = Field(default_factory=dict)
    hwmon_chip_names: dict[NodeRole, dict[str, str]] = Field(default_factory=dict)
    hwmon_sensors_found: dict[NodeRole, dict[str, str | None]] = Field(default_factory=dict)
    cpufreq_cores_found: dict[NodeRole, tuple[int, ...]] = Field(default_factory=dict)
    """見張りの開始時に発見した cpufreq のコア番号 (台ごと。issue #10、決めごとの 8 と同じ形)。"""
    gpu_temperature_max_c: dict[NodeRole, int | None] = Field(default_factory=dict)
    gpu_sm_clock_range_mhz: dict[NodeRole, tuple[int, int] | None] = Field(default_factory=dict)
    gpu_power_max_w: dict[NodeRole, float | None] = Field(default_factory=dict)
    thermal_zone_max_c: dict[NodeRole, dict[int, float]] = Field(default_factory=dict)
    hwmon_temp_max_c: dict[NodeRole, dict[str, float]] = Field(default_factory=dict)
    cpu_cluster_max_freq_khz: dict[NodeRole, dict[CpuCluster, int]] = Field(default_factory=dict)
    """台ごとの X925/A725 の周波数の上限の最大値 (kHz。読めた値だけの全観察を通した最大。
    1 度も読めなかった群はキーを持たない。issue #10)。"""
    thermal_over_threshold_samples: NonNegativeInt = 0
    thermal_over_threshold_span: tuple[datetime, datetime] | None = None

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at が started_at より前になっている")
        return self


# --- thinking の確かめ --------------------------------------------------


class ThinkingTrial(_Frozen):
    """thinking の確かめの 1 回 (requirements 9.2、10.5)。

    記録するのは、thinking のブロックの文字数、入力と出力のトークンの数、終わりの理由
    だけで、本文は保存しない。
    """

    variant: ThinkingVariant
    trial_index: PositiveInt
    http_status: int | None = None
    thinking_chars: NonNegativeInt | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    stop_reason: str | None = None


class ThinkingOutcome(_Frozen):
    """thinking の深さの確かめの結果 (design.md 「thinking」)。

    同じ入力、温度 0 で、5 通りを 3 回ずつ送る。`effective` は、効いたと言える条件
    (範囲が重ならない) に照らした判定で、判定できなければ空にする。
    """

    model: str = Field(min_length=1)
    trials: tuple[ThinkingTrial, ...] = Field(min_length=1)
    effective: bool | None = None
    detail: str = ""
