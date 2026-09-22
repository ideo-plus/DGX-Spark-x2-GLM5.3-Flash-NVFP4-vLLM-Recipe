"""1 台の縮小の確認 (`serve probe <構成>`。design.md 「確認 › probe」、tasks.md 4.1)。

重みを取得する前に、1 台の Spark だけで、モデルのアテンションの形を保ったまま層の数を減らし、
中身のない重み (`--load-format dummy`) で、固定したイメージの推論サーバーの起動を試みる
(requirements 5.1)。いちばんの懸念 (このモデルのアテンションに対応する計算の部品が、この GPU
で選べるか) に、いちばん安い段階で当たるための道具である。

## 結果は 3 つ (design.md 「probe」、requirements 5.2、5.3、5.5)

| 結末 | いつ | 何が入るか |
|---|---|---|
| `ready` | 起動した (受け付けの開始まで届いた) | 記録から読んだ観察と、短い要求 1 つの結果 |
| `failed` | 失敗した (コンテナが終了した) | 観察と `known_failure` |
| `inconclusive` | 判定できない (確認そのものができなかった) | 何が起きたかの文と、読めた観察 |

`failed` の `known_failure` は、知っている失敗のどれでもなければ `unclassified` になる
(`lifecycle.observation` が、コンテナの終了と合わせて付ける)。この値が、段 0 と段 1 で同じか
どうかが、打ち切りの判定の材料になる (requirements 8.7)。

**`inconclusive` に入れるもの** (requirements 5.5 の「縮小の確認そのものが、このモデルの懸念と
は別の理由でできない場合」):

- 関門が断った (置き場所がない、GPU が空いていない、照合の記録がない、など)
- 終了せずに、待ちの上限に達した
- 待っても直らない食い違い (名乗るモデルの名前が構成と違う、投機的デコードの指標が出ている)。
  どちらも構成を直さなければ変わらないので、モデルの懸念とは別の理由である
- 了承のあとに、起こす呼び出しや読み取りが届かなかった (ssh の切断、`docker run` の失敗)

## 結末と例外の、終了コードへの写し方 (5.1 が、この表のとおりに写す)

| 結末 / 例外 | 終了コード | 理由 |
|---|---|---|
| `ProbeOutcome.status` が `ready` | 0 | 起動した (段 2 に進める) |
| `ProbeOutcome.status` が `failed` | 2 | **実行して失敗した**。縮小の確認としては判定が出ている |
| `ProbeOutcome.status` が `inconclusive` | 1 | **前提の不足、断った**。判定が出ていない |
| `config.ConfigError` (`plan.PlanError` を含む)、`ValueError` | 1 | 構成の誤り (触る前に断る) |
| `guards.ApprovalError` | 1 | 計測者が了承しなかった |
| `ProbeError` | 2 | 片付けが終わらなかった (コンテナが残っているかもしれない) |
| `KeyboardInterrupt` | 130 | 中断 (片付けてから伝える) |

**`failed` を 2、`inconclusive` を 1 にした理由** (design.md 「Error Handling」の表に合わせた。
手順書 (tasks.md 6.3) が、終了コードで段を進める判断をするので、ここは大事である):

- `failed` は、推論サーバーを**実際に起こして**失敗した結果である。design の表の 2 (実行して
  失敗した。「時間切れ、コンテナの終了」) がそのまま当たる。手順書は、2 を見たら段 1 (より
  新しいイメージでの同じ確認) に進み、2 つの `known_failure` が同じなら打ち切りの判定の材料に
  する (requirements 8.7)
- `inconclusive` は、縮小の確認という手続きの前提が満たせなかったことで、design の表の 1
  (前提の不足、断った。「関門の不通過、了承されなかった」) に当たる。手順書は、1 を見たら、
  足りない前提 (配布、取得、照合、了承) を直してやり直すか、そのことを記録して段 2 (2 台での
  起動) に進む (requirements 5.5)
- **片付けが終わらなかったときは、終了コードを 2 に上げる** (design.md 「Error Handling」の
  「片付けそのものが失敗したら、残ったコンテナの名前を示して終了コード 2」)。`ready` と
  `inconclusive` は、結果を返さずに `ProbeError` にする。`failed` は、もともと 2 なので、
  失敗の種類を返すほうが計測者に役立つので、片付けの問題を `detail` に書いて結果を返す

## 守る決まり

- **`kind = "probe"` の構成を、1 つのノードで起こす**。2 台以上の構成と、`kind` が `probe` で
  ない構成は、**Spark に触る前に**断る (design.md 「types / config」: `probe` のノードは 1 つ)
- **1 つの起動の仕組み** (design.md 「Architecture Integration」): `serve start` と同じ道を
  通る。この module は `lifecycle` の**公開の口だけ**で組み立て、下線つきの名前を 1 つも
  触らない (`lifecycle.py` は、この作業では書き換えない)。順は「一覧と関門 → 計画と了承 →
  起動の記録の配布 → `docker run` → 待ち → 短い要求 → 片付け」である
- **どの結果でも、記録を回収してから、必ず止めて消す** (requirements 5.4、design.md 「probe」の
  「確認のコンテナを動かしたままにしない」)。成功でも `lifecycle.wrap_up` を呼ぶ。片付けの
  決まり (中断が、どこで、何度来ても、できるところまで進めて、最後に中断を伝える。止めてよい
  相手は、流す直前に自分の一覧で確かめる) は、`lifecycle.wrap_up` / `clean_up` に任せる
  (tasks.md の Implementation Notes 3.4。`image._roll_back` の作りは写さない)
- **短い要求は 1 つだけ**送り、返り切ったかだけを記録する (requirements 5.4)。重みは
  `--load-format dummy` なので、応答の中身は意味を持たない。見るのは、HTTP の状態と、
  終わりの理由だけである
- **応答の本文は、どこにも出さない** (requirements 10.5)。`lifecycle.send_smoke` が返す
  `SmokeShown.text` を、結果の型にも、誤りの文にも、`report` (画面) にも入れない
  (`serve smoke` は画面に出すが、こちらは中身のない重みなので、読む意味がない)
- 待ちの上限は、構成の `ready_timeout_s` で、`timeout_s` でその回だけ上書きできる
  (design.md 「cli」の `--timeout`)
- リポジトリの commit と、未コミットの変更の有無は**引数で受ける** (この module は `git` を
  呼ばない)。`record_dir` は、絶対の道筋だけを受ける (配る元を、配る前に空にするため)

依存の向きにより、この module が読み込む `serving_kit` は `types`、`config`、`remote`、`plan`、
`observe`、`guards`、`logs`、`lifecycle` だけである (`netcheck`、`watch`、`thinking`、`cli` は
読み込まない)。

## design の文からの、意図した決めごと

1. **短い要求が 200 を返さなくても `ready` のままにする**。受け付けの開始は、すでに判定が
   ついている (requirements 5.4 は「誤りなく応答を返し切ることだけを確かめて記録する」と
   定める)。HTTP の状態と終わりの理由は `ProbeOutcome.reply` に入るので、計測者が読める
2. **推論サーバーの版は `/version` から読む** (requirements 5.2 の「使われた推論サーバーの版」)。
   `observe` は、記録の中の版の行の原文が分からないので、つねに空を返す (tasks.md の
   Implementation Notes 2.2)。読み方は `lifecycle` と同じだが、`lifecycle.py` を書き換えられ
   ないので、この module にも同じ短い読み取りを持つ (形が変わったら、2 か所を直す)
3. **記録の回収の置き場所の名前は、`lifecycle.wrap_up` が決める**ので、`serving/var/<日時>-
   start-<構成>/` になる (`wrap_up` はコマンドの名前を受けない)。構成の名前 (`probe-*`) で
   見分けられるので、いまは受け入れる
4. **観察は、どの結末でも読む** (requirements 5.2)。`failed` と `inconclusive` では、分類に
   使う長い末尾 (`lifecycle.OBSERVE_TAIL_LINES`) から読む。起こす途中で失敗した経路だけは、
   長い末尾を読まない (片付けを遅らせるだけなので。`lifecycle` の決めごとと同じ)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO

import httpx

from serving_kit import lifecycle
from serving_kit.config import ConfigError
from serving_kit.guards import (
    READ_TIMEOUT_S,
    Confirmer,
    build_approved_plan,
    request_approval,
    run_gates,
)
from serving_kit.logs import DEFAULT_TAIL_LINES, LOG_READ_TIMEOUT_S
from serving_kit.plan import build_plans
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    GateResult,
    Lang,
    LaunchObservation,
    NodeDef,
    NodeRole,
    ProbeOutcome,
    ProbeStatus,
    WeightsManifest,
)

__all__ = ["PROBE_LANG", "ProbeError", "run_probe"]


# --- 決まった値 -----------------------------------------------------------

PROBE_LANG: Final[Lang] = "en"
"""短い要求に使う言語。

1 つだけ送る (requirements 5.4)。中身のない重みなので、日本語の文字化け (requirements 7.6) は、
ここでは確かめられない。日本語は、2 台で起動したあとの `serve smoke` で見る。
"""

_KIND_PROBE: Final[str] = "probe"
"""この module が受ける構成の種類 (design.md 「probe」)。"""

_STATUS_READY: Final[ProbeStatus] = "ready"
_STATUS_FAILED: Final[ProbeStatus] = "failed"
_STATUS_INCONCLUSIVE: Final[ProbeStatus] = "inconclusive"

_INCONCLUSIVE_NOTE: Final[str] = (
    "縮小の確認としての判定は出せなかったので、このことを記録して、次の段に進むかどうかを"
    "計測者が決める (requirements 5.5)"
)


class ProbeError(Exception):
    """縮小の確認を実行して失敗した (終了コード 2)。

    いまのところ、**片付けが終わらなかった**ときだけである (自分のコンテナを止められなかった、
    消せなかった、止めてよい相手を確かめるための一覧を読めなかった)。コンテナが残っている
    かもしれないので、誤りの文に、残りうるコンテナの名前と理由を入れる。
    """


# --- 小さな助け -----------------------------------------------------------


def _report(stream: TextIO, text: str) -> None:
    """進捗と警告を知らせる (標準出力は、後の処理が読むので混ぜない)。"""
    stream.write(f"{text}\n")
    stream.flush()


def _check_config(config: ConfigDef) -> str:
    """Spark に触る前に、構成が縮小の確認のものかを確かめ、名乗るモデルの名前を返す。

    design.md 「probe」: `kind = "probe"` の構成を、1 つのノードで起こす。
    """
    if config.kind != _KIND_PROBE:
        raise ConfigError(
            f"1 台の縮小の確認に使えるのは、kind = '{_KIND_PROBE}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if len(config.nodes) != 1:
        raise ConfigError(
            f"縮小の確認は 1 台だけで行う (構成 '{config.name}' の nodes は"
            f" {len(config.nodes)} 台: {', '.join(config.nodes)})"
        )
    if config.served_model_name is None:  # pragma: no cover - 型の検証が、先に断る
        raise ConfigError(f"構成 '{config.name}' に served_model_name がない")
    return config.served_model_name


def _check_record_dir(record_dir: Path) -> None:
    """`record_dir` が絶対の道筋であることを、Spark に触る前に確かめる。

    起動の記録を配る元は、配る前に空にする (消す操作)。相対の道筋を受けると、プロセスの作業
    ディレクトリに対して効いてしまう。`lifecycle.record_pushes` も同じ検査をするが、そちらは
    関門のあとなので、ここで先に断る。
    """
    if not record_dir.is_absolute():
        raise ValueError(
            f"record_dir は、絶対の道筋で渡す (配る元を、配る前に空にするため): '{record_dir}'"
        )


def _node_of(nodes: Mapping[NodeRole, NodeDef], config: ConfigDef, role: NodeRole) -> NodeDef:
    """構成が使う役割のノードの定義を取る (なければ、触る前に断る)。"""
    node = nodes.get(role)
    if node is None:
        raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
    return node


def _refusal_detail(refused: Sequence[GateResult]) -> str:
    """断った関門を、台と関門の名前つきで並べる。"""
    reasons = " / ".join(f"{gate.node or '-'} の {gate.gate}: {gate.detail}" for gate in refused)
    return f"関門が断ったので、縮小の確認を始めなかった: {reasons}。{_INCONCLUSIVE_NOTE}"


def _shown_count(value: int | None) -> str:
    return "不明" if value is None else str(value)


def _smoke_facts(shown: lifecycle.SmokeShown) -> str:
    """短い要求 1 つの、**保存してよい事実だけ**を並べる (**本文を入れない**)。

    `lifecycle` にも同じ形の文を作る助けがあるが、下線つきの名前なので使えない。入れるのは、
    `SmokeShown.reply` (保存してよい部分) と `problem` (理由。本文を含まない) だけである。
    """
    reply = shown.reply
    parts = ["届かなかった" if reply.http_status is None else f"HTTP {reply.http_status}"]
    if reply.stop_reason is not None:
        parts.append(f"終わりの理由 {reply.stop_reason}")
    parts.append(
        f"入力 {_shown_count(reply.input_tokens)} / 出力 {_shown_count(reply.output_tokens)}"
        " トークン"
    )
    if reply.replacement_char:
        parts.append("置き換え文字 (U+FFFD) を含む")
    if shown.problem:
        parts.append(shown.problem)
    return (
        "短い要求を 1 つ送った (" + "、".join(parts) + ")。応答の本文は、中身のない重みでは"
        "意味を持たないので、どこにも残さない"
    )


def _shown_log_dir(done: lifecycle.WrapUp) -> str:
    """記録の置き場所を、見せる文にする。"""
    if done.log_dir is None:
        return "記録は回収していない。"
    return f"記録は {done.log_dir} に回収した。"


def _show_tails(stream: TextIO, tails: Mapping[NodeRole, str]) -> None:
    """記録の末尾を見せる (`ProbeOutcome` は、末尾を持たないため)。"""
    for role, text in tails.items():
        _report(stream, f"--- {role} の記録の末尾 ---\n{text}")


def _read_version(client: httpx.Client, base_url: str) -> str | None:
    """`/version` から、推論サーバーの版を読む (読めなければ空。決めごとの 2)。

    **落ちない**。つながらない、200 でない、JSON として読めない、のどれでも空を返す。
    """
    try:
        response = client.get(f"{base_url}/version")
    except (httpx.HTTPError, httpx.InvalidURL):
        return None
    if response.status_code != httpx.codes.OK:
        return None
    try:
        payload: object = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    return version if isinstance(version, str) and version else None


def _send_one(client: httpx.Client, base_url: str, *, model: str) -> lifecycle.SmokeShown:
    """短い要求を**ちょうど 1 つ**送る (requirements 5.4)。

    `lifecycle.send_smoke` は、1 つの要求を送って落ちずに結果を返す公開の口である
    (`lifecycle.smoke` は英語と日本語の 2 つを送るので、こちらは使わない)。
    """
    return lifecycle.send_smoke(
        client,
        base_url,
        model=model,
        lang=PROBE_LANG,
        prompt=lifecycle.SMOKE_PROMPTS[PROBE_LANG],
        max_tokens=lifecycle.SMOKE_MAX_TOKENS,
    )


def _clean(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: lifecycle.Controls,
    *,
    var_root: Path,
    started_at: datetime,
    classify: bool,
    show_tails: bool,
) -> lifecycle.WrapUp:
    """記録を読んで回収し、コンテナを止めて消す (requirements 5.4)。

    片付けの決まりは、すべて `lifecycle.wrap_up` が守る (中断が、どこで、何度来ても、できる
    ところまで進めて、受けた中断を `WrapUp.interrupted` で返す)。**投げるのは、呼ぶ側**である。
    """
    done = lifecycle.wrap_up(
        runner,
        config,
        nodes,
        plans,
        controls,
        var_root=var_root,
        started_at=started_at,
        classify=classify,
    )
    if show_tails:
        _show_tails(controls.report, done.shown)
    if done.log_dir is not None:
        _report(controls.report, f"記録は {done.log_dir} に回収した")
    return done


def _finish(
    status: ProbeStatus,
    config: ConfigDef,
    done: lifecycle.WrapUp,
    *,
    detail: str,
    observation: LaunchObservation | None = None,
    shown: lifecycle.SmokeShown | None = None,
) -> ProbeOutcome:
    """片付けの結果を見てから、`ProbeOutcome` を作る。

    片付けが終わらなかったときは、コンテナが残っているかもしれないので、終了コードを 2 に
    上げる (`ready` と `inconclusive` は `ProbeError` にする)。`failed` は、もともと 2 なので、
    失敗の種類を返すほうが役に立つ (module の docstring の表)。

    保存するのは `SmokeShown.reply` だけで、**本文 (`SmokeShown.text`) は入れない**
    (requirements 10.5)。
    """
    head = f"{detail}。{_shown_log_dir(done)}"
    if done.problems:
        trouble = "片付けが終わらなかった: " + " / ".join(done.problems)
        if status != _STATUS_FAILED:
            raise ProbeError(
                f"{head}{trouble}。コンテナが残っているかもしれないので、`serve status` で"
                "確かめて、`serve stop` で片付ける"
            )
        body = f"{head}{trouble}"
    else:
        body = f"{head}自分のコンテナを止めて消した"
    return ProbeOutcome(
        status=status,
        config_name=config.name,
        detail=body,
        observation=observation,
        reply=None if shown is None else shown.reply,
    )


# --- 入口 -----------------------------------------------------------------


def run_probe(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    confirmer: Confirmer,
    var_root: Path,
    record_dir: Path,
    repo_commit: str,
    repo_dirty: bool,
    manifest: WeightsManifest | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float = lifecycle.READY_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    client: httpx.Client | None = None,
    smoke_client: httpx.Client | None = None,
    http_timeout_s: float = lifecycle.HTTP_TIMEOUT_S,
    smoke_timeout_s: float = lifecycle.SMOKE_TIMEOUT_S,
    start_timeout_s: float = lifecycle.START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> ProbeOutcome:
    """1 台の縮小の確認を流す (`serve probe <構成>`。requirements 5.1〜5.5)。

    進む順と、結末の分け方と、終了コードへの写し方は、module の docstring にある。**どの結末
    でも、記録を回収してから、必ず止めて消す**。

    引数:
        runner: 遠隔の実行役。
        config: `kind = "probe"` の、1 台の構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        started_at: 起こす時刻 (ラベルと起動の記録と、回収の置き場所の名前に使う。時差の
            付いた日時)。
        confirmer: 計画を見せて、了承を得る口。
        var_root: 記録の回収の置き場所の根 (`serving/var/`。`SshRunner` を作ったときと同じ値に
            すること)。
        record_dir: 起動の記録を作る、Mac の側の置き場所 (**絶対の道筋**)。
        repo_commit: リポジトリの commit (**呼ぶ側が調べて渡す**)。
        repo_dirty: 未コミットの変更があるかどうか (同上)。
        manifest: 重みのマニフェスト (`gate_weights_verified` と `gate_disk_space` が、
            `probe_files` の範囲だけを見る)。
        timeout_s: 受け付けの開始を待つ上限を、**その回だけ**上書きする。省くと、構成の
            `ready_timeout_s` を使う。
        poll_interval_s: 受け付けの開始を見に行く間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        report: 進捗と、記録の末尾を知らせる先。既定は `sys.stderr`。
        client: 受け付けの判定に使う HTTP のクライアント。省くと作り、この関数の中で閉じる。
        smoke_client: 短い要求に使う HTTP のクライアント (生成を待つので、判定より長い時間切れ
            にする)。省くと作り、この関数の中で閉じる。
        http_timeout_s: 1 回の判定の問い合わせの時間切れ。
        smoke_timeout_s: 短い要求 1 つの時間切れ。
        start_timeout_s: `docker run -d` の時間切れ。
        read_timeout_s: 関門と、一覧と、コンテナの状態の読み取りの時間切れ。
        log_timeout_s: 記録の全量の回収の時間切れ。
        tail_lines: 計測者に見せる、記録の末尾の行数。

    返り値:
        `ready` (終了コード 0)、`inconclusive` (1)、`failed` (2)。

    例外:
        config.ConfigError: `kind` が `probe` でない構成、2 台以上の構成、HTTP のポートを
            決められない構成、引数の列を組み立てられない構成 (どれも終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        ProbeError: 片付けが終わらなかった (終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。コンテナは、片付けてから伝える。
        ValueError: 構成が使う役割のノードの定義がない、`record_dir` が相対の道筋のとき
            (終了コード 1)。
    """
    served_model_name = _check_config(config)
    _check_record_dir(record_dir)
    role = config.nodes[0]
    node = _node_of(nodes, config, role)
    port = lifecycle.http_port(config)
    base_url = f"http://{node.lan_addr}:{port}"
    plans = build_plans(config, nodes, started_at)

    # 1. 一覧と関門 (読み取りだけ。自分のコンテナが残っていれば `gate_own_state` が断る)
    gates = run_gates(runner, config, nodes, plans, manifest=manifest, timeout_s=read_timeout_s)
    refused = tuple(gate for gate in gates if not gate.passed)
    if refused:
        return ProbeOutcome(
            status=_STATUS_INCONCLUSIVE,
            config_name=config.name,
            detail=_refusal_detail(refused),
        )

    # 2. 計画と了承 (ここより前に、状態を変える呼び出しは 1 つも出ていない)
    approved = build_approved_plan(plans, extra_forward=lifecycle.record_pushes(config, record_dir))
    request_approval(confirmer, runner, approved, nodes)

    owns_client = client is None
    owns_smoke_client = smoke_client is None
    controls = lifecycle.Controls(
        client=lifecycle.new_client(http_timeout_s) if client is None else client,
        timeout_s=float(config.ready_timeout_s) if timeout_s is None else timeout_s,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        tail_lines=tail_lines,
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
        report=sys.stderr if report is None else report,
    )
    sender = lifecycle.new_client(smoke_timeout_s) if smoke_client is None else smoke_client
    try:
        return _probe(
            runner,
            config,
            nodes,
            plans,
            controls,
            sender,
            base_url=base_url,
            served_model_name=served_model_name,
            var_root=var_root,
            record_dir=record_dir,
            started_at=started_at,
            repo_commit=repo_commit,
            repo_dirty=repo_dirty,
        )
    finally:
        if owns_client:
            controls.client.close()
        if owns_smoke_client:
            sender.close()


def _probe(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: lifecycle.Controls,
    sender: httpx.Client,
    *,
    base_url: str,
    served_model_name: str,
    var_root: Path,
    record_dir: Path,
    started_at: datetime,
    repo_commit: str,
    repo_dirty: bool,
) -> ProbeOutcome:
    """了承のあとの「置く → 起こす → 待つ → (短い要求) → 片付ける」。

    `serve start` (tasks.md 3.4) と同じ片付けの道を通るが、**成功のときも片付ける**ところが
    違う (design.md 「probe」: 確認のコンテナを動かしたままにしない)。

    起動の記録を置くのは `try` の**外**である (まだコンテナが 1 つもないので、片付けるものが
    ない)。置けなかったことは、縮小の確認そのものができなかったこと (`inconclusive`) にする。
    """
    record = lifecycle.launch_record(
        config, plans, started_at, repo_commit=repo_commit, repo_dirty=repo_dirty
    )
    try:
        lifecycle.push_launch_records(runner, config, nodes, record, record_dir)
    except lifecycle.LifecycleError as exc:
        return ProbeOutcome(
            status=_STATUS_INCONCLUSIVE,
            config_name=config.name,
            detail=f"起動の記録を置けなかったので、縮小の確認を始めなかった: {exc}。"
            f"{_INCONCLUSIVE_NOTE}",
        )

    shown: lifecycle.SmokeShown | None = None
    version: str | None = None
    try:
        lifecycle.start_all(runner, config, nodes, plans, controls)
        targets = lifecycle.started_targets(runner, config, nodes, plans, controls)
        waited = lifecycle.wait_ready(
            runner,
            targets,
            controls,
            base_url=base_url,
            served_model_name=served_model_name,
        )
        if waited.failure is None:
            # 止める前に読む (片付けのあとでは、どちらも届かない)
            shown = _send_one(sender, base_url, model=served_model_name)
            version = _read_version(controls.client, base_url)
    except (RemoteError, lifecycle.RunFailed) as exc:
        # 分類の結果を使わない経路なので、長い末尾は読まない (決めごとの 4)
        done = _clean(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            classify=False,
            show_tails=True,
        )
        if done.interrupted is not None:
            # 片付けの最中の中断は、実行しての失敗より優先して伝える (130)
            raise done.interrupted from exc
        reason = (
            str(exc)
            if isinstance(exc, lifecycle.RunFailed)
            else f"了承のあとに、読み取りが届かなかった: {exc}"
        )
        return _finish(
            _STATUS_INCONCLUSIVE,
            config,
            done,
            detail=f"{reason}。{_INCONCLUSIVE_NOTE}",
        )
    except BaseException as exc:
        # 中断 (Ctrl-C) と、思わぬ誤りのときも、確認のコンテナを残さない
        done = _clean(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            classify=False,
            show_tails=True,
        )
        if done.interrupted is not None and done.interrupted is not exc:
            # 待ちの中断 (exc) のあとに、片付けの最中にもう一度来た中断
            raise done.interrupted from exc
        raise

    done = _clean(
        runner,
        config,
        nodes,
        plans,
        controls,
        var_root=var_root,
        started_at=started_at,
        classify=True,
        show_tails=waited.failure is not None,
    )
    if done.interrupted is not None:
        raise done.interrupted
    if waited.failure is None:
        return _ready_outcome(config, done, waited, shown, version)
    if waited.exited:
        return _finish(
            _STATUS_FAILED,
            config,
            done,
            detail=waited.failure,
            observation=lifecycle.observation(config, done.classified, waited.exited),
        )
    return _finish(
        _STATUS_INCONCLUSIVE,
        config,
        done,
        detail=f"{waited.failure}。{_INCONCLUSIVE_NOTE}",
        observation=lifecycle.observation(config, done.classified, ()),
    )


def _ready_outcome(
    config: ConfigDef,
    done: lifecycle.WrapUp,
    waited: lifecycle.Waited,
    shown: lifecycle.SmokeShown | None,
    version: str | None,
) -> ProbeOutcome:
    """起動したときの結果 (requirements 5.2、5.4)。

    観察は、分類に使う長い末尾から読み、版だけは `/version` から埋める (決めごとの 2)。
    """
    found = lifecycle.observation(config, done.classified, ())
    if found is not None and version is not None:
        found = found.model_copy(update={"vllm_version": version})
    facts = "" if shown is None else f"。{_smoke_facts(shown)}"
    return _finish(
        _STATUS_READY,
        config,
        done,
        detail=(
            f"構成 '{config.name}' の縮小の確認は、1 台で要求を受け付けられる状態になった"
            f" ({waited.readiness.detail}){facts}"
        ),
        observation=found,
        shown=shown,
    )
