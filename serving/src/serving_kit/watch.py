"""連続の負荷の間、外から見張る (`serve watch <構成>`。design.md 「確認 › watch」、tasks.md 4.5)。

すでに動いている推論サーバーを、決めた間隔で外から観察し続け、応答しなくなる・固まる (2 台の
GPU が使われ続けたまま生成のトークンの数が進まなくなる) ことが起きたら、時刻とその前後の
観察を残す (requirements 7.7)。**読み取りだけ**である: 何も止めない、起こし直さない (関門も
了承も要らない)。状態を変える呼び出し (`mutating=True`、`docker run` / `stop` / `rm`、`push`)
は、この module から 1 つも出ない。

## 1 回の観察 (design.md 「watch」)

head の `/health` (時間切れ `health_timeout_s`。既定 10 秒)、`/metrics` の
`vllm:generation_tokens_total` / `vllm:num_requests_running` / `vllm:num_requests_waiting`、
2 台の `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` (`GPU_UTIL_ARGV`。読める
列は、これだけである。requirements 2.2、2.4) を読み、1 つの `types.WatchSample` にして、1 行の
JSON で `samples_path` (`serving/var/<UTC>-watch-<構成>/samples.jsonl`。置き場所は
`logs.var_dir` の決まりで作る) に**観察のたびに** (1 行ごとに開いて書き出す。design の
「途中で落ちても、そこまでの観察が残るように」を守る) 足す。

読み取りが 1 つ届かなくても (`RemoteError`、つながらない、200 でない)、その項目だけを空にして
続ける (requirements 2.3、2.4 の「よそのものを読まない」に反しない範囲の、ただ 1 つの問い合わせ
しか出さない)。

## 判定 (design.md 「watch」、tasks.md 4.5)

- **`unresponsive`**: `/health` の確認が、**続けて** `unresponsive_threshold` 回 (既定 3)
  失敗した。**成功が 1 回でも挟まれば、続けての数を 0 から数え直す** (合計の失敗の数では
  判定しない)
- **`stalled`**: 生成のトークンの数 (`vllm:generation_tokens_total`) が、最後に増えた時刻から
  `stall_window_s` 秒 (既定 5 分) のあいだ増えず、**その窓のあいだ、`/metrics` を読めた観察の
  すべてで**、処理中の要求 (`num_requests_running`) が 1 つ以上あり、**かつ**、**その窓のあいだ
  の、GPU の使用率を読めた観察のすべてで**、2 台とも `stall_gpu_threshold_pct` (既定 90) 以上
  だったとき。**窓の途中で、1 度でも、処理中の要求が 0 になった観察や、2 台のどちらかが
  しきい値を割った観察があれば、窓の最後の値が条件を満たしていても「固まった」にしない**
  (要求がなくなったのは、負荷がかかっていないだけで正常。GPU が休んでいるのも、固まりでは
  なく、負荷がかかっていないか、別の原因である)。窓の中で、処理中の要求を**読めた**観察が
  1 つもなければ (`/metrics` がずっと読めない)、固まったとは言えない (確かめられていない)
  ので「固まった」にしない。GPU の使用率を**読めた**観察が 1 つもなければ、生成のトークンの
  数と処理中の要求だけで判定し、その旨を出来事の `detail` に書く。読めた観察と読めなかった
  観察が混ざるときは、読めた観察だけで判定し、読めなかった観察があったことを `detail` に書く
  (決めごとの 4、4a)
- **生成のトークンの数が減った** (サーバーが起こし直されたとみられる) ときは、固まりとは扱わず、
  窓を (GPU の履歴も、処理中の要求の履歴も込みで) 数え直す。起こし直したこと自体は
  `WatchOutcome.detail` に書く (requirements 10.5 により、送った内容や応答の本文は、どの記録
  にも書かない)
- `/metrics` が一時的に読めない (トークンの数、処理中の要求の数が空になる) 回を挟んでも、窓は
  測り直さない (読めなかった回の前後で、値が同じであることを前提にする。決めごとの 4)。その
  回の GPU の使用率は、読めれば、そのまま窓に積む (`/metrics` と `nvidia-smi` は独立に読むため)
- **どちらの判定も、条件が続いているあいだは 1 件のまま** (立ち上がりの回だけ `WatchEvent` を
  1 つ作る)。いったん条件が外れて、また起きたら、別の 1 件になる
- `unresponsive_threshold`、`stall_window_s`、`stall_gpu_threshold_pct`、`interval_s` は、
  引数で上書きできる (試験が、間隔を縮めるために使う)

出来事が起きた回は、時刻と、その前後の観察 (`_CONTEXT_SAMPLE_COUNT` 回ぶん) を `WatchEvent` に
残し、`logs.collect_logs` で 2 台の記録を回収する (対象は、`guards.list_own_containers` の一覧
から。回収は、出来事が起きたその時の時刻を渡して、そのたびに別の置き場所に書く)。**回収が
失敗しても、見張りは続く** (`collect_logs` 自身が `RemoteError` を内側で吸収するので、ここで
捕まえるのは `ValueError` / `OSError` だけである)。

見張りは、`duration_s` (既定 2 時間) に達するまで続く。**中断 (Ctrl-C) が来たら、そこまでの
観察から要約 (`WatchOutcome`) を作って `result.json` に書いてから、中断を伝える**。1 回の観察が
間隔より長くかかっても (ssh の時間切れなど)、次の観察までの待ちを、経過ぶんだけ短くするだけで
(負にはしない)、詰まらない (design.md の決めごとにはないが、この module の決めごと)。

## 依存の向き

`types`、`config`、`remote`、`plan`、`logs`、`lifecycle` を読み込む (`observe`、`guards` は、
`logs`・`lifecycle` の内側で使われるだけで、この module からは直に読み込まない)。`probe`、
`netcheck`、`thinking`、`cli` は読み込まない (design.md の依存の向き)。

## design の文からの、意図した決めごと

1. **`kind = "serve"` の構成だけを受ける**。見張るのは、動いている推論サーバーの HTTP と
   `nvidia-smi` だけであり、`probe` / `job` / `fetch` / `inspect` には待ち受ける口がない
2. **`/metrics` の読み取りは、この module 自身が行う** (`lifecycle._metric_value` は下線つきの
   名前で使えないため。tasks.md 4.5 の指示のとおり)。生成のトークンの数は counter なので、
   `lifecycle` のように `int` へ丸めず `float` のまま持つ (`types.WatchSample` の形に合わせる)
3. **GPU の使用率の「読める」は、台ごとに、一度でも読めたかどうかで判定する** (`gpu_seen`)。
   片方の台に 1 度だけ入れなかった (`RemoteError`) ときに、それだけで「読めない」と早合点して
   トークンの数だけの判定に落ちないようにするため。**GB10 がそもそも対応していない**
   (`utilization.gpu` の列が `int` に変換できない文字列を返す) ときは、一度も読めないまま
   終わるので、`gpu_utilization_available = False` になる
4. **「その間、GPU が使われ続けていた」は、窓 (トークンの数が最後に変わってからの、すべての
   観察) の全体を見る** (design.md の「その間」を、1 回の値だけでなく、窓の始まりからの
   すべての観察で確かめる。親の判断、2026-09-22)。窓は `window_gpu_status`
   (`_gpu_status` が 1 回ごとに追加する `bool | None`) に積み、判定は、読めた
   (`bool` の) 要素だけを見る。窓は、トークンの数が増えた・減ったときに、GPU の履歴も含めて
   測り直す (`window_gpu_status = []`)。`/metrics` が読めない回 (トークンの数が空) は、窓を
   測り直さない (前後の値が同じであることを前提にする) が、GPU の使用率は、その回に読めれば、
   そのまま窓に積む (`nvidia-smi` は `/metrics` と独立に読むため)。窓の中に、GPU の使用率を
   読めた回が 1 つもなければ、生成のトークンの数と処理中の要求だけで判定する
4a. **「処理中の要求があった」も、GPU と同じ書き方で、窓の全体を見る** (親の判断、
   2026-09-22)。窓は `window_requests_status` (`_requests_status` が 1 回ごとに追加する
   `bool | None`) に積み、GPU と同じ場所 (`since_change_mono is not None` のとき) で積み、
   同じ場所 (トークンが増えた・減ったとき) で測り直す。**GPU と違う点**: 処理中の要求は
   「固まった」の主たる条件なので、窓の中で 1 度も読めていなければ (`readable_requests_in_window`
   が空)、確かめられていないとして「固まった」に**しない** (`requests_criterion_ok = False`)。
   GPU は補助の条件なので、読めなければ判定から外して素通りさせる
   (`gpu_criterion_ok = True`) のと、ここが違う
5. **要約の置き場所は `result.json`** (design.md 「記録の置き場所」が、置き場所の直下に置くと
   決めている「コマンドの結果」の名前をそのまま使う。`serve watch` に固有の名前を新たに作らない)
6. **前後の観察は「起きた回を含めて、直近 `_CONTEXT_SAMPLE_COUNT` 回」**とする (design は正確な
   数を定めていない)
7. **数の引数は、すべて、Spark にも推論サーバーにも触る前に確かめる** (`_check_positive_finite`
   ほか。親の判断、2026-09-22)。`duration_s`、`interval_s`、`stall_window_s`、
   `health_timeout_s`、`gpu_timeout_s`、`log_timeout_s` は、有限の正の数 (`NaN`、`inf`、0、
   負の数を断る)。`unresponsive_threshold` は、bool を除く本物の int で、1 以上 (0 以下では、
   1 回も失敗しないうちに「応答しない」を誤って出す。`NaN`、`inf`、小数、`bool` は、`int` との
   比較を素通りしてしまうので、先に型そのものを確かめる)。`stall_gpu_threshold_pct` は
   0〜100 の範囲 (使用率なので)。
   `interval_s` は `duration_s` 以下にする (でなければ、決めた時間のあいだに観察が間延びする)。
   0 以下や無限大の `interval_s` を断らないと、眠らずに回り続け、推論サーバーと 2 台の
   Spark への問い合わせを叩き続ける (レビューの指摘 1)。0 以下の `stall_window_s` や
   `unresponsive_threshold` を断らないと、健全な状態でもすぐに「固まった」「応答しない」を
   誤って出す (レビューの指摘 2)。**`gpu_timeout_s` と `interval_s` の大小関係は、断らない**
   (`DEFAULT_GPU_TIMEOUT_S` を既定の間隔より短くしたのは既定値どうしの選び方の理由であって、
   呼ぶ側が短い `interval_s` を選ぶこと自体は誤りではない。試験が、間隔を縮めて確かめるのに
   使うため、ここを断ると、値を決めた組み合わせでしか呼べなくなる)

## 終了コードへの写し方 (5.1 が、この表のとおりに写す。design.md 「Error Handling」に合わせる)

| `WatchOutcome` / 例外 | 終了コード | 意味 |
|---|---|---|
| `events` が空のまま `duration_s` に達した | 0 | 正常。応答しなくも固まりもしなかった |
| `events` に 1 件以上ある | 2 | **実行しての失敗** (`bool(outcome.events)` で判定できる) |
| `config.ConfigError`、`ValueError` | 1 | 前提の不足。触る前に断る |
| `KeyboardInterrupt` | 130 | 中断。要約は `result.json` に書いてから伝える |

**`events` が 1 件以上あると 2 にする理由**: 見張りの道具そのものは正しく動いたが、対象の
推論サーバーが、応答しなくなる・固まるのどちらかを、決めた時間のあいだに少なくとも 1 度
起こしたという「実行しての失敗」を、終了コードだけでも見分けられるようにするため。前提の
不足 (`kind` が `serve` でない、`--port` を 1 つに決められない、数の引数が退化している
(決めごとの 7)、構成が使う役割のノードの定義が `nodes` にない) は、どれも Spark にも
推論サーバーにも触る前に断る。
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

import httpx

from serving_kit import lifecycle
from serving_kit.config import ConfigError
from serving_kit.logs import LOG_READ_TIMEOUT_S, collect_logs, var_dir
from serving_kit.plan import build_plans
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    NodeDef,
    NodeRole,
    WatchEvent,
    WatchFinding,
    WatchOutcome,
    WatchSample,
)

__all__ = [
    "COMMAND_NAME",
    "DEFAULT_DURATION_S",
    "DEFAULT_GPU_TIMEOUT_S",
    "DEFAULT_HEALTH_TIMEOUT_S",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_STALL_GPU_THRESHOLD_PCT",
    "DEFAULT_STALL_WINDOW_S",
    "DEFAULT_UNRESPONSIVE_THRESHOLD",
    "GPU_UTIL_ARGV",
    "RESULT_FILE_NAME",
    "SAMPLES_FILE_NAME",
    "watch",
]


# --- 決まった値 -------------------------------------------------------------

COMMAND_NAME: Final[str] = "watch"
"""記録の置き場所の名前に使う、`serve` のサブコマンドの名前 (`logs.var_dir`)。"""

SAMPLES_FILE_NAME: Final[str] = "samples.jsonl"
"""1 回ずつの観察を積む先の名前 (design.md 「watch」)。"""

RESULT_FILE_NAME: Final[str] = "result.json"
"""見張りの要約を書く先の名前 (design.md 「記録の置き場所」の「コマンドの結果」。決めごとの 5)。"""

DEFAULT_INTERVAL_S: Final[float] = 30.0
"""既定の観察の間隔 (design.md 「watch」: `serve watch --interval 30s`)。"""

DEFAULT_DURATION_S: Final[float] = 2.0 * 60.0 * 60.0
"""既定の見張りの長さ (design.md 「watch」: `serve watch --duration 2h`)。"""

DEFAULT_HEALTH_TIMEOUT_S: Final[float] = 10.0
"""`/health` の時間切れ (design.md 「watch」)。`/metrics` の読み取りにも同じ値を使う。"""

DEFAULT_GPU_TIMEOUT_S: Final[float] = 5.0
"""`nvidia-smi` の読み取りの時間切れ。既定の間隔 (30 秒) より短くする (tasks.md 4.5 の制約:
1 回の観察が間隔より長くかかっても詰まらないようにするため)。"""

DEFAULT_UNRESPONSIVE_THRESHOLD: Final[int] = 3
"""「応答しない」と判定する、連続の失敗の回数 (design.md 「watch」)。"""

DEFAULT_STALL_WINDOW_S: Final[float] = 5.0 * 60.0
"""「固まった」と判定する、生成のトークンの数が増えない長さ (design.md 「watch」: 5 分)。"""

DEFAULT_STALL_GPU_THRESHOLD_PCT: Final[int] = 90
"""「固まった」と判定する、GPU の使用率のしきい値 (design.md 「watch」)。"""

GPU_UTIL_ARGV: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=utilization.gpu",
    "--format=csv,noheader",
)
"""GPU の使用率を読む、ただ 1 つの問い合わせ (design.md 「watch」。見本は
`tests/fixtures/spark/*/nvidia-smi-gpu-util.txt`、形は `0 %`)。"""

_KIND_SERVE: Final[str] = "serve"
"""この module が受ける構成の種類 (決めごとの 1)。"""

_HEAD: Final[NodeRole] = "head"
"""HTTP の宛先になる役割 (design.md 「watch」: head の `/health` と `/metrics`)。"""

_METRIC_TOKENS: Final[str] = "vllm:generation_tokens_total"
_METRIC_RUNNING: Final[str] = "vllm:num_requests_running"
_METRIC_WAITING: Final[str] = "vllm:num_requests_waiting"
"""`/metrics` から読む指標の名前 (design.md 「watch」)。"""

_CONTEXT_SAMPLE_COUNT: Final[int] = 3
"""出来事に添える、前後の観察の数 (決めごとの 6。起きた回を含む)。"""

_MIN_UNRESPONSIVE_THRESHOLD: Final[int] = 1
"""`unresponsive_threshold` に許す最小 (決めごとの 7)。"""

_MIN_GPU_THRESHOLD_PCT: Final[int] = 0
_MAX_GPU_THRESHOLD_PCT: Final[int] = 100
"""`stall_gpu_threshold_pct` に許す範囲 (使用率なので 0〜100。決めごとの 7)。"""

_GPU_UNAVAILABLE_NOTE: Final[str] = (
    "2 台のうち、少なくとも 1 台の GPU の使用率を、一度も読めなかった。"
    "判定は、生成のトークンの数だけで行った (design.md 「watch」)"
)


def _default_now() -> datetime:
    """壁の時計の既定 (試験は、差し替えて実際の時刻を進めない)。"""
    return datetime.now(UTC)


def _default_report() -> TextIO:
    """知らせの既定の出し先 (呼んだときの `sys.stderr`)。"""
    return sys.stderr


def _report(stream: TextIO, text: str) -> None:
    """進捗と、出来事の知らせを書く (標準出力は、後の処理が読むので混ぜない)。"""
    stream.write(f"{text}\n")
    stream.flush()


def _check_positive_finite(value: float, name: str) -> None:
    """数の引数が、有限の正の数であることを、Spark にも推論サーバーにも触る前に確かめる
    (決めごとの 7)。`NaN`、`inf`、`0`、負の数は、どれも受けない (`0` 以下や無限大では、
    眠らずに回り続ける、窓が測れない、時間切れがすぐに来る、のどれかになる)。
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は、有限の正の数にする (NaN、無限大、0 以下は受けない): {value}")


# --- /metrics と /health の読み取り (この module 自身が行う。決めごとの 2) ------------


def _metric_value(text: str, name: str) -> float | None:
    """Prometheus のテキストから、1 つの指標の値を読む (読めなければ空)。

    `lifecycle._metric_value` と同じ考え方だが、下線つきの名前を跨いで使えないので、ここに
    持つ (tasks.md 4.5 の指示)。生成のトークンの数は counter で `float` のまま持つ必要がある
    ので、`int` には丸めない。
    """
    for line in text.splitlines():
        if not line.startswith(name) or line.startswith("#"):
            continue
        rest = line[len(name) :]
        if rest[:1] not in ("{", " "):
            continue  # 名前が前方一致しただけの、別の指標
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            return float(fields[1])
        except ValueError:
            return None
    return None


def _check_health(client: httpx.Client, base_url: str) -> bool:
    """`/health` が 200 を返すか (design.md 「watch」の「応答の確認」)。"""
    try:
        response = client.get(f"{base_url}/health")
    except (httpx.HTTPError, httpx.InvalidURL):
        return False
    return response.status_code == httpx.codes.OK


def _read_metrics(
    client: httpx.Client, base_url: str
) -> tuple[float | None, int | None, int | None]:
    """`/metrics` から、生成のトークンの数、処理中と待ちの要求の数を読む (読めなければ 3 つとも
    空にする。断らない)。"""
    try:
        response = client.get(f"{base_url}/metrics")
    except (httpx.HTTPError, httpx.InvalidURL):
        return None, None, None
    if response.status_code != httpx.codes.OK:
        return None, None, None
    text = response.text
    tokens = _metric_value(text, _METRIC_TOKENS)
    running = _metric_value(text, _METRIC_RUNNING)
    waiting = _metric_value(text, _METRIC_WAITING)
    return (
        tokens,
        None if running is None else int(running),
        None if waiting is None else int(waiting),
    )


# --- nvidia-smi の読み取り ---------------------------------------------------


def _parse_gpu_utilization_pct(text: str) -> int | None:
    """`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` の 1 行を読む。

    見本 (`tests/fixtures/spark/*/nvidia-smi-gpu-util.txt`) の形は `0 %`。GB10 が対応して
    いない場合に出る文字列 (`[Not Supported]` など) は `int` に変換できないので、読めなかった
    こととして空を返す (断らない)。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = stripped.removesuffix("%").strip()
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _read_gpu_utilization_pct(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float
) -> int | None:
    """1 台の GPU の使用率を読む (読めなかった、つながらなかったときは空。断らない)。"""
    try:
        result = runner.run(node, GPU_UTIL_ARGV, timeout_s=timeout_s, mutating=False)
    except RemoteError:
        return None
    if result.exit_code != 0:
        return None
    return _parse_gpu_utilization_pct(result.stdout)


# --- 1 回の観察 ---------------------------------------------------------------


def _observe_once(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    client: httpx.Client,
    base_url: str,
    *,
    gpu_timeout_s: float,
    now: Callable[[], datetime],
) -> WatchSample:
    """1 回ぶんの観察 (design.md 「watch」の「1 回の観察」)。"""
    taken_at = now()
    health_ok = _check_health(client, base_url)
    tokens, running, waiting = _read_metrics(client, base_url)
    gpu: dict[NodeRole, int | None] = {
        role: _read_gpu_utilization_pct(runner, nodes[role], timeout_s=gpu_timeout_s)
        for role in config.nodes
    }
    return WatchSample(
        taken_at_utc=taken_at,
        health_ok=health_ok,
        generation_tokens_total=tokens,
        running_requests=running,
        waiting_requests=waiting,
        gpu_utilization_pct=gpu,
    )


def _append_sample(path: Path, sample: WatchSample) -> None:
    """観察を、samples.jsonl に 1 行足す (途中で落ちても、そこまでが残るように、1 回ごとに
    開いて書き出す)。"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(sample.model_dump_json())
        handle.write("\n")


def _write_result(run_dir: Path, outcome: WatchOutcome) -> None:
    """見張りの要約を `result.json` に書く (決めごとの 5)。"""
    text = json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    (run_dir / RESULT_FILE_NAME).write_text(text + "\n", encoding="utf-8")


def _event_context(samples: Sequence[WatchSample]) -> tuple[WatchSample, ...]:
    """出来事に添える、前後の観察 (決めごとの 6: 起きた回を含めて、直近の数回ぶん)。"""
    return tuple(samples[-_CONTEXT_SAMPLE_COUNT:])


def _gpu_status(gpu_values: Sequence[int | None], threshold: int) -> bool | None:
    """1 回ぶんの観察の、GPU の使用率の状態 (決めごとの 4)。

    2 台とも読めたときだけ判定できる (design.md 「watch」の「2 台の GPU の使用率」)。
    片方でも読めなければ、その回は判定できないとして `None` を返す (窓の中で「読めなかった
    回」として扱う。断らない)。
    """
    if any(value is None for value in gpu_values):
        return None
    return all(value >= threshold for value in gpu_values if value is not None)


def _requests_status(running_requests: int | None) -> bool | None:
    """1 回ぶんの観察の、処理中の要求の状態 (決めごとの 4a。GPU と同じ書き方)。

    `/metrics` が読めなかった回は `None` (窓の中で「読めなかった回」として扱う)。読めた回は、
    処理中の要求が 1 以上かどうかを返す。
    """
    if running_requests is None:
        return None
    return running_requests > 0


def _stall_detail(stall_window_s: float, window_gpu_status: Sequence[bool | None]) -> str:
    """「固まった」の出来事の説明 (決めごとの 4)。

    窓の中に、GPU の使用率を 1 度も読めた回がなければ、生成のトークンの数だけで判定したことを
    書く。読めた回と読めなかった回が混ざっていれば (判定そのものは、読めた回だけで行う)、
    読めなかった回があったことを書く。
    """
    parts = [
        f"処理中の要求があるのに、生成のトークンの数が {stall_window_s:g} 秒のあいだ増えなかった"
    ]
    readable = [status for status in window_gpu_status if status is not None]
    if not readable:
        parts.append(
            "この窓では、GPU の使用率を一度も読めなかったので、"
            "生成のトークンの数と処理中の要求だけで判定した"
        )
    elif any(status is None for status in window_gpu_status):
        parts.append("窓の途中に、GPU の使用率を読めなかった観察があった")
    return "。".join(parts)


def _new_event(
    finding: WatchFinding, at_sample: WatchSample, samples: Sequence[WatchSample], *, detail: str
) -> WatchEvent:
    return WatchEvent(
        finding=finding,
        at_utc=at_sample.taken_at_utc,
        context=_event_context(samples),
        detail=detail,
    )


def _collect_on_event(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    *,
    var_root: Path,
    event_time: datetime,
    config_name: str,
    log_timeout_s: float,
    event: WatchEvent,
) -> WatchEvent:
    """出来事が起きた回に、2 台の記録を回収し、置き場所 (または失敗の理由) を `detail` に
    書き足す (`logs.collect_logs`。対象は `guards.list_own_containers` の一覧から。回収の
    たびに、その時の時刻を渡すので、出来事ごとに別の置き場所になる)。

    **回収が失敗しても、見張りは続く**。`collect_logs` 自身が、ssh の届かなさ (`RemoteError`)
    を内側で吸収するので、ここで捕まえるのは、構成が使う役割のノードの定義が足りない
    (`ValueError`) と、Mac 側のファイルの書き込みが失敗した (`OSError`) ときだけである。
    """
    try:
        log_dir = collect_logs(
            runner,
            nodes,
            plans,
            var_root=var_root,
            started_at=event_time,
            command=COMMAND_NAME,
            config_name=config_name,
            timeout_s=log_timeout_s,
        )
        note = f"記録は {log_dir} に回収した"
    except (ValueError, OSError) as exc:
        note = f"記録の回収に失敗した (見張りは続ける): {exc}"
    detail = f"{event.detail}。{note}" if event.detail else note
    return event.model_copy(update={"detail": detail})


# --- 入口 ---------------------------------------------------------------------


def watch(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    var_root: Path,
    duration_s: float = DEFAULT_DURATION_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    unresponsive_threshold: int = DEFAULT_UNRESPONSIVE_THRESHOLD,
    stall_window_s: float = DEFAULT_STALL_WINDOW_S,
    stall_gpu_threshold_pct: int = DEFAULT_STALL_GPU_THRESHOLD_PCT,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    gpu_timeout_s: float = DEFAULT_GPU_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    now: Callable[[], datetime] | None = None,
    report: TextIO | None = None,
) -> WatchOutcome:
    """連続の負荷の間、外から見張る (`serve watch <構成>`。requirements 7.7)。

    **読み取りだけ**である。関門も了承も要らない (状態を変える呼び出しを 1 つも出さない)。
    `duration_s` に達するまで、`interval_s` ごとに観察を続け、`samples_path` に 1 行ずつ足す。
    「応答しない」「固まった」を見つけたら、時刻とその前後の観察を `WatchOutcome.events` に残し、
    2 台の記録を回収する。決めた時間に達したら、要約 (`WatchOutcome`) を返し、`result.json` に
    書く。**中断 (Ctrl-C) が来ても、そこまでの要約を書いてから、中断を伝える**。

    引数:
        runner: 遠隔の実行役 (読み取りしか呼ばない)。
        config: `kind = "serve"` の構成 (見張る対象。`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義 (`config.nodes` が使う役割のぶんを持つこと)。
        var_root: 記録の置き場所の根 (`serving/var/`)。
        duration_s: 見張りを続ける長さ。
        interval_s: 観察の間隔。
        unresponsive_threshold: 「応答しない」と判定する、連続の失敗の回数。
        stall_window_s: 「固まった」と判定する、トークンの数が増えない長さ。
        stall_gpu_threshold_pct: 「固まった」と判定する、GPU の使用率のしきい値。
        health_timeout_s: `/health` と `/metrics` の、1 回の問い合わせの時間切れ。
        gpu_timeout_s: `nvidia-smi` の、1 回の読み取りの時間切れ。
        log_timeout_s: 出来事が起きたときの、記録の全量の回収の時間切れ。
        client: 読み取りに使う HTTP のクライアント。省くと作り、この関数の中で閉じる。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        now: 壁の時計 (時差付きの `datetime` を返す)。既定は `datetime.now(UTC)`。
        report: 出来事と中断を知らせる先。既定は `sys.stderr`。

    返り値:
        `WatchOutcome`。

    例外:
        config.ConfigError: `kind` が `serve` でない構成、HTTP のポートを 1 つに決められない
            構成、引数の列を組み立てられない構成。
        ValueError: 構成が使う役割のノードの定義が `nodes` にない、`config.nodes` に `head` が
            ないとき。数の引数のどれかが退化している (決めごとの 7: `duration_s` /
            `interval_s` / `stall_window_s` / `health_timeout_s` / `gpu_timeout_s` /
            `log_timeout_s` が有限の正の数でない、`unresponsive_threshold` が bool を除く本物の
            int でない、または 1 未満、`stall_gpu_threshold_pct` が 0〜100 の外、`interval_s`
            が `duration_s` を超える) とき。
        KeyboardInterrupt: 中断。ここまでの要約を `result.json` に書いてから伝える。
    """
    if config.kind != _KIND_SERVE:
        raise ConfigError(
            f"見張りは、kind = '{_KIND_SERVE}' の構成にしか使えない"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if _HEAD not in config.nodes:
        raise ValueError(
            f"構成 '{config.name}' が {_HEAD} を使わない (見張りは {_HEAD} の HTTP を読む)"
        )
    for role in config.nodes:
        if role not in nodes:
            raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
    _check_positive_finite(duration_s, "duration_s")
    _check_positive_finite(interval_s, "interval_s")
    _check_positive_finite(stall_window_s, "stall_window_s")
    _check_positive_finite(health_timeout_s, "health_timeout_s")
    _check_positive_finite(gpu_timeout_s, "gpu_timeout_s")
    _check_positive_finite(log_timeout_s, "log_timeout_s")
    if isinstance(unresponsive_threshold, bool) or not isinstance(unresponsive_threshold, int):
        raise ValueError(
            "unresponsive_threshold は、bool を除く本物の int にする"
            f" (NaN、無限大、小数は受けない): {unresponsive_threshold!r}"
        )
    if unresponsive_threshold < _MIN_UNRESPONSIVE_THRESHOLD:
        raise ValueError(
            f"unresponsive_threshold は {_MIN_UNRESPONSIVE_THRESHOLD} 以上にする (0 以下では、"
            f"1 回も失敗しないうちに「応答しない」を誤って出す): {unresponsive_threshold}"
        )
    if not _MIN_GPU_THRESHOLD_PCT <= stall_gpu_threshold_pct <= _MAX_GPU_THRESHOLD_PCT:
        raise ValueError(
            f"stall_gpu_threshold_pct は {_MIN_GPU_THRESHOLD_PCT}〜{_MAX_GPU_THRESHOLD_PCT}"
            f" の範囲にする: {stall_gpu_threshold_pct}"
        )
    if interval_s > duration_s:
        raise ValueError(
            "interval_s は duration_s 以下にする"
            f" (でなければ、決めた時間の中で観察が間延びする): interval_s={interval_s},"
            f" duration_s={duration_s}"
        )

    port = lifecycle.http_port(config)
    base_url = f"http://{nodes[_HEAD].lan_addr}:{port}"

    now_fn: Callable[[], datetime] = _default_now if now is None else now
    clock_fn: Callable[[], float] = time.monotonic if clock is None else clock
    sleep_fn: Callable[[float], None] = time.sleep if sleep is None else sleep
    report_stream: TextIO = _default_report() if report is None else report

    started_at = now_fn()
    plans = build_plans(config, nodes, started_at)
    run_dir = var_dir(var_root, started_at, COMMAND_NAME, config.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / SAMPLES_FILE_NAME

    owns_client = client is None
    http_client = lifecycle.new_client(health_timeout_s) if client is None else client

    samples: list[WatchSample] = []
    events: list[WatchEvent] = []
    notes: list[str] = []
    gpu_seen: dict[NodeRole, bool] = dict.fromkeys(config.nodes, False)
    last_tokens: float | None = None
    since_change_mono: float | None = None
    window_gpu_status: list[bool | None] = []
    window_requests_status: list[bool | None] = []
    active_unresponsive = False
    active_stalled = False
    consecutive_health_failures = 0

    def build_outcome(finished_at: datetime) -> WatchOutcome:
        gpu_available = all(gpu_seen[role] for role in config.nodes)
        parts = [*notes, *(() if gpu_available else (_GPU_UNAVAILABLE_NOTE,))]
        return WatchOutcome(
            config_name=config.name,
            started_at=started_at,
            finished_at=finished_at,
            samples_path=samples_path,
            sample_count=len(samples),
            events=tuple(events),
            gpu_utilization_available=gpu_available,
            detail="。".join(parts),
        )

    try:
        start_mono = clock_fn()
        tick = 0
        while True:
            sample = _observe_once(
                runner,
                config,
                nodes,
                http_client,
                base_url,
                gpu_timeout_s=gpu_timeout_s,
                now=now_fn,
            )
            samples.append(sample)
            _append_sample(samples_path, sample)

            for role in config.nodes:
                if sample.gpu_utilization_pct.get(role) is not None:
                    gpu_seen[role] = True

            # --- 「応答しない」: 続けての失敗の数だけで判定する ---
            consecutive_health_failures = 0 if sample.health_ok else consecutive_health_failures + 1
            unresponsive_now = consecutive_health_failures >= unresponsive_threshold
            if unresponsive_now and not active_unresponsive:
                event = _new_event(
                    "unresponsive",
                    sample,
                    samples,
                    detail=f"応答の確認 (/health) が {unresponsive_threshold} 回続けて失敗した",
                )
                event = _collect_on_event(
                    runner,
                    nodes,
                    plans,
                    var_root=var_root,
                    event_time=now_fn(),
                    config_name=config.name,
                    log_timeout_s=log_timeout_s,
                    event=event,
                )
                events.append(event)
                _report(report_stream, f"見張り: {event.detail}")
            active_unresponsive = unresponsive_now

            # --- トークンの数の基準を更新する (増えた/減った/変わらない) ---
            current_mono = clock_fn()
            tokens = sample.generation_tokens_total
            if tokens is not None:
                if last_tokens is None or tokens > last_tokens:
                    last_tokens = tokens
                    since_change_mono = current_mono
                    window_gpu_status = []
                    window_requests_status = []
                elif tokens < last_tokens:
                    notes.append(
                        f"{sample.taken_at_utc.isoformat()}: 生成のトークンの数が"
                        f" {last_tokens:g} から {tokens:g} に減った"
                        " (推論サーバーが起こし直されたとみられる)"
                    )
                    last_tokens = tokens
                    since_change_mono = current_mono
                    window_gpu_status = []
                    window_requests_status = []
                # tokens == last_tokens のときは、窓を動かさない

            # --- 「固まった」: 処理中の要求があるのに、トークンの数が窓のあいだ増えない ---
            # (窓の全体を見る。design.md の「その間」を、この回だけでなく、窓が始まって
            # からのすべての観察で確かめる。決めごとの 4、4a)
            gpu_values = [sample.gpu_utilization_pct.get(role) for role in config.nodes]
            if since_change_mono is not None:
                window_gpu_status.append(_gpu_status(gpu_values, stall_gpu_threshold_pct))
                window_requests_status.append(_requests_status(sample.running_requests))

            elapsed_since_change = (
                None if since_change_mono is None else current_mono - since_change_mono
            )
            readable_gpu_in_window = [status for status in window_gpu_status if status is not None]
            gpu_criterion_ok = all(readable_gpu_in_window) if readable_gpu_in_window else True
            readable_requests_in_window = [
                status for status in window_requests_status if status is not None
            ]
            # GPU と違って、処理中の要求は「固まった」の主たる条件なので、窓の中で 1 度も
            # 読めていなければ (すべて None)、確かめられていないとして「固まった」にしない
            # (決めごとの 4a。GPU は補助の条件なので、読めなければ判定から外して素通りさせる)
            requests_criterion_ok = bool(readable_requests_in_window) and all(
                readable_requests_in_window
            )
            stalled_now = (
                requests_criterion_ok
                and elapsed_since_change is not None
                and elapsed_since_change >= stall_window_s
                and gpu_criterion_ok
            )
            if stalled_now and not active_stalled:
                event = _new_event(
                    "stalled",
                    sample,
                    samples,
                    detail=_stall_detail(stall_window_s, window_gpu_status),
                )
                event = _collect_on_event(
                    runner,
                    nodes,
                    plans,
                    var_root=var_root,
                    event_time=now_fn(),
                    config_name=config.name,
                    log_timeout_s=log_timeout_s,
                    event=event,
                )
                events.append(event)
                _report(report_stream, f"見張り: {event.detail}")
            active_stalled = stalled_now

            tick += 1
            if clock_fn() - start_mono >= duration_s:
                break
            remaining = (start_mono + tick * interval_s) - clock_fn()
            if remaining > 0:
                sleep_fn(remaining)
    except KeyboardInterrupt:
        outcome = build_outcome(now_fn())
        _write_result(run_dir, outcome)
        _report(report_stream, "中断されたので、ここまでの見張りの要約を書いた")
        raise
    finally:
        if owns_client:
            http_client.close()

    outcome = build_outcome(now_fn())
    _write_result(run_dir, outcome)
    return outcome
