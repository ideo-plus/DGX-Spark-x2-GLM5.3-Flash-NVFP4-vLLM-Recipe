"""連続の負荷の間、外から見張る (`serve watch <構成>`。design.md 「確認 › watch」、tasks.md 4.5)。

すでに動いている推論サーバーを、決めた間隔で外から観察し続け、応答しなくなる・固まる (2 台の
GPU が使われ続けたまま生成のトークンの数が進まなくなる)・熱くなる、ことが起きたら、時刻と
その前後の観察を残す (requirements 7.7)。**読み取りだけ**である: 何も止めない、起こし直さ
ない (関門も了承も要らない)。状態を変える呼び出し (`mutating=True`、`docker run` / `stop` /
`rm`、`push`) は、この module から 1 つも出ない。

## 1 回の観察 (design.md 「watch」)

head の `/health` (時間切れ `health_timeout_s`。既定 10 秒)、`/metrics` の
`vllm:generation_tokens_total` / `vllm:num_requests_running` / `vllm:num_requests_waiting`
を読み、2 台それぞれで次を読む (計測中の熱源を切り分けるための拡張)。

1. **GPU**: 温度・SM クロック・電力・使用率を、1 回の `nvidia-smi` の問い合わせ
   (`GPU_UTIL_ARGV`) で読む。「固まった」の判定に使う使用率は、この問い合わせと同じ値
2. **熱区域**: 見張りの開始時に発見した全区域 (`/sys/class/thermal/thermal_zone<N>/temp`)
   の値を、区域番号つきで読む (最高値だけにまとめない)
3. **hwmon**: 見張りの開始時に発見した、`mlx5`/`nvme`/`acpitz` の hwmon の `temp*_input`
   の値を、センサーの識別子つきで読む
4. **CPU**: `/proc/stat` の `cpu<N>` 行から、コアごとの使用率を、直前に読めた観察との差で
   計算する (初回は空)
5. **cpufreq のコアの周波数の上限**: 見張りの開始時に発見した全コア
   (`/sys/devices/system/cpu/cpu<N>/cpufreq/scaling_max_freq`) の値を、コア番号つきで
   kHz の整数で読む (issue #10: GPU クロックの上限とあわせて、計測中の CPU の上限を記録から
   確かめるため)

1 つの `types.WatchSample` にして、1 行の JSON で `samples_path`
(`serving/var/<UTC>-watch-<構成>/samples.jsonl`。置き場所は `logs.var_dir` の決まりで作る)
に**観察のたびに** (1 行ごとに開いて書き出す。design の「途中で落ちても、そこまでの観察が
残るように」を守る) 足す。

読み取りが 1 つ届かなくても (`RemoteError`、つながらない、200 でない、`cat` が 0 以外で
終わる) その項目だけを空にして続ける (requirements 2.3、2.4 の「よそのものを読まない」に
反しない範囲の、決まった問い合わせしか出さない)。

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
- **`thermal`**: どれかの台の、どれかの熱区域が `thermal_threshold_c` (既定 90℃) 以上に
  なった。`unresponsive`/`stalled` と違い、記録の回収 (`_collect_on_event`) をしない (熱では
  推論サーバーは壊れていないため。決めごとの 12)
- **生成のトークンの数が減った** (サーバーが起こし直されたとみられる) ときは、固まりとは扱わず、
  窓を (GPU の履歴も、処理中の要求の履歴も込みで) 数え直す。起こし直したこと自体は
  `WatchOutcome.detail` に書く (requirements 10.5 により、送った内容や応答の本文は、どの記録
  にも書かない)
- `/metrics` が一時的に読めない (トークンの数、処理中の要求の数が空になる) 回を挟んでも、窓は
  測り直さない (読めなかった回の前後で、値が同じであることを前提にする。決めごとの 4)。その
  回の GPU の使用率は、読めれば、そのまま窓に積む (`/metrics` と `nvidia-smi` は独立に読むため)
- **どの判定も、条件が続いているあいだは 1 件のまま** (立ち上がりの回だけ `WatchEvent` を
  1 つ作る)。いったん条件が外れて、また起きたら、別の 1 件になる
- `unresponsive_threshold`、`stall_window_s`、`stall_gpu_threshold_pct`、`thermal_threshold_c`、
  `interval_s` は、引数で上書きできる (試験が、間隔を縮めるために使う)

出来事が起きた回は、時刻と、その前後の観察 (`_CONTEXT_SAMPLE_COUNT` 回ぶん) を `WatchEvent` に
残す。`unresponsive`/`stalled` は、`logs.collect_logs` で 2 台の記録を回収する (対象は、
`guards.list_own_containers` の一覧から。回収は、出来事が起きたその時の時刻を渡して、そのたび
に別の置き場所に書く)。**回収が失敗しても、見張りは続く** (`collect_logs` 自身が `RemoteError`
を内側で吸収するので、ここで捕まえるのは `ValueError` / `OSError` だけである)。

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
   `nvidia-smi`・`cat` だけであり、`probe` / `job` / `fetch` / `inspect` には待ち受ける口がない
2. **`/metrics` の読み取りは、この module 自身が行う** (`lifecycle._metric_value` は下線つきの
   名前で使えないため。tasks.md 4.5 の指示のとおり)。生成のトークンの数は counter なので、
   `lifecycle` のように `int` へ丸めず `float` のまま持つ (`types.WatchSample` の形に合わせる)
3. **GPU の使用率の「読める」は、台ごとに、一度でも読めたかどうかで判定する** (`gpu_seen`)。
   片方の台に 1 度だけ入れなかった (`RemoteError`) ときに、それだけで「読めない」と早合点して
   トークンの数だけの判定に落ちないようにするため。**GB10 がそもそも対応していない**
   (4 列の CSV が読めない) ときは、一度も読めないまま終わるので、`gpu_utilization_available =
   False` になる
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
   `health_timeout_s`、`gpu_timeout_s`、`log_timeout_s`、`thermal_threshold_c` は、有限の正の
   数 (`NaN`、`inf`、0、負の数を断る)。`unresponsive_threshold` は、bool を除く本物の int で、
   1 以上 (0 以下では、1 回も失敗しないうちに「応答しない」を誤って出す。`NaN`、`inf`、小数、
   `bool` は、`int` との比較を素通りしてしまうので、先に型そのものを確かめる)。
   `stall_gpu_threshold_pct` は 0〜100 の範囲 (使用率なので)。
   `interval_s` は `duration_s` 以下にする (でなければ、決めた時間のあいだに観察が間延びする)。
   0 以下や無限大の `interval_s` を断らないと、眠らずに回り続け、推論サーバーと 2 台の
   Spark への問い合わせを叩き続ける (レビューの指摘 1)。0 以下の `stall_window_s` や
   `unresponsive_threshold` を断らないと、健全な状態でもすぐに「固まった」「応答しない」を
   誤って出す (レビューの指摘 2)。**`gpu_timeout_s` と `interval_s` の大小関係は、断らない**
   (`DEFAULT_GPU_TIMEOUT_S` を既定の間隔より短くしたのは既定値どうしの選び方の理由であって、
   呼ぶ側が短い `interval_s` を選ぶこと自体は誤りではない。試験が、間隔を縮めて確かめるのに
   使うため、ここを断ると、値を決めた組み合わせでしか呼べなくなる)
8. **熱区域・hwmon・cpufreq のコアの発見は、見張りの開始時に 1 回だけ** (`start_mono` を
   取る前。中断が発見の最中に来ても、そこまでの要約が書かれる)。熱区域・hwmon・cpufreq の
   コア自体の番号は 0 から (上限 32、`_THERMAL_ZONE_LIMIT`、`_HWMON_LIMIT`、
   `_CPU_CORE_LIMIT`)、hwmon の温度センサーの番号は 1 から (上限 32、`_HWMON_TEMP_LIMIT`)
   順に試す。見つけた一覧は、見張りの
   あいだ使い、`result.json` に記録する (観察のたびには繰り返さない)。発見の途中で台に届かな
   い (`RemoteError`) と、その台の発見を打ち切り、役割つきの文を `WatchOutcome.detail` に
   残す。`exit_code != 0` (存在しない) は、これまでどおり素通りする (`detail` に書かない)。
   見張りそのものは止めず、打ち切った台の観察も続ける (`_discover_node`)
9. **熱区域・hwmon・cpufreq のコアの読みは、1 回の `cat` に複数の道筋を並べる**。行数が、
   発見した数と合わなければ、その項目全体を空にする (対応が取れないため)
10. **CPU の使用率は、直前に読めた観察との差**。初回や、直前の観察が読めなかったときは空に
   する (読めない回を挟んだあとも、直前に読めた回との差として計算を続ける)
11. **`thermal` の終了コードへの写しは、既存の表のとおり** (`events` に 1 件でもあれば 2)。
   ただし、`thermal` が起きても計測そのものは止めない (止めるかは、要約と出来事を見て対話側
   が決める)
12. **`thermal` の型は `types.ThermalFinding`**。`types.WatchFinding` は変えない (`unresponsive`
   /`stalled` は記録の回収を伴うが、`thermal` は伴わないという意味の違いを、型でも分ける)
13. **`result.json` の `cpu_cluster_max_freq_khz` は、台ごとの X925/A725 の上限の最大値**
   (全観察を通した、読めた値だけの最大。`watch.X925_CORES` が GB10 の X925 のコア番号を
   固定で持ち、残りを A725 として扱う。issue #10: GPU の上限は既存の `gpu_sm_clock_range_mhz`
   で確かめられるので、CPU 側の上限をこの欄で確かめられるようにする)

## 終了コードへの写し方 (5.1 が、この表のとおりに写す。design.md 「Error Handling」に合わせる)

| `WatchOutcome` / 例外 | 終了コード | 意味 |
|---|---|---|
| `events` が空のまま `duration_s` に達した | 0 | 正常。応答しなくも固まりも熱くもならなかった |
| `events` に 1 件以上ある | 2 | **実行しての失敗** (`bool(outcome.events)` で判定できる) |
| `config.ConfigError`、`ValueError` | 1 | 前提の不足。触る前に断る |
| `KeyboardInterrupt` | 130 | 中断。要約は `result.json` に書いてから伝える |

**`events` が 1 件以上あると 2 にする理由**: 見張りの道具そのものは正しく動いたが、対象の
推論サーバーが、応答しなくなる・固まる・熱くなる、のいずれかを、決めた時間のあいだに少なく
とも 1 度起こしたという「実行しての失敗」を、終了コードだけでも見分けられるようにするため。
前提の不足 (`kind` が `serve` でない、`--port` を 1 つに決められない、数の引数が退化している
(決めごとの 7)、構成が使う役割のノードの定義が `nodes` にない) は、どれも Spark にも
推論サーバーにも触る前に断る。
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    CpuCluster,
    NodeDef,
    NodeRole,
    ThermalFinding,
    WatchEvent,
    WatchFinding,
    WatchOutcome,
    WatchSample,
)

__all__ = [
    "COMMAND_NAME",
    "CPU_DIR",
    "DEFAULT_DURATION_S",
    "DEFAULT_GPU_TIMEOUT_S",
    "DEFAULT_HEALTH_TIMEOUT_S",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_STALL_GPU_THRESHOLD_PCT",
    "DEFAULT_STALL_WINDOW_S",
    "DEFAULT_THERMAL_THRESHOLD_C",
    "DEFAULT_UNRESPONSIVE_THRESHOLD",
    "GPU_UTIL_ARGV",
    "HWMON_DIR",
    "PROC_STAT_PATH",
    "RESULT_FILE_NAME",
    "SAMPLES_FILE_NAME",
    "THERMAL_ZONE_DIR",
    "X925_CORES",
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
"""`nvidia-smi`・`cat` の 1 回の読み取りの時間切れ。既定の間隔 (30 秒) より短くする
(tasks.md 4.5 の制約: 1 回の観察が間隔より長くかかっても詰まらないようにするため)。"""

DEFAULT_UNRESPONSIVE_THRESHOLD: Final[int] = 3
"""「応答しない」と判定する、連続の失敗の回数 (design.md 「watch」)。"""

DEFAULT_STALL_WINDOW_S: Final[float] = 5.0 * 60.0
"""「固まった」と判定する、生成のトークンの数が増えない長さ (design.md 「watch」: 5 分)。"""

DEFAULT_STALL_GPU_THRESHOLD_PCT: Final[int] = 90
"""「固まった」と判定する、GPU の使用率のしきい値 (design.md 「watch」)。"""

DEFAULT_THERMAL_THRESHOLD_C: Final[float] = 90.0
"""「熱」と判定する、熱区域の温度のしきい値の既定 (℃。単位は摂氏。判定と要約)。"""

GPU_UTIL_ARGV: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
    "--format=csv,noheader,nounits",
)
"""GPU の 4 項目 (温度・SM クロック・電力・使用率) を読む、ただ 1 つの問い合わせ
(design.md 「watch」)。列の順は温度・SM・電力・使用率。実機の見本は未採取。"""

THERMAL_ZONE_DIR: Final[str] = "/sys/class/thermal/"
"""熱区域の読み取り先 (`remote._CAT_PREFIXES` が前方一致で許す場所)。"""

HWMON_DIR: Final[str] = "/sys/class/hwmon/"
"""hwmon の読み取り先 (`remote._CAT_PREFIXES` が前方一致で許す場所)。"""

PROC_STAT_PATH: Final[str] = "/proc/stat"
"""CPU の使用率の読み取り先 (`remote._CAT_EXACT_PATHS` が完全一致で許す場所)。"""

CPU_DIR: Final[str] = "/sys/devices/system/cpu/"
"""cpufreq のコアごとの周波数の上限の読み取り先の根 (`remote._CAT_EXACT_PATTERNS` が
正規表現の完全一致で許す場所。issue #10: GPU クロックの上限とあわせて、計測中の CPU の
上限を記録から確かめるため)。"""

_THERMAL_ZONE_LIMIT: Final[int] = 32
_HWMON_LIMIT: Final[int] = 32
_CPU_CORE_LIMIT: Final[int] = 32
"""発見が試す、熱区域・hwmon・CPU コア自体の番号の上限 (決めごとの 8。0 から、この数の手前
まで (0〜31) 試す。GB10 の実コア数は 20 だが、余裕を持たせた上限にする)。"""

_HWMON_TEMP_LIMIT: Final[int] = 32
"""発見が試す、hwmon の温度センサーの番号の上限 (決めごとの 8。1 から、この数まで
(1〜32) 試す。熱区域・hwmon 自体の番号と違い、1 始まりであることに注意)。"""

_HWMON_CHIP_NAMES: Final[frozenset[str]] = frozenset({"mlx5", "nvme", "acpitz"})
"""採用する hwmon の `name` (ConnectX-7、NVMe、ACPI 熱区域相当の hwmon)。"""

X925_CORES: Final[frozenset[int]] = frozenset({5, 6, 7, 8, 9, 15, 16, 17, 18, 19})
"""X925 の高性能コアの番号 (GB10 固有の既知の事実。issue #10、
docs/results/2026-09-23-thermal-source.md)。残りのコアは A725 として扱う。"""

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

_CpuCounters = dict[int, tuple[int, int]]
"""コア番号から `(idle + iowait, 先頭 8 欄の和)` への対応 (`_parse_cpu_counters` の返り値)。"""


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


def _to_int(text: str) -> int | None:
    """文字列を `int` に変える (変換できなければ空。断らない)。"""
    try:
        return int(text)
    except ValueError:
        return None


def _to_float(text: str) -> float | None:
    """文字列を `float` に変える (変換できなければ空。断らない)。"""
    try:
        return float(text)
    except ValueError:
        return None


def _max_present[T: (int, float)](values: Iterable[T | None]) -> T | None:
    """読めた値だけの最大値 (1 つも読めなければ空)。GPU の温度 (`int`) と電力 (`float`) の、
    どちらの最大値の計算もこの 1 つに集める。
    """
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _int_range(values: Iterable[int | None]) -> tuple[int, int] | None:
    """読めた値だけの最小と最大 (1 つも読めなければ空)。"""
    present = [value for value in values if value is not None]
    return (min(present), max(present)) if present else None


def _max_by_key[K: (int, str)](per_sample: Iterable[Mapping[K, float | None]]) -> dict[K, float]:
    """観察ごとの、キーつきの値の列から、キーごとの最大値を求める (読めた値だけ)。熱区域
    (`int` のキー) と hwmon (`str` のキー) の、どちらのキーごとの最大値の計算もこの 1 つに
    集める。1 度も読めなかったキーは持たない (要約に空のキーを作らない)。
    """
    result: dict[K, float] = {}
    for values in per_sample:
        for key, value in values.items():
            if value is None:
                continue
            if key not in result or value > result[key]:
                result[key] = value
    return result


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


def _parse_gpu_query(text: str) -> tuple[int | None, int | None, float | None, int | None]:
    """`GPU_UTIL_ARGV` の 1 行を、温度・SM クロック・電力・使用率の 4 列として読む。

    最初の空でない行を `,` で分け、4 列でなければ全部空にする (旧い 1 列だけの応答も、ここで
    「読めない」扱いになる。実機の見本ではない旧い形式を読む fallback は持たない)。列ごとに
    変換できなければ、その列だけ空にする (`[Not Supported]`、`[N/A]` など。断らない)。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != 4:
            return None, None, None, None
        temperature = _to_int(fields[0])
        sm_clock = _to_int(fields[1])
        power = _to_float(fields[2])
        utilization = _to_int(fields[3])
        return temperature, sm_clock, power, utilization
    return None, None, None, None


def _read_gpu(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float
) -> tuple[int | None, int | None, float | None, int | None]:
    """1 台の GPU の 4 項目を読む (読めなかった、つながらなかったときは全部空。断らない)。"""
    try:
        result = runner.run(node, GPU_UTIL_ARGV, timeout_s=timeout_s, mutating=False)
    except RemoteError:
        return None, None, None, None
    if result.exit_code != 0:
        return None, None, None, None
    return _parse_gpu_query(result.stdout)


# --- cat の読み取りと、熱区域・hwmon の発見 -----------------------------------


def _cat_if_present(
    runner: RemoteRunner, node: NodeDef, paths: Sequence[str], *, timeout_s: float
) -> str | None:
    """`cat` で、存在を区別しながら読む。`exit_code != 0` は「存在しない」として `None` を
    返すが、`RemoteError` (時間切れ、接続の失敗。台そのものに届かなかったこと) は上に伝える。

    発見 (`_discover_thermal_zones`、`_discover_hwmon`) が使う。観察用の `_cat` は、この
    2 つの違いをまとめて空にするが、発見は「存在しない」と「その台に届かなかった」を区別する
    必要があるため、ここでは `RemoteError` を吸収しない。
    """
    result = runner.run(node, ("cat", *paths), timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return None
    return result.stdout


def _cat(
    runner: RemoteRunner, node: NodeDef, paths: Sequence[str], *, timeout_s: float
) -> str | None:
    """`cat` で 1 つ以上の道筋を読む (読めなかった、つながらなかったときは空。断らない)。"""
    try:
        return _cat_if_present(runner, node, paths, timeout_s=timeout_s)
    except RemoteError:
        return None


def _discover_thermal_zones(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float
) -> tuple[int, ...]:
    """熱区域の番号を、0 から `_THERMAL_ZONE_LIMIT` の手前まで、1 つずつ試して見つける
    (決めごとの 8。存在は `cat` の exit 0 で判定する。見張りの開始時に 1 回だけ呼ぶ)。

    台に届かない (`RemoteError`) ときは、それまでに見つけた区域を返さずに上へ伝える
    (呼び出し元の `_discover_node` が、打ち切りとして扱う)。
    """
    found: list[int] = []
    for number in range(_THERMAL_ZONE_LIMIT):
        path = f"{THERMAL_ZONE_DIR}thermal_zone{number}/temp"
        if _cat_if_present(runner, node, (path,), timeout_s=timeout_s) is not None:
            found.append(number)
    return tuple(found)


def _discover_hwmon(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float
) -> tuple[dict[str, str], dict[str, str | None]]:
    """hwmon の番号を 0 から `_HWMON_LIMIT` の手前まで試し、`name` が `_HWMON_CHIP_NAMES` の
    ものだけ採用する。採用した chip について、温度センサーの番号を 1 から `_HWMON_TEMP_LIMIT`
    まで試し、`temp<M>_input` が読めれば発見し、識別子は `"hwmon<N>/temp<M>"`。
    `temp<M>_label` が読めればラベルを添える (読めなければ空)。見張りの開始時に 1 回だけ呼ぶ。

    台に届かない (`RemoteError`) ときは、それまでに見つけたものを返さずに上へ伝える
    (呼び出し元の `_discover_node` が、打ち切りとして扱う)。
    """
    chip_names: dict[str, str] = {}
    sensors: dict[str, str | None] = {}
    for number in range(_HWMON_LIMIT):
        hwmon = f"hwmon{number}"
        raw_name = _cat_if_present(runner, node, (f"{HWMON_DIR}{hwmon}/name",), timeout_s=timeout_s)
        if raw_name is None:
            continue
        name = raw_name.strip()
        if name not in _HWMON_CHIP_NAMES:
            continue
        chip_names[hwmon] = name
        for temp_number in range(1, _HWMON_TEMP_LIMIT + 1):
            temp = f"temp{temp_number}"
            input_path = f"{HWMON_DIR}{hwmon}/{temp}_input"
            if _cat_if_present(runner, node, (input_path,), timeout_s=timeout_s) is None:
                continue
            sensor_id = f"{hwmon}/{temp}"
            label = _cat_if_present(
                runner, node, (f"{HWMON_DIR}{hwmon}/{temp}_label",), timeout_s=timeout_s
            )
            sensors[sensor_id] = None if label is None else label.strip()
    return chip_names, sensors


def _scaling_max_freq_path(core: int) -> str:
    """CPU コア 1 つの、周波数の上限の読み取り先 (`remote._CAT_EXACT_PATTERNS` が正規表現の
    完全一致で許す場所。issue #10)。"""
    return f"{CPU_DIR}cpu{core}/cpufreq/scaling_max_freq"


def _discover_cpufreq_cores(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float
) -> tuple[int, ...]:
    """cpufreq のコア番号を、0 から `_CPU_CORE_LIMIT` の手前まで、1 つずつ試して見つける
    (決めごとの 8 と同じ形。存在は `cat` の exit 0 で判定する。見張りの開始時に 1 回だけ呼ぶ)。

    台に届かない (`RemoteError`) ときは、それまでに見つけたコアを返さずに上へ伝える
    (呼び出し元の `_discover_node` が、打ち切りとして扱う)。
    """
    found: list[int] = []
    for core in range(_CPU_CORE_LIMIT):
        path = _scaling_max_freq_path(core)
        if _cat_if_present(runner, node, (path,), timeout_s=timeout_s) is not None:
            found.append(core)
    return tuple(found)


def _discover_node(
    runner: RemoteRunner, role: NodeRole, node: NodeDef, *, timeout_s: float
) -> tuple[tuple[int, ...], dict[str, str], dict[str, str | None], tuple[int, ...], str | None]:
    """1 台ぶんの熱区域・hwmon・cpufreq のコアの発見。台に届かなければ (`RemoteError`)、
    その台の発見を打ち切る。打ち切ったときは、完了していた種類の一覧はそのまま残し、途中
    だった種類と、まだ始めていない種類は空にする (`熱区域` → `hwmon` → `cpufreq` の順で試すため)。

    返り値の最後の要素は、打ち切ったことを役割つきで示す文 (完了していれば `None`)。
    """
    zones: tuple[int, ...] = ()
    chip_names: dict[str, str] = {}
    sensors: dict[str, str | None] = {}
    cpufreq_cores: tuple[int, ...] = ()
    try:
        zones = _discover_thermal_zones(runner, node, timeout_s=timeout_s)
        chip_names, sensors = _discover_hwmon(runner, node, timeout_s=timeout_s)
        cpufreq_cores = _discover_cpufreq_cores(runner, node, timeout_s=timeout_s)
    except RemoteError as exc:
        note = (
            f"{role}: 熱区域・hwmon・cpufreq の発見が完了しなかった"
            f" (この台の一覧は完全ではない): {exc}"
        )
        return zones, chip_names, sensors, cpufreq_cores, note
    return zones, chip_names, sensors, cpufreq_cores, None


def _read_int_by_key[K: (int, str)](
    runner: RemoteRunner, node: NodeDef, paths: Mapping[K, str], *, timeout_s: float
) -> dict[K, int | None]:
    """発見した区域・センサー・コアの値を、1 回の `cat` にまとめて整数として読む
    (決めごとの 9)。熱区域・hwmon・cpufreq で、読みの規則は同じ (この関数 1 つに集める)。
    `paths` は、キー (区域番号、hwmon センサーの識別子、CPU コア番号のいずれか) から、
    読み取り先の道筋への対応 (空なら `cat` を流さず空を返す)。読めなかった回や、行数が
    キーの数と合わない回は、全キーを空にする (キーどうしの対応が取れないため)。変換できない
    行は、そのキーだけ空。
    """
    if not paths:
        return {}
    keys = list(paths)
    text = _cat(runner, node, [paths[key] for key in keys], timeout_s=timeout_s)
    if text is None:
        return dict.fromkeys(keys, None)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(keys):
        return dict.fromkeys(keys, None)
    result: dict[K, int | None] = {}
    for key, line in zip(keys, lines, strict=True):
        result[key] = _to_int(line.strip())
    return result


def _read_milli_celsius[K: (int, str)](
    runner: RemoteRunner, node: NodeDef, paths: Mapping[K, str], *, timeout_s: float
) -> dict[K, float | None]:
    """発見した区域・センサーの値を読み、m℃ の整数を℃に変える (`/1000`)。読みの規則
    そのものは `_read_int_by_key` に集めている (熱区域・hwmon・cpufreq で共通)。
    """
    raw = _read_int_by_key(runner, node, paths, timeout_s=timeout_s)
    return {key: (None if milli is None else milli / 1000.0) for key, milli in raw.items()}


def _read_thermal_zones(
    runner: RemoteRunner, node: NodeDef, zones: Sequence[int], *, timeout_s: float
) -> dict[int, float | None]:
    """発見した熱区域の値を読む (`_read_milli_celsius` に、区域番号から道筋への対応を渡す
    だけ)。"""
    paths = {number: f"{THERMAL_ZONE_DIR}thermal_zone{number}/temp" for number in zones}
    return _read_milli_celsius(runner, node, paths, timeout_s=timeout_s)


def _read_hwmon_temps(
    runner: RemoteRunner, node: NodeDef, sensors: Sequence[str], *, timeout_s: float
) -> dict[str, float | None]:
    """発見した hwmon センサーの値を読む (`_read_milli_celsius` に、識別子から道筋への対応を
    渡すだけ)。"""
    paths = {sensor_id: f"{HWMON_DIR}{sensor_id}_input" for sensor_id in sensors}
    return _read_milli_celsius(runner, node, paths, timeout_s=timeout_s)


def _read_scaling_max_freq(
    runner: RemoteRunner, node: NodeDef, cores: Sequence[int], *, timeout_s: float
) -> dict[int, int | None]:
    """発見した cpufreq のコアの周波数の上限を読む (`_read_int_by_key` に、コア番号から
    道筋への対応を渡すだけ。issue #10)。"""
    paths = {core: _scaling_max_freq_path(core) for core in cores}
    return _read_int_by_key(runner, node, paths, timeout_s=timeout_s)


def _cluster_of(core: int) -> CpuCluster:
    """CPU コア番号から、X925/A725 のどちらの群に属するかを返す (GB10 固有。issue #10)。"""
    return "x925" if core in X925_CORES else "a725"


def _cluster_max_khz(samples: Sequence[WatchSample], role: NodeRole) -> dict[CpuCluster, int]:
    """台ごとの X925/A725 の周波数の上限の最大値 (全観察を通した、読めた値だけの最大。
    1 度も読めなかった群はキーを持たない。issue #10)。"""
    result: dict[CpuCluster, int] = {}
    for sample in samples:
        for core, khz in sample.cpu_scaling_max_freq_khz.get(role, {}).items():
            if khz is None:
                continue
            cluster = _cluster_of(core)
            if cluster not in result or khz > result[cluster]:
                result[cluster] = khz
    return result


# --- /proc/stat の読み取り (CPU) ---------------------------------------------


def _parse_cpu_counters(text: str) -> _CpuCounters:
    """`/proc/stat` から、コアごとの `(idle + iowait, 先頭 8 欄の和)` を読む。

    `cpu<N>` 行だけを対象にし (合計行 `cpu ` や `intr`/`ctxt` などの他の行は含めない)、
    読めない行は無視する (断らない)。
    """
    counters: _CpuCounters = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 9 or fields[0] == "cpu" or not fields[0].startswith("cpu"):
            continue
        core_text = fields[0].removeprefix("cpu")
        if not core_text.isdigit():
            continue
        try:
            values = [int(field) for field in fields[1:9]]
        except ValueError:
            continue
        idle_total = values[3] + values[4]
        total = sum(values)
        counters[int(core_text)] = (idle_total, total)
    return counters


def _cpu_utilization(
    previous: Mapping[int, tuple[int, int]] | None, current: Mapping[int, tuple[int, int]]
) -> dict[int, float]:
    """コアごとの使用率を、直前に読めた観察との差から出す (決めごとの 10: 初回、または直前が
    読めなかったときは空)。両方にあるコアで、合計の差が正のものだけを対象にする (差が 0 以下
    のコアは使用率が求まらないので含めない)。
    """
    if previous is None:
        return {}
    result: dict[int, float] = {}
    for core, (idle, total) in current.items():
        base = previous.get(core)
        if base is None:
            continue
        prev_idle, prev_total = base
        delta_total = total - prev_total
        if delta_total <= 0:
            continue
        delta_idle = idle - prev_idle
        pct = 100.0 * (1.0 - delta_idle / delta_total)
        result[core] = min(100.0, max(0.0, pct))
    return result


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
    zones: Mapping[NodeRole, tuple[int, ...]],
    sensors: Mapping[NodeRole, tuple[str, ...]],
    cpufreq_cores: Mapping[NodeRole, tuple[int, ...]],
    previous_cpu: Mapping[NodeRole, _CpuCounters | None],
) -> tuple[WatchSample, dict[NodeRole, _CpuCounters | None]]:
    """1 回ぶんの観察 (design.md 「watch」の「1 回の観察」に、熱区域・hwmon・CPU・cpufreq
    のコアの周波数の上限を加えた形)。

    台ごとに `nvidia-smi` → 熱区域 → hwmon → `/proc/stat` → cpufreq のコアの順で読む。
    1 つが読めなくても、他は続ける。返り値の 2 つ目は、台ごとに、この回に読めた
    `/proc/stat` の生カウンタ (読めなければ `None`)。呼び出し側は、読めた台だけ次回の
    `previous_cpu` を更新する。
    """
    taken_at = now()
    health_ok = _check_health(client, base_url)
    tokens, running, waiting = _read_metrics(client, base_url)

    gpu_utilization_pct: dict[NodeRole, int | None] = {}
    gpu_temperature_c: dict[NodeRole, int | None] = {}
    gpu_sm_clock_mhz: dict[NodeRole, int | None] = {}
    gpu_power_w: dict[NodeRole, float | None] = {}
    thermal_zones_c: dict[NodeRole, dict[int, float | None]] = {}
    hwmon_temps_c: dict[NodeRole, dict[str, float | None]] = {}
    cpu_utilization_pct: dict[NodeRole, dict[int, float]] = {}
    cpu_scaling_max_freq_khz: dict[NodeRole, dict[int, int | None]] = {}
    current_cpu: dict[NodeRole, _CpuCounters | None] = {}

    for role in config.nodes:
        node = nodes[role]

        temperature, sm_clock, power, utilization = _read_gpu(runner, node, timeout_s=gpu_timeout_s)
        gpu_temperature_c[role] = temperature
        gpu_sm_clock_mhz[role] = sm_clock
        gpu_power_w[role] = power
        gpu_utilization_pct[role] = utilization

        thermal_zones_c[role] = _read_thermal_zones(
            runner, node, zones.get(role, ()), timeout_s=gpu_timeout_s
        )
        hwmon_temps_c[role] = _read_hwmon_temps(
            runner, node, sensors.get(role, ()), timeout_s=gpu_timeout_s
        )

        proc_text = _cat(runner, node, (PROC_STAT_PATH,), timeout_s=gpu_timeout_s)
        counters = None if proc_text is None else _parse_cpu_counters(proc_text)
        current_cpu[role] = counters
        cpu_utilization_pct[role] = (
            {} if counters is None else _cpu_utilization(previous_cpu.get(role), counters)
        )

        cpu_scaling_max_freq_khz[role] = _read_scaling_max_freq(
            runner, node, cpufreq_cores.get(role, ()), timeout_s=gpu_timeout_s
        )

    sample = WatchSample(
        taken_at_utc=taken_at,
        health_ok=health_ok,
        generation_tokens_total=tokens,
        running_requests=running,
        waiting_requests=waiting,
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_temperature_c=gpu_temperature_c,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_power_w=gpu_power_w,
        thermal_zones_c=thermal_zones_c,
        hwmon_temps_c=hwmon_temps_c,
        cpu_utilization_pct=cpu_utilization_pct,
        cpu_scaling_max_freq_khz=cpu_scaling_max_freq_khz,
    )
    return sample, current_cpu


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


def _zones_at_or_above(
    thermal_zones_c: Mapping[NodeRole, Mapping[int, float | None]], threshold_c: float
) -> dict[NodeRole, list[tuple[int, float]]]:
    """1 回の観察で、閾値以上だった熱区域を、台ごとに区域番号の順で集める (読めた値だけ)。

    「熱」の判定 (`thermal_now`) と、出来事の説明 (`_thermal_detail`) は、この 1 つの結果から
    作る (モジュール docstring の「判定」の `thermal` の項。同じ条件を 2 か所で別々に評価し
    ない)。閾値以上の区域が 1 つもない台は、キーを持たない。
    """
    result: dict[NodeRole, list[tuple[int, float]]] = {}
    for role, zones in thermal_zones_c.items():
        hot = sorted(
            (number, value)
            for number, value in zones.items()
            if value is not None and value >= threshold_c
        )
        if hot:
            result[role] = hot
    return result


def _thermal_detail(
    hot_zones: Mapping[NodeRole, Sequence[tuple[int, float]]], threshold_c: float
) -> str:
    """「熱」の出来事の説明 (「熱区域が 90 ℃ 以上: head 区域 0 = 90.5 ℃、区域 4 = 90.5 ℃」の
    形)。`_zones_at_or_above` が求めた、判定と同じ結果を受け取って文にするだけ。
    """
    parts = [
        f"{role} {'、'.join(f'区域 {number} = {value:g} ℃' for number, value in zones)}"
        for role, zones in hot_zones.items()
    ]
    return f"熱区域が {threshold_c:g} ℃ 以上: {'、'.join(parts)}"


def _new_event(
    finding: WatchFinding | ThermalFinding,
    at_sample: WatchSample,
    samples: Sequence[WatchSample],
    *,
    detail: str,
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

    `thermal` はここを呼ばない (熱では推論サーバーは壊れていないため。決めごとの 12)。
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
    thermal_threshold_c: float = DEFAULT_THERMAL_THRESHOLD_C,
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
    見張りの開始時に、熱区域と hwmon の一覧を 1 回だけ発見してから、`duration_s` に達する
    まで、`interval_s` ごとに観察を続け、`samples_path` に 1 行ずつ足す。「応答しない」
    「固まった」「熱」を見つけたら、時刻とその前後の観察を `WatchOutcome.events` に残す
    (「応答しない」「固まった」は、2 台の記録も回収する)。決めた時間に達したら、要約
    (`WatchOutcome`) を返し、`result.json` に書く。**中断 (Ctrl-C) が来ても、そこまでの要約を
    書いてから、中断を伝える**。

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
        thermal_threshold_c: 「熱」と判定する、熱区域の温度のしきい値 (℃)。
        health_timeout_s: `/health` と `/metrics` の、1 回の問い合わせの時間切れ。
        gpu_timeout_s: `nvidia-smi`・`cat` の、1 回の読み取りの時間切れ。
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
            `log_timeout_s` / `thermal_threshold_c` が有限の正の数でない、
            `unresponsive_threshold` が bool を除く本物の int でない、または 1 未満、
            `stall_gpu_threshold_pct` が 0〜100 の外、`interval_s` が `duration_s` を超える) とき。
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
    _check_positive_finite(thermal_threshold_c, "thermal_threshold_c")
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
    active_thermal = False
    consecutive_health_failures = 0
    zones: dict[NodeRole, tuple[int, ...]] = {}
    chip_names: dict[NodeRole, dict[str, str]] = {}
    sensors_found: dict[NodeRole, dict[str, str | None]] = {}
    cpufreq_found: dict[NodeRole, tuple[int, ...]] = {}
    previous_cpu: dict[NodeRole, _CpuCounters | None] = dict.fromkeys(config.nodes, None)
    thermal_over_threshold_samples = 0
    thermal_first_at: datetime | None = None
    thermal_last_at: datetime | None = None

    def build_outcome(finished_at: datetime) -> WatchOutcome:
        gpu_available = all(gpu_seen[role] for role in config.nodes)
        parts = [*notes, *(() if gpu_available else (_GPU_UNAVAILABLE_NOTE,))]

        gpu_temperature_max_c: dict[NodeRole, int | None] = {}
        gpu_sm_clock_range_mhz: dict[NodeRole, tuple[int, int] | None] = {}
        gpu_power_max_w: dict[NodeRole, float | None] = {}
        thermal_zone_max_c: dict[NodeRole, dict[int, float]] = {}
        hwmon_temp_max_c: dict[NodeRole, dict[str, float]] = {}
        cpu_cluster_max_freq_khz: dict[NodeRole, dict[CpuCluster, int]] = {}
        for role in config.nodes:
            gpu_temperature_max_c[role] = _max_present(
                s.gpu_temperature_c.get(role) for s in samples
            )
            gpu_sm_clock_range_mhz[role] = _int_range(s.gpu_sm_clock_mhz.get(role) for s in samples)
            gpu_power_max_w[role] = _max_present(s.gpu_power_w.get(role) for s in samples)

            zone_max = _max_by_key(s.thermal_zones_c.get(role, {}) for s in samples)
            if zone_max:
                thermal_zone_max_c[role] = zone_max

            hwmon_max = _max_by_key(s.hwmon_temps_c.get(role, {}) for s in samples)
            if hwmon_max:
                hwmon_temp_max_c[role] = hwmon_max

            cluster_max = _cluster_max_khz(samples, role)
            if cluster_max:
                cpu_cluster_max_freq_khz[role] = cluster_max

        thermal_span: tuple[datetime, datetime] | None = None
        if thermal_first_at is not None and thermal_last_at is not None:
            thermal_span = (thermal_first_at, thermal_last_at)

        return WatchOutcome(
            config_name=config.name,
            started_at=started_at,
            finished_at=finished_at,
            samples_path=samples_path,
            sample_count=len(samples),
            events=tuple(events),
            gpu_utilization_available=gpu_available,
            detail="。".join(parts),
            thermal_threshold_c=thermal_threshold_c,
            thermal_zones_found=dict(zones),
            hwmon_chip_names=dict(chip_names),
            hwmon_sensors_found=dict(sensors_found),
            cpufreq_cores_found=dict(cpufreq_found),
            gpu_temperature_max_c=gpu_temperature_max_c,
            gpu_sm_clock_range_mhz=gpu_sm_clock_range_mhz,
            gpu_power_max_w=gpu_power_max_w,
            thermal_zone_max_c=thermal_zone_max_c,
            hwmon_temp_max_c=hwmon_temp_max_c,
            cpu_cluster_max_freq_khz=cpu_cluster_max_freq_khz,
            thermal_over_threshold_samples=thermal_over_threshold_samples,
            thermal_over_threshold_span=thermal_span,
        )

    try:
        # 熱区域と hwmon の発見は、見張りの開始時に 1 回だけ (決めごとの 8)。start_mono を
        # 取る前に行うので、発見の所要は最初の間隔に食い込まず、発見の最中の中断でも、
        # そこまでの要約 (発見は空のまま) が書かれる。
        for role in config.nodes:
            (
                found_zones,
                found_chip_names,
                found_sensors,
                found_cpufreq_cores,
                discovery_note,
            ) = _discover_node(runner, role, nodes[role], timeout_s=gpu_timeout_s)
            zones[role] = found_zones
            chip_names[role] = found_chip_names
            sensors_found[role] = found_sensors
            cpufreq_found[role] = found_cpufreq_cores
            if discovery_note is not None:
                notes.append(discovery_note)

        start_mono = clock_fn()
        tick = 0
        while True:
            sample, current_cpu = _observe_once(
                runner,
                config,
                nodes,
                http_client,
                base_url,
                gpu_timeout_s=gpu_timeout_s,
                now=now_fn,
                zones=zones,
                sensors={role: tuple(sensors_found[role]) for role in config.nodes},
                cpufreq_cores=cpufreq_found,
                previous_cpu=previous_cpu,
            )
            samples.append(sample)
            _append_sample(samples_path, sample)
            for role in config.nodes:
                counters = current_cpu.get(role)
                if counters is not None:
                    previous_cpu[role] = counters

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

            # --- 「熱」: どれかの台の、どれかの熱区域がしきい値以上 (決めごとの 12) ---
            hot_zones = _zones_at_or_above(sample.thermal_zones_c, thermal_threshold_c)
            thermal_now = bool(hot_zones)
            if thermal_now:
                thermal_over_threshold_samples += 1
                if thermal_first_at is None:
                    thermal_first_at = sample.taken_at_utc
                thermal_last_at = sample.taken_at_utc
            if thermal_now and not active_thermal:
                event = _new_event(
                    "thermal",
                    sample,
                    samples,
                    detail=_thermal_detail(hot_zones, thermal_threshold_c),
                )
                # thermal では _collect_on_event を呼ばない (推論サーバーは壊れていないため)
                events.append(event)
                _report(report_stream, f"見張り: {event.detail}")
            active_thermal = thermal_now

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
