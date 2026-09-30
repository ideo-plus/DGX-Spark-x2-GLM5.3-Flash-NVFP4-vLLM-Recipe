"""thinking の深さの渡し方が、実際に出力を変えるかを確かめる (`serve thinking`。design.md
「確認 › thinking」、tasks.md 4.6、requirements 9.2、10.5)。

## 何をするか

同じ、複数ターンの会話 (`_CONVERSATION`)、`temperature 0` で、次の 6 通り (`types.ThinkingVariant`)
を、決めた回数 (既定 3) ずつ、head の `/v1/messages` に送る (ストリームでない応答)。

| # | `ThinkingVariant` | 足す項目 | 期待 (design.md 「thinking」) |
|---|---|---|---|
| 1 | `none` | (無し) | 既定 = `Reasoning Effort: Max` |
| 2 | `output_config_low` | `output_config.effort = "low"` | `Low` に変わる (本来の口) |
| 3 | `chat_template_low` | `chat_template_kwargs.reasoning_effort = "low"` | 2 と同じはず |
| 4 | `output_config_medium` | `output_config.effort = "medium"` | 変わらないはず (`max` に落ちる) |
| 5 | `clear_thinking` | `chat_template_kwargs.clear_thinking = True` | 深さ同じ、`input_tokens`減 |
| 6 | `chat_template_off` | `chat_template_kwargs.enable_thinking = False` | `output_tokens`減 ※ |

※ 6 番目の効いた証拠は、`output_tokens` が 1 番目 (`none`) より減ることである。思考の文字数は、
どちらの構成 (公式のテンプレートでも、変数を読むテンプレートでも) でも 0 になるので、
効いたかどうかの区別には使えない (下の「6 番目の限界」)。

共通の項目は `model`、`max_tokens`、`temperature` (0)、`messages`、`stream` (`False`) だけ。
`seed`、`thinking` は送らない (research.md §g: `/v1/messages` の `AnthropicMessagesRequest` は
Anthropic 形式の `thinking` を持たず、pydantic の既定 (`extra="ignore"`) で黙って捨てる。渡しても
意味がない)。`enable_thinking` は、要求の**最上位**には今も送らない。6 番目の通りだけ、
`chat_template_kwargs` の中に `enable_thinking` を送る (issue #131 のコメント 1)。

## 同じ入力 (親の判断: 決めごとの 2)

6 通りとも、**同じ、合成の固定の会話** (`_CONVERSATION`: user → 過去の thinking のブロックと
本文を持つ assistant → user) を送る。5 番目 (`clear_thinking`) の `input_tokens` を、1 番目
(`none`) と比べられるようにするため (5 番目だけ別の入力にすると、「同じ入力」でなくなり、
比べる相手もなくなる)。会話はこの module の定数であり、**モデルの出力ではない**。呼ぶ側は
`messages=` で差し替えられる (較正など、別の会話で確かめたいときのため)。

**過去の thinking のブロックの形は、Anthropic の形
(`{"type": "thinking", "thinking": …, "signature": …}`) にした。** research.md §g (と、そこが
引く vLLM の `anthropic/protocol.py` の記述) は、`/v1/messages` の要求の**最上位**が受け取る
項目 (`chat_template_kwargs`、`output_config` など) までは実物で裏を取っているが、`messages`
の 1 件、1 件の **`content` ブロックの型**までは裏を取れていない。したがって、この形が実機で
受け付けられるか、テンプレートの `reasoning_content` 変数に正しく写るか (research.md §g-2 が
読んだテンプレートの `{%- if … reasoning_content is defined -%}` の分岐に乗るか) は、**未確認**
である (Status Report の CONCERNS に明記する)。`signature` の値はテンプレートの本物の署名では
なく、この module の合成の定数だとわかる文字列にしてある。

## 送る順 (決めごとの 4)

回ごとに 6 通りを `_VARIANTS_ORDER` (types.py の `ThinkingVariant` の列挙と同じ順) で送る
(1 回目の 1〜6、2 回目の 1〜6、…)。時間とともに変わる影響 (キャッシュ、暖まり) が、1 つの
通りに偏らないようにする。

## 記録すること (決めごとの 5。requirements 10.5)

記録するのは `types.ThinkingTrial` の項目 (thinking のブロックの文字数の合計、`input_tokens`、
`output_tokens`、`stop_reason`、HTTP の状態) だけである。**応答の本文、thinking の文、誤りの
応答の本文は、結果のファイルにも、`detail` にも、`report` への表示にも、例外の文にも入れない**。
`report` に出す進捗の行 (`_trial_line`) は、数値と印 (HTTP の状態、`stop_reason` の名前、
thinking の文字数という**数**) だけを並べる。送った会話の文 (合成の定数) も、結果には入れない。

## HTTP の失敗 (決めごとの 7)

1 回の失敗 (接続の失敗、時間切れ、2xx 以外、JSON でない、`usage` がない) は、その回を、値の
欠けた `ThinkingTrial` (`variant` と `trial_index` と、わかるところまでの `http_status` だけを
持つ) として記録し、残りを続ける (止めない)。**`usage` が丸ごと無い応答は、失敗として扱う**
(その回のほかの項目もすべて空にする。個々のキーが部分的に欠けているだけなら、読めた項目だけを
使う)。thinking のブロックがない 200 の応答は、`thinking_chars = 0` (これは失敗ではない)。

## 判定 (`effective` と `detail`。決めごとの 6)

- 「範囲が重ならない」= 一方の (既定 3 回の) 最大 < もう一方の最小
- **`max_tokens` で打ち切られた回の値は、下限である**。ある比べで「範囲が重ならない」と
  言えるのは、値が低い側のグループに、`stop_reason == "max_tokens"` の回が 1 つも無いときだけ
  (低い側が打ち切られていれば、本当の値はもっと高いかもしれず、その比べは判定できない)
- 2 対 1、3 対 1 のそれぞれを、thinking の文字数と `output_tokens` の**どちらか**が重ならなければ
  「重ならない」とする (`_combine`)
- `effective = True`: 2 対 1 か 3 対 1 の**どちらか**が「重ならない」
- `effective = False`: 2 対 1 も 3 対 1 も判定でき (値が揃い、打ち切りにも阻まれず)、**どちらも**
  「重なる」
- `effective = None`: それ以外 (値の欠け、HTTP の失敗、打ち切りで、2 対 1 と 3 対 1 の少なくとも
  一方が判定できない)
- `detail` には、5 つの比べ (2 対 1、3 対 1、4 対 1、5 対 1、6 対 1) を、範囲の数字つきで書く。
  4 対 1 は「重なる」が期待どおり (重ならなければ「期待と違う」)。5 対 1 は `input_tokens` だけで
  比べ、5 の最大 < 1 の最小なら「減った」(期待どおり)。6 対 1 (`chat_template_off`) は
  `output_tokens` だけで比べ、6 の最大 < 1 の最小なら「減った」(期待どおり)。
  6 番目の thinking の文字数は、どちらの構成 (公式のテンプレートでも、変数を読む
  テンプレートでも) でも、パーサーが思考のブロックを作らないため 0 になり、効いたかどうかの
  区別には使えない (1 番目の `none` は思考が有効なので 0 にならない。issue #131。
  下の「6 番目の限界」)。
  **4 対 1、5 対 1、6 対 1 は、`effective` の値を左右しない**
  (決めごとの 6 のとおり、`effective` は 2 対 1 と 3 対 1 だけで決める)

## 6 番目 (`chat_template_off`) の限界 (issue #131)

公式のチャットテンプレートの構成に送ると、`enable_thinking` がテンプレートに届かず、
生成の書き出しは無条件に `<think>` のまま開くため、この通りは効かない (思考が本文に漏れる)。
効いたと確かめられるのは、`enable_thinking`/`thinking` を読むテンプレート (issue #131 の
`glm53-tp2-mtp3-marlin-thinking-toggle`) を相手にしたときだけである。どちらの構成でも
`thinking_chars` は 0 になる (`glm47` のパーサーは、思考が無効なら思考のブロックを作らない
ため)。効いたかどうかは、`output_tokens` が 1 番目より減るかで読む (6 対 1)。

## 中断 (決めごとの 8)

`KeyboardInterrupt` は、それまでの回を `result.json` に書き出してから、そのまま上げ直す。
1 回も終わっていなければ、書き出さずに上げ直す (`ThinkingOutcome.trials` は 1 つ以上が要るため)。

## 引数の確かめ (決めごとの 9。HTTP の要求を 1 つも出す前に行う)

`trials` と `max_tokens` は、bool を除く本物の int で 1 以上。`timeout_s` は `math.isfinite` で
正 (NaN、無限大、0、負を断る。**片側だけの比較は NaN と無限大を素通しするので使わない**)。
`base_url` は `http://` か `https://` で始まり、ホストを持つ。`model` は空でない文字列。
`messages` を渡す場合は空でない。

## 依存の向き

`serving_kit.types` (`ThinkingOutcome`、`ThinkingTrial`、`ThinkingVariant`) と、
`serving_kit.lifecycle` の公開の名前 (`new_client` だけ。HTTP のクライアントの決まりごと
(`trust_env=False`、転送を追わない) を、`smoke`/`watch` と揃えるため) しか読み込まない。
`config`、`remote`、`plan`、`guards`、`logs` は読み込まない (design.md の依存の向き:
`… → lifecycle → probe, netcheck, watch, thinking → cli`。この部品は Spark に一切触らず、
head への HTTP だけで完結するので、Spark の遠隔の実行役 (`RemoteRunner`) も、構成の定義
(`ConfigDef`/`NodeDef`) も要らない。呼ぶ側 (5.1) が、選んだ構成から `base_url` と `model` を
組み立てて渡す)。

## この module 自身の決めごと (design.md が定めない細部)

1. **入口の形**: `run_thinking(*, base_url, model, var_root, trials=3, max_tokens=1024,
   timeout_s=180.0, messages=None, client=None, now=None, report=None) -> ThinkingOutcome`。
   `probe.run_probe` にならい、関数名は `run_thinking` にした
2. **既定値**: `max_tokens` の既定 1024 は `bench/config/profiles.toml` の `quality`/`code` の
   `max_tokens` と揃えた (thinking の分だけ、`smoke` の 64 では短すぎる)。`timeout_s` の既定
   180 秒は、`lifecycle.SMOKE_TIMEOUT_S` (120 秒、`max_tokens=64`) より長くした (thinking は、
   既定の Reasoning Effort: Max で、より多くのトークンを生成しうるため)
3. **置き場所**: `logs.var_dir` は読み込まない (依存の向きの外)。この module 専用に
   `<var_root>/<UTC の日時>-thinking/result.json` に書く (`logs.py` と同じ時刻の形
   `%Y%m%dT%H%M%SZ` を、ここでも使う。「構成」に相当するものがこの入口にはないので、
   `<UTC>-<コマンド>-<構成>` の「構成」の部分は付けない。`model` は空白や `/` を含みうる
   任意の文字列なので、そのまま置き場所の名前には使わない)
4. **`ThinkingOutcome` に持たせない情報は、`report` の進捗行にも数値・印だけを出す**
   (`_trial_line`)。最後に `outcome.detail` (これも数値と印だけ) を `report` に出す

## 5.1 への手がかり (終了コードは、ここでは決めない。design.md 「Error Handling」)

`ThinkingOutcome.effective` が `True`/`False` なら「判定できた」、`None` なら「判定できなかった」
(HTTP の失敗や値の欠けで、6 通りの比べが決着しなかった)。5.1 は、これをもとに終了コードを選ぶ。
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TextIO
from urllib.parse import urlsplit

import httpx

from serving_kit.lifecycle import new_client
from serving_kit.types import ThinkingOutcome, ThinkingTrial, ThinkingVariant

__all__ = [
    "COMMAND_NAME",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_TRIALS",
    "RESULT_FILE_NAME",
    "run_thinking",
]


# --- 決まった値 ---------------------------------------------------------------

COMMAND_NAME: Final[str] = "thinking"
"""記録の置き場所の名前に使う、`serve` のサブコマンドの名前 (この module の決めごとの 3)。"""

RESULT_FILE_NAME: Final[str] = "result.json"
"""確かめの要約を書く先の名前 (design.md 「記録の置き場所」の「コマンドの結果」と同じ名前)。"""

DEFAULT_TRIALS: Final[int] = 3
"""既定の回数 (design.md 「thinking」: 各 3 回ずつ送る)。"""

DEFAULT_MAX_TOKENS: Final[int] = 1024
"""既定の応答の長さの上限 (この module の決めごとの 2)。"""

DEFAULT_TIMEOUT_S: Final[float] = 180.0
"""既定の 1 要求あたりの時間切れ (この module の決めごとの 2)。"""

_MESSAGES_PATH: Final[str] = "/v1/messages"

_VARIANTS_ORDER: Final[tuple[ThinkingVariant, ...]] = (
    "none",
    "output_config_low",
    "chat_template_low",
    "output_config_medium",
    "clear_thinking",
    "chat_template_off",
)
"""1 回ぶんで送る 6 通りの順 (`types.ThinkingVariant` の列挙と同じ順。決めごとの 4。
issue #131 で `chat_template_off` を末尾に足した)。"""

_STOP_REASON_MAX_TOKENS: Final[str] = "max_tokens"
"""`max_tokens` で打ち切られたことを示す `stop_reason` の値 (Anthropic の Messages API の
値。vLLM の `/v1/messages` は、この API の形をまねているので、この値をそのまま使う。実機で
確かめるまでは想定であることに注意 (Status Report の CONCERNS))。"""

_THINKING_BLOCK_TYPE: Final[str] = "thinking"

_TIMESTAMP_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"
"""置き場所の名前に使う UTC の時刻の形 (`logs.py` の決めごとの 1 と同じ形。この module は
`logs` を読み込まないので、ここに複製する)。"""

_CONVERSATION: Final[tuple[dict[str, object], ...]] = (
    {
        "role": "user",
        "content": "42、7、19 を、小さい順に並べてください。理由も一言添えてください。",
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "42, 7, 19 を比べる。7 が最小で 42 が最大、19 はその間。"
                    "よって小さい順は 7, 19, 42 になる。"
                ),
                "signature": "thinking-fixture-not-a-real-signature",
            },
            {
                "type": "text",
                "text": "小さい順に並べると 7, 19, 42 です。7 が最小で 42 が最大だからです。",
            },
        ],
    },
    {
        "role": "user",
        "content": (
            "ありがとうございます。では、その 3 つの数の合計はいくつですか。"
            "計算の過程も一言添えてください。"
        ),
    },
)
"""6 通りとも同じ、合成の固定の会話 (module の docstring の「同じ入力」。モデルの出力ではない)。"""


_CompareVerdict = Literal["candidate_below", "candidate_above", "overlap"]
_CombinedVerdict = Literal["disjoint", "overlap"]


# --- 既定の壁の時計と、進捗の出し先 -------------------------------------------


def _default_now() -> datetime:
    """壁の時計の既定 (試験は、差し替えて実際の時刻を進めない)。"""
    return datetime.now(UTC)


def _default_report() -> TextIO:
    """知らせの既定の出し先 (呼んだときの `sys.stderr`)。"""
    return sys.stderr


def _report(stream: TextIO, text: str) -> None:
    """進捗を書く (標準出力は、後の処理が読むので混ぜない。requirements 10.5: 数値と印だけ)。"""
    stream.write(f"{text}\n")
    stream.flush()


# --- 引数の確かめ (決めごとの 9。HTTP の要求より前に行う) ----------------------


def _check_base_url(value: str) -> None:
    """`base_url` が `http://`/`https://` で始まり、ホストを持つことを確かめる。"""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"base_url を読めない: {value!r} ({exc})") from exc
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url は http:// か https:// で始める: {value!r}")
    if not host:
        raise ValueError(f"base_url にホストがない: {value!r}")


def _check_positive_int(value: int, name: str) -> None:
    """`bool` を除く本物の `int` で、1 以上であることを確かめる (決めごとの 9)。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} は、bool を除く本物の int にする (NaN、無限大、小数は受けない): {value!r}"
        )
    if value < 1:
        raise ValueError(f"{name} は 1 以上にする: {value}")


def _check_positive_finite(value: float, name: str) -> None:
    """有限の正の数であることを確かめる (`NaN`、`inf`、0、負の数は受けない。片側だけの比較は
    `NaN`/`inf` を素通ししてしまうので使わない)。
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は、有限の正の数にする (NaN、無限大、0 以下は受けない): {value}")


# --- 要求の本文の組み立て ------------------------------------------------------


def _build_body(
    variant: ThinkingVariant,
    *,
    model: str,
    max_tokens: int,
    messages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """1 回ぶんの要求の本文 (module の docstring の表)。共通の項目に、通りごとの項目を 1 つだけ
    足す (`none` は何も足さない)。`seed`、`thinking` は送らない。`enable_thinking` は、最上位
    には今も送らない (6 番目の `chat_template_off` だけ、`chat_template_kwargs` の中に送る)。
    """
    body: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [dict(message) for message in messages],
        "stream": False,
    }
    if variant == "output_config_low":
        body["output_config"] = {"effort": "low"}
    elif variant == "chat_template_low":
        body["chat_template_kwargs"] = {"reasoning_effort": "low"}
    elif variant == "output_config_medium":
        body["output_config"] = {"effort": "medium"}
    elif variant == "clear_thinking":
        body["chat_template_kwargs"] = {"clear_thinking": True}
    elif variant == "chat_template_off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


# --- 応答の読み取り (決めごとの 7) --------------------------------------------


def _usage_count(usage: object, key: str) -> int | None:
    """`usage` から、トークンの数を読む (読めなければ空。`lifecycle._usage_count` と同じ考え方
    だが、下線つきの名前を跨いで使えないので、ここに複製する)。
    """
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _thinking_chars(content: object) -> int:
    """`content` の中の thinking のブロックの文字数の合計 (無ければ 0。断らない)。"""
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, dict) and block.get("type") == _THINKING_BLOCK_TYPE:
            text = block.get("thinking")
            if isinstance(text, str):
                total += len(text)
    return total


def _send_trial(
    client: httpx.Client,
    url: str,
    variant: ThinkingVariant,
    trial_index: int,
    body: dict[str, object],
) -> ThinkingTrial:
    """1 回を送って読む。**落ちない**。つながらない、200 でない、JSON でない、`usage` が
    無いのどれでも、値の欠けた `ThinkingTrial` を返す (決めごとの 7)。応答の本文はここでしか
    読まず、`ThinkingTrial` には数値と印だけが残る。
    """
    try:
        response = client.post(url, json=body)
    except (httpx.HTTPError, httpx.InvalidURL):
        return ThinkingTrial(variant=variant, trial_index=trial_index)
    status = response.status_code
    if status != httpx.codes.OK:
        return ThinkingTrial(variant=variant, trial_index=trial_index, http_status=status)
    try:
        payload: object = response.json()
    except ValueError:
        return ThinkingTrial(variant=variant, trial_index=trial_index, http_status=status)
    if not isinstance(payload, dict):
        return ThinkingTrial(variant=variant, trial_index=trial_index, http_status=status)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return ThinkingTrial(variant=variant, trial_index=trial_index, http_status=status)
    stop_reason = payload.get("stop_reason")
    return ThinkingTrial(
        variant=variant,
        trial_index=trial_index,
        http_status=status,
        thinking_chars=_thinking_chars(payload.get("content")),
        input_tokens=_usage_count(usage, "input_tokens"),
        output_tokens=_usage_count(usage, "output_tokens"),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


def _shown(value: int | None) -> str:
    return "不明" if value is None else str(value)


def _trial_line(trial: ThinkingTrial) -> str:
    """進捗の 1 行 (**数値と印だけ**。requirements 10.5)。"""
    status = "届かなかった" if trial.http_status is None else f"HTTP {trial.http_status}"
    parts = [f"thinking: {trial.variant} の {trial.trial_index} 回目: {status}"]
    if trial.stop_reason is not None:
        parts.append(f"終わりの理由 {trial.stop_reason}")
    parts.append(f"thinking {_shown(trial.thinking_chars)} 文字")
    parts.append(f"入力 {_shown(trial.input_tokens)} / 出力 {_shown(trial.output_tokens)} トークン")
    return "、".join(parts)


# --- 判定 (決めごとの 6) -------------------------------------------------------


@dataclass(frozen=True)
class _MetricRange:
    """1 つの通りの、1 つの指標の 3 回ぶんの値 (`values` が `None` なら判定できない: 値の欠け、
    またはその通りの回が 1 つも無い)。"""

    values: tuple[int, ...] | None
    truncated: bool
    """このグループの中に、`max_tokens` で打ち切られた回が 1 つでもあるか。"""


def _metric_range(
    trials: Sequence[ThinkingTrial], extract: Callable[[ThinkingTrial], int | None]
) -> _MetricRange:
    """1 つの通りの、1 つの指標の範囲を作る。値が 1 つでも欠けていれば `None` にする
    (決めごとの 6: 値の欠けでは判定できない)。"""
    if not trials:
        return _MetricRange(values=None, truncated=False)
    checked: list[int] = []
    for trial in trials:
        value = extract(trial)
        if value is None:
            return _MetricRange(values=None, truncated=False)
        checked.append(value)
    truncated = any(trial.stop_reason == _STOP_REASON_MAX_TOKENS for trial in trials)
    return _MetricRange(values=tuple(checked), truncated=truncated)


def _range_text(metric: _MetricRange) -> str:
    if metric.values is None:
        return "判定できない"
    lo, hi = min(metric.values), max(metric.values)
    base = f"[{lo}, {hi}]"
    return f"{base} (max_tokens で打ち切られた回を含む)" if metric.truncated else base


def _compare_ranges(candidate: _MetricRange, baseline: _MetricRange) -> _CompareVerdict | None:
    """2 つの範囲を比べる。**値が低い側のグループが打ち切られていたら、その比べは判定できない**
    (打ち切られた値は下限であり、本当の値はもっと高いかもしれないため。決めごとの 6)。
    """
    if candidate.values is None or baseline.values is None:
        return None
    cand_lo, cand_hi = min(candidate.values), max(candidate.values)
    base_lo, base_hi = min(baseline.values), max(baseline.values)
    if cand_hi < base_lo:
        return None if candidate.truncated else "candidate_below"
    if base_hi < cand_lo:
        return None if baseline.truncated else "candidate_above"
    return "overlap"


def _combine(*results: _CompareVerdict | None) -> _CombinedVerdict | None:
    """複数の指標の比べを 1 つにまとめる。**どれか 1 つでも「重ならない」なら「重ならない」**
    (2 対 1 / 3 対 1 は、文字数か `output_tokens` のどちらかで良いため)。判定できた指標が
    1 つも無ければ `None`。
    """
    present = [result for result in results if result is not None]
    if any(result != "overlap" for result in present):
        return "disjoint"
    if present:
        return "overlap"
    return None


def _judge_pair(
    trials: Sequence[ThinkingTrial], candidate: ThinkingVariant, baseline: ThinkingVariant
) -> tuple[_CombinedVerdict | None, str]:
    """1 つの比べ (`candidate` 対 `baseline`) の判定と、範囲つきの説明文。"""
    candidate_trials = [trial for trial in trials if trial.variant == candidate]
    baseline_trials = [trial for trial in trials if trial.variant == baseline]
    chars_c = _metric_range(candidate_trials, lambda trial: trial.thinking_chars)
    chars_b = _metric_range(baseline_trials, lambda trial: trial.thinking_chars)
    tokens_c = _metric_range(candidate_trials, lambda trial: trial.output_tokens)
    tokens_b = _metric_range(baseline_trials, lambda trial: trial.output_tokens)
    verdict = _combine(_compare_ranges(chars_c, chars_b), _compare_ranges(tokens_c, tokens_b))
    text = (
        f"thinking 文字数 {_range_text(chars_c)} 対 {_range_text(chars_b)}、"
        f"output_tokens {_range_text(tokens_c)} 対 {_range_text(tokens_b)}"
    )
    return verdict, text


def _fifth_text(trials: Sequence[ThinkingTrial]) -> str:
    """5 対 1 の説明文 (`input_tokens` だけで比べる。決めごとの 6)。"""
    candidate = [trial for trial in trials if trial.variant == "clear_thinking"]
    baseline = [trial for trial in trials if trial.variant == "none"]
    cand_range = _metric_range(candidate, lambda trial: trial.input_tokens)
    base_range = _metric_range(baseline, lambda trial: trial.input_tokens)
    verdict = _compare_ranges(cand_range, base_range)
    note = {
        "candidate_below": "減った (期待どおり)",
        "candidate_above": "増えた (期待と違う)",
        "overlap": "変わらなかった (期待と違う)",
        None: "判定できない",
    }[verdict]
    return f"5 対 1: input_tokens {_range_text(cand_range)} 対 {_range_text(base_range)} ({note})"


def _sixth_text(trials: Sequence[ThinkingTrial]) -> str:
    """6 対 1 (`chat_template_off`) の説明文。thinking の文字数の範囲も書くが、判定は
    `output_tokens` だけで行う (決めごとの 6。issue #131)。

    `glm47` のパーサーは、思考が無効なら思考のブロックを作らないため、6 番目の
    `thinking_chars` は、どちらの構成でも 0 になる (公式テンプレートの構成に送った場合を含む)。
    したがって効いたかどうかの区別には使えない (1 番目の `none` は思考が有効なので 0 にならない)。
    効いたかどうかは、`output_tokens` が 1 番目より減るかどうかで読む。
    """
    candidate = [trial for trial in trials if trial.variant == "chat_template_off"]
    baseline = [trial for trial in trials if trial.variant == "none"]
    chars_c = _metric_range(candidate, lambda trial: trial.thinking_chars)
    chars_b = _metric_range(baseline, lambda trial: trial.thinking_chars)
    tokens_c = _metric_range(candidate, lambda trial: trial.output_tokens)
    tokens_b = _metric_range(baseline, lambda trial: trial.output_tokens)
    verdict = _compare_ranges(tokens_c, tokens_b)
    note = {
        "candidate_below": "減った (期待どおり)",
        "candidate_above": "増えた (期待と違う)",
        "overlap": "変わらなかった (期待と違う)",
        None: "判定できない",
    }[verdict]
    return (
        f"6 対 1: thinking 文字数 {_range_text(chars_c)} 対 {_range_text(chars_b)}、"
        f"output_tokens {_range_text(tokens_c)} 対 {_range_text(tokens_b)} ({note})"
    )


def _judge(trials: Sequence[ThinkingTrial]) -> tuple[bool | None, str]:
    """`effective` と `detail` を決める (決めごとの 6)。"""
    second_verdict, second_text = _judge_pair(trials, "output_config_low", "none")
    third_verdict, third_text = _judge_pair(trials, "chat_template_low", "none")
    fourth_verdict, fourth_text = _judge_pair(trials, "output_config_medium", "none")

    effective: bool | None
    if second_verdict == "disjoint" or third_verdict == "disjoint":
        effective = True
    elif second_verdict == "overlap" and third_verdict == "overlap":
        effective = False
    else:
        effective = None

    plain_note: dict[_CombinedVerdict | None, str] = {
        "disjoint": "重ならない",
        "overlap": "重なる",
        None: "判定できない",
    }
    fourth_note: dict[_CombinedVerdict | None, str] = {
        "disjoint": "重ならない (期待と違う)",
        "overlap": "重なる (期待どおり)",
        None: "判定できない",
    }
    round_count = max((trial.trial_index for trial in trials), default=0)
    parts = [
        f"6 通りを、回ごとに 1〜6 の順で {round_count} 回ずつ送った",
        f"2 対 1: {second_text} ({plain_note[second_verdict]})",
        f"3 対 1: {third_text} ({plain_note[third_verdict]})",
        f"4 対 1: {fourth_text} ({fourth_note[fourth_verdict]})",
        _fifth_text(trials),
        _sixth_text(trials),
    ]
    if effective is True:
        parts.append("2 か 3 のどちらかで範囲が重ならないので、効いたと言える")
    elif effective is False:
        parts.append("2 も 3 も判定でき、どちらも重なるので、効いたとは言えない")
    else:
        parts.append(
            "値の欠け、HTTP の失敗、または max_tokens での打ち切りのため、"
            "効いたかどうかを判定できない"
        )
    return effective, "。".join(parts)


# --- 置き場所と書き出し --------------------------------------------------------


def _run_dir(var_root: Path, started_at: datetime) -> Path:
    """記録の置き場所 (この module の決めごとの 3)。"""
    stamp = started_at.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)
    return var_root / f"{stamp}-{COMMAND_NAME}"


def _write_result(run_dir: Path, outcome: ThinkingOutcome) -> None:
    """確かめの要約を `result.json` に書く。"""
    text = json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    (run_dir / RESULT_FILE_NAME).write_text(text + "\n", encoding="utf-8")


def _build_outcome(model: str, trials: Sequence[ThinkingTrial]) -> ThinkingOutcome:
    effective, detail = _judge(trials)
    return ThinkingOutcome(model=model, trials=tuple(trials), effective=effective, detail=detail)


# --- 入口 -----------------------------------------------------------------------


def run_thinking(
    *,
    base_url: str,
    model: str,
    var_root: Path,
    trials: int = DEFAULT_TRIALS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    messages: Sequence[Mapping[str, object]] | None = None,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] | None = None,
    report: TextIO | None = None,
) -> ThinkingOutcome:
    """thinking の深さの渡し方を確かめる (`serve thinking`。requirements 9.2、10.5)。

    同じ、合成の固定の会話、`temperature 0` で、6 通り (`_VARIANTS_ORDER`) を `trials` 回ずつ、
    `{base_url}/v1/messages` に送る (ストリームでない応答)。**Spark には触らない** (HTTP だけ)。

    引数:
        base_url: head の推論サーバーの URL (`http://` か `https://`、ホストを持つこと)。
        model: 送る `model` の値 (空でないこと)。
        var_root: 記録の置き場所の根 (`serving/var/`)。
        trials: 通りごとに送る回数 (bool を除く本物の int、1 以上)。
        max_tokens: 応答の長さの上限 (bool を除く本物の int、1 以上)。
        timeout_s: 1 要求あたりの時間切れ (有限の正の数)。
        messages: 差し替える会話。省くと `_CONVERSATION` を使う (渡す場合は空でないこと)。
        client: HTTP のクライアント。省くと `lifecycle.new_client(timeout_s)` で作り、この
            関数の中で閉じる (呼ぶ側が渡したクライアントは閉じない)。
        now: 壁の時計 (時差付きの `datetime` を返す)。既定は `datetime.now(UTC)`。
        report: 進捗を知らせる先 (**数値と印だけ**。requirements 10.5)。既定は `sys.stderr`。

    返り値:
        `ThinkingOutcome`。**終了コードは、5.1 が `effective` から決める** (module の docstring
        の「5.1 への手がかり」)。

    例外:
        ValueError: 引数のどれかが、決めごとの 9 の条件を満たさないとき (Spark にも推論
            サーバーにも触る前に断る)。
        KeyboardInterrupt: 中断。1 回以上終わっていれば、そこまでの要約を `result.json` に
            書いてから上げ直す (決めごとの 8)。
    """
    _check_base_url(base_url)
    if not model:
        raise ValueError("model は空であってはいけない")
    _check_positive_int(trials, "trials")
    _check_positive_int(max_tokens, "max_tokens")
    _check_positive_finite(timeout_s, "timeout_s")
    if messages is not None and not messages:
        raise ValueError("messages を渡す場合は、空であってはいけない")

    now_fn: Callable[[], datetime] = _default_now if now is None else now
    report_stream: TextIO = _default_report() if report is None else report
    conversation: Sequence[Mapping[str, object]] = _CONVERSATION if messages is None else messages

    started_at = now_fn()
    run_dir = _run_dir(var_root, started_at)
    run_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url}{_MESSAGES_PATH}"
    owns_client = client is None
    http_client = new_client(timeout_s) if client is None else client

    trial_list: list[ThinkingTrial] = []
    try:
        for round_index in range(1, trials + 1):
            for variant in _VARIANTS_ORDER:
                body = _build_body(
                    variant, model=model, max_tokens=max_tokens, messages=conversation
                )
                trial = _send_trial(http_client, url, variant, round_index, body)
                trial_list.append(trial)
                _report(report_stream, _trial_line(trial))
    except KeyboardInterrupt:
        if trial_list:
            outcome = _build_outcome(model, trial_list)
            _write_result(run_dir, outcome)
            _report(report_stream, "中断されたので、ここまでの確かめの要約を書いた")
        raise
    finally:
        if owns_client:
            http_client.close()

    outcome = _build_outcome(model, trial_list)
    _write_result(run_dir, outcome)
    _report(report_stream, f"thinking: {outcome.detail}")
    return outcome
