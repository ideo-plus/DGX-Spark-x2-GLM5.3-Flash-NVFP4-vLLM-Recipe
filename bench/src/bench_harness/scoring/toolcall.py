"""ツール呼び出しの応答を 9 種類に分ける (task 6.3: scoring/toolcall、5.1、6.3)。

`classify_tool_call(result, task, markers)` が、1 つの応答 (`StreamResult`) を
`ToolCallOutcome` の 9 種類のどれか **1 つ** に分ける。純粋な関数で、時計も
環境変数もネットワークも読まない。依存するのは標準ライブラリ、pydantic、
`jsonschema`、`bench_harness.types` だけである (design.md の依存の向き)。

## 検査の順序 (design.md scoring/toolcall)

上から順に、**最初に当てはまった** 1 つになる。下の表 (`_CHECKS`) が、
design.md の 1〜9 の並びをそのまま写したものである。

| # | 当てはまる条件 | 種類 |
|---|---|---|
| 1 | `result.error` がある | `REQUEST_FAILED` |
| 2 | ブロックが 1 つもない、または `stop_reason` が `max_tokens` | `EMPTY_OR_TRUNCATED` |
| 3 | `tool_use` がなく、本文に記法の目印が含まれる | `MARKUP_LEAKED` |
| 4 | `tool_use` のブロックがない | `NO_CALL` |
| 5 | 最初の `tool_use` の名前が、渡したツールの定義にない | `UNKNOWN_TOOL` |
| 6 | 引数が JSON として読めない | `ARGS_UNPARSEABLE` |
| 7 | 引数が、そのツールの `input_schema` に合わない | `ARGS_SCHEMA_INVALID` |
| 8 | ツールの名前か引数が正解と違う、または `tool_use` が 2 つ以上 | `WRONG_CALL` |
| 9 | それ以外 | `CORRECT` |

それぞれの検査は、**手前の検査が当たらなかったこと**を前提にする (前提を自分で
確かめ直さない)。たとえば 5 は「`tool_use` が 1 つ以上ある」(4 が当たらなかった)
ことを前提にするので、`tool_use` が 1 つもない応答をここに渡すと
`UNKNOWN_TOOL` になる。順序を入れ替えると結果が変わるということであり、
`tests/unit/test_scoring_toolcall.py` が、隣り合う 8 組すべてについて、入れ
替えると落ちる例を持っている。

`ToolCallOutcome` の**宣言の順**は、要求 6.3 の文の並びであって、この検査の順序
ではない (型は凍結されているので直せない)。順序の正本は、この module の
`_CHECKS` と design.md である。

## 決めたこと (design.md が書いていない点は、厳しいほうに寄せる)

正確さの割合を、あとから甘くできない側に倒す。

- **引数の比較は、完全な一致**である。正解 (`ToolTask.expected_input`) と、鍵の
  集合も値も同じでなければ `WRONG_CALL` にする。定義 (`input_schema`) にある
  任意の引数でも、正解に入っていない引数を足せば `WRONG_CALL` である
- **JSON の型を混ぜない**。`True` と `1`、`1` と `1.0`、`"5"` と `5` は、どれも
  違う値として扱う (Python の `True == 1`、`1 == 1.0` に引きずられない)
- **文字列は、そのまま比べる**。大文字小文字、前後の空白、Unicode の正規化
  (NFC/NFKC)、経路の区切りの書き換えは、いっさいしない
- **記法の目印は、`text` のブロックだけで探す**。design.md は「本文」としか書いて
  いないが、`thinking` を含めるかどうかで変わるのは `MARKUP_LEAKED` と `NO_CALL`
  の内訳だけで、崩れた割合 (6.4) も正確さ (5.1) も変わらない。モデルが人に見せる
  本文に漏れたことを見る検査なので、狭いほう (`text` だけ) を採る
- **空の目印は無視する**。`TargetDef.tool_markup_markers` は空の文字列を禁じて
  いないが、空の目印はすべての本文に当たってしまい、検査が意味を失う

## 投げるもの、投げないもの

応答 (`StreamResult`) は対象サーバーから来る外の値なので、**どんな形でも投げ
ない** (名前が `None`、引数が写像でない、深い入れ子、極端に長い値、`tool_use` の
あとに本文、など)。投げるのは、課題 (`ToolTask`) の側が壊れているときだけで、
そのときは `ToolTaskError` を出して止まる (`fail closed`)。モデルのせいにして
数えてしまうと、正確さの数字が静かに歪むためである。

- 目録 (`task.tools`) のスキーマが JSON Schema (Draft 2020-12) として正しくない
- 目録に同じ名前のツールが 2 つある
- 正解のツール (`expected_tool`) が目録にない
- 正解の引数 (`expected_input`) が、そのツールのスキーマに合わない
  (このまま測ると、正解どおりの応答が `ARGS_SCHEMA_INVALID` になってしまう)

GLM と `glm47` のパーサーの組み合わせでは、壊れた記法はサーバーの中で落とされる
ので、`MARKUP_LEAKED` と `ARGS_UNPARSEABLE` は出にくい (research.md)。9 種類とも
実装するが、結果を読むときはこの偏りを念頭に置く。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

# jsonschema は型の注釈を配っていない (`py.typed` がない)。開発の依存に
# `types-jsonschema` を足したら、この 2 つの `type: ignore` は外すこと。
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError, ValidationError  # type: ignore[import-untyped]
from pydantic import JsonValue

from bench_harness.types import (
    ContentBlock,
    StreamResult,
    ToolCallOutcome,
    ToolCallVerdict,
    ToolDef,
    ToolTask,
)

__all__ = [
    "ToolTaskError",
    "classify_tool_call",
]

MAX_DETAIL_CHARS: Final[int] = 200
"""`ToolCallVerdict.detail` の長さの上限。応答の本文をそのまま持ち回らない。"""

_MAX_LABEL_CHARS: Final[int] = 40
"""説明に入れる名前や鍵の、1 つあたりの長さの上限。"""

_MAX_LABELS: Final[int] = 5
"""説明に並べる鍵の数の上限 (残りは件数で示す)。"""

_VALIDATOR_CACHE_SIZE: Final[int] = 256
"""スキーマごとに 1 回だけ検証器を組み立てるための、覚えておく数。"""

_TRUNCATED_MAX_TOKENS: Final[str] = "max_tokens"
"""途中で切れたことを表す `stop_reason`。"""


class ToolTaskError(Exception):
    """課題 (`ToolTask`) の側の誤り。モデルの応答のせいにせず、ここで止める。"""


def classify_tool_call(result: StreamResult, task: ToolTask, markers: list[str]) -> ToolCallVerdict:
    """1 つの応答を、9 種類のどれか 1 つに分ける (design.md scoring/toolcall)。

    `markers` は、対象サーバーの定義 (`TargetDef.tool_markup_markers`) から
    受け取る、呼び出しの記法の目印である。空の文字列は無視する。

    応答がどんな形でも投げない。課題が壊れているときだけ `ToolTaskError` を
    投げる (module の docstring を参照)。
    """
    response = _Response(
        result=result,
        task=task,
        catalog=_catalog(task),
        markers=tuple(marker for marker in markers if marker),
        tool_uses=tuple(block for block in result.blocks if block.type == "tool_use"),
        text="".join(block.text or "" for block in result.blocks if block.type == "text"),
    )
    for outcome, check in _CHECKS:
        detail = check(response)
        if detail is not None:
            return ToolCallVerdict(outcome=outcome, detail=_short(detail))
    return ToolCallVerdict(
        outcome=ToolCallOutcome.CORRECT, detail=_short(f"tool={_label(task.expected_tool)}")
    )


@dataclass(frozen=True)
class _Response:
    """1 つの応答の、検査に使う見方。検査ごとに数え直さないよう、先に作る。"""

    result: StreamResult
    task: ToolTask
    catalog: Mapping[str, ToolDef]
    markers: tuple[str, ...]
    tool_uses: tuple[ContentBlock, ...]
    text: str

    @property
    def first_tool_use(self) -> ContentBlock | None:
        """最初の `tool_use` のブロック。1 つもなければ `None`。"""
        return self.tool_uses[0] if self.tool_uses else None


# --- 検査 (design.md の 1〜9 と、同じ並び) ----------------------------------


def _check_request_failed(response: _Response) -> str | None:
    """1. `result.error` がある → `REQUEST_FAILED`。

    途中までの本文や呼び出しが残っていても、中身では分けない。要求そのものの
    失敗はモデルの崩れではなく、割合の分母から外すものだからである (6.4)。
    """
    error = response.result.error
    if error is None:
        return None
    status = "" if error.http_status is None else f" status={error.http_status}"
    return f"kind={error.kind}{status}"


def _check_empty_or_truncated(response: _Response) -> str | None:
    """2. ブロックが 1 つもない、または `stop_reason` が `max_tokens`。

    `max_tokens` で切れた応答は、呼び出しの形が整っていても途中で切れたものと
    して扱う (モデルが自分で終わっていない)。
    """
    reasons: list[str] = []
    if not response.result.blocks:
        reasons.append("no_blocks")
    if response.result.stop_reason == _TRUNCATED_MAX_TOKENS:
        reasons.append(f"stop_reason={_TRUNCATED_MAX_TOKENS}")
    if not reasons:
        return None
    return " ".join(reasons)


def _check_markup_leaked(response: _Response) -> str | None:
    """3. `tool_use` がなく、本文に記法の目印が含まれる → `MARKUP_LEAKED`。

    当てはまった目印は、`markers` の並びの最初のものを説明に入れる (並びだけで
    決まるので、同じ入力からは必ず同じ説明になる)。
    """
    if response.tool_uses:
        return None
    for marker in response.markers:
        if marker in response.text:
            return f"marker={_label(marker)}"
    return None


def _check_no_call(response: _Response) -> str | None:
    """4. `tool_use` のブロックがない → `NO_CALL`。"""
    if response.tool_uses:
        return None
    return f"blocks={_block_counts(response.result.blocks)}"


def _check_unknown_tool(response: _Response) -> str | None:
    """5. 最初の `tool_use` の名前が、渡したツールの定義にない → `UNKNOWN_TOOL`。

    4 が当たらなかったこと (`tool_use` が 1 つ以上あること) を前提にする。名前が
    `None` (文字列でない名前をクライアントが落とした) と空の文字列は、目録の
    名前 (`ToolDef.name` は 1 文字以上) と一致しないので、ここに落ちる。
    """
    block = response.first_tool_use
    name = None if block is None else block.tool_name
    if name is not None and name in response.catalog:
        return None
    return f"name={_label(name)}"


def _check_args_unparseable(response: _Response) -> str | None:
    """6. 引数が JSON として読めない → `ARGS_UNPARSEABLE`。

    `tool_input is None` が、その唯一の合図である (注 2.1)。生の文字列
    (`tool_input_raw`) を自分で解析し直さない。断片が 1 つも来なかった呼び出しは
    クライアントが `{}` にするので、ここではなく 7 で分かれる。
    """
    block = response.first_tool_use
    if block is None or block.tool_input is not None:
        return None
    raw = block.tool_input_raw or ""
    return f"tool={_label(block.tool_name)} raw_chars={len(raw)}"


def _check_args_schema_invalid(response: _Response) -> str | None:
    """7. 引数が、そのツールの `input_schema` に合わない → `ARGS_SCHEMA_INVALID`。

    5 が当たらなかったこと (名前が目録にあること) と、6 が当たらなかったこと
    (引数が写像であること) を前提にする。
    """
    block = response.first_tool_use
    if block is None:
        return None
    tool = response.catalog.get(block.tool_name or "")
    if tool is None:
        return None
    violation = _first_violation(tool, block.tool_input)
    if violation is None:
        return None
    return f"tool={_label(tool.name)} {violation}"


def _check_wrong_call(response: _Response) -> str | None:
    """8. 名前か引数が正解と違う、または `tool_use` が 2 つ以上 → `WRONG_CALL`。"""
    block = response.first_tool_use
    if block is None:
        return None
    reasons: list[str] = []
    if len(response.tool_uses) > 1:
        reasons.append(f"extra_calls={len(response.tool_uses) - 1}")
    expected_tool = response.task.expected_tool
    if block.tool_name != expected_tool:
        reasons.append(f"tool={_label(block.tool_name)} expected={_label(expected_tool)}")
    else:
        mismatch = _argument_mismatch(response.task.expected_input, block.tool_input)
        if mismatch is not None:
            reasons.append(mismatch)
    if not reasons:
        return None
    return " ".join(reasons)


_Check = Callable[[_Response], str | None]
"""1 つの検査。当てはまれば説明を、当てはまらなければ `None` を返す。"""

_CHECKS: Final[tuple[tuple[ToolCallOutcome, _Check], ...]] = (
    (ToolCallOutcome.REQUEST_FAILED, _check_request_failed),  # 1
    (ToolCallOutcome.EMPTY_OR_TRUNCATED, _check_empty_or_truncated),  # 2
    (ToolCallOutcome.MARKUP_LEAKED, _check_markup_leaked),  # 3
    (ToolCallOutcome.NO_CALL, _check_no_call),  # 4
    (ToolCallOutcome.UNKNOWN_TOOL, _check_unknown_tool),  # 5
    (ToolCallOutcome.ARGS_UNPARSEABLE, _check_args_unparseable),  # 6
    (ToolCallOutcome.ARGS_SCHEMA_INVALID, _check_args_schema_invalid),  # 7
    (ToolCallOutcome.WRONG_CALL, _check_wrong_call),  # 8
)
"""design.md scoring/toolcall の 1〜8 と、同じ並び (9 は、どれも当たらなかった場合)。"""


# --- 引数の比較 -------------------------------------------------------------


def _argument_mismatch(
    expected: Mapping[str, JsonValue], actual: Mapping[str, JsonValue] | None
) -> str | None:
    """正解の引数と実際の引数の食い違いを、短い説明で返す (同じなら `None`)。

    鍵の集合と値が、完全に一致したときだけ正解にする。値は JSON の型に厳密に
    比べる (module の docstring の「決めたこと」を参照)。説明に入れるのは鍵の
    名前だけで、値は入れない (長い応答の本文を持ち回らないため)。
    """
    if actual is None:
        return "args=not_object"
    missing = sorted(key for key in expected if key not in actual)
    extra = sorted(key for key in actual if key not in expected)
    differs = sorted(
        key
        for key, value in expected.items()
        if key in actual and not _json_equal(value, actual[key])
    )
    parts = [
        f"{name}=[{_labels(keys)}]"
        for name, keys in (("missing", missing), ("extra", extra), ("differs", differs))
        if keys
    ]
    if not parts:
        return None
    return "args " + " ".join(parts)


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    """2 つの JSON の値が、型まで含めて同じかを返す。

    JSON の真偽値、整数、小数、文字列、`null` は、たがいに別の値である
    (Python では `True == 1`、`1 == 1.0` だが、JSON の値としては違う)。写像と
    配列は、中身まで見る。深い入れ子でも `RecursionError` にならないよう、
    再帰せずに自前の山で回す。
    """
    stack: list[tuple[JsonValue, JsonValue]] = [(left, right)]
    while stack:
        first, second = stack.pop()
        if isinstance(first, bool) or isinstance(second, bool):
            if not (isinstance(first, bool) and isinstance(second, bool)) or first is not second:
                return False
        elif isinstance(first, dict):
            if not isinstance(second, dict) or len(first) != len(second):
                return False
            for key, value in first.items():
                if key not in second:
                    return False
                stack.append((value, second[key]))
        elif isinstance(first, list):
            if not isinstance(second, list) or len(first) != len(second):
                return False
            stack.extend(zip(first, second, strict=True))
        elif isinstance(first, int):
            if not isinstance(second, int) or isinstance(second, bool) or first != second:
                return False
        elif isinstance(first, float):
            # 小数は小数とだけ比べる (NaN はそれ自身とも一致しない)
            if not isinstance(second, float) or first != second:
                return False
        elif isinstance(first, str):
            if not isinstance(second, str) or first != second:
                return False
        elif first is None:
            if second is not None:
                return False
        else:  # pragma: no cover - JsonValue はこれ以外の型を持たない
            return False
    return True


# --- スキーマ (jsonschema) --------------------------------------------------


def _first_violation(tool: ToolDef, arguments: JsonValue) -> str | None:
    """引数がツールの定義に合わなければ、いちばん先の違反を短い説明で返す。

    どの違反を選ぶかは、場所と検査の名前と本文の並びで決める。`iter_errors` の
    返す順や、写像の鍵の並びに左右されないようにするためである。
    """
    validator = _validator_for(tool)
    try:
        errors = list(validator.iter_errors(arguments))
    except RecursionError:
        return "args too deeply nested for the schema"
    except ValidationError as exc:  # pragma: no cover - iter_errors は投げない
        return f"{exc.json_path}: {exc.message}"
    except Exception as exc:  # スキーマの側の問題 (解決できない $ref など)
        raise ToolTaskError(f"ツール {tool.name} の input_schema で検証できない: {exc}") from exc
    if not errors:
        return None
    worst = min(errors, key=lambda error: (error.json_path, str(error.validator), error.message))
    return f"{worst.json_path}: {worst.message}"


def _validator_for(tool: ToolDef) -> Draft202012Validator:
    """ツールの `input_schema` の検証器を返す (同じスキーマなら組み立て直さない)。"""
    try:
        schema_json = json.dumps(tool.input_schema, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - JsonValue は JSON になる
        raise ToolTaskError(
            f"ツール {tool.name} の input_schema を JSON にできない: {exc}"
        ) from exc
    try:
        return _compile_schema(schema_json)
    except SchemaError as exc:
        raise ToolTaskError(
            f"ツール {tool.name} の input_schema が JSON Schema として正しくない: {exc.message}"
        ) from exc


@lru_cache(maxsize=_VALIDATOR_CACHE_SIZE)
def _compile_schema(schema_json: str) -> Draft202012Validator:
    """正規の形にしたスキーマの文字列から、検証器を組み立てる。

    スキーマそのものが JSON Schema (Draft 2020-12) として正しいことを先に
    確かめる。正しくなければ `SchemaError` を投げ、応答を分類しない
    (道具の側の誤りを、モデルの崩れとして数えないため)。
    """
    schema = json.loads(schema_json)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _catalog(task: ToolTask) -> Mapping[str, ToolDef]:
    """課題のツールの目録を、名前で引ける形にし、課題の側の誤りをここで弾く。"""
    catalog: dict[str, ToolDef] = {}
    for tool in task.tools:
        if tool.name in catalog:
            raise ToolTaskError(f"課題の tools に、同じ名前のツールが 2 つある: {tool.name}")
        catalog[tool.name] = tool
        _validator_for(tool)  # スキーマの誤りは、分類を始める前に見つける
    expected = catalog.get(task.expected_tool)
    if expected is None:
        raise ToolTaskError(
            f"課題の expected_tool が tools にない: {task.expected_tool} (tools={sorted(catalog)})"
        )
    violation = _first_violation(expected, dict(task.expected_input))
    if violation is not None:
        raise ToolTaskError(
            f"課題の expected_input が {expected.name} の input_schema に合わない: {violation}"
        )
    return catalog


# --- 説明の組み立て ---------------------------------------------------------


def _block_counts(blocks: Sequence[ContentBlock]) -> str:
    """ブロックの種類ごとの数 (`text=1,thinking=0` のような形)。"""
    counts = {kind: 0 for kind in ("text", "thinking", "tool_use")}
    for block in blocks:
        counts[block.type] += 1
    return ",".join(f"{kind}={count}" for kind, count in counts.items())


def _labels(keys: Sequence[str]) -> str:
    """鍵の名前を並べる (多すぎるときは、残りを件数で示す)。"""
    shown = ",".join(_label(key) for key in keys[:_MAX_LABELS])
    if len(keys) <= _MAX_LABELS:
        return shown
    return f"{shown},+{len(keys) - _MAX_LABELS}"


def _label(value: str | None) -> str:
    """名前や鍵を、説明に入れられる長さに切り詰める。"""
    if value is None:
        return "None"
    if len(value) <= _MAX_LABEL_CHARS:
        return value
    return value[:_MAX_LABEL_CHARS] + "…"


def _short(text: str) -> str:
    """説明を 1 行に畳み、上限の長さまで切り詰める (末尾の `…` を含めて上限)。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_DETAIL_CHARS:
        return collapsed
    return collapsed[: MAX_DETAIL_CHARS - 1] + "…"
