"""scoring/toolcall の試験 (task 6.3)。

`classify_tool_call` は、1 つの応答を 9 種類のどれか 1 つに分ける。ここで
確かめるのは、次の 4 つである。

1. 9 種類それぞれの最小の例が、期待の種類に分かれること
2. 複数の検査に当てはまる応答が、design.md の**順序のとおり**、先の検査の
   種類になること (隣り合う 8 組すべてについて、入れ替えると結果が変わる例を
   置く。順序の入れ替えを、試験が必ず見つける)
3. 引数の比較が、JSON の型に厳密であること (`True` と `1`、`1` と `1.0`、
   `"5"` と `5` を混ぜない)。過不足のある引数、鍵の並び、日本語の文字列
4. 外から来る応答では、どんな形でも例外を投げないこと。投げるのは、課題
   (`ToolTask`) の側が壊れているときだけであること
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from bench_harness.scoring import toolcall as toolcall_module
from bench_harness.scoring.toolcall import ToolTaskError, classify_tool_call
from bench_harness.types import (
    ContentBlock,
    RequestError,
    StreamResult,
    StreamTiming,
    ToolCallOutcome,
    ToolDef,
    ToolTask,
)

# --- 助け -------------------------------------------------------------------

MARKERS: Final[list[str]] = ["<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>"]
"""対象サーバーの定義の既定値 (`TargetDef.tool_markup_markers`) と同じ目印。"""

_WRITE_FILE_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "append": {"type": "boolean"},
    },
    "required": ["path", "content"],
}
"""`additionalProperties` を書かない (定義にない引数は、スキーマでは通る)。"""


def _timing() -> StreamTiming:
    return StreamTiming(sent_at_utc=datetime.now(UTC), sent_at_ns=0, end_ns=1)


def _result(
    blocks: Sequence[ContentBlock] = (),
    *,
    stop_reason: str | None = "tool_use",
    error: RequestError | None = None,
) -> StreamResult:
    return StreamResult(timing=_timing(), blocks=list(blocks), stop_reason=stop_reason, error=error)


def _tool_use(
    name: str | None, tool_input: dict[str, JsonValue] | None, *, raw: str | None = None
) -> ContentBlock:
    """`tool_use` のブロックを作る。

    `raw` を省くと、`tool_input` を JSON にしたものを生の引数にする (クライアント
    2.1 が作る形)。`tool_input=None` は「JSON として読めなかった」の合図。
    """
    if raw is None:
        raw = "" if tool_input is None else json.dumps(tool_input, ensure_ascii=False)
    return ContentBlock(type="tool_use", tool_name=name, tool_input_raw=raw, tool_input=tool_input)


def _text(text: str) -> ContentBlock:
    return ContentBlock(type="text", text=text)


def _thinking(text: str) -> ContentBlock:
    return ContentBlock(type="thinking", text=text)


def _task(
    *,
    expected_tool: str = "write_file",
    expected_input: dict[str, JsonValue] | None = None,
    tools: Sequence[ToolDef] | None = None,
) -> ToolTask:
    """手で書いた課題 (6.2 の生成器は使わない)。似た名前のツールを目録に混ぜる。"""
    if expected_input is None:
        expected_input = {"path": "src/main.py", "content": "print(1)"}
    if tools is None:
        tools = [
            ToolDef(
                name="write_file",
                description="1 つのファイルを書く",
                input_schema=_WRITE_FILE_SCHEMA,
            ),
            ToolDef(
                name="write_files",
                description="複数のファイルを書く (紛らわしい名前)",
                input_schema={
                    "type": "object",
                    "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                    "required": ["paths"],
                },
            ),
            ToolDef(
                name="read_file",
                description="1 つのファイルを読む",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ]
    return ToolTask(
        prompt="src/main.py に print(1) と書いてください",
        tools=list(tools),
        expected_tool=expected_tool,
        expected_input=expected_input,
    )


def _correct_block() -> ContentBlock:
    return _tool_use("write_file", {"path": "src/main.py", "content": "print(1)"})


def _classify(
    result: StreamResult, task: ToolTask | None = None, markers: list[str] | None = None
) -> ToolCallOutcome:
    verdict = classify_tool_call(result, task or _task(), MARKERS if markers is None else markers)
    return verdict.outcome


# --- 9 種類の最小の例 -------------------------------------------------------

_NINE_EXAMPLES: Final[tuple[tuple[str, StreamResult, ToolCallOutcome], ...]] = (
    (
        "correct",
        _result([_tool_use("write_file", {"path": "src/main.py", "content": "print(1)"})]),
        ToolCallOutcome.CORRECT,
    ),
    (
        "no_call",
        _result([_text("ファイルを書く必要はないと思います。")], stop_reason="end_turn"),
        ToolCallOutcome.NO_CALL,
    ),
    (
        "unknown_tool",
        _result([_tool_use("write_file_v2", {"path": "src/main.py", "content": "print(1)"})]),
        ToolCallOutcome.UNKNOWN_TOOL,
    ),
    (
        "args_unparseable",
        _result([_tool_use("write_file", None, raw='{"path": "src/main.py", "cont')]),
        ToolCallOutcome.ARGS_UNPARSEABLE,
    ),
    (
        "args_schema_invalid",
        _result([_tool_use("write_file", {"path": 12, "content": "print(1)"})]),
        ToolCallOutcome.ARGS_SCHEMA_INVALID,
    ),
    (
        "markup_leaked",
        _result(
            [_text("<tool_call>write_file\n<arg_key>path</arg_key>")],
            stop_reason="end_turn",
        ),
        ToolCallOutcome.MARKUP_LEAKED,
    ),
    (
        "wrong_call",
        _result([_tool_use("read_file", {"path": "src/main.py"})]),
        ToolCallOutcome.WRONG_CALL,
    ),
    ("empty_or_truncated", _result([], stop_reason="end_turn"), ToolCallOutcome.EMPTY_OR_TRUNCATED),
    (
        "request_failed",
        _result(
            [], stop_reason=None, error=RequestError(kind="timeout_idle", message="無音が続いた")
        ),
        ToolCallOutcome.REQUEST_FAILED,
    ),
)


@pytest.mark.parametrize(
    ("result", "expected"),
    [pytest.param(result, expected, id=name) for name, result, expected in _NINE_EXAMPLES],
)
def test_minimal_example_of_each_outcome(result: StreamResult, expected: ToolCallOutcome) -> None:
    """9 種類のそれぞれの最小の例が、期待の種類に分かれる (6.3)。"""
    assert _classify(result) is expected


def test_nine_examples_cover_every_outcome() -> None:
    """例が 9 種類を漏れなく覆っている (種類を足したら、この試験が落ちる)。"""
    covered = {expected for _, _, expected in _NINE_EXAMPLES}
    assert covered == set(ToolCallOutcome)


def test_verdict_detail_is_filled_and_short() -> None:
    """判定には、人が生データを読んで理由がわかる短い説明が付く。"""
    for name, result, _ in _NINE_EXAMPLES:
        verdict = classify_tool_call(result, _task(), MARKERS)
        assert verdict.detail, f"{name}: detail が空"
        assert len(verdict.detail) <= 200, f"{name}: detail が長すぎる"
        assert "\n" not in verdict.detail


# --- 順序: 隣り合う 8 組 ----------------------------------------------------
#
# design.md scoring/toolcall の順序 (1〜9)。各試験は、隣り合う 2 つの検査の
# 両方に当てはまる応答を作り、先の検査の種類になることを確かめる。入れ替えると
# 必ず落ちる。


def test_order_1_request_failed_beats_2_empty_or_truncated() -> None:
    """失敗した要求は、ブロックがなくても `REQUEST_FAILED` (要求の失敗は崩れに数えない)。"""
    result = _result([], stop_reason=None, error=RequestError(kind="protocol", message="切れた"))
    assert _classify(result) is ToolCallOutcome.REQUEST_FAILED


def test_order_2_empty_or_truncated_beats_3_markup_leaked() -> None:
    """途中で切れた応答に記法が残っていても、`EMPTY_OR_TRUNCATED` が先。"""
    result = _result([_text("<tool_call>write_file<arg_key>path")], stop_reason="max_tokens")
    assert _classify(result) is ToolCallOutcome.EMPTY_OR_TRUNCATED


def test_order_3_markup_leaked_beats_4_no_call() -> None:
    """`tool_use` がなく、記法が本文に漏れていれば `MARKUP_LEAKED` が先。"""
    result = _result([_text("では <tool_call>write_file を呼びます")], stop_reason="end_turn")
    assert _classify(result) is ToolCallOutcome.MARKUP_LEAKED


def test_order_4_no_call_beats_5_unknown_tool() -> None:
    """`tool_use` が 1 つもない応答は、「名前が定義にない」ではなく `NO_CALL`。"""
    result = _result([_text("なにもしません")], stop_reason="end_turn")
    assert _classify(result) is ToolCallOutcome.NO_CALL


def test_order_5_unknown_tool_beats_6_args_unparseable() -> None:
    """定義にない名前で、引数も読めないときは `UNKNOWN_TOOL` が先。"""
    result = _result([_tool_use("write_file_v2", None, raw="{broken")])
    assert _classify(result) is ToolCallOutcome.UNKNOWN_TOOL


def test_order_6_args_unparseable_beats_7_args_schema_invalid() -> None:
    """引数が読めないときは、スキーマに照らす前に `ARGS_UNPARSEABLE`。"""
    result = _result([_tool_use("write_file", None, raw='{"path": ')])
    assert _classify(result) is ToolCallOutcome.ARGS_UNPARSEABLE


def test_order_7_args_schema_invalid_beats_8_wrong_call() -> None:
    """定義に合わない引数は、正解と違っていても `ARGS_SCHEMA_INVALID` が先。"""
    result = _result([_tool_use("write_file", {"path": "other.py"})])  # content がない
    assert _classify(result) is ToolCallOutcome.ARGS_SCHEMA_INVALID


def test_order_8_wrong_call_beats_9_correct() -> None:
    """正しい呼び出しが含まれていても、`tool_use` が 2 つ以上あれば `WRONG_CALL`。"""
    result = _result([_correct_block(), _tool_use("read_file", {"path": "src/main.py"})])
    verdict = classify_tool_call(result, _task(), MARKERS)
    assert verdict.outcome is ToolCallOutcome.WRONG_CALL
    assert "extra_calls=1" in verdict.detail


def test_the_first_tool_use_is_the_one_that_is_judged() -> None:
    """design.md の 5: 判定するのは**最初の** `tool_use`。最後のものではない。

    1 つ目が目録にないツール、2 つ目が正解どおりの呼び出し。最後の呼び出しで判定
    すると `WRONG_CALL` (extra_calls) になるので、取り違えを検出できる。
    """
    result = _result([_tool_use("delete_everything", {"path": "src/main.py"}), _correct_block()])
    assert _classify(result) is ToolCallOutcome.UNKNOWN_TOOL


def test_schema_invalid_wins_over_wrong_call_for_extra_properties() -> None:
    """`additionalProperties: false` の定義では、余計な引数はスキーマで弾かれる。"""
    strict_schema: dict[str, JsonValue] = dict(_WRITE_FILE_SCHEMA)
    strict_schema["additionalProperties"] = False
    task = _task(tools=[ToolDef(name="write_file", input_schema=strict_schema)])
    result = _result(
        [_tool_use("write_file", {"path": "src/main.py", "content": "print(1)", "mode": "w"})]
    )
    assert _classify(result, task) is ToolCallOutcome.ARGS_SCHEMA_INVALID


# --- 引数の比較 -------------------------------------------------------------


def test_true_is_not_one() -> None:
    """JSON の真偽値と整数を混ぜない (Python の `True == 1` に引きずられない)。"""
    task = _task(expected_input={"path": "a.py", "content": "x", "append": True})
    result = _result([_tool_use("write_file", {"path": "a.py", "content": "x", "append": 1})])
    # スキーマ (append は boolean) にも合わないので、まずそこで分かれる
    assert _classify(result, task) is ToolCallOutcome.ARGS_SCHEMA_INVALID

    loose = _task(
        expected_input={"path": "a.py", "content": "x", "flag": True},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    loose_result = _result([_tool_use("write_file", {"path": "a.py", "content": "x", "flag": 1})])
    assert _classify(loose_result, loose) is ToolCallOutcome.WRONG_CALL


def test_one_is_not_one_point_zero() -> None:
    """整数と小数を混ぜない (`1 == 1.0` に引きずられない)。"""
    task = _task(
        expected_input={"limit": 1},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    result = _result([_tool_use("write_file", {"limit": 1.0})])
    assert _classify(result, task) is ToolCallOutcome.WRONG_CALL


def test_string_five_is_not_number_five() -> None:
    """文字列と数を混ぜない。"""
    task = _task(
        expected_input={"limit": 5},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    result = _result([_tool_use("write_file", {"limit": "5"})])
    assert _classify(result, task) is ToolCallOutcome.WRONG_CALL


def test_nested_values_compare_strictly() -> None:
    """入れ子の中でも、型の厳密さは変わらない。"""
    task = _task(
        expected_input={"opts": {"flags": [True, 2]}},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    same = _result([_tool_use("write_file", {"opts": {"flags": [True, 2]}})])
    assert _classify(same, task) is ToolCallOutcome.CORRECT
    swapped = _result([_tool_use("write_file", {"opts": {"flags": [1, 2]}})])
    assert _classify(swapped, task) is ToolCallOutcome.WRONG_CALL


def test_extra_argument_is_wrong_call() -> None:
    """正解にない引数を足した呼び出しは、正解にしない (厳密な一致)。"""
    result = _result(
        [_tool_use("write_file", {"path": "src/main.py", "content": "print(1)", "append": False})]
    )
    verdict = classify_tool_call(result, _task(), MARKERS)
    assert verdict.outcome is ToolCallOutcome.WRONG_CALL
    assert "append" in verdict.detail


def test_optional_argument_omitted_is_correct() -> None:
    """定義にある任意の引数 (`append`) を省いた、正解どおりの呼び出しは `CORRECT`。"""
    assert _classify(_result([_correct_block()])) is ToolCallOutcome.CORRECT


def test_missing_expected_argument_is_wrong_call() -> None:
    """正解にある引数が足りなければ `WRONG_CALL` (スキーマでは任意の場合)。"""
    task = _task(
        expected_input={"path": "a.py", "content": "x"},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    result = _result([_tool_use("write_file", {"path": "a.py"})])
    verdict = classify_tool_call(result, task, MARKERS)
    assert verdict.outcome is ToolCallOutcome.WRONG_CALL
    assert "content" in verdict.detail


def test_key_order_does_not_matter() -> None:
    """鍵の並びは、結果に影響しない。"""
    result = _result([_tool_use("write_file", {"content": "print(1)", "path": "src/main.py"})])
    assert _classify(result) is ToolCallOutcome.CORRECT


def test_japanese_path_compares_exactly() -> None:
    """日本語の文字列は、そのまま厳密に比べる (正規化も切り詰めもしない)。"""
    task = _task(expected_input={"path": "資料/設計書.md", "content": "見出し"})
    same = _result([_tool_use("write_file", {"path": "資料/設計書.md", "content": "見出し"})])
    assert _classify(same, task) is ToolCallOutcome.CORRECT
    different = _result([_tool_use("write_file", {"path": "資料/設計書.md", "content": "見出 し"})])
    assert _classify(different, task) is ToolCallOutcome.WRONG_CALL


def test_wrong_tool_name_is_wrong_call() -> None:
    """定義にはあるが、呼ぶべきでないツール (紛らわしい名前) は `WRONG_CALL`。"""
    task = _task(
        expected_input={"paths": ["src/main.py"]},
        expected_tool="write_files",
    )
    result = _result([_tool_use("read_file", {"path": "src/main.py"})])
    verdict = classify_tool_call(result, task, MARKERS)
    assert verdict.outcome is ToolCallOutcome.WRONG_CALL
    assert "read_file" in verdict.detail


# --- 外から来る応答で、例外を投げない ---------------------------------------


def test_tool_input_that_is_not_an_object_is_unparseable() -> None:
    """写像でない引数 (配列、文字列、数、null) は、クライアントが `None` にする (注 2.1)。"""
    for raw in ("[1, 2, 3]", '"文字列"', "42", "null"):
        result = _result([_tool_use("write_file", None, raw=raw)])
        assert _classify(result) is ToolCallOutcome.ARGS_UNPARSEABLE


def test_empty_tool_input_is_schema_invalid() -> None:
    """断片が 1 つも来なかった呼び出し (`{}`) は、読めない引数ではなく、定義に合わない引数。"""
    result = _result([_tool_use("write_file", {}, raw="")])
    assert _classify(result) is ToolCallOutcome.ARGS_SCHEMA_INVALID


def test_deeply_nested_input_does_not_raise() -> None:
    """深く入れ子になった引数でも、例外を投げずに分類する。"""
    nested: JsonValue = {"leaf": 1}
    for _ in range(200):
        nested = {"n": nested}
    task = _task(
        expected_input={"opts": {"leaf": 1}},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    result = _result([_tool_use("write_file", {"opts": nested})])
    assert _classify(result, task) is ToolCallOutcome.WRONG_CALL


def test_huge_input_does_not_raise() -> None:
    """鍵の多い引数でも、例外を投げずに分類する。"""
    huge: dict[str, JsonValue] = {f"k{index}": index for index in range(2000)}
    task = _task(
        expected_input={"path": "a.py"},
        tools=[ToolDef(name="write_file", input_schema={"type": "object"})],
    )
    result = _result([_tool_use("write_file", huge)])
    assert _classify(result, task) is ToolCallOutcome.WRONG_CALL


def test_tool_name_of_wrong_type_or_empty_is_unknown_tool() -> None:
    """名前が文字列でなかった (クライアントが `None` にする) 呼び出しと、空の名前。"""
    for name in (None, ""):
        result = _result([_tool_use(name, {"path": "src/main.py", "content": "print(1)"})])
        assert _classify(result) is ToolCallOutcome.UNKNOWN_TOOL


def test_text_before_and_after_the_tool_use_block() -> None:
    """本文とツール呼び出しが混ざっていても、最初の `tool_use` で判定する。"""
    result = _result(
        [
            _thinking("path と content を渡す"),
            _text("書き込みます。"),
            _correct_block(),
            _text("書き込みました。"),
        ]
    )
    assert _classify(result) is ToolCallOutcome.CORRECT


def test_markers_are_looked_for_in_text_blocks_only() -> None:
    """記法の目印は、本文 (text) だけで探す。thinking の中の記法では分けない。"""
    result = _result([_thinking("<tool_call>write_file<arg_key>path")], stop_reason="end_turn")
    assert _classify(result) is ToolCallOutcome.NO_CALL


def test_thinking_only_response_is_no_call() -> None:
    """thinking だけの応答は、ブロックがあるので `NO_CALL` (空ではない)。"""
    result = _result([_thinking("考え中")], stop_reason="end_turn")
    assert _classify(result) is ToolCallOutcome.NO_CALL


def test_truncated_tool_call_with_max_tokens() -> None:
    """`max_tokens` で切れた応答は、呼び出しの形が整っていても `EMPTY_OR_TRUNCATED`。"""
    result = _result([_correct_block()], stop_reason="max_tokens")
    assert _classify(result) is ToolCallOutcome.EMPTY_OR_TRUNCATED


def test_failed_request_is_not_classified_by_its_content() -> None:
    """途中までの本文や呼び出しが残っていても、失敗した要求は `REQUEST_FAILED`。"""
    for blocks in (
        [_correct_block()],
        [_text("<tool_call>write_file")],
        [_tool_use("write_file_v2", None, raw="{brok")],
    ):
        result = _result(
            blocks,
            stop_reason=None,
            error=RequestError(kind="http", http_status=503, message="混んでいる"),
        )
        assert _classify(result) is ToolCallOutcome.REQUEST_FAILED


def test_stop_reason_none_is_handled() -> None:
    """`stop_reason` が `None` の応答 (注 2.1) も、ふつうに分類する。"""
    result = _result([_correct_block()], stop_reason=None)
    assert _classify(result) is ToolCallOutcome.CORRECT


def test_empty_marker_list_never_leaks() -> None:
    """目印が空の一覧なら、記法の漏れでは分けない。"""
    result = _result([_text("<tool_call>write_file")], stop_reason="end_turn")
    assert _classify(result, markers=[]) is ToolCallOutcome.NO_CALL


def test_empty_string_marker_is_ignored() -> None:
    """空の文字列の目印は、すべての本文に当たってしまうので、使わない。"""
    result = _result([_text("ふつうの返事です")], stop_reason="end_turn")
    assert _classify(result, markers=["", "<tool_call>"]) is ToolCallOutcome.NO_CALL


def test_marker_from_the_target_definition_is_used() -> None:
    """記法の目印は、対象サーバーの定義から受け取ったものを使う。"""
    result = _result([_text("[[call]] write_file")], stop_reason="end_turn")
    assert _classify(result, markers=["[[call]]"]) is ToolCallOutcome.MARKUP_LEAKED
    assert _classify(result) is ToolCallOutcome.NO_CALL


def test_two_tool_uses_with_the_same_name_are_extra_calls() -> None:
    """同じ呼び出しが 2 つ来た場合 (識別子は `ContentBlock` に残らない) も `WRONG_CALL`。"""
    result = _result([_correct_block(), _correct_block()])
    assert _classify(result) is ToolCallOutcome.WRONG_CALL


def test_long_and_odd_values_do_not_break_the_detail() -> None:
    """長い名前や長い引数でも、投げず、説明を短く保つ。"""
    result = _result([_tool_use("x" * 5000, {"path": "y" * 20000})])
    verdict = classify_tool_call(result, _task(), MARKERS)
    assert verdict.outcome is ToolCallOutcome.UNKNOWN_TOOL
    assert len(verdict.detail) <= 200


# --- 課題の側の誤りは、モデルのせいにしない ---------------------------------


def test_a_huge_schema_violation_does_not_put_the_body_into_the_detail() -> None:
    """jsonschema の違反の文には、値の実体が入る。説明は 200 文字で切る。

    切らないと、1 MB の引数がそのまま試行のレコード (`trials.jsonl`) に入る。
    """
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    task = _task(
        expected_tool="count",
        expected_input={"a": 1},
        tools=[ToolDef(name="count", input_schema=schema)],
    )
    result = _result([_tool_use("count", {"a": "y" * 1_000_000})])

    verdict = classify_tool_call(result, task, MARKERS)

    assert verdict.outcome is ToolCallOutcome.ARGS_SCHEMA_INVALID
    assert 0 < len(verdict.detail) <= 200


def test_an_unresolvable_ref_in_a_decoy_schema_raises_when_the_decoy_is_called() -> None:
    """囮のツールのスキーマが解決できない `$ref` を持つとき、モデルの誤りにしない。

    `check_schema` は `$ref` を解決しないので、囮が呼ばれて初めて分かる。そのときも
    `WRONG_CALL` に丸めずに、課題の誤りとして投げる (fail closed)。
    """
    decoy = ToolDef(
        name="decoy",
        input_schema={"type": "object", "properties": {"a": {"$ref": "#/$defs/nowhere"}}},
    )
    task = _task(tools=[ToolDef(name="write_file", input_schema=_WRITE_FILE_SCHEMA), decoy])
    result = _result([_tool_use("decoy", {"a": 1})])

    with pytest.raises(ToolTaskError, match="decoy"):
        classify_tool_call(result, task, MARKERS)


def test_invalid_schema_raises() -> None:
    """目録のスキーマが JSON Schema として壊れていたら、投げる (設定の誤り)。"""
    task = _task(tools=[ToolDef(name="write_file", input_schema={"type": "objekt"})])
    with pytest.raises(ToolTaskError, match="input_schema"):
        classify_tool_call(_result([_correct_block()]), task, MARKERS)


def test_expected_tool_not_in_catalog_raises() -> None:
    """正解のツールが目録にないと、どの応答も正解にならないので、投げる。"""
    task = _task(expected_tool="move_file")
    with pytest.raises(ToolTaskError, match="expected_tool"):
        classify_tool_call(_result([_correct_block()]), task, MARKERS)


def test_expected_input_violating_the_schema_raises() -> None:
    """正解の引数がスキーマに合わないと、正しい応答が `ARGS_SCHEMA_INVALID` になるので、投げる。"""
    task = _task(expected_input={"path": "src/main.py"})  # content がない
    with pytest.raises(ToolTaskError, match="expected_input"):
        classify_tool_call(_result([_correct_block()]), task, MARKERS)


def test_duplicate_tool_names_raise() -> None:
    """目録に同じ名前のツールが 2 つあると、どちらの定義で検証するか決まらない。"""
    task = _task(
        tools=[
            ToolDef(name="write_file", input_schema=_WRITE_FILE_SCHEMA),
            ToolDef(name="write_file", input_schema={"type": "object"}),
        ]
    )
    with pytest.raises(ToolTaskError, match="write_file"):
        classify_tool_call(_result([_correct_block()]), task, MARKERS)


def test_a_broken_task_raises_even_when_the_request_failed() -> None:
    """課題が壊れていれば、要求が失敗した応答でも投げる (黙って数えない)。"""
    task = _task(expected_tool="move_file")
    result = _result([], stop_reason=None, error=RequestError(kind="connect"))
    with pytest.raises(ToolTaskError):
        classify_tool_call(result, task, MARKERS)


# --- 何度呼んでも同じ -------------------------------------------------------


def test_same_input_gives_the_same_verdict() -> None:
    """同じ入力からは、種類も説明も同じ判定になる。"""
    for _, result, _ in _NINE_EXAMPLES:
        first = classify_tool_call(result, _task(), MARKERS)
        second = classify_tool_call(result, _task(), MARKERS)
        assert first == second


def test_dict_ordering_does_not_change_the_verdict() -> None:
    """鍵の並びや目録の並びを変えても、判定は変わらない。"""
    task = _task()
    reversed_tools = _task(tools=list(reversed(task.tools)))
    result = _result([_tool_use("write_file", {"content": "print(1)", "path": "src/main.py"})])
    assert classify_tool_call(result, task, MARKERS) == classify_tool_call(
        result, reversed_tools, MARKERS
    )

    invalid = _result([_tool_use("write_file", {"path": 1, "content": 2})])
    flipped = _result([_tool_use("write_file", {"content": 2, "path": 1})])
    assert classify_tool_call(invalid, task, MARKERS) == classify_tool_call(flipped, task, MARKERS)


# --- 依存の向き (design.md) -------------------------------------------------


def test_toolcall_imports_nothing_outside_the_allowed_set() -> None:
    """`scoring` は `types` 以外の `bench_harness` を読み込まない (design.md)。"""
    source = Path(toolcall_module.__file__ or "")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対 import は使わない (依存の向きを見えなくする)"
            assert node.module is not None
            modules.add(node.module)

    for module in modules:
        root = module.split(".")[0]
        if root == "bench_harness":
            assert module == "bench_harness.types", f"読み込んではいけない module: {module}"
        else:
            assert root in sys.stdlib_module_names or root in ("pydantic", "jsonschema"), (
                f"標準ライブラリ、pydantic、jsonschema 以外を読み込んでいる: {module}"
            )
