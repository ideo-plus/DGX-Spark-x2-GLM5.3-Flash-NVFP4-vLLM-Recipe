"""ツールの目録と、正解の決まった課題を生成する部品の試験 (task 6.2)。

決定性 (同じ `(seed, index)` で同じ課題、プロセスと `PYTHONHASHSEED` を
またいでも同じ)、目録の形 (JSON Schema として妥当、名前が重複しない)、
課題の妥当性 (正解の引数がスキーマに合う、値が指示の文に文字どおり現れる、
指示の文が正解でない値を語らない、ツールの名前が指示の文に漏れない、
兄弟どうしの言い回しが取り違えられていない、1 つの課題の引数どうしが
互いに違う値である)、分布 (すべてのツールが正解になる、似た名前の組を
十分な割合で狙う) を確かめる。golden なハッシュの試験は、目録の並びや
個数、課題の生成の手順を変えたときの意図しない変化を検出する。

## レビュー (round 1) で見つかった穴と、この試験での対応

採点の部品 (`scoring/toolcall`) は引数の完全一致で判定するため、指示の文が
正解でない値を「語って」しまうと (省略できる引数の値が指示に書かれているのに
`expected_input` にない)、指示のとおりに従ったモデルが不正解になる。この
穴を機械的に検出するため、指示の文の値の書き方を 1 つの決まりに統一した上
(文字列は二重引用符、整数は裸の10進数字)、`test_wide_range_prompt_names_no_unexpected_value`
で、指示の文に現れる値らしきものの集合と `expected_input` の値の集合が
過不足なく一致することを確かめる。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Final, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from pydantic import HttpUrl, JsonValue

from bench_harness.client.messages import build_request_body
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
from bench_harness.types import (
    InputMessage,
    MessagesRequest,
    TargetDef,
    TextBlockParam,
    ToolDef,
    ToolTask,
)

# jsonschema (jsonschema~=4.26.0) は py.typed を同梱していないため、mypy の
# import-untyped を上で無視している (型スタブの導入は、この task の境界の
# 外にある pyproject.toml の変更が要る)。


def _schema_properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], schema["properties"])


def _schema_required(schema: dict[str, JsonValue]) -> list[str]:
    return cast(list[str], schema["required"])


# --- 目録の形 -----------------------------------------------------------------


def test_catalog_has_unique_names() -> None:
    names = [tool.name for tool in TOOL_CATALOG]
    assert len(names) == len(set(names))


def test_catalog_is_not_trivially_small() -> None:
    # 5.1 が測る「似た名前から選べるか」を試すには、ある程度の大きさが要る
    assert len(TOOL_CATALOG) >= 10


def test_every_tool_schema_is_valid_json_schema() -> None:
    for tool in TOOL_CATALOG:
        Draft202012Validator.check_schema(tool.input_schema)


def test_every_tool_schema_is_a_closed_object_schema() -> None:
    for tool in TOOL_CATALOG:
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        properties = _schema_properties(schema)
        required = _schema_required(schema)
        for name in required:
            assert name in properties, f"{tool.name}: required だが properties にない"


def test_catalog_covers_a_variety_of_argument_types() -> None:
    """string、integer (上下限つき)、boolean、enum、配列、省略可能な項目が、目録のどこかにある。"""
    seen_types: set[str] = set()
    has_bounded_integer = False
    has_enum = False
    has_array = False
    has_optional = False
    for tool in TOOL_CATALOG:
        schema = tool.input_schema
        properties = _schema_properties(schema)
        required = set(_schema_required(schema))
        for name, raw_prop in properties.items():
            prop = cast(dict[str, JsonValue], raw_prop)
            prop_type = prop["type"]
            assert isinstance(prop_type, str)
            seen_types.add(prop_type)
            if prop_type == "integer" and "minimum" in prop and "maximum" in prop:
                has_bounded_integer = True
            if "enum" in prop:
                has_enum = True
            if prop_type == "array":
                has_array = True
            if name not in required:
                has_optional = True
    assert {"string", "integer", "boolean", "array"} <= seen_types
    assert has_bounded_integer
    assert has_enum
    assert has_array
    assert has_optional


def test_catalog_has_a_required_enum_argument() -> None:
    """design.md corpus: 列挙の値も、指示の中に文字どおり書ける。少なくとも 1 つは必須にする

    (必須でない列挙は、指示に書かなければ `expected_input` に入れずに済むが、
    その分「列挙の値を文字どおり指示に書く」経路を試験しないままになる)。
    """
    required_enum_tools = []
    for tool in TOOL_CATALOG:
        schema = tool.input_schema
        properties = _schema_properties(schema)
        required = set(_schema_required(schema))
        for name in required:
            prop = cast(dict[str, JsonValue], properties[name])
            if "enum" in prop:
                required_enum_tools.append(tool.name)
    assert required_enum_tools, "必須の列挙の引数を持つツールがない"


_SIMILAR_NAME_GROUPS: Final[tuple[frozenset[str], ...]] = (
    frozenset({"read_file", "read_file_range", "read_files"}),
    frozenset({"search_text", "search_files"}),
    frozenset({"git_status", "git_diff", "git_log"}),
    frozenset({"run_command", "run_tests"}),
)


def test_confusable_groups_exist_in_catalog() -> None:
    names = {tool.name for tool in TOOL_CATALOG}
    for group in _SIMILAR_NAME_GROUPS:
        assert group <= names, f"目録に足りない: {group - names}"
        assert len(group) >= 2


def test_tool_defs_serialize_to_anthropic_tools_request_format() -> None:
    """`ToolDef` が `build_request_body` を通して、送る本文の `tools` にそのまま乗る。

    design.md corpus は「`ToolDef` がすでにその形にならないなら、変換の口を
    足す」と言っている。ここでは、変換の口を足さずに済むことを確かめる
    (client/messages.py は変更していない)。
    """
    request = MessagesRequest(
        model="glm-5.3-flash",
        max_tokens=16,
        messages=[InputMessage(role="user", content=[TextBlockParam(text="hi")])],
        tools=list(TOOL_CATALOG),
    )
    body = build_request_body(request)
    tools_in_body = body["tools"]
    assert isinstance(tools_in_body, list)
    assert len(tools_in_body) == len(TOOL_CATALOG)
    for sent, tool in zip(tools_in_body, TOOL_CATALOG, strict=True):
        assert isinstance(sent, dict)
        assert sent["name"] == tool.name
        assert sent["description"] == tool.description
        assert sent["input_schema"] == tool.input_schema


# --- 決定性 -------------------------------------------------------------------


def test_same_seed_and_index_yield_identical_task() -> None:
    a = make_tool_task(index=17, seed=2026)
    b = make_tool_task(index=17, seed=2026)
    assert a.model_dump_json() == b.model_dump_json()


_HASH_SCRIPT: Final[str] = """
from bench_harness.corpus.tools import make_tool_task

task = make_tool_task(index={index}, seed={seed})
print(task.model_dump_json())
"""


def test_deterministic_across_process_and_pythonhashseed() -> None:
    """プロセスをまたいでも、`PYTHONHASHSEED` を変えても、同じ `ToolTask` になる。

    `hash()` に依存していれば、ここで揺れる (2 プロセスだけの、軽い確認)。
    """
    index, seed = 401, 7
    expected = make_tool_task(index=index, seed=seed).model_dump_json()

    script = _HASH_SCRIPT.format(index=index, seed=seed)
    for hash_seed in ("0", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert result.stdout.strip() == expected, f"PYTHONHASHSEED={hash_seed} で結果が違う"


def test_different_index_almost_always_changes_task() -> None:
    seed = 5
    dumps = [make_tool_task(index=i, seed=seed).model_dump_json() for i in range(1000)]
    collisions = len(dumps) - len(set(dumps))
    # 目録 17 個、豊富な値の空間があるので、1000 件のうち衝突は 1% 未満のはず
    assert collisions / len(dumps) < 0.01, f"index を変えても変わらなかった件数: {collisions}"


def test_different_seed_almost_always_changes_task() -> None:
    index = 5
    dumps = [make_tool_task(index=index, seed=s).model_dump_json() for s in range(1000)]
    collisions = len(dumps) - len(set(dumps))
    assert collisions / len(dumps) < 0.01, f"seed を変えても変わらなかった件数: {collisions}"


# --- 課題の妥当性 (広い範囲、複数の種) -----------------------------------------

_WIDE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2026)
_WIDE_INDEX_COUNT: Final[int] = 500


def _all_wide_tasks() -> list[tuple[int, int, ToolTask]]:
    tasks: list[tuple[int, int, ToolTask]] = []
    for seed in _WIDE_SEEDS:
        for index in range(_WIDE_INDEX_COUNT):
            tasks.append((seed, index, make_tool_task(index=index, seed=seed)))
    return tasks


_WIDE_TASKS: Final[list[tuple[int, int, ToolTask]]] = _all_wide_tasks()
_TOOLS_BY_NAME: Final[dict[str, ToolDef]] = {tool.name: tool for tool in TOOL_CATALOG}
_ALL_NAMES: Final[tuple[str, ...]] = tuple(tool.name for tool in TOOL_CATALOG)


def test_wide_range_expected_input_validates_against_schema() -> None:
    for _seed, _index, task in _WIDE_TASKS:
        tool = _TOOLS_BY_NAME[task.expected_tool]
        Draft202012Validator(tool.input_schema).validate(task.expected_input)


def test_wide_range_every_expected_value_appears_literally_in_prompt() -> None:
    for seed, index, task in _WIDE_TASKS:
        prompt = task.prompt
        for value in task.expected_input.values():
            if isinstance(value, list):
                for item in value:
                    assert str(item) in prompt, (
                        f"seed={seed} index={index}: {item!r} が指示の文にない: {prompt!r}"
                    )
            else:
                assert str(value) in prompt, (
                    f"seed={seed} index={index}: {value!r} が指示の文にない: {prompt!r}"
                )


# 指示の文の中の値の書き方の決まり (module docstring 「指示は、正解の引数だけを
# 指し示す」): 文字列は必ず二重引用符で囲み、整数は裸の 10 進の数字で書く。
_QUOTED_RE: Final[re.Pattern[str]] = re.compile(r'"([^"]*)"')
_BARE_INT_RE: Final[re.Pattern[str]] = re.compile(r"\d+")


def _expected_string_atoms(expected_input: dict[str, JsonValue]) -> Counter[str]:
    """`expected_input` の、文字列の値 (配列の要素を含む) の多重集合。"""
    atoms: Counter[str] = Counter()
    for value in expected_input.values():
        if isinstance(value, str):
            atoms[value] += 1
        elif isinstance(value, list):
            for item in value:
                assert isinstance(item, str), f"配列の要素が文字列でない: {item!r}"
                atoms[item] += 1
    return atoms


def _expected_int_atoms(expected_input: dict[str, JsonValue]) -> Counter[str]:
    """`expected_input` の、整数の値 (10 進の文字列にしたもの) の多重集合。"""
    atoms: Counter[str] = Counter()
    for value in expected_input.values():
        # bool は int の派生型なので先に弾く (この module は bool を expected_input に入れない)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            atoms[str(value)] += 1
    return atoms


def test_wide_range_prompt_names_no_unexpected_value() -> None:
    """指示の文は、正解の引数の値だけを指し示す (レビュー round 1 finding 1 への対応)。

    採点の部品は引数の完全一致で判定するため、指示の文が正解でない値を
    「語って」しまうと (省略できる引数の値が指示に書かれているのに
    `expected_input` にない)、指示のとおりに従ったモデルが不正解になる。
    二重引用符で囲まれた部分文字列の多重集合と、引用符の外にある裸の整数の
    多重集合が、`expected_input` の値の多重集合と過不足なく一致することを
    確かめる。ずれたら、指示の文が正解でない値を語っているか、正解の値が
    `expected_input` から抜けている。
    """
    for seed, index, task in _WIDE_TASKS:
        prompt = task.prompt
        quoted = Counter(_QUOTED_RE.findall(prompt))
        expected_strings = _expected_string_atoms(task.expected_input)
        assert quoted == expected_strings, (
            f"seed={seed} index={index}: {task.expected_tool}: 引用符の中の値が食い違う "
            f"(指示={quoted}, expected_input={expected_strings})"
        )

        remainder = _QUOTED_RE.sub(" ", prompt)
        bare_ints = Counter(_BARE_INT_RE.findall(remainder))
        expected_ints = _expected_int_atoms(task.expected_input)
        assert bare_ints == expected_ints, (
            f"seed={seed} index={index}: {task.expected_tool}: 引用符の外の整数が食い違う "
            f"(指示={bare_ints}, expected_input={expected_ints})"
        )


def test_wide_range_arguments_within_a_task_are_pairwise_distinct() -> None:
    """1 つの課題の引数どうしが、互いに違う値であること (レビュー round 1 finding 5 への対応)。

    `_distinct_paths` の重複の除去が抜けると (`move_file` の source と
    destination が同じパスになる、`read_files` の配列に同じパスが 2 回入る、
    など) ここで検出する。配列そのものの要素どうしの重複と、引数をまたいだ
    重複 (配列の要素を含む) の両方を確かめる。
    """
    for seed, index, task in _WIDE_TASKS:
        atoms: list[str | int] = []
        for value in task.expected_input.values():
            if isinstance(value, list):
                str_items = [item for item in value if isinstance(item, str)]
                assert len(str_items) == len(set(str_items)), (
                    f"seed={seed} index={index}: {task.expected_tool} の配列の要素が重複: {value}"
                )
                atoms.extend(str_items)
            elif isinstance(value, str | int) and not isinstance(value, bool):
                atoms.append(value)
        assert len(atoms) == len(set(atoms)), (
            f"seed={seed} index={index}: {task.expected_tool} の引数どうしが重複: "
            f"{task.expected_input}"
        )


def test_wide_range_expected_tool_exists_exactly_once() -> None:
    for _seed, _index, task in _WIDE_TASKS:
        assert task.expected_tool in _ALL_NAMES
        assert _ALL_NAMES.count(task.expected_tool) == 1


def test_wide_range_no_tool_name_leaks_into_prompt() -> None:
    """正解のツールの名前も、似た名前の兄弟の名前も、指示の文に文字どおり現れない。"""
    for seed, index, task in _WIDE_TASKS:
        prompt = task.prompt
        for name in _ALL_NAMES:
            assert name not in prompt, (
                f"seed={seed} index={index}: ツールの名前 {name!r} が指示の文に漏れた: {prompt!r}"
            )


def test_wide_range_tools_field_is_full_catalog_in_fixed_order() -> None:
    for _seed, _index, task in _WIDE_TASKS:
        assert [t.name for t in task.tools] == list(_ALL_NAMES)


# 各ツールの指示の文に固有の言い回し。design.md「モデルが名前を見て選ぶ」とは
# 書いていないため、ツールの名前 (snake_case) は指示に現れない代わりに、
# 動詞と対象の言い回しでツールを絞り込む。ここで、その言い回しが「自分の
# ツールにだけ現れ、ほかのどのツールにも現れない」ことを固定する。似た名前の
# 兄弟の文言を取り違えると (レビュー round 1 finding 4: `read_file` に
# `read_files` の文言、など) ここで落ちる。
_TOOL_SIGNATURE_PHRASE: Final[dict[str, str]] = {
    "read_file": "Show the full contents of the file at",
    "read_file_range": "Show only lines",
    "read_files": "Show the full contents of each of these files",
    "search_text": "Find every place in the codebase where the text",
    "search_files": "whose name matches the pattern",
    "git_status": "currently have uncommitted changes",
    "git_diff": "surrounding context around each change",
    "git_log": "most recent commits touching",
    "run_command": "Execute the shell command",
    "run_tests": "Run the test suite for",
    "write_file": "containing exactly the text",
    "edit_file": "replace the exact text",
    "delete_file": "Permanently delete the file at",
    "move_file": "Move the file from",
    "create_directory": "Create a new, empty directory at",
    "fetch_url": "Issue an HTTP",
    "list_directory": "List the entries directly inside the directory",
}


def test_wide_range_prompt_has_tool_specific_signature_phrase() -> None:
    """兄弟どうしの指示の文言を取り違えると (レビュー round 1 finding 4) ここで落ちる。"""
    assert set(_TOOL_SIGNATURE_PHRASE) == set(_ALL_NAMES)
    for seed, index, task in _WIDE_TASKS:
        prompt = task.prompt
        own_phrase = _TOOL_SIGNATURE_PHRASE[task.expected_tool]
        assert own_phrase in prompt, (
            f"seed={seed} index={index}: {task.expected_tool} 固有の言い回しがない: {prompt!r}"
        )
        for other_name, other_phrase in _TOOL_SIGNATURE_PHRASE.items():
            if other_name == task.expected_tool:
                continue
            assert other_phrase not in prompt, (
                f"seed={seed} index={index}: {task.expected_tool} の指示に "
                f"{other_name} の言い回しが混ざっている: {prompt!r}"
            )


# --- 分布 ---------------------------------------------------------------------


def test_every_tool_is_expected_at_least_once_over_1000_indices() -> None:
    seed = 0
    expected_tools = {make_tool_task(index=i, seed=seed).expected_tool for i in range(1000)}
    assert expected_tools == set(_ALL_NAMES)


def test_at_least_half_of_tasks_target_a_tool_with_a_similar_named_sibling() -> None:
    grouped_names: set[str] = set()
    for group in _SIMILAR_NAME_GROUPS:
        grouped_names |= group

    seed = 0
    n = 1000
    grouped_hits = sum(
        1 for i in range(n) if make_tool_task(index=i, seed=seed).expected_tool in grouped_names
    )
    fraction = grouped_hits / n
    assert fraction >= 0.5, f"似た名前の組を狙った割合: {fraction:.3f}"


# --- golden なハッシュ (GENERATOR_VERSION の上げ忘れを検出する) -----------------

_GOLDEN_CATALOG_HASH: Final[str] = (
    "78422b1d98782ef37ba9277455d48a73796e8863b2b9e679b339af5d995d899f"
)


def test_golden_hash_pins_catalog_shape_and_order() -> None:
    """目録の並びや個数を変えたら、ここが落ちる。

    落ちたら、変更が意図どおりか確かめたうえで、このハッシュを更新し、
    `bench_harness.types.GENERATOR_VERSION` も上げること (corpus/synth.py の
    決まりと同じ)。
    """
    serialized = json.dumps(
        [tool.model_dump(mode="json") for tool in TOOL_CATALOG],
        sort_keys=True,
    )
    actual = hashlib.sha256(serialized.encode()).hexdigest()
    assert actual == _GOLDEN_CATALOG_HASH


_GOLDEN_TASK_HASHES: Final[dict[str, str]] = {
    "seed=123,index=7": "5eeaea90801c1b4963e12dc41ccda971b8c5f49ad8f5bb45776aaf44cf7dbcf3",
}


def test_golden_hash_pins_a_sample_task() -> None:
    """1 つの `(seed, index)` の `ToolTask` を pin する、生成の手順の変更の見張り。"""
    task = make_tool_task(index=7, seed=123)
    actual = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
    assert actual == _GOLDEN_TASK_HASHES["seed=123,index=7"]


# --- 構成の確認 (試験自身が壊れていないこと) -----------------------------------


def test_target_def_accepts_a_generated_url_shape() -> None:
    """`fetch_url` が作る URL が、対象サーバーの定義と同じ `HttpUrl` の型に合う (形だけの確認)。"""
    TargetDef(name="x", base_url=HttpUrl("https://api.internal.example/v1/health"), model="m")
