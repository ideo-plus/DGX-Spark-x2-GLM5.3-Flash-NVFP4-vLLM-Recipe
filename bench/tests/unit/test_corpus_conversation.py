"""takt の作業を模した会話の組み立ての試験 (task 7.1: corpus/conversation)。

確かめるのは 6 つのまとまり。

1. **決定性** (11.4): 同じ引数なら、プロセスや `PYTHONHASHSEED` をまたいでも
   同じ会話になる。golden なハッシュが、生成の手順の意図しない変更を検出する
2. **先頭の一致** (6.1、design.md「長い会話の検査の 1 段階」): 同じ種では、
   長い段階の会話が、短い段階の会話をそのまま先頭に含む。発話の列としても、
   クライアント (httpx の `json=`) が実際に送るバイト列としても一致する
3. **要求として妥当な形** (6.2): 履歴のツール呼び出しがすべてツールの引数の
   定義に合い、`tool_use` と `tool_result` が 1 対 1 で対応し、役割が交互に
   並び、最後が assistant で終わる (7.2 が user の指示を 1 つ足せる)
4. **呼び出しと結果の辻褄** (6.1、レビュー round 2): 17 個のツールのそれぞれに
   ついて、合成の結果がその呼び出しの引数と矛盾しないこと (範囲の行数、
   `limit` の件数、差分の見出しのパス、一覧のパスの下にあること、glob との
   一致、1 つのパスの確認、コマンドの反響など)。まとめの 1 文も、呼んだツールと
   引数に触れる
5. **採点の公平さとの整合** (注 6.2 / 6.3): 履歴の中で見せる呼び出しは、その
   指示に対する「正解の呼び出し」そのもので、省略できる引数を勝手に足さない。
   履歴の課題の番号は、7.2 が最後の 1 手に使う番号と決して重ならない
6. **長さの狙い** (6.1、6.8): `config/profiles.toml` の `quick` の比で、実際の
   段階のはしご (2 万〜12 万、会話 5 本) を作り、狙いとのずれが
   `Profile.length_tolerance` の内側に収まること。狙いを細かく動かしても、
   手番の数が減らず、短い会話が長い会話の先頭になること

仕掛けを確かめる試験は、その仕掛けがないと結果が変わる状況を作る (注 3.4)。
先頭の一致の試験は、切れ目が違う手番に落ちる狙いの長さの組を使い、役割の
交互の試験は隣り合うすべての組を見て、識別子の対応の試験はすべての対を見る。
ツールごとの辻褄の試験は、そのツールの手番が 1 つ以上あることを必ず確かめてから
判定する (0 件の素通りを防ぐ)。
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Final

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from pydantic import HttpUrl, JsonValue

from bench_harness.client.messages import build_request_body
from bench_harness.config import select_profile
from bench_harness.corpus.conversation import (
    MESSAGES_PER_ROUND,
    SYSTEM_PROMPT,
    build_conversation,
    derive_conversation_seed,
    history_task_index,
    history_tool_task,
)
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
from bench_harness.types import (
    ContentKind,
    ConversationPrefix,
    InputMessage,
    MessagesRequest,
    Profile,
    TargetDef,
    TextBlockParam,
    ToolDef,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

# jsonschema (jsonschema~=4.26.0) は py.typed を同梱していないため、mypy の
# import-untyped を上で無視している (test_corpus_tools.py と同じ)。

_SEEDS: Final[tuple[int, ...]] = (0, 1, 2026)
"""試験で使う `Profile.seed` の代わり。3 種類以上で確かめる。"""

_SMALL_TARGETS: Final[tuple[int, int, int]] = (4_000, 6_000, 9_000)
"""速さのための、小さい身代わりの狙いの長さ (前置きより十分に大きい)。"""

_REALISTIC_TARGETS: Final[tuple[int, int, int]] = (40_000, 60_000, 120_000)
"""実際の段階の長さ (6.1: 2 万〜12 万トークン)。"""

_QUICK_PROFILE: Final[Profile] = select_profile("quick")
"""`config/profiles.toml` の実際の設定 (計測で 7.2 が渡す比を、試験でも使う)。"""

_QUICK_RATIOS: Final[dict[ContentKind, float]] = dict(_QUICK_PROFILE.chars_per_token)

_TOOLS_BY_NAME: Final[dict[str, ToolDef]] = {tool.name: tool for tool in TOOL_CATALOG}


# --- 助け ---------------------------------------------------------------------


def _conversation(
    target_tokens: int,
    seed: int,
    conversation_index: int = 0,
    ratios: dict[ContentKind, float] | None = None,
) -> ConversationPrefix:
    return build_conversation(
        target_tokens=target_tokens,
        conversation_seed=derive_conversation_seed(seed, conversation_index),
        chars_per_token=ratios,
    )


def _rounds(prefix: ConversationPrefix) -> list[list[InputMessage]]:
    """発話の列を、1 手番 (`MESSAGES_PER_ROUND` 個) ずつに切り分ける。"""
    messages = prefix.messages
    return [
        list(messages[i : i + MESSAGES_PER_ROUND])
        for i in range(0, len(messages), MESSAGES_PER_ROUND)
    ]


def _round_count(prefix: ConversationPrefix) -> int:
    return len(prefix.messages) // MESSAGES_PER_ROUND


def _text_block(message: InputMessage) -> TextBlockParam:
    assert len(message.content) == 1, f"ブロックが 1 つでない: {message.content}"
    block = message.content[0]
    assert isinstance(block, TextBlockParam), f"本文のブロックでない: {block}"
    return block


def _tool_use_block(message: InputMessage) -> ToolUseBlockParam:
    assert len(message.content) == 1, f"ブロックが 1 つでない: {message.content}"
    block = message.content[0]
    assert isinstance(block, ToolUseBlockParam), f"ツール呼び出しのブロックでない: {block}"
    return block


def _tool_result_block(message: InputMessage) -> ToolResultBlockParam:
    assert len(message.content) == 1, f"ブロックが 1 つでない: {message.content}"
    block = message.content[0]
    assert isinstance(block, ToolResultBlockParam), f"ツールの結果のブロックでない: {block}"
    return block


def _all_text(prefix: ConversationPrefix) -> str:
    """会話として対象サーバーに届く文字列を、すべてつないだもの。"""
    parts: list[str] = [prefix.system]
    parts.append(json.dumps([tool.model_dump(mode="json") for tool in prefix.tools]))
    for message in prefix.messages:
        for block in message.content:
            if isinstance(block, TextBlockParam):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlockParam):
                parts.append(block.name)
                parts.append(json.dumps(block.input))
            else:
                parts.append(block.content)
    return "\n".join(parts)


def _wire_body(prefix: ConversationPrefix, final_prompt: str, max_tokens: int = 256) -> str:
    """7.2 が送る本文を、クライアントと同じやり方で直列化する。

    クライアントは `build_request_body(request)` の返り値を httpx の `json=` に
    渡す。httpx (`httpx/_content.py` の `encode_json`) は
    `json.dumps(json, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`
    で符号化するので、鍵の並びは `MessagesRequest` の項目の定義順になる
    (保存の部品の `sort_keys=True` の直列化とは別物)。
    """
    request = MessagesRequest(
        model="glm-5.3-flash",
        max_tokens=max_tokens,
        system=prefix.system,
        tools=list(prefix.tools),
        messages=[
            *prefix.messages,
            InputMessage(role="user", content=[TextBlockParam(text=final_prompt)]),
        ],
    )
    body = build_request_body(request)
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _history_span(prefix: ConversationPrefix) -> str:
    """送る本文の中の、`"messages":[` から履歴の最後の発話の終わりまで。"""
    messages_json = json.dumps(
        [message.model_dump(mode="json") for message in prefix.messages],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return '"messages":' + messages_json[:-1]  # 閉じ括弧を除く


# --- 1. 決定性 ----------------------------------------------------------------


def test_same_arguments_give_the_same_conversation() -> None:
    for seed in _SEEDS:
        a = _conversation(_SMALL_TARGETS[1], seed)
        b = _conversation(_SMALL_TARGETS[1], seed)
        assert a.model_dump_json() == b.model_dump_json()


_HASH_SCRIPT: Final[str] = """
from bench_harness.corpus.conversation import build_conversation, derive_conversation_seed

prefix = build_conversation(
    target_tokens={target},
    conversation_seed=derive_conversation_seed({seed}, 0),
)
print(prefix.model_dump_json())
"""


def test_deterministic_across_process_and_pythonhashseed() -> None:
    """`hash()`、大域の乱数、時刻に依存していれば、ここで揺れる (2 プロセスの軽い確認)。"""
    target, seed = _SMALL_TARGETS[0], 7
    expected = build_conversation(
        target_tokens=target, conversation_seed=derive_conversation_seed(seed, 0)
    ).model_dump_json()

    script = _HASH_SCRIPT.format(target=target, seed=seed)
    for hash_seed in ("0", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert result.stdout.strip() == expected, f"PYTHONHASHSEED={hash_seed} で結果が違う"


_GOLDEN_CONVERSATION_HASH: Final[str] = (
    "99e82d77228532979aafb3e8bfcaf2e347383308da2bef498f443ce2de54fe47"
)
_GOLDEN_QUICK_CONVERSATION_HASH: Final[str] = (
    "08cbe4515ce5b43ba0b51034bbc2b22d27cf94554742619145123fb1d40ea9a7"
)


def test_golden_hash_pins_a_small_conversation() -> None:
    """生成の手順を変えたら、ここが落ちる (`Profile` の初期値の比)。

    落ちたら、変更が意図どおりか確かめたうえで、2 つのハッシュを更新すること
    (corpus/synth.py、corpus/tools.py と同じ決まり)。
    """
    prefix = build_conversation(
        target_tokens=4_000, conversation_seed=derive_conversation_seed(0, 0)
    )
    actual = hashlib.sha256(prefix.model_dump_json().encode()).hexdigest()
    assert actual == _GOLDEN_CONVERSATION_HASH


def test_golden_hash_pins_a_small_conversation_with_profile_ratios() -> None:
    """計測で実際に使う比 (`config/profiles.toml` の `quick`) での固定のハッシュ。"""
    prefix = build_conversation(
        target_tokens=4_000,
        conversation_seed=derive_conversation_seed(0, 0),
        chars_per_token=_QUICK_RATIOS,
    )
    actual = hashlib.sha256(prefix.model_dump_json().encode()).hexdigest()
    assert actual == _GOLDEN_QUICK_CONVERSATION_HASH


# --- 2. 先頭の一致 ------------------------------------------------------------


def test_longer_target_contains_shorter_as_an_exact_message_prefix() -> None:
    """同じ種では、長い段階の会話が、短い段階の会話をそのまま先頭に含む (6.1)。

    狙いの長さは、切れ目が**違う手番**に落ちる組を使う (同じ手番で切れる組
    では、手番が狙いの長さに依存する壊れ方を検出できない)。
    """
    short_target, mid_target, long_target = _SMALL_TARGETS
    for seed in _SEEDS:
        short = _conversation(short_target, seed)
        mid = _conversation(mid_target, seed)
        long = _conversation(long_target, seed)

        counts = (_round_count(short), _round_count(mid), _round_count(long))
        assert counts[0] < counts[1] < counts[2], f"手番の数が違わない (seed={seed}): {counts}"

        for shorter, longer in ((short, mid), (mid, long), (short, long)):
            assert longer.system == shorter.system
            assert longer.tools == shorter.tools
            head = longer.messages[: len(shorter.messages)]
            assert head == list(shorter.messages), f"先頭が一致しない (seed={seed})"


def test_wire_body_key_order_puts_messages_before_system_and_tools() -> None:
    """送る本文の鍵の並びは、`MessagesRequest` の項目の定義順になる。

    この並びのおかげで、共有する履歴の終わりまでが、送るバイト列の先頭から
    丸ごと一致する (`system` と `tools` はそのうしろ)。
    """
    prefix = _conversation(_SMALL_TARGETS[0], _SEEDS[0])
    request = MessagesRequest(
        model="glm-5.3-flash",
        max_tokens=256,
        system=prefix.system,
        tools=list(prefix.tools),
        messages=list(prefix.messages),
    )
    assert list(build_request_body(request))[:3] == ["model", "max_tokens", "messages"]


def test_wire_byte_prefix_covers_the_whole_shared_history() -> None:
    """クライアントが実際に送るバイト列でも、共有する履歴の終わりまでが一致する。

    `model` と `max_tokens` が同じなら (7.2 は同じ対象サーバーと
    `AgentSettings.max_tokens` を使う)、短い段階の履歴の最後の発話まで、
    先頭からのバイト列が丸ごと一致する。7.2 が足す最後の 1 手は、そのうしろ。
    """
    short_target, _mid, long_target = _SMALL_TARGETS
    for seed in _SEEDS:
        short = _conversation(short_target, seed)
        long = _conversation(long_target, seed)
        assert _round_count(short) < _round_count(long)

        short_body = _wire_body(short, 'Show the full contents of the file at "a/b/c.py".')
        long_body = _wire_body(long, 'Run the test suite for "a/b/c.py" and report.')

        span = _history_span(short)
        assert span in short_body
        assert span in long_body
        end = short_body.index(span) + len(span)
        assert short_body[:end] == long_body[:end], f"送る本文の先頭が一致しない (seed={seed})"
        # 共有する部分が、短い会話の履歴の終わりまで (次は、それぞれの続きの発話)
        assert short_body[end] == ","
        assert long_body[end] == ","
        assert end > len(short_body) * 0.5


def test_stage_prefix_holds_at_realistic_sizes_and_building_is_fast() -> None:
    """実際の段階の長さ (4 万 → 6 万 → 12 万) でも、先頭が一致する (完了の状態)。

    12 万トークンの会話の組み立てにかかる時間も見る (計測の実行のたびに何本も
    作るので、遅いと計測の流れを妨げる)。
    """
    seed = _SEEDS[0]
    started = time.perf_counter()
    built = [_conversation(target, seed) for target in _REALISTIC_TARGETS]
    elapsed = time.perf_counter() - started

    for shorter, longer in zip(built, built[1:], strict=False):
        assert _round_count(shorter) < _round_count(longer)
        assert longer.messages[: len(shorter.messages)] == list(shorter.messages)
        assert longer.system == shorter.system
        assert longer.tools == shorter.tools

    # このマシンはふだんからロードアベレージが高いので、狙い (12 万で 1 秒未満。
    # 実測は 3 本あわせて 0.03 秒ほど) よりゆるい上限で判定する (注 1.4 の
    # 「遅い側は狭い幅で全数判定しない」)
    assert elapsed < 2.0, f"4 万 + 6 万 + 12 万の組み立てに {elapsed:.2f} 秒かかった"


_MONOTONIC_TARGETS: Final[tuple[int, ...]] = tuple(range(1_000, 130_001, 1_300))
"""狙いを細かく動かすはしご (100 段 × 2 本の種 = 200 回の組み立て)。"""


def test_rounds_are_monotonic_and_nested_across_many_targets() -> None:
    """狙いを細かく上げていくと、手番は減らず、会話は前の会話を先頭に含む。

    段階の刻みが設定で変わっても (6.8)、キャッシュに当たる性質が保たれること
    を、切れ目の境界をまたぐ細かいはしごで確かめる。
    """
    for seed in (0, 1):
        previous: ConversationPrefix | None = None
        for target in _MONOTONIC_TARGETS:
            prefix = _conversation(target, seed, ratios=_QUICK_RATIOS)
            if previous is not None:
                assert _round_count(prefix) >= _round_count(previous), (
                    f"seed={seed} target={target}: 手番が減った"
                )
                assert prefix.messages[: len(previous.messages)] == list(previous.messages), (
                    f"seed={seed} target={target}: 前の狙いの会話を先頭に含まない"
                )
            previous = prefix


# --- 3. 要求として妥当な形 ----------------------------------------------------


def test_every_tool_call_in_the_history_validates_against_its_schema() -> None:
    """履歴のツール呼び出しが、すべてそのツールの引数の定義に合う (完了の状態)。"""
    for seed in _SEEDS:
        prefix = _conversation(20_000, seed)
        rounds = _rounds(prefix)
        assert len(rounds) >= 10
        for index, messages in enumerate(rounds):
            call = _tool_use_block(messages[1])
            tool = _TOOLS_BY_NAME.get(call.name)
            assert tool is not None, f"目録にないツール (seed={seed}, round={index}): {call.name}"
            Draft202012Validator(tool.input_schema).validate(call.input)


def test_every_tool_use_is_answered_by_a_tool_result_with_the_same_id() -> None:
    """`tool_use` の識別子が会話の中で一意で、直後の `tool_result` と対応する。"""
    for seed in _SEEDS:
        prefix = _conversation(_SMALL_TARGETS[2], seed)
        seen: set[str] = set()
        for index, messages in enumerate(_rounds(prefix)):
            call = _tool_use_block(messages[1])
            result = _tool_result_block(messages[2])
            assert result.tool_use_id == call.id, f"識別子が対応しない (seed={seed}, {index=})"
            assert call.id not in seen, f"識別子が重複した (seed={seed}, {index=}): {call.id}"
            seen.add(call.id)
        assert len(seen) == _round_count(prefix)


def test_roles_alternate_and_the_history_ends_with_an_assistant_text() -> None:
    """役割が user から始まって交互に並び、最後は assistant の短いまとめで終わる。

    最後が user の `tool_result` で終わると、7.2 が user の指示を足したときに
    user が 2 つ続いてしまう (Anthropic 互換の要求として危うい)。
    """
    for seed in _SEEDS:
        prefix = _conversation(_SMALL_TARGETS[1], seed)
        messages = list(prefix.messages)
        assert messages, "発話が 1 つもない"
        assert messages[0].role == "user"
        for before, after in zip(messages, messages[1:], strict=False):
            assert before.role != after.role, f"同じ役割が続いた (seed={seed}): {before.role}"
        assert messages[-1].role == "assistant"
        last = _text_block(messages[-1])
        assert last.text.strip(), "最後の assistant の発話が空"


def test_conversation_is_made_of_whole_rounds() -> None:
    """切れ目は必ず手番の境界に落ちる (手番の途中で切ると、先頭の一致が壊れる)。"""
    for seed in _SEEDS:
        for target in _SMALL_TARGETS:
            prefix = _conversation(target, seed)
            assert len(prefix.messages) % MESSAGES_PER_ROUND == 0
            assert _round_count(prefix) >= 1
            for messages in _rounds(prefix):
                roles = [message.role for message in messages]
                assert roles == ["user", "assistant", "user", "assistant"]
                _text_block(messages[0])
                _tool_use_block(messages[1])
                _tool_result_block(messages[2])
                _text_block(messages[3])


def test_appending_one_user_instruction_yields_a_valid_request() -> None:
    """7.2 の使い方 (会話の最後に課題を 1 つ足して送る) が、そのまま通る。"""
    prefix = _conversation(_SMALL_TARGETS[0], _SEEDS[0])
    task = make_tool_task(index=0, seed=0)
    request = MessagesRequest(
        model="glm-5.3-flash",
        max_tokens=1024,
        system=prefix.system,
        tools=list(task.tools),
        messages=[
            *prefix.messages,
            InputMessage(role="user", content=[TextBlockParam(text=task.prompt)]),
        ],
    )
    body = build_request_body(request)
    assert body["system"] == prefix.system
    tools_in_body = body["tools"]
    assert isinstance(tools_in_body, list)
    assert len(tools_in_body) == len(TOOL_CATALOG)
    messages_in_body = body["messages"]
    assert isinstance(messages_in_body, list)
    assert len(messages_in_body) == len(prefix.messages) + 1
    last = messages_in_body[-1]
    assert isinstance(last, dict)
    assert last["role"] == "user"


def test_history_tools_are_the_full_catalog_in_a_fixed_order() -> None:
    prefix = _conversation(_SMALL_TARGETS[0], _SEEDS[0])
    assert [tool.name for tool in prefix.tools] == [tool.name for tool in TOOL_CATALOG]


# --- 4. 呼び出しと結果の辻褄 (17 個のツールを 1 つずつ) -----------------------


class _Observed:
    """ツールごとに集めた、1 手番の呼び出しと結果。"""

    __slots__ = ("arguments", "result", "round_index", "seed", "summary")

    def __init__(
        self,
        seed: int,
        round_index: int,
        arguments: dict[str, JsonValue],
        result: str,
        summary: str,
    ) -> None:
        self.seed = seed
        self.round_index = round_index
        self.arguments = arguments
        self.result = result
        self.summary = summary

    def __repr__(self) -> str:  # pragma: no cover - 失敗したときだけ使う
        return f"<seed={self.seed} round={self.round_index} args={self.arguments}>"


def _observe(target_tokens: int, seeds: tuple[int, ...]) -> dict[str, list[_Observed]]:
    """実際の段階の大きさの会話から、ツールごとに手番を集める。"""
    observed: dict[str, list[_Observed]] = {tool.name: [] for tool in TOOL_CATALOG}
    for seed in seeds:
        prefix = _conversation(target_tokens, seed, ratios=_QUICK_RATIOS)
        for index, messages in enumerate(_rounds(prefix)):
            call = _tool_use_block(messages[1])
            observed[call.name].append(
                _Observed(
                    seed=seed,
                    round_index=index,
                    arguments=dict(call.input),
                    result=_tool_result_block(messages[2]).content,
                    summary=_text_block(messages[3]).text,
                )
            )
    return observed


_OBSERVED_CACHE: Final[dict[str, list[_Observed]]] = {}


def _observed() -> dict[str, list[_Observed]]:
    """3 つの種 × 6 万トークンの会話から集めた手番 (最初に使うときに 1 回だけ作る)。

    module の読み込み時ではなく、使うときに作る。読み込み時に作ると、会話の形が
    壊れたときに、どの試験が落ちたのかが見えない (収集の段階のエラーになる)。
    """
    if not _OBSERVED_CACHE:
        _OBSERVED_CACHE.update(_observe(60_000, _SEEDS))
    return _OBSERVED_CACHE


_NUMBERED_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+) \| (.*)$")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{8} \d{4}-\d{2}-\d{2} ")
_HUNK_RE: Final[re.Pattern[str]] = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@$")
_LOG_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ")
_TEST_SUMMARY_RE: Final[re.Pattern[str]] = re.compile(r"^\d+ passed in \d+\.\d+s$")


def _rounds_for(tool: str) -> list[_Observed]:
    """そのツールの手番 (1 つもなければ、試験として意味がないので落とす)。"""
    rounds = _observed()[tool]
    assert rounds, f"{tool} の手番が 1 つもない (試験が素通りしている)"
    return rounds


def _numbered_lines(text: str) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for line in text.split("\n"):
        matched = _NUMBERED_RE.match(line)
        if matched:
            pairs.append((int(matched.group(1)), matched.group(2)))
    return pairs


def _str_arg(observed: _Observed, name: str) -> str:
    value = observed.arguments[name]
    assert isinstance(value, str), f"{name} が文字列でない: {value!r}"
    return value


def _int_arg(observed: _Observed, name: str) -> int:
    value = observed.arguments[name]
    assert isinstance(value, int) and not isinstance(value, bool), f"{name} が整数でない: {value!r}"
    return value


def test_read_file_result_shows_the_requested_path_numbered_from_one() -> None:
    for observed in _rounds_for("read_file"):
        path = _str_arg(observed, "path")
        assert observed.result.startswith(f"==> {path} <=="), observed
        numbers = [number for number, _ in _numbered_lines(observed.result)]
        assert numbers, observed
        assert numbers == list(range(1, len(numbers) + 1)), observed


def test_read_file_range_result_covers_exactly_the_requested_range() -> None:
    """行数は `end_line - start_line + 1`。上限を超える範囲は、注記つきで切り詰める。"""
    full_seen = truncated_seen = False
    for observed in _rounds_for("read_file_range"):
        path = _str_arg(observed, "path")
        start = _int_arg(observed, "start_line")
        end = _int_arg(observed, "end_line")
        wanted = end - start + 1
        assert observed.result.startswith(f"==> {path} (lines {start}-{end}) <=="), observed
        numbers = [number for number, _ in _numbered_lines(observed.result)]
        assert numbers, observed
        assert numbers == list(range(start, start + len(numbers))), observed
        if len(numbers) == wanted:
            full_seen = True
            assert numbers[-1] == end, observed
            assert "not shown" not in observed.result, observed
        else:
            truncated_seen = True
            assert len(numbers) < wanted, observed
            notice = f"... {wanted - len(numbers)} more lines up to {end} not shown"
            assert notice in observed.result, observed
    assert full_seen, "範囲をすべて見せた手番がない (切り詰めの道しか試していない)"
    assert truncated_seen, "切り詰めた手番がない (注記の道を試していない)"


def test_read_files_result_has_one_section_per_requested_path_in_order() -> None:
    for observed in _rounds_for("read_files"):
        raw = observed.arguments["paths"]
        assert isinstance(raw, list)
        paths = [item for item in raw if isinstance(item, str)]
        headers = [
            line[4:-4]
            for line in observed.result.split("\n")
            if line.startswith("==> ") and line.endswith(" <==")
        ]
        assert headers == paths, observed


def test_search_text_hits_all_contain_the_query() -> None:
    for observed in _rounds_for("search_text"):
        query = _str_arg(observed, "query")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines, observed
        for line in lines:
            assert query in line, observed


def test_search_files_hits_are_under_the_path_and_match_the_pattern() -> None:
    for observed in _rounds_for("search_files"):
        pattern = _str_arg(observed, "pattern")
        scope = _str_arg(observed, "path")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines, observed
        for line in lines:
            assert line.startswith(f"{scope}/"), observed
            assert fnmatch.fnmatch(line, pattern), f"{line!r} が {pattern!r} に一致しない"


def test_git_status_lines_are_under_the_requested_path() -> None:
    for observed in _rounds_for("git_status"):
        scope = _str_arg(observed, "path")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines[0] == f"# uncommitted changes under {scope}", observed
        for line in lines[1:]:
            assert line[3:].startswith(f"{scope}/"), observed


def test_git_diff_paths_are_under_the_path_and_context_matches() -> None:
    """差分のパスが `path` の下にあり、文脈の行数が `context_lines` × 2 になる。"""
    for observed in _rounds_for("git_diff"):
        scope = _str_arg(observed, "path")
        context = _int_arg(observed, "context_lines")
        hunks = [hunk for hunk in observed.result.split("--- a/") if hunk.strip()]
        assert hunks, observed
        for hunk in hunks:
            lines = [line for line in f"--- a/{hunk}".split("\n") if line]
            assert lines[0].startswith(f"--- a/{scope}/"), observed
            assert lines[1] == lines[0].replace("--- a/", "+++ b/", 1), observed
            matched = _HUNK_RE.match(lines[2])
            assert matched is not None, f"かたまりの見出しがない: {lines[2]!r}"
            assert int(matched.group(2)) == 2 * context + 1, observed
            assert int(matched.group(4)) == 2 * context + 1, observed
            context_lines = [line for line in lines[3:] if line.startswith(" ")]
            assert len(context_lines) == 2 * context, observed
            assert sum(1 for line in lines[3:] if line.startswith("-")) == 1, observed
            assert sum(1 for line in lines[3:] if line.startswith("+")) == 1, observed


def test_git_log_returns_exactly_the_requested_number_of_commits() -> None:
    capped_seen = exact_seen = False
    for observed in _rounds_for("git_log"):
        scope = _str_arg(observed, "path")
        limit = _int_arg(observed, "limit")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines[0] == f"# commits touching {scope}", observed
        commits = [line for line in lines if _COMMIT_RE.match(line)]
        if len(commits) == limit:
            exact_seen = True
            assert "not shown" not in observed.result, observed
        else:
            capped_seen = True
            assert len(commits) < limit, observed
            assert f"... {limit - len(commits)} older commits not shown" in observed.result
    assert exact_seen, "`limit` をそのまま出した手番がない"
    assert capped_seen, "切り詰めた手番がない"


def test_run_command_echoes_the_command_and_matches_its_verb() -> None:
    verbs: Counter[str] = Counter()
    for observed in _rounds_for("run_command"):
        command = _str_arg(observed, "command")
        target = command.rsplit(" ", 1)[-1]
        lines = [line for line in observed.result.split("\n") if line]
        assert lines[0] == f"$ {command}", observed
        body = lines[1:]
        assert body, observed
        if command.startswith("du -sh"):
            verbs["du"] += 1
            assert body == [f"{body[0].split('K')[0]}K\t{target}"], observed
        elif command.startswith("wc -l"):
            verbs["wc"] += 1
            assert len(body) == 1 and body[0].endswith(f" {target}"), observed
            assert body[0].strip().split(" ")[0].isdigit(), observed
        elif command.startswith("grep -rn"):
            verbs["grep"] += 1
            for line in body:
                assert line.startswith(f"{target}:"), observed
                assert "TODO" in line, observed
        else:
            verbs["find"] += 1
            directory = target.rsplit("/", 1)[0]
            assert body[0] == target, observed
            for line in body:
                assert line.startswith(directory), observed
    assert len(verbs) >= 3, f"試した動詞が少なすぎる: {verbs}"


def test_run_tests_log_ends_with_a_summary_line_on_its_own() -> None:
    """まとめの行が、切れたログの行にくっつかない (レビュー round 2 の指摘)。"""
    for observed in _rounds_for("run_tests"):
        target = _str_arg(observed, "target")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines[0] == f"# tests for {target}", observed
        assert _TEST_SUMMARY_RE.match(lines[-1]), f"まとめの行が壊れている: {lines[-1]!r}"
        for line in lines[1:-1]:
            assert _LOG_LINE_RE.match(line), f"ログの行が途中で切れている: {line!r}"


def test_write_file_confirmation_counts_the_content_bytes() -> None:
    for observed in _rounds_for("write_file"):
        path = _str_arg(observed, "path")
        content = _str_arg(observed, "content")
        expected = f"wrote {len(content.encode('utf-8'))} bytes to {path}\n"
        assert observed.result == expected, observed


def test_edit_file_diff_names_the_edited_file_and_shows_both_texts() -> None:
    for observed in _rounds_for("edit_file"):
        path = _str_arg(observed, "path")
        old_text = _str_arg(observed, "old_text")
        new_text = _str_arg(observed, "new_text")
        lines = observed.result.split("\n")
        assert lines[0] == f"--- a/{path}", observed
        assert lines[1] == f"+++ b/{path}", observed
        assert f"-{old_text}" in lines, observed
        assert f"+{new_text}" in lines, observed


def test_delete_file_confirmation_names_exactly_one_path() -> None:
    for observed in _rounds_for("delete_file"):
        path = _str_arg(observed, "path")
        assert observed.result == f"deleted {path}\n", observed


def test_move_file_confirmation_names_source_and_destination() -> None:
    for observed in _rounds_for("move_file"):
        source = _str_arg(observed, "source")
        destination = _str_arg(observed, "destination")
        assert observed.result == f"moved {source} -> {destination}\n", observed


def test_create_directory_confirmation_names_the_directory() -> None:
    for observed in _rounds_for("create_directory"):
        path = _str_arg(observed, "path")
        assert observed.result == f"created directory {path}\n", observed


def test_fetch_url_response_matches_the_method_and_url() -> None:
    methods: Counter[str] = Counter()
    for observed in _rounds_for("fetch_url"):
        url = _str_arg(observed, "url")
        method = _str_arg(observed, "method")
        methods[method] += 1
        lines = observed.result.split("\n")
        assert lines[0] == f"> {method} {url}", observed
        status = next(line for line in lines if line.startswith("< HTTP/1.1"))
        if method == "DELETE":
            assert "204 No Content" in status, observed
            assert "{" not in observed.result, observed
        elif method == "POST":
            assert "201 Created" in status, observed
            assert '"accepted":true' in observed.result, observed
        elif method == "PUT":
            assert "200 OK" in status, observed
            assert '"accepted":true' in observed.result, observed
        else:
            assert "200 OK" in status, observed
            assert observed.result.rstrip().endswith("]"), observed
    assert len(methods) >= 3, f"試した method が少なすぎる: {methods}"


def test_list_directory_entries_are_direct_children_of_the_path() -> None:
    for observed in _rounds_for("list_directory"):
        path = _str_arg(observed, "path")
        lines = [line for line in observed.result.split("\n") if line]
        assert lines[0] == f"# {path}", observed
        for line in lines[1:]:
            name = line.split(" ")[0]
            assert "/" not in name.rstrip("/"), f"直下でない項目: {name!r}"


def test_every_tool_in_the_catalog_is_demonstrated_somewhere() -> None:
    """目録のどのツールも、合成の結果を作る道を持っている。"""
    for tool in TOOL_CATALOG:
        rounds = _rounds_for(tool.name)
        for observed in rounds:
            assert observed.result.strip(), f"結果が空: {tool.name}"


def test_result_sizes_stay_varied_from_short_confirmations_to_multi_kilobyte_files() -> None:
    sizes = [len(observed.result) for rounds in _observed().values() for observed in rounds]
    assert min(sizes) < 100, f"短い結果がない: min={min(sizes)}"
    assert max(sizes) > 3_000, f"数 KB の結果がない: max={max(sizes)}"
    assert len({size // 500 for size in sizes}) >= 5, "大きさの幅が狭い"


def test_summaries_mention_the_call_and_vary_from_round_to_round() -> None:
    """まとめの 1 文が、呼んだツールの引数に触れ、手番ごとに変わる。"""
    summaries: list[str] = []
    for rounds in _observed().values():
        for observed in rounds:
            summaries.append(observed.summary)
            values: list[str] = []
            for value in observed.arguments.values():
                if isinstance(value, list):
                    values.extend(str(item) for item in value)
                elif not isinstance(value, bool):
                    values.append(str(value))
            assert any(value in observed.summary for value in values), (
                f"まとめが引数に触れていない: {observed.summary!r} / {observed.arguments}"
            )
            # 1 文であること。引数の中の「.」(パスの拡張子、`find .` のコマンド)
            # は数えず、型紙の側に文の切れ目 (「. 」) がないことを見る
            assert observed.summary.endswith("."), f"文が閉じていない: {observed.summary!r}"
            without_values = observed.summary
            for value in sorted(values, key=len, reverse=True):
                without_values = without_values.replace(value, " ")
            assert ". " not in without_values, f"1 文でない: {observed.summary!r}"
    distinct = len(set(summaries))
    assert distinct >= len(summaries) * 0.9, f"まとめの種類が少ない: {distinct}/{len(summaries)}"


# --- 5. 複数の会話と、前置きの共有 --------------------------------------------


def test_different_conversation_indices_differ_from_the_first_round() -> None:
    """1 段階に複数の会話を使う (design.md: 既定 5 本)。1 手番目から違う。"""
    seed = _SEEDS[0]
    prefixes = [_conversation(_SMALL_TARGETS[1], seed, index) for index in range(5)]
    first_rounds = [
        json.dumps([m.model_dump(mode="json") for m in _rounds(prefix)[0]], sort_keys=True)
        for prefix in prefixes
    ]
    assert len(set(first_rounds)) == len(prefixes), "会話の番号を変えても 1 手番目が同じ"
    whole = {prefix.model_dump_json() for prefix in prefixes}
    assert len(whole) == len(prefixes)
    seeds = {prefix.conversation_seed for prefix in prefixes}
    assert len(seeds) == len(prefixes)


def test_preamble_is_identical_across_conversations_and_targets() -> None:
    """システムプロンプトとツールの目録は、会話にも長さにも依存しない (前置き自体を使い回す)。"""
    prefixes = [
        _conversation(target, seed, index)
        for target in _SMALL_TARGETS
        for seed in _SEEDS
        for index in range(2)
    ]
    assert {prefix.system for prefix in prefixes} == {SYSTEM_PROMPT}
    assert all(list(prefix.tools) == list(TOOL_CATALOG) for prefix in prefixes)


def test_system_prompt_says_nothing_about_the_conversation_length() -> None:
    """長さに触れる言葉が前置きに入ると、段階をまたいだ先頭の一致が壊れる。"""
    lowered = SYSTEM_PROMPT.lower()
    for word in ("20k", "40k", "60k", "120k", "token", "length", "stage"):
        assert word not in lowered, f"前置きが長さに触れている: {word!r}"


def test_rounds_within_a_conversation_are_all_different() -> None:
    """手番ごとに種が変わる (すべての手番が同じ内容になる壊れ方を検出する)。"""
    prefix = _conversation(_SMALL_TARGETS[2], _SEEDS[0])
    dumps = [
        json.dumps([m.model_dump(mode="json") for m in messages], sort_keys=True)
        for messages in _rounds(prefix)
    ]
    assert len(set(dumps)) == len(dumps), "同じ手番が 2 回現れた"


# --- 6. 採点の公平さとの整合 --------------------------------------------------

# 指示の文の中の値の書き方の決まり (corpus/tools.py の docstring)。
_QUOTED_RE: Final[re.Pattern[str]] = re.compile(r'"([^"]*)"')
_BARE_INT_RE: Final[re.Pattern[str]] = re.compile(r"\d+")


def _string_atoms(arguments: dict[str, JsonValue]) -> Counter[str]:
    atoms: Counter[str] = Counter()
    for value in arguments.values():
        if isinstance(value, str):
            atoms[value] += 1
        elif isinstance(value, list):
            for item in value:
                assert isinstance(item, str), f"配列の要素が文字列でない: {item!r}"
                atoms[item] += 1
    return atoms


def _int_atoms(arguments: dict[str, JsonValue]) -> Counter[str]:
    atoms: Counter[str] = Counter()
    for value in arguments.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            atoms[str(value)] += 1
    return atoms


def test_demonstrated_calls_are_exactly_the_expected_calls() -> None:
    """履歴で見せる呼び出しは、その指示に対する正解そのもの (注 6.2 / 6.3)。

    採点は引数の完全一致なので、履歴で省略できる引数を足して見せると、模倣した
    モデルが最後の 1 手で `WRONG_CALL` になる。指示の文に現れる値 (二重引用符の
    中の文字列、引用符の外の裸の整数) の多重集合と、見せた呼び出しの引数の値の
    多重集合が、過不足なく一致することを確かめる。
    """
    for seed in _SEEDS:
        conversation_seed = derive_conversation_seed(seed, 0)
        prefix = build_conversation(
            target_tokens=_SMALL_TARGETS[2], conversation_seed=conversation_seed
        )
        for index, messages in enumerate(_rounds(prefix)):
            instruction = _text_block(messages[0]).text
            call = _tool_use_block(messages[1])
            task = history_tool_task(conversation_seed, index)

            assert instruction == task.prompt
            assert call.name == task.expected_tool
            assert call.input == task.expected_input

            quoted = Counter(_QUOTED_RE.findall(instruction))
            assert quoted == _string_atoms(call.input), (
                f"seed={seed} round={index}: 指示の文の文字列と、見せた引数が食い違う "
                f"(指示={quoted}, 呼び出し={_string_atoms(call.input)})"
            )
            remainder = _QUOTED_RE.sub(" ", instruction)
            bare = Counter(_BARE_INT_RE.findall(remainder))
            assert bare == _int_atoms(call.input), (
                f"seed={seed} round={index}: 指示の文の整数と、見せた引数が食い違う "
                f"(指示={bare}, 呼び出し={_int_atoms(call.input)})"
            )


def test_assistant_turn_shows_only_the_tool_call() -> None:
    """呼び出しの手番は `tool_use` 1 つだけ (本文を添えると、上限で切られやすくなる)。"""
    prefix = _conversation(_SMALL_TARGETS[1], _SEEDS[1])
    for messages in _rounds(prefix):
        assert len(messages[1].content) == 1
        _tool_use_block(messages[1])


def test_history_task_indices_can_never_collide_with_the_final_task_index() -> None:
    """履歴の課題の番号は必ず負で、7.2 が最後の 1 手に使う番号 (0 以上) と重ならない。"""
    indices = [history_task_index(round_index) for round_index in range(500)]
    assert all(index < 0 for index in indices)
    assert len(set(indices)) == len(indices)

    final_prompts = {
        make_tool_task(index=index, seed=seed).prompt for seed in _SEEDS for index in range(200)
    }
    history_prompts: set[str] = set()
    for seed in _SEEDS:
        for conversation_index in range(3):
            prefix = _conversation(_SMALL_TARGETS[2], seed, conversation_index)
            for messages in _rounds(prefix):
                history_prompts.add(_text_block(messages[0]).text)
    overlap = final_prompts & history_prompts
    assert not overlap, f"履歴の指示が、最後の 1 手の課題と重なった: {sorted(overlap)[:3]}"


def test_no_tool_markup_marker_leaks_into_the_conversation() -> None:
    """記法の目印が会話に漏れると、引き写したモデルの応答が `MARKUP_LEAKED` になる。"""
    # `config/targets.toml` は上書きしていないので、既定の目印がそのまま使われる
    markers = TargetDef(
        name="candidate-d", base_url=HttpUrl("http://10.0.1.60:8001"), model="glm-5.3-flash"
    ).tool_markup_markers
    assert markers, "確かめる目印がない"
    for seed in _SEEDS:
        text = _all_text(_conversation(20_000, seed, ratios=_QUICK_RATIOS))
        for marker in markers:
            assert marker not in text, f"目印が漏れた (seed={seed}): {marker!r}"


# --- 7. 長さの狙いと、設定の比 ------------------------------------------------


def test_real_stage_ladder_with_profile_ratios_stays_within_the_tolerance() -> None:
    """`config/profiles.toml` の `quick` で、実際のはしご (2 万〜12 万 × 5 本) を作る。

    比が設定の値と違うと、狙いとのずれが `length_tolerance` を超えて
    `LENGTH_OFF_TARGET` の印が付きはじめる (3.1 の共通の印)。
    """
    agent = _QUICK_PROFILE.agent
    targets = list(range(agent.start_tokens, agent.end_tokens + 1, agent.step_tokens))
    assert len(targets) >= 6
    worst = 0.0
    for conversation_index in range(agent.conversations_per_stage):
        conversation_seed = derive_conversation_seed(_QUICK_PROFILE.seed, conversation_index)
        previous: ConversationPrefix | None = None
        for target in targets:
            prefix = build_conversation(
                target_tokens=target,
                conversation_seed=conversation_seed,
                chars_per_token=_QUICK_RATIOS,
            )
            assert len(prefix.messages) % MESSAGES_PER_ROUND == 0
            error = abs(prefix.approx_tokens - target) / target
            worst = max(worst, error)
            assert error <= _QUICK_PROFILE.length_tolerance, (
                f"会話 {conversation_index} の段階 {target}: ずれ {error:.1%}"
            )
            if previous is not None:
                assert _round_count(prefix) > _round_count(previous)
                assert prefix.messages[: len(previous.messages)] == list(previous.messages)
            previous = prefix
    assert worst > 0.0  # 見積もりがちょうど当たることはない (試験が空回りしていない)


def test_a_stage_step_of_20k_tokens_contains_several_rounds() -> None:
    """段階の刻み (既定 2 万トークン) に、複数の手番が入る (6.8)。"""
    for seed in _SEEDS:
        smaller = _conversation(20_000, seed, ratios=_QUICK_RATIOS)
        larger = _conversation(40_000, seed, ratios=_QUICK_RATIOS)
        added = _round_count(larger) - _round_count(smaller)
        assert added >= 5, f"2 万トークンの刻みで増えた手番: {added}"
        assert _round_count(smaller) >= 5


def test_chars_per_token_changes_the_realised_length() -> None:
    """1 トークンあたりの文字数 (設定から渡す比) が、実際に長さの狙いに効く。"""
    conversation_seed = derive_conversation_seed(_SEEDS[0], 0)
    doubled = {kind: value * 2 for kind, value in _QUICK_RATIOS.items()}

    base = build_conversation(
        target_tokens=20_000, conversation_seed=conversation_seed, chars_per_token=_QUICK_RATIOS
    )
    wide = build_conversation(
        target_tokens=20_000, conversation_seed=conversation_seed, chars_per_token=doubled
    )
    # 1 トークンあたりの文字数が 2 倍なら、同じトークン数の狙いに倍の文字が要る
    assert len(_all_text(wide)) > len(_all_text(base)) * 1.5


def test_invalid_arguments_raise_value_error() -> None:
    conversation_seed = derive_conversation_seed(0, 0)
    for bad_target in (0, -1):
        try:
            build_conversation(target_tokens=bad_target, conversation_seed=conversation_seed)
        except ValueError:
            pass
        else:  # pragma: no cover - 失敗したときだけ通る
            raise AssertionError(f"target_tokens={bad_target} で ValueError にならなかった")

    incomplete: dict[ContentKind, float] = {ContentKind.PROSE_EN: 4.0}
    try:
        build_conversation(
            target_tokens=4_000,
            conversation_seed=conversation_seed,
            chars_per_token=incomplete,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("足りない chars_per_token で ValueError にならなかった")

    try:
        derive_conversation_seed(0, -1)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の会話の番号で ValueError にならなかった")

    try:
        history_task_index(-1)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の手番の番号で ValueError にならなかった")


def test_non_finite_chars_per_token_raises_before_any_generation() -> None:
    """nan と inf は、文章を 1 文字も作る前に弾かれる (止まらなくなるのを防ぐ)。

    弾いているのは `TemplateCorpus.__init__` (`corpus/synth.py`) で、
    `build_conversation` は手番を積む前にそれを作る。だから、この試験は
    何も生成せずに一瞬で終わる。
    """
    conversation_seed = derive_conversation_seed(0, 0)
    for bad in (math.nan, math.inf, -math.inf, 0.0, -1.0):
        ratios = dict(_QUICK_RATIOS)
        ratios[ContentKind.CODE] = bad
        started = time.perf_counter()
        try:
            build_conversation(
                target_tokens=120_000,
                conversation_seed=conversation_seed,
                chars_per_token=ratios,
            )
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"chars_per_token={bad} で ValueError にならなかった")
        assert time.perf_counter() - started < 1.0, f"{bad} の判定に時間がかかりすぎた"


def test_a_tiny_target_still_yields_one_whole_round() -> None:
    """前置きより小さい狙いでも、手番 1 つぶんの会話を返す (要求として成り立つ形)。"""
    prefix = build_conversation(target_tokens=1, conversation_seed=derive_conversation_seed(0, 0))
    assert _round_count(prefix) == 1
    assert prefix.messages[0].role == "user"
    assert prefix.messages[-1].role == "assistant"
