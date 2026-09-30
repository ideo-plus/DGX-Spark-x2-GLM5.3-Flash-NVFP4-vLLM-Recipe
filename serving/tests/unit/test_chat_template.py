"""`enable_thinking`/`thinking` を読むチャットテンプレートの試験 (issue #131)。

確かめること (完了契約 K1、K2、K3、K6):

- K1: 写し (`tests/fixtures/chat-template/glm53-flash.official.jinja`) の SHA-256 と大きさが、
  コミット済みのマニフェスト (`serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`、
  `serving/weights/k2s4.manifest.json`) の `chat_template.jinja` の値と一致する。新しい
  テンプレート (`serving/payload/chat-template/glm53-flash-thinking-toggle.jinja`) は、
  生成の書き出しの塊だけが写しと違い、それ以外は 1 バイトも変わらない
- K2: `enable_thinking`/`thinking` のどちらかが指定されていて (`null` でない)、どちらの値も
  真でないときだけ、生成の書き出しが空の思考の区間 (`<think></think>`) になる。これは
  パーサー (`vllm/parser/glm47_moe.py:192-198`) の読み方と同じである。それ以外の変数
  (値が食い違う組、明示の `null`、文字列の `"false"` を含む) では、公式テンプレートと
  1 文字も変わらない。Jinja2 は `serving/pyproject.toml` の開発用の依存にだけ足す
- K3: テンプレートの出所 (リビジョン、元の SHA-256、MIT) と、開発時の依存の jinja2 が
  `LICENSES.md` に載る
- K6: vLLM の確認 (2 つの経路、パーサー、`--chat-template` の受け付け方) の file:line と、
  未確認の項目が `serving/payload/UPSTREAM.md` に記録されている

この試験は、Spark にも推論サーバーにも触らない (取得済みの写しと、ローカルの Jinja2 の
描画だけで完結する)。他のレシピ (MiaAI-Lab、tonyd2wild、mmastrac など) のテンプレート・
スクリプト・設定は、この試験でも参照しない (クリーンルームの決めごと)。
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Final

import jinja2
import jinja2.ext
import pytest

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

OFFICIAL_FIXTURE_PATH: Final[Path] = (
    SERVING_DIR / "tests" / "fixtures" / "chat-template" / "glm53-flash.official.jinja"
)
"""Hugging Face から 1 バイトも変えずに取った写し。"""

NEW_TEMPLATE_PATH: Final[Path] = (
    SERVING_DIR / "payload" / "chat-template" / "glm53-flash-thinking-toggle.jinja"
)
"""実装が `serve push` で配る、`enable_thinking`/`thinking` を読む新しいテンプレート。"""

LICENSES_PATH: Final[Path] = REPO_ROOT / "LICENSES.md"
UPSTREAM_PATH: Final[Path] = SERVING_DIR / "payload" / "UPSTREAM.md"
PYPROJECT_PATH: Final[Path] = SERVING_DIR / "pyproject.toml"

ORIGIN_REVISION: Final[str] = "18d55bfd5a2194887738da73753975c9d3842f46"
"""`RedHatAI/GLM-5.3-Flash-NVFP4` の、テンプレートを取ったリビジョン。"""

ORIGIN_SHA256: Final[str] = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"
"""公式の `chat_template.jinja` の SHA-256 (`serving/weights/*.manifest.json` と同じ値)。"""

ORIGIN_SIZE_BYTES: Final[int] = 10950

OFFICIAL_GENERATION_PROMPT_BLOCK: Final[str] = (
    "{%- if add_generation_prompt -%}\n    <|assistant|>{{- '<think>' -}}\n{%- endif -%}"
)
"""公式テンプレートの、生成の書き出しの塊 (ちょうど 1 回だけ現れる)。"""

OFFICIAL_GENERATION_TAIL: Final[str] = "<|assistant|><think>"
"""`add_generation_prompt=true` で公式テンプレートを描画したときの、末尾の文字列。"""

NEW_GENERATION_TAIL_WHEN_OFF: Final[str] = "<|assistant|><think></think>"
"""`add_generation_prompt=true`、思考オフの変数で新しいテンプレートを描画したときの末尾。"""


# --- 会話 (受け入れ基準: system・ツール定義・ツール呼び出しと結果を含む) -------------

_TOOLS: Final[list[dict[str, Any]]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "天気を調べる",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_CONVERSATION: Final[list[dict[str, Any]]] = [
    {"role": "system", "content": "あなたは天気を調べるアシスタントです。"},
    {"role": "user", "content": "東京の天気を教えて"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "晴れ、22度"},
    {"role": "user", "content": "ありがとう。じゃあ大阪は?"},
]


# --- 描画の環境 --------------------------------------------------------------------
#
# 実際の描画 (transformers 側) の環境はソース木の外にあり、未確認である。ここでは、公式と
# 新しいテンプレートを**同じ**環境で描画して比べるので、環境の差はどちらにも同じように効き、
# 比べの結果からは打ち消される (計画の確認事項)。`tojson` フィルターは、テンプレートが
# ツール呼び出しの組み立てで使うので登録する。`loopcontrols` 拡張は、テンプレートが
# `{%- break -%}` を使うので要る。


def _tojson_filter(
    value: Any,
    ensure_ascii: bool = True,
    indent: Any = None,
    separators: Any = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = _tojson_filter
    return env


def _read(path: Path, *, missing_hint: str = "") -> str:
    if not path.is_file():
        pytest.fail(f"ファイルがない{missing_hint}: {path}")
    return path.read_text(encoding="utf-8")


def _official_text() -> str:
    return _read(OFFICIAL_FIXTURE_PATH)


def _new_text() -> str:
    return _read(
        NEW_TEMPLATE_PATH, missing_hint=" (配るテンプレートがない。消えたか、名前が変わった)"
    )


def _render(text: str, **overrides: Any) -> str:
    """会話・ツール定義・`add_generation_prompt=True` を既定にして描画する。"""
    env = _environment()
    template = env.from_string(text)
    kwargs: dict[str, Any] = {
        "messages": _CONVERSATION,
        "tools": _TOOLS,
        "add_generation_prompt": True,
    }
    kwargs.update(overrides)
    return template.render(**kwargs)


# --- K1: 写しの固定と、新しいテンプレートとの差分 ------------------------------------


def test_the_official_fixture_copy_is_pinned_to_the_manifest_sha256() -> None:
    """写しの SHA-256 と大きさが、コミット済みのマニフェストの値と一致する。

    写しを手で打ち直す (1 文字でも変える) と、ここで落ちる。
    """
    data = OFFICIAL_FIXTURE_PATH.read_bytes()

    assert hashlib.sha256(data).hexdigest() == ORIGIN_SHA256
    assert len(data) == ORIGIN_SIZE_BYTES


def test_the_official_generation_prompt_block_appears_exactly_once() -> None:
    """写しの生成の書き出しの塊が、ちょうど 1 回だけ現れる (差分の比べの前提)。"""
    text = _official_text()

    assert text.count(OFFICIAL_GENERATION_PROMPT_BLOCK) == 1


def test_the_new_template_differs_from_the_official_copy_only_in_the_generation_prompt_block() -> (
    None
):
    """新しいテンプレートは、生成の書き出しの塊だけが公式と違う。

    塊の前後 (ツール呼び出しの組み立てなど、ほかのすべてのバイト) が公式と違えば、ここで
    落ちる。新しい塊を公式の塊に戻すと、写しと完全に一致する (どちらの塊もちょうど 1 回)。
    """
    official = _official_text()
    prefix, _, suffix = official.partition(OFFICIAL_GENERATION_PROMPT_BLOCK)
    assert prefix + OFFICIAL_GENERATION_PROMPT_BLOCK + suffix == official  # 分割の前提の確認

    new_text = _new_text()

    assert new_text.startswith(prefix), "生成の書き出しの塊より前が、公式と違う"
    assert new_text.endswith(suffix), "生成の書き出しの塊より後が、公式と違う"
    middle = new_text[len(prefix) : len(new_text) - len(suffix)]
    assert middle != OFFICIAL_GENERATION_PROMPT_BLOCK, "生成の書き出しの塊が、公式のまま"
    reverted = (
        new_text[: len(prefix)]
        + OFFICIAL_GENERATION_PROMPT_BLOCK
        + new_text[len(new_text) - len(suffix) :]
    )
    assert reverted == official


# --- K2: enable_thinking/thinking を読む分岐 -----------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_thinking": False},
        {"thinking": False},
        {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"},
    ],
    ids=["enable_thinking_false", "thinking_false_only", "both_and_thinking_mode_disabled"],
)
def test_thinking_off_variables_render_the_generation_prompt_as_an_empty_think_block(
    overrides: dict[str, Any],
) -> None:
    """思考オフの変数では、生成の書き出しが空の思考の区間になり、それより前は
    公式テンプレートを同じ変数で描画した文字列と一致する。"""
    official_rendered = _render(_official_text(), **overrides)
    new_rendered = _render(_new_text(), **overrides)

    assert official_rendered.endswith(OFFICIAL_GENERATION_TAIL)
    assert new_rendered.endswith(NEW_GENERATION_TAIL_WHEN_OFF)
    assert (
        new_rendered[: -len(NEW_GENERATION_TAIL_WHEN_OFF)]
        == official_rendered[: -len(OFFICIAL_GENERATION_TAIL)]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"enable_thinking": True},
        {"reasoning_effort": "low", "enable_thinking": True},
        {"thinking_mode": "disabled"},
        {"enable_thinking": "false"},
        {"thinking": True},
        {"enable_thinking": True, "thinking": True},
        {"enable_thinking": True, "thinking": False},
        {"enable_thinking": False, "thinking": True},
        {"enable_thinking": None},
    ],
    ids=[
        "no_vars",
        "enable_thinking_true",
        "reasoning_effort_low_and_enable_thinking_true",
        "thinking_mode_disabled_only",
        "enable_thinking_string_false",
        "thinking_true",
        "both_true",
        "enable_true_thinking_false",
        "enable_false_thinking_true",
        "enable_thinking_none",
    ],
)
def test_variables_that_do_not_turn_thinking_off_render_identically_to_the_official_template(
    overrides: dict[str, Any],
) -> None:
    """思考オフに当たらない変数では、新しいテンプレートは公式と 1 文字も
    変わらない。文字列としての `"false"`、`thinking_mode` だけ、明示の `None`、値が食い違う組
    (パーサーは `bool(thinking) or bool(enable_thinking)` で思考ありと読む) では切り替わらない。"""
    official_rendered = _render(_official_text(), **overrides)
    new_rendered = _render(_new_text(), **overrides)

    assert new_rendered == official_rendered


def test_generation_prompt_false_with_thinking_off_also_renders_identically() -> None:
    """`add_generation_prompt=false` の組は、生成の書き出しの分岐そのものに
    入らないので、`enable_thinking=false` と組んでも公式と 1 文字も変わらない。"""
    overrides = {"add_generation_prompt": False, "enable_thinking": False}

    official_rendered = _render(_official_text(), **overrides)
    new_rendered = _render(_new_text(), **overrides)

    assert new_rendered == official_rendered


def test_pyproject_declares_jinja2_as_a_dev_only_dependency() -> None:
    """Jinja2 は `serving/pyproject.toml` の開発用の依存にだけ足し、実行時の依存にしない。"""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    runtime_deps = data["project"]["dependencies"]
    dev_deps = data["dependency-groups"]["dev"]

    assert not any(dep.lower().startswith("jinja2") for dep in runtime_deps), runtime_deps
    assert any(dep.lower().startswith("jinja2") for dep in dev_deps), dev_deps


# --- K3: 出所と Jinja2 が LICENSES.md に載る ----------------------------------------


def test_licenses_md_records_the_template_origin_and_the_jinja2_dev_dependency() -> None:
    """テンプレートの行に、リビジョン・元の SHA-256・MIT が並び、serving-kit の開発時の
    依存の表に jinja2 がある。"""
    text = LICENSES_PATH.read_text(encoding="utf-8")

    origin_rows = [
        line
        for line in text.splitlines()
        if "glm53-flash-thinking-toggle" in line or "chat_template.jinja" in line
    ]
    assert origin_rows, "LICENSES.md にテンプレートの出所の行がない"
    assert any(
        ORIGIN_REVISION in row and ORIGIN_SHA256 in row and "MIT" in row for row in origin_rows
    ), origin_rows

    serving_kit_start = text.find("## serving-kit の依存する部品")
    assert serving_kit_start >= 0, "LICENSES.md に serving-kit の節がない"
    serving_kit_section = text[serving_kit_start:]
    dev_start = serving_kit_section.find("### 開発時の依存")
    assert dev_start >= 0, "serving-kit の節に開発時の依存の表がない"
    dev_section = serving_kit_section[dev_start:]
    dev_end = dev_section.find("\n## ", 1)
    dev_table = dev_section if dev_end < 0 else dev_section[:dev_end]
    assert "jinja2" in dev_table.lower(), dev_table


# --- K6: vLLM の確認の file:line と、未確認の項目 -----------------------------------

_UPSTREAM_FILE_REFS: Final[tuple[str, ...]] = (
    "anthropic/protocol.py",
    "chat_completion/protocol.py",
    "renderers/hf.py",
    "glm47_moe.py",
    "cli_args.py",
)
"""`/v1/messages`、`/v1/chat/completions` の 2 つの経路、`--chat-template` の受け付け方、
パーサーの確認に使った、固定した vLLM のソースのファイル (計画の「参照資料の調査結果」)。"""


def test_upstream_doc_records_the_two_request_paths_the_parser_and_the_chat_template_flag() -> None:
    """`UPSTREAM.md` に、2 つの経路・パーサー・`--chat-template` の受け付け方の file:line と、
    未確認の項目がある。"""
    if not UPSTREAM_PATH.is_file():
        pytest.fail(f"UPSTREAM.md がない (消えたか、名前が変わった): {UPSTREAM_PATH}")
    text = UPSTREAM_PATH.read_text(encoding="utf-8")

    for needle in _UPSTREAM_FILE_REFS:
        assert re.search(rf"{re.escape(needle)}:\d+", text), f"{needle} に file:line の記載がない"
    assert "--chat-template" in text
    assert "未確認" in text
