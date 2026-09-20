"""ツールの定義の目録と、正解の決まった課題を生成する (task 6.2: corpus/tools)。

## 目録 (`TOOL_CATALOG`)

コーディングを手伝うエージェントが使いそうな道具を模した、17 個のツールの
固定の目録。名前、説明、`input_schema` (JSON Schema、`type: object` /
`properties` / `required` / `additionalProperties: false`) は、すべてこの
module で自作したものである (クリーンルーム)。既存の製品やエージェントの
枠組み、ベンチマークのツールの定義や説明文を、写したり参考にしたりしていない。

似た名前のツールを、意図して 4 つの組にして混ぜてある (5.1 が測る「呼ぶべき
ツールを 1 つだけ選べるか」を試すため)。

- `read_file` / `read_file_range` / `read_files`
- `search_text` / `search_files`
- `git_status` / `git_diff` / `git_log`
- `run_command` / `run_tests`

残り 7 個は単独: `write_file`、`edit_file`、`delete_file`、`move_file`、
`create_directory`、`fetch_url`、`list_directory`。

目録は種に関わらず常に同じ内容、同じ並びである (design.md corpus 「目録は
決定的」)。並びと個数は `GENERATOR_VERSION` の対象になる (2.5 の決まりに
ならう)。目録を足したり削ったり並べ替えたりしたら、この module の golden な
ハッシュの試験が落ちるので、意図どおりか確かめたうえで `GENERATOR_VERSION`
を上げ、ハッシュを更新すること。

## 課題 (`make_tool_task`)

`make_tool_task(index, seed)` は、正解のツールが 1 つだけに決まる
`ToolTask` を返す。乱数は `random.Random(seed)` を直接は使わず、
`(seed, index, purpose)` を `hashlib.sha256` で符号化した値から種を作る
(`corpus/synth.py` の `prefix_nonce`、`suites/base.py` の `trial_seed` と
同じ考え方。`hash()` は使わない)。同じ `(seed, index)` なら、プロセスや
`PYTHONHASHSEED` をまたいでも、必ず同じ `ToolTask` になる。

### 指示は、正解の引数だけを指し示す (レビューで見つかった穴への対応)

採点の部品 (`scoring/toolcall`、task 6.3) は、引数の完全一致 (exact
equality) でツールの呼び出しを判定する。`input_schema` で省略できる項目
であっても、モデルが指示の文を読んで「値が決まっている」と判断して埋めて
しまえば、`expected_input` にその項目がない限り不正解 (`WRONG_CALL`) に
なる。したがって、この module が守る決まりは 1 つ:

> **指示の文は、正解の引数の値だけを指し示す。それ以外の値は何も示さず、
> 省略できる項目についてのふるまい (上書きするか、再帰的に扱うか、新規と
> 既存のどちらかなど) も一切示唆しない。**

これを機械的に確かめられるように、指示の文の中の値の書き方を 1 つの
決まりに統一している。

- **文字列の値** (パス、識別子、コマンド、URL、列挙の値を含む) は、必ず
  二重引用符 `"..."` で囲む
- **整数の値** は、10 進の裸の数字として埋め込む (引用符で囲まない)
- 指示の文に現れる、二重引用符で囲まれた部分文字列の集合と、裸の整数の
  集合は、`expected_input` の文字列の値 (配列の要素を含む) と整数の値の
  集合に、過不足なく (多重度も含めて) 一致する。指示の文は、正解でない
  値を一切「語らない」

省略できる項目のうち、指示の文のどこにも意味が触れられていないもの
(`search_text.case_sensitive`、`read_file.encoding`、`delete_file.recursive`
など) は、`expected_input` に値を入れない。逆に、指示の文が具体的な場所
(「どこの」) に触れる場合 (`git_status`、`git_diff`、`git_log`、
`search_files` の省略できる `path`) は、その値を必ず `expected_input` に
含める。省略できる項目のうち、ふつうの言い回しがその項目の値を暗に決めて
しまう場合 (新規のファイルを作る指示は `overwrite=false` を、深い経路への
ディレクトリの作成は `parents=true` を、「直下だけ」という言い回しは
`recursive=false` を、それぞれ暗に示しうる) は、その項目自体を目録の
`input_schema` から外してある (`git_diff.staged`、`write_file.overwrite`、
`create_directory.parents`、`list_directory.recursive`)。

ブール値は `expected_input` に一切現れない (design.md がブール値を指示の
文にどう文字どおり書くかを決めていないため)。列挙の値は、design.md
「引数の値 (識別子、パス、数、列挙の値) は、指示の中に文字どおり書く」の
とおり、文字列と同じ扱いで二重引用符に入れて書ける。`fetch_url` の
`method` は、この形の唯一の**必須**の列挙の引数である (`["GET", "POST",
"PUT", "DELETE"]` から 1 つを選び、`expected_input` に入れ、指示の文にも
文字どおり書く)。

指示の文は、目録のどのツールの名前 (`snake_case` の識別子) も文字どおりには
含めない。design.md は「モデルが名前を見て選ぶ」とは書いていないため、
安全側に倒して、正解のツールの名前も、似た名前の兄弟の名前も、指示の文には
一切書かない (動詞と対象の説明だけで、1 つのツールに絞り込めるようにする)。

`ToolTask.tools` には、常に目録のすべて (17 個、常に同じ並び) を渡す。
`ToolDef` は `model_dump(mode="json")` で `{"name", "description",
"input_schema"}` の形になり、`client.messages.build_request_body` が
そのまま `tools` に混ぜられる (Anthropic の `tools` の要求の形と同じ)。
この module 専用の変換の関数は要らない。

依存するのは標準ライブラリと pydantic、`bench_harness.types` だけ。
`client`、`suites`、`runner`、`analysis`、`scoring` のどの module も
読み込まない (依存の向き: types → config → corpus)。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Final

from pydantic import JsonValue

from bench_harness.types import ToolDef, ToolTask

__all__ = ["TOOL_CATALOG", "make_tool_task"]

_TASK_PURPOSE: Final[str] = "tool_task"
"""`_derive_seed` に渡す、この module 専用の種の名前空間。"""


# --- 種を作る (`hash()` を使わない) -----------------------------------------


def _encode_part(part: str | int) -> bytes:
    """種の部品を、型と長さを含めて一意に符号化する。

    `corpus/synth.py` の `_encode_part`、`suites/base.py` の `_encode_part`
    と同じ形。型 (文字列 `s`、整数 `i`) と符号化のあとの長さを前置きするので、
    単純な連結では区別できない組み合わせ (`("a", "bc")` と `("ab", "c")` など)
    も違う符号化になる。
    """
    if isinstance(part, str):
        data = part.encode("utf-8")
        return b"s" + str(len(data)).encode("ascii") + b":" + data
    data = str(part).encode("ascii")
    return b"i" + str(len(data)).encode("ascii") + b":" + data


def _derive_seed(seed: int, index: int, purpose: str) -> int:
    """`(seed, index, purpose)` から、決定的な `random.Random` の種を作る。

    `hash()`、大域の乱数、時刻には依存しない。同じ引数なら、プロセスや
    `PYTHONHASHSEED` をまたいでも同じ値を返す。
    """
    parts: tuple[str | int, ...] = (seed, index, purpose)
    digest = hashlib.sha256(b"".join(_encode_part(part) for part in parts)).digest()
    return int.from_bytes(digest[:8], "big")


# --- 値の語彙 (架空。既存の製品やデータの引き写しではない) --------------------

_DIRS: Final[tuple[str, ...]] = (
    "src",
    "lib",
    "pkg",
    "app",
    "services",
    "internal",
    "cmd",
    "scripts",
)
_COMPONENTS: Final[tuple[str, ...]] = (
    "billing",
    "auth",
    "search",
    "ledger",
    "catalog",
    "payments",
    "notify",
    "sync",
    "cache",
    "router",
    "worker",
    "gateway",
    "scheduler",
    "inventory",
    "session",
    "metrics",
)
_STEMS: Final[tuple[str, ...]] = (
    "handler",
    "client",
    "server",
    "worker",
    "model",
    "schema",
    "utils",
    "config",
    "manager",
    "adapter",
)
_EXTENSIONS: Final[tuple[str, ...]] = (".py", ".ts", ".go", ".rs", ".java", ".rb")
_MARKER_WORDS: Final[tuple[str, ...]] = (
    "retry-budget",
    "cache-ttl",
    "rate-limit",
    "auth-token",
    "queue-depth",
    "lease-timeout",
    "batch-size",
    "backoff-delay",
)
_GLOBS: Final[tuple[str, ...]] = (
    "*.py",
    "*.ts",
    "*.go",
    "*.md",
    "**/*.test.ts",
    "**/*_test.py",
    "*.yaml",
    "*.json",
)
_SHELL_VERBS: Final[tuple[str, ...]] = ("du -sh", "grep -rn TODO", "wc -l", "find . -newer HEAD")
_HOSTS: Final[tuple[str, ...]] = (
    "api.internal.example",
    "status.gateway.example",
    "metrics.collector.example",
    "files.storage.example",
)
_URL_PATHS: Final[tuple[str, ...]] = (
    "v1/health",
    "v1/orders",
    "v2/inventory/sync",
    "status",
    "v1/ledger/entries",
)
_HTTP_METHODS: Final[tuple[str, ...]] = ("GET", "POST", "PUT", "DELETE")
"""`fetch_url` の `method` の列挙の値。`ToolDef` の `enum` と同じ並びにする。"""


def _path(rng: random.Random) -> str:
    return (
        f"{rng.choice(_DIRS)}/{rng.choice(_COMPONENTS)}/"
        f"{rng.choice(_STEMS)}_{rng.randint(0, 999):03d}{rng.choice(_EXTENSIONS)}"
    )


def _distinct_paths(rng: random.Random, count: int) -> list[str]:
    """`count` 個の、互いに違うパスを返す (衝突したら引き直す)。"""
    paths: list[str] = []
    seen: set[str] = set()
    while len(paths) < count:
        candidate = _path(rng)
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)
    return paths


def _marker(rng: random.Random) -> str:
    return f"{rng.choice(_MARKER_WORDS)}-{rng.randint(1000, 9999)}"


def _content_line(rng: random.Random) -> str:
    return f"# {rng.choice(_COMPONENTS)}: {_marker(rng)}"


def _dir_path(rng: random.Random) -> str:
    """`_path` と同じ広さの値の空間を持つ、拡張子のないディレクトリの経路。

    `list_directory` / `create_directory` の正解の引数と、`git_status` /
    `git_diff` / `git_log` / `search_files` の指示の文に添える「どこの」の
    両方に使う。後者の `path` は省略できる項目 (`input_schema`) だが、指示の
    文がその場所に触れる以上、正解の呼び出しもその値を含むはずなので、
    呼び出し元は返り値を必ず `expected_input["path"]` にも入れる (レビューで
    見つかった穴: 指示の文にだけ書いて `expected_input` に入れ忘れると、
    指示のとおりに `path` を渡したモデルが不正解になってしまう)。値の組み
    合わせを増やして、`(seed, index)` が違う課題どうしが偶然そっくりになる
    (目印の JSON がまるごと同じになる) ことも防ぐ。`_path` のようにディレク
    トリの末尾を切り捨てると値の空間が `_DIRS × _COMPONENTS` (128 通り) まで
    縮むので、そうしない。
    """
    return (
        f"{rng.choice(_DIRS)}/{rng.choice(_COMPONENTS)}/"
        f"{rng.choice(_STEMS)}_{rng.randint(0, 999):03d}"
    )


# --- ツールごとの課題の生成 ---------------------------------------------------

_ToolGenerator = Callable[[random.Random], tuple[dict[str, JsonValue], str]]
"""`rng` から `(正解の引数, 指示の文)` を作る関数の型。"""


def _gen_read_file(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _path(rng)
    prompt = f'Show the full contents of the file at "{path}".'
    return {"path": path}, prompt


def _gen_read_file_range(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _path(rng)
    start = rng.randint(1, 19_000)
    end = start + rng.randint(1, 500)
    prompt = f'Show only lines {start} through {end} of the file at "{path}".'
    return {"path": path, "start_line": start, "end_line": end}, prompt


def _gen_read_files(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    paths = _distinct_paths(rng, 3)
    quoted = ", ".join(f'"{p}"' for p in paths)
    prompt = f"Show the full contents of each of these files: {quoted}."
    path_values: list[JsonValue] = list(paths)
    return {"paths": path_values}, prompt


def _gen_search_text(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    marker = _marker(rng)
    prompt = f'Find every place in the codebase where the text "{marker}" appears.'
    return {"query": marker}, prompt


def _gen_search_files(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    pattern = rng.choice(_GLOBS)
    scope = _dir_path(rng)
    prompt = f'Find every file under "{scope}" whose name matches the pattern "{pattern}".'
    return {"pattern": pattern, "path": scope}, prompt


def _gen_git_status(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    scope = _dir_path(rng)
    prompt = f'Check which files under "{scope}" currently have uncommitted changes.'
    return {"path": scope}, prompt


def _gen_git_diff(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    context_lines = rng.randint(0, 20)
    scope = _dir_path(rng)
    prompt = (
        f'Show the uncommitted changes under "{scope}", with {context_lines} lines of '
        "surrounding context around each change."
    )
    return {"context_lines": context_lines, "path": scope}, prompt


def _gen_git_log(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    limit = rng.randint(1, 200)
    scope = _dir_path(rng)
    prompt = f'List the {limit} most recent commits touching "{scope}".'
    return {"limit": limit, "path": scope}, prompt


def _gen_run_command(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    command = f"{rng.choice(_SHELL_VERBS)} {_path(rng)}"
    timeout_s = rng.randint(5, 300)
    prompt = (
        f'Execute the shell command "{command}", and allow at most {timeout_s} seconds '
        "for it to finish."
    )
    return {"command": command, "timeout_s": timeout_s}, prompt


def _gen_run_tests(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    target = _path(rng)
    prompt = f'Run the test suite for "{target}" and report whether it passes.'
    return {"target": target}, prompt


def _gen_write_file(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _path(rng)
    content = _content_line(rng)
    prompt = f'Create a new file at "{path}" containing exactly the text "{content}".'
    return {"path": path, "content": content}, prompt


def _gen_edit_file(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _path(rng)
    old_text = _content_line(rng)
    new_text = f"{old_text} (updated)"  # old_text と必ず違う値にする
    prompt = f'In the file at "{path}", replace the exact text "{old_text}" with "{new_text}".'
    return {"path": path, "old_text": old_text, "new_text": new_text}, prompt


def _gen_delete_file(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _path(rng)
    prompt = f'Permanently delete the file at "{path}".'
    return {"path": path}, prompt


def _gen_move_file(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    source, destination = _distinct_paths(rng, 2)
    prompt = f'Move the file from "{source}" to "{destination}".'
    return {"source": source, "destination": destination}, prompt


def _gen_create_directory(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    parent = _dir_path(rng)
    path = f"{parent}/new_{rng.randint(0, 999):03d}"
    prompt = f'Create a new, empty directory at "{path}".'
    return {"path": path}, prompt


def _gen_fetch_url(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    trace = rng.randint(100_000, 999_999)
    url = f"https://{rng.choice(_HOSTS)}/{rng.choice(_URL_PATHS)}?trace={trace}"
    method = rng.choice(_HTTP_METHODS)
    prompt = f'Issue an HTTP "{method}" request to the URL "{url}" and return the response body.'
    return {"url": url, "method": method}, prompt


def _gen_list_directory(rng: random.Random) -> tuple[dict[str, JsonValue], str]:
    path = _dir_path(rng)
    prompt = f'List the entries directly inside the directory "{path}".'
    return {"path": path}, prompt


def _schema(properties: dict[str, JsonValue], required: list[str]) -> dict[str, JsonValue]:
    # list[str] は list[JsonValue] と不変 (invariant) なので明示的に変換する
    required_values: list[JsonValue] = list(required)
    return {
        "type": "object",
        "properties": properties,
        "required": required_values,
        "additionalProperties": False,
    }


# --- 目録 (name, description, input_schema, generator) -----------------------

_CATALOG_SPEC: Final[tuple[tuple[ToolDef, _ToolGenerator], ...]] = (
    (
        ToolDef(
            name="read_file",
            description="Return the full contents of a single file, given its path.",
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                    "encoding": {"type": "string", "enum": ["utf-8", "ascii", "latin-1"]},
                },
                ["path"],
            ),
        ),
        _gen_read_file,
    ),
    (
        ToolDef(
            name="read_file_range",
            description=(
                "Return only a contiguous range of lines (inclusive, 1-indexed) from a "
                "single file, given its path and the first and last line numbers."
            ),
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                    "start_line": {"type": "integer", "minimum": 1, "maximum": 20_000},
                    "end_line": {"type": "integer", "minimum": 1, "maximum": 20_000},
                },
                ["path", "start_line", "end_line"],
            ),
        ),
        _gen_read_file_range,
    ),
    (
        ToolDef(
            name="read_files",
            description=(
                "Return the full contents of two or more files at once, given their paths."
            ),
            input_schema=_schema(
                {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                    },
                },
                ["paths"],
            ),
        ),
        _gen_read_files,
    ),
    (
        ToolDef(
            name="search_text",
            description=(
                "Search file contents in the repository for a literal substring and "
                "return every matching location."
            ),
            input_schema=_schema(
                {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "case_sensitive": {"type": "boolean"},
                },
                ["query"],
            ),
        ),
        _gen_search_text,
    ),
    (
        ToolDef(
            name="search_files",
            description=(
                "Search file NAMES (not contents) in the repository for a glob pattern "
                "and return every matching path."
            ),
            input_schema=_schema(
                {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                },
                ["pattern"],
            ),
        ),
        _gen_search_files,
    ),
    (
        ToolDef(
            name="git_status",
            description="Report which tracked files have uncommitted changes.",
            input_schema=_schema({"path": {"type": "string", "minLength": 1}}, []),
        ),
        _gen_git_status,
    ),
    (
        ToolDef(
            name="git_diff",
            description=(
                "Show the uncommitted changes in the working tree, with a chosen number "
                "of surrounding context lines per hunk."
            ),
            input_schema=_schema(
                {
                    "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                    "path": {"type": "string", "minLength": 1},
                },
                ["context_lines"],
            ),
        ),
        _gen_git_diff,
    ),
    (
        ToolDef(
            name="git_log",
            description="List the most recent commits in the repository's history.",
            input_schema=_schema(
                {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "path": {"type": "string", "minLength": 1},
                },
                ["limit"],
            ),
        ),
        _gen_git_log,
    ),
    (
        ToolDef(
            name="run_command",
            description=(
                "Execute an arbitrary shell command as-is and return its output. Does not "
                "interpret the command as a test invocation."
            ),
            input_schema=_schema(
                {
                    "command": {"type": "string", "minLength": 1},
                    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600},
                    "cwd": {"type": "string", "minLength": 1},
                },
                ["command", "timeout_s"],
            ),
        ),
        _gen_run_command,
    ),
    (
        ToolDef(
            name="run_tests",
            description=(
                "Run the project's test suite (or a subset of it) for a given path or "
                "module and report pass/fail."
            ),
            input_schema=_schema(
                {
                    "target": {"type": "string", "minLength": 1},
                    "pattern": {"type": "string", "minLength": 1},
                },
                ["target"],
            ),
        ),
        _gen_run_tests,
    ),
    (
        ToolDef(
            name="write_file",
            description="Create a new file at a path, with exactly the given content.",
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                },
                ["path", "content"],
            ),
        ),
        _gen_write_file,
    ),
    (
        ToolDef(
            name="edit_file",
            description=(
                "Replace one exact, literal occurrence of text in an existing file with new text."
            ),
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                    "old_text": {"type": "string", "minLength": 1},
                    "new_text": {"type": "string"},
                },
                ["path", "old_text", "new_text"],
            ),
        ),
        _gen_edit_file,
    ),
    (
        ToolDef(
            name="delete_file",
            description="Permanently delete a single file, given its path.",
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                    "recursive": {"type": "boolean"},
                },
                ["path"],
            ),
        ),
        _gen_delete_file,
    ),
    (
        ToolDef(
            name="move_file",
            description="Move (or rename) a file from one path to another.",
            input_schema=_schema(
                {
                    "source": {"type": "string", "minLength": 1},
                    "destination": {"type": "string", "minLength": 1},
                },
                ["source", "destination"],
            ),
        ),
        _gen_move_file,
    ),
    (
        ToolDef(
            name="create_directory",
            description="Create a new, empty directory at the given path.",
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                },
                ["path"],
            ),
        ),
        _gen_create_directory,
    ),
    (
        ToolDef(
            name="fetch_url",
            description="Issue an HTTP request to a URL and return the response body.",
            input_schema=_schema(
                {
                    "url": {"type": "string", "minLength": 1},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 120},
                },
                ["url", "method"],
            ),
        ),
        _gen_fetch_url,
    ),
    (
        ToolDef(
            name="list_directory",
            description="List the entries directly inside a directory (not recursive).",
            input_schema=_schema(
                {
                    "path": {"type": "string", "minLength": 1},
                },
                ["path"],
            ),
        ),
        _gen_list_directory,
    ),
)

TOOL_CATALOG: Final[tuple[ToolDef, ...]] = tuple(tool for tool, _ in _CATALOG_SPEC)
"""固定の目録 (17 個、常に同じ並び)。並びと個数は `GENERATOR_VERSION` の対象。"""

_GENERATORS: Final[dict[str, _ToolGenerator]] = {tool.name: gen for tool, gen in _CATALOG_SPEC}


def make_tool_task(index: int, seed: int) -> ToolTask:
    """`(seed, index)` から、正解が 1 つに決まる `ToolTask` を作る (design.md corpus)。

    同じ `(seed, index)` なら、プロセスや `PYTHONHASHSEED` をまたいでも同じ
    `ToolTask` になる (`model_dump_json()` で比較できる)。`tools` には、常に
    `TOOL_CATALOG` の全体を、同じ並びで渡す。
    """
    rng = random.Random(_derive_seed(seed, index, _TASK_PURPOSE))
    tool = rng.choice(TOOL_CATALOG)
    expected_input, prompt = _GENERATORS[tool.name](rng)
    return ToolTask(
        prompt=prompt,
        tools=list(TOOL_CATALOG),
        expected_tool=tool.name,
        expected_input=expected_input,
    )
