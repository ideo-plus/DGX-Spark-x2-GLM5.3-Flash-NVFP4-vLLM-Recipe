"""takt の作業を模した会話を、狙った長さまで組み立てる (task 7.1: corpus/conversation)。

`build_conversation(target_tokens, conversation_seed)` は、コーディングを手伝う
エージェントの長い作業 (takt の使い方) を模した合成の会話を返す。同じ前置き
(システムプロンプトと `TOOL_CATALOG`) のあとに、「指示 → ツール呼び出し →
合成の結果 → 短いまとめ」の手番を、狙った長さになるまで積む (6.1)。ツールは
実際には一切実行せず、結果は合成の文章である (6.2)。

システムプロンプト、手番の言い回し、合成の結果の書式は、すべてこの module で
自作したものである (クリーンルーム)。takt、Claude Code、その他のエージェントの
製品や枠組みのプロンプトや出力の書式を、写したり参考にしたりしていない。

## 1 手番の形 (`MESSAGES_PER_ROUND` = 4)

| # | 役割 | 中身 |
|---|------|------|
| 0 | user | 指示 (`make_tool_task` の `prompt` をそのまま) |
| 1 | assistant | `tool_use` のブロック 1 つだけ (正解のツールと、正解の引数) |
| 2 | user | `tool_result` のブロック 1 つだけ (合成の結果。下の節を参照) |
| 3 | assistant | 1 文の短いまとめ (本文のブロック 1 つ。呼んだツールと引数に触れる) |

**7.2 への申し送り (この形が守る契約)**

- 会話は必ず user の指示から始まり、役割が交互に並び、**assistant の本文で
  終わる**。したがって 7.2 は、会話の後ろに user の発話 (最後の 1 手の課題) を
  1 つ足すだけで、Anthropic 互換の要求として妥当な列になる (user が 2 つ続か
  ない)。`tool_result` は、対応する `tool_use` の直後の発話に必ず入っている
- 呼び出しの手番には本文を添えない。`tool_use` を 1 つだけ見せることで、
  最後の 1 手でもモデルが前置きの本文なしに呼び出しを出しやすくする
  (注 6.3: 本文を続けるモデルは `max_tokens` に当たって `EMPTY_OR_TRUNCATED`
  に落ちやすい)
- 前置き (`system` と `tools`) は、**どの会話でも、どの長さでも同じ**である。
  `tools` は `TOOL_CATALOG` の全体を、常に同じ並びで返す (7.2 が最後の 1 手で
  渡す `ToolTask.tools` と同じ並びなので、要求の `tools` は段階をまたいで
  完全に同じになる)
- `ConversationPrefix.approx_tokens` は、**前置き + 履歴**のおよその長さで、
  7.2 が足す最後の 1 手 (課題の指示) は**含まない**。上限に収まるかを見る
  とき (`fits_context(limit, input_tokens, max_tokens)`) は、最後の 1 手の
  ぶんを足してから渡すこと。`fixed_tokens` を渡したときの `approx_tokens` は
  **包み + 履歴**の長さで、こちらは 7.2 が足す最後の 1 手まで含む (issue #9)

## 合成のツールの結果 (6.1「ファイルの中身、コマンドの出力」)

結果は、**その呼び出しの引数と辻褄が合う**ように作る。履歴の中で、呼び出しと
食い違う結果 (別のファイルの差分、指示していないディレクトリの一覧、範囲を
無視した全文) を見せると、モデルの振る舞いが変わりかねず、崩れの割合の計測が
汚れる。ツールごとの決まりは次のとおり。

- `read_file`: 見出しが `path`。本文は 1 から通し番号を振った行
- `read_file_range`: 見出しが `path` と範囲。`start_line` から通し番号を振り、
  行数はちょうど `end_line - start_line + 1` (`_MAX_RESULT_LINES` を超える
  範囲は、そこまでを見せて切り詰めの注記を添える)
- `read_files`: `paths` の順に 1 つずつの節。見出しはそれぞれのパス
- `search_text`: 当たりの各行が `query` を文字どおり含む
- `search_files`: 当たりの各行が `path` の下にあり、`pattern` に一致する
  (`fnmatch`)
- `git_status`: 各行のパスが `path` の下にある
- `git_diff`: 差分の見出しのパスが `path` の下にあり、各かたまりの文脈の
  行数がちょうど `context_lines` × 2 (前後)
- `git_log`: 見出しが `path`。コミットの行数がちょうど `limit` (上限を超える
  場合は切り詰めの注記)
- `run_command`: 1 行目がコマンドの反響 (`$ <command>`)。出力の形はコマンドの
  動詞に合う (`du -sh`、`wc -l`、`grep -rn TODO`、`find`)
- `run_tests`: 見出しが `target`。ログの行のあと、**必ず行の切れ目で改行して
  から**まとめの 1 行 (`N passed in X.Xs`)
- `write_file`: `content` のバイト数と `path` を報告する 1 行
- `edit_file`: 差分の見出しが `path`。消えた行が `old_text`、足された行が
  `new_text`
- `delete_file`: `path` 1 つを消したことの 1 行
- `move_file`: `source` から `destination` へ動かしたことの 1 行
- `create_directory`: `path` を作ったことの 1 行
- `fetch_url`: 1 行目が `> <method> <url>`。状態行と本文は `method` に合う
  (`DELETE` は 204 で本文なし)
- `list_directory`: 見出しが `path`。項目は `path` の直下の名前だけ
  (区切りを含まない)

大きさは、引数で決まるもの (範囲、`limit`、1 行の確認) はそのまま、それ以外は
`_RESULT_SIZE_TOKENS` から手番ごとに抽選する。実測で 20 文字〜4 KB ほどの幅に
なる。

## 先頭の一致 (段階をまたいだプレフィックスキャッシュ)

`k` 番目の手番の中身は `(conversation_seed, k)` だけで決まり、狙った長さには
**一切依存しない**。切れ目は必ず手番の境界に落ちる。したがって、同じ
`conversation_seed` なら、長い段階の会話は短い段階の会話を、発話の列として
そのまま先頭に含む (design.md「長い会話の検査の 1 段階」)。

対象サーバーのキャッシュに効くのは、**チャットテンプレートで組み立てられた
プロンプトの先頭**である。それが段階をまたいで一致するのは、次の 2 つによる。

1. `system` と `tools` が、どの会話でも、どの段階でも、まったく同じ (ふつう、
   チャットテンプレートはこれをプロンプトの先頭に置く)
2. 短い段階の発話の列が、長い段階の発話の列の、そのままの先頭になっている

送る本文のバイト列にも、副次的な事実として同じ性質がある。クライアントは
`build_request_body(request)` の返り値を httpx の `json=` に渡す。httpx は
`json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`
で直列化するので、鍵の並びは `MessagesRequest` の項目の定義順
(`model` → `max_tokens` → `messages` → `system` → `tools` → …) になる。
`model` と `max_tokens` が段階をまたいで同じなら (7.2 は同じ対象サーバーと
`AgentSettings.max_tokens` を使う)、**短い段階の履歴の最後の発話まで、
先頭からのバイト列が丸ごと一致する**。7.2 が足す最後の 1 手は、そのうしろに
来る。保存の部品 (`store/rawstore.py`) は `sort_keys=True` の別の直列化で
内容のハッシュを取るが、それは重複の除去のためで、送るバイト列ではない。

## 課題の番号の住み分け (7.2 と衝突させない)

履歴の指示は `make_tool_task` から作る。採点 (6.3) は引数の完全一致なので、
履歴で見せる呼び出しは、その指示に対する正解そのものでなければならない
(省略できる引数を足して見せると、模倣したモデルが最後の 1 手で `WRONG_CALL`
になる。注 6.2 / 6.3)。そのため、指示は `ToolTask.prompt` をそのまま使い、
呼び出しは `expected_tool` と `expected_input` をそのまま見せる。

住み分けは二重にしてある。

1. **種の名前空間** (これが本体): 履歴の課題の種は
   `_derive_seed(conversation_seed, 0, "conversation_history_task")` で、
   7.2 が使う `Profile.seed` とは別の名前空間から来る
2. **番号の符号** (二重の守り): 履歴が使う番号は
   `history_task_index(k) = -(k + 1)` で必ず負である (`make_tool_task` は
   負の番号も受け付ける)。7.2 は最後の 1 手に 0 以上の番号を使うこと

## 長さの狙い方

狙いは、`Profile.chars_per_token` (`chars_per_token` の引数。省略すると
`Profile` の初期値) で文字数に直して積む。前置き (システムプロンプトと
ツールの目録の JSON) も見積もりに入れる。手番は 1 つずつ足し、**狙いに最も
近い手番の境界**で止める (足す前と足した後を比べ、遠くなるなら足さない)。
手番の大きさは一定ではないので、実際の長さは狙いから最大で 1 手番ぶん
(実際にはその半分ほど) ずれる。

`fixed_tokens` を渡すと、前置きの見積もり (`_preamble_tokens`) の代わりに、
その値を「手番以外に要求へ入る決まった分量」として使う (issue #9)。7.2 は、
`system` + `tools` + 最後の 1 手 + チャットテンプレートの包みを対象サーバーに
数えさせた値を渡す。文字数と比からは導けない固定の分量 (テンプレートの展開) を、
実際のトークン数で差し引くためである。省くと、これまでどおり前置きを文字数で
見積もる。

`history_scale` は、履歴の見積もりに対する対象サーバーの実測の比 (issue #24)。
7.2 が段階 × 会話ごとに組み立てた会話を数えさせ、`(数えた長さ − 包み) ÷ 履歴の
見積もり` で決めた値を渡す。手番を足すたびの見積もりにこの比を掛けるので、止める
位置が「実測では狙いに近い手番の境界」になる。既定の 1.0 は今までと同じ見積もりで、
出力の内容 (手番そのもの) は比によって変わらない。よって `GENERATOR_VERSION` は
上げない。

`config/profiles.toml` の `quick` の比で測った実測値 (400 手番): 1 手番は
最小 47、中央値 190、平均 394、最大 1528 トークン。会話 5 本での手番の数は、
2 万トークンの段階で 36〜61、12 万トークンの段階で 286〜311 になり、狙いとの
ずれは、2 万から 12 万までのどの段階、どの会話でも 1.6% 以内
(`Profile.length_tolerance` の既定 5% の内側) である。

見積もりは文字数からの近似である。実際の長さは、対象サーバーが返した入力の
トークン数で記録する (6.7)。比は `bench calibrate` で測り直せる。

## GENERATOR_VERSION

型紙、語彙、乱数の消費の順序、手番の形を変えて出力が変わったら、
`bench_harness.types.GENERATOR_VERSION` を上げ、
`tests/unit/test_corpus_conversation.py` の golden なハッシュを更新すること
(corpus/synth.py、corpus/tools.py と同じ決まり)。

依存するのは標準ライブラリと pydantic、`bench_harness.types`、
`bench_harness.corpus.synth`、`bench_harness.corpus.tools` だけ。`client`、
`suites`、`runner`、`analysis`、`scoring` のどの module も読み込まない
(依存の向き: types → config → corpus)。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue

from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
from bench_harness.types import (
    ContentKind,
    ConversationPrefix,
    InputMessage,
    Profile,
    TextBlockParam,
    ToolResultBlockParam,
    ToolTask,
    ToolUseBlockParam,
)

__all__ = [
    "MESSAGES_PER_ROUND",
    "SYSTEM_PROMPT",
    "build_conversation",
    "derive_conversation_seed",
    "history_task_index",
    "history_tool_task",
]

MESSAGES_PER_ROUND: Final[int] = 4
"""1 手番の発話の数 (指示、呼び出し、結果、まとめ)。切れ目はこの境界に落ちる。"""

_CONVERSATION_PURPOSE: Final[str] = "conversation"
_HISTORY_TASK_PURPOSE: Final[str] = "conversation_history_task"
_ROUND_PURPOSE: Final[str] = "conversation_round"
_CONTENT_PURPOSE: Final[str] = "conversation_round_content"
_TOOL_USE_ID_PURPOSE: Final[str] = "conversation_tool_use_id"

_MAX_TARGET_TOKENS: Final[int] = 2_000_000
"""受け付ける狙いの上限 (これを超える要求は、設定の誤りとして弾く)。"""

_MAX_ROUNDS: Final[int] = 100_000
"""手番の数の安全弁 (異常な `chars_per_token` で、無限に積まないため)。"""

_RANGE_SKIP_LINES: Final[int] = 3
"""範囲の読み取りで飛ばす、ファイルの先頭の行数 (module の docstring のぶん)。"""

_MAX_RESULT_LINES: Final[int] = 60
"""引数で行数が決まる結果 (範囲の読み取り、履歴の一覧) の、見せる行数の上限。

実物のエージェントの道具も、長い出力は途中で切って注記を添える。ここで切ら
ないと、500 行の範囲や 200 件の履歴が 1 手番で数千トークンになり、2 万トークン
の段階で狙いから外れる。切るときは、何行 (何件) 見せていないかを必ず書く。
"""

_DEFAULT_PROFILE: Final[Profile] = Profile(name="corpus.conversation:default-chars-per-token")
"""`chars_per_token` を省いたときに使う初期値 (corpus/needle.py と同じ形)。

計測では 7.2 が `Profile.chars_per_token` (`config/profiles.toml` の値) を
必ず渡す。この初期値は、口を単体で使うときの当て木である。
"""


# --- 種を作る (`hash()` を使わない。corpus/tools.py と同じ形) -------------------


def _encode_part(part: str | int) -> bytes:
    """種の部品を、型と長さを含めて一意に符号化する (`corpus/tools.py` と同じ形)。"""
    if isinstance(part, str):
        data = part.encode("utf-8")
        return b"s" + str(len(data)).encode("ascii") + b":" + data
    data = str(part).encode("ascii")
    return b"i" + str(len(data)).encode("ascii") + b":" + data


def _digest(seed: int, index: int, purpose: str) -> bytes:
    parts: tuple[str | int, ...] = (seed, index, purpose)
    return hashlib.sha256(b"".join(_encode_part(part) for part in parts)).digest()


def _derive_seed(seed: int, index: int, purpose: str) -> int:
    """`(seed, index, purpose)` から、決定的な `random.Random` の種を作る。

    `hash()`、大域の乱数、時刻には依存しない。同じ引数なら、プロセスや
    `PYTHONHASHSEED` をまたいでも同じ値を返す。
    """
    return int.from_bytes(_digest(seed, index, purpose)[:8], "big")


def derive_conversation_seed(seed: int, conversation_index: int) -> int:
    """`(Profile.seed, 会話の番号)` から、1 本の会話の種を作る。

    1 段階につき複数の会話を使う (design.md: 既定 5 本) ときは、`0` から
    `conversations_per_stage - 1` までの番号でこれを呼ぶ。番号が違えば、
    1 手番目から違う会話になる。段階が変わっても、同じ番号なら同じ種なので、
    長い段階の会話が短い段階の会話を先頭に含む。
    """
    if conversation_index < 0:
        raise ValueError(
            f"derive_conversation_seed: conversation_index は 0 以上である必要がある "
            f"(conversation_index={conversation_index})"
        )
    return _derive_seed(seed, conversation_index, _CONVERSATION_PURPOSE)


def history_task_index(round_index: int) -> int:
    """履歴の手番が `make_tool_task` に渡す番号。必ず負になる (module docstring)。"""
    if round_index < 0:
        raise ValueError(
            f"history_task_index: round_index は 0 以上である必要がある (round_index={round_index})"
        )
    return -(round_index + 1)


def history_tool_task(conversation_seed: int, round_index: int) -> ToolTask:
    """`k` 番目の手番で見せる、正解の決まった課題を返す。

    指示 (`prompt`) と、見せる呼び出し (`expected_tool`、`expected_input`) は、
    どちらもこの `ToolTask` から取る。7.2 の採点と同じ「正解」を見せるための
    唯一の口である。種は 7.2 とは別の名前空間から作り、番号は負にする
    (module docstring「課題の番号の住み分け」)。
    """
    seed = _derive_seed(conversation_seed, 0, _HISTORY_TASK_PURPOSE)
    return make_tool_task(index=history_task_index(round_index), seed=seed)


def _tool_use_id(conversation_seed: int, round_index: int) -> str:
    """会話の中で一意で、決定的な `tool_use` の識別子。"""
    return f"toolu_bench_{_digest(conversation_seed, round_index, _TOOL_USE_ID_PURPOSE).hex()[:24]}"


# --- システムプロンプト (自作。長さや段階に触れない) --------------------------

SYSTEM_PROMPT: Final[str] = "\n".join(
    (
        "You are a coding agent working inside one source repository, driven by an orchestrator.",
        "",
        "How this session works:",
        "",
        "- The orchestrator sends one instruction at a time, and each instruction describes a",
        "  single step of a larger investigation or change.",
        "- For every instruction, call exactly one tool from the list you were given, and pass",
        "  exactly the arguments that the instruction states.",
        "- Do not invent values the instruction does not state, and do not pass optional",
        "  arguments the instruction does not mention.",
        "- The result of the call comes back in the next turn. Read it, then reply with one",
        "  short sentence that records what it told you.",
        "- Never answer from memory when a tool can answer instead, and never call more than",
        "  one tool for a single instruction.",
        "",
        "The repository belongs to a small team and is only visible through these tools: you",
        "have no other access to the machine, the network, or the team's own notes.",
    )
)
"""すべての会話、すべての段階で同じ前置き (これ自体をキャッシュに乗せる)。"""


# --- 語彙 (架空。既存の製品やデータの引き写しではない) ------------------------

_DIRS: Final[tuple[str, ...]] = ("src", "lib", "app", "internal", "services", "pkg")
_COMPONENTS: Final[tuple[str, ...]] = (
    "billing",
    "auth",
    "search",
    "ledger",
    "catalog",
    "router",
    "worker",
    "gateway",
    "session",
    "inventory",
)
_STEMS: Final[tuple[str, ...]] = (
    "handler",
    "client",
    "server",
    "model",
    "schema",
    "utils",
    "config",
    "adapter",
)
_EXTENSIONS: Final[tuple[str, ...]] = (".py", ".ts", ".go", ".rs")
_SYMBOLS: Final[tuple[str, ...]] = (
    "retry_budget",
    "cache_ttl",
    "rate_limit",
    "queue_depth",
    "lease_timeout",
    "batch_size",
    "backoff_delay",
    "shard_count",
)
_STATUS_CODES: Final[tuple[str, ...]] = (" M", " D", "??", "A ", "R ")
_COMMIT_SUBJECTS: Final[tuple[str, ...]] = (
    "tighten the retry budget",
    "split the batch writer",
    "drop the unused adapter",
    "record the queue depth",
    "widen the lease timeout",
    "fold the router into the gateway",
    "guard the empty shard case",
    "rename the capacity helper",
)
_AUTHORS: Final[tuple[str, ...]] = ("r.ito", "k.mori", "s.abe", "t.fujii", "n.oda")
_TODO_NOTES: Final[tuple[str, ...]] = (
    "TODO: drop this once the adapter lands",
    "TODO: confirm the bound with the owner",
    "TODO: replace with the shared helper",
    "TODO: cover the empty case",
)


def _relative_path(rng: random.Random) -> str:
    """`{component}/{stem}_{nnn}{ext}` の形の、ディレクトリのない相対の経路。"""
    return (
        f"{rng.choice(_COMPONENTS)}/{rng.choice(_STEMS)}_"
        f"{rng.randint(0, 999):03d}{rng.choice(_EXTENSIONS)}"
    )


def _random_path(rng: random.Random) -> str:
    return f"{rng.choice(_DIRS)}/{_relative_path(rng)}"


def _under(rng: random.Random, scope: str) -> str:
    """`scope` の下にある経路 (`scope` で始まることが、試験で確かめられる)。"""
    return f"{scope.rstrip('/')}/{_relative_path(rng)}"


def _entry_name(rng: random.Random) -> str:
    """ディレクトリの直下の名前 (区切りを含まない)。"""
    return f"{rng.choice(_STEMS)}_{rng.randint(0, 999):03d}{rng.choice(_EXTENSIONS)}"


def _fake_date(rng: random.Random) -> str:
    """実際の日時に依存しない、固定の年の架空の日付。"""
    return f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _dirname(path: str) -> str:
    head, _, tail = path.rpartition("/")
    return head if tail and head else path


def _pattern_suffix(pattern: str) -> str:
    """glob の最後の `*` より後ろ (`*.py` → `.py`、`**/*_test.py` → `_test.py`)。"""
    if "*" not in pattern:
        return pattern
    return pattern.rsplit("*", 1)[-1]


# --- 合成のツールの結果 -------------------------------------------------------

_RESULT_SIZE_TOKENS: Final[tuple[int, ...]] = (40, 120, 300, 700, 1200)
"""大きさが引数で決まらない結果の、狙いのトークン数。"""

_RESULT_SIZE_WEIGHTS: Final[tuple[int, ...]] = (3, 4, 4, 3, 2)
"""大きさの選ばれやすさ (短いコマンドの出力から、数 KB のファイルの中身まで)。"""


@dataclass(frozen=True)
class _ResultRequest:
    """1 つの `tool_result` を作るための入れ物。"""

    corpus: TemplateCorpus
    rng: random.Random
    size_tokens: int
    arguments: dict[str, JsonValue]
    seed: int
    chars_per_token: Mapping[ContentKind, float]

    def budget_chars(self, kind: ContentKind) -> int:
        return max(1, int(self.size_tokens * self.chars_per_token[kind]))


_ResultBuilder = Callable[[_ResultRequest], tuple[str, ContentKind]]
"""`(合成の結果の本文, その内容の種類)` を返す関数の型。"""


def _arg_str(arguments: Mapping[str, JsonValue], name: str, fallback: str) -> str:
    value = arguments.get(name)
    return value if isinstance(value, str) else fallback


def _arg_int(arguments: Mapping[str, JsonValue], name: str, fallback: int) -> int:
    value = arguments.get(name)
    if isinstance(value, bool):  # bool は int の派生型なので先に弾く
        return fallback
    return value if isinstance(value, int) else fallback


def _arg_strs(arguments: Mapping[str, JsonValue], name: str) -> list[str]:
    value = arguments.get(name)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _split_lines(text: str) -> list[str]:
    """文字数の狙いで切られた末尾の半端な行を捨てて、行に分ける。

    `TemplateCorpus.log` と `code` は、狙った文字数に合わせるために最後の行を
    途中で切ることがある。そのまま使うと、`run_tests` のまとめの行が切れた行に
    くっついた `2024-01-01T22:1469 passed in 36.7s` のような、実物ではありえない
    出力になる (レビュー round 2 の指摘)。
    """
    lines = text.split("\n")
    if not text.endswith("\n") and len(lines) > 1:
        lines = lines[:-1]  # 途中で切れた最後の行を捨てる
    return [line for line in lines if line != ""] or [text.strip()]


def _code_lines(request: _ResultRequest, minimum: int, tokens: int) -> list[str]:
    """合成のコードの行を、少なくとも `minimum` 行ぶん作る。"""
    lines: list[str] = []
    attempt = 0
    wanted = max(1, minimum)
    while len(lines) < wanted:
        chunk = request.corpus.code(
            max(20, tokens * (attempt + 1)),
            _derive_seed(request.seed, attempt, "code_chunk"),
        )
        lines = _split_lines(chunk)
        attempt += 1
        if attempt > 8:  # pragma: no cover - 比が異常なときの安全弁
            break
    return lines


def _numbered(lines: Sequence[str], first_number: int) -> list[str]:
    """行に通し番号を振る (`read_file` と `read_file_range` で同じ書式)。"""
    return [f"{first_number + offset:>6} | {line}" for offset, line in enumerate(lines)]


def _fill_lines(request: _ResultRequest, kind: ContentKind, make_line: Callable[[], str]) -> str:
    """`make_line` の行を、その種類の文字数の狙いに届くまで積む (必ず 1 行は出す)。"""
    budget = request.budget_chars(kind)
    lines: list[str] = []
    total = 0
    while total < budget:
        line = make_line()
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) + "\n"


def _result_read_file(request: _ResultRequest) -> tuple[str, ContentKind]:
    """ファイルの全文 (1 から通し番号を振る)。"""
    path = _arg_str(request.arguments, "path", _random_path(request.rng))
    lines = _code_lines(request, 1, request.size_tokens)
    body = "\n".join([f"==> {path} <==", *_numbered(lines, 1)])
    return body + "\n", ContentKind.CODE


def _result_read_file_range(request: _ResultRequest) -> tuple[str, ContentKind]:
    """範囲の読み取り。行数は `end_line - start_line + 1` (上限で切り詰める)。"""
    path = _arg_str(request.arguments, "path", _random_path(request.rng))
    start = _arg_int(request.arguments, "start_line", 1)
    end = max(start, _arg_int(request.arguments, "end_line", start))
    wanted = end - start + 1
    shown = min(wanted, _MAX_RESULT_LINES)
    # 先頭の数行 (module の docstring) を飛ばす。ファイルの途中の範囲なのに
    # docstring から始まると、実物の出力として不自然になる
    lines = _code_lines(request, shown + _RANGE_SKIP_LINES, shown * 12)[_RANGE_SKIP_LINES:][:shown]
    parts = [f"==> {path} (lines {start}-{end}) <==", *_numbered(lines, start)]
    if shown < wanted:
        parts.append(f"... {wanted - shown} more lines up to {end} not shown (output limit) ...")
    return "\n".join(parts) + "\n", ContentKind.CODE


def _result_read_files(request: _ResultRequest) -> tuple[str, ContentKind]:
    """`paths` の順に 1 つずつの節を出す。"""
    paths = _arg_strs(request.arguments, "paths") or [_random_path(request.rng) for _ in range(2)]
    per_file = max(20, request.size_tokens // len(paths))
    sections: list[str] = []
    for index, path in enumerate(paths):
        seed = _derive_seed(request.seed, index, "file_section")
        lines = _split_lines(request.corpus.code(per_file, seed))
        sections.append("\n".join([f"==> {path} <==", *_numbered(lines, 1)]))
    return "\n\n".join(sections) + "\n", ContentKind.CODE


def _result_search_text(request: _ResultRequest) -> tuple[str, ContentKind]:
    """本文の検索の当たり。どの行も `query` を文字どおり含む。"""
    query = _arg_str(request.arguments, "query", "retry_budget")
    rng = request.rng

    def line() -> str:
        return (
            f"{_random_path(rng)}:{rng.randint(1, 4000)}: "
            f'{rng.choice(_SYMBOLS)} = lookup("{query}", default={rng.randint(1, 900)})'
        )

    return _fill_lines(request, ContentKind.LOG, line), ContentKind.LOG


def _result_search_files(request: _ResultRequest) -> tuple[str, ContentKind]:
    """名前の検索の当たり。`path` の下にあり、`pattern` に一致するものだけ。"""
    pattern = _arg_str(request.arguments, "pattern", "*.py")
    scope = _arg_str(request.arguments, "path", "src/app")
    suffix = _pattern_suffix(pattern)
    rng = request.rng

    def line() -> str:
        stem = f"{rng.choice(_STEMS)}_{rng.randint(0, 999):03d}"
        return f"{scope.rstrip('/')}/{rng.choice(_COMPONENTS)}/{stem}{suffix}"

    return _fill_lines(request, ContentKind.LOG, line), ContentKind.LOG


def _result_git_status(request: _ResultRequest) -> tuple[str, ContentKind]:
    """未コミットの変更。どの行のパスも `path` の下にある。"""
    scope = _arg_str(request.arguments, "path", "src/app")
    rng = request.rng

    def line() -> str:
        return f"{rng.choice(_STATUS_CODES)} {_under(rng, scope)}"

    body = _fill_lines(request, ContentKind.LOG, line)
    return f"# uncommitted changes under {scope}\n{body}", ContentKind.LOG


def _result_git_diff(request: _ResultRequest) -> tuple[str, ContentKind]:
    """差分。パスは `path` の下、文脈の行数は前後それぞれ `context_lines`。"""
    scope = _arg_str(request.arguments, "path", "src/app")
    context = max(0, _arg_int(request.arguments, "context_lines", 3))
    rng = request.rng
    budget = request.budget_chars(ContentKind.CODE)
    hunks: list[str] = []
    total = 0
    while total < budget:
        path = _under(rng, scope)
        start = rng.randint(1, 900)
        symbol = rng.choice(_SYMBOLS)
        span = 2 * context + 1
        before = [f"     {symbol}_{index} = {rng.randint(1, 900)}" for index in range(context)]
        after = [f"     return {symbol}_{index}" for index in range(context)]
        hunk = "\n".join(
            [
                f"--- a/{path}",
                f"+++ b/{path}",
                f"@@ -{start},{span} +{start},{span} @@",
                *before,
                f"-    {symbol} = {rng.randint(1, 900)}",
                f"+    {symbol} = {rng.randint(1, 900)}",
                *after,
            ]
        )
        hunks.append(hunk)
        total += len(hunk) + 1
    return "\n".join(hunks) + "\n", ContentKind.CODE


def _result_git_log(request: _ResultRequest) -> tuple[str, ContentKind]:
    """履歴。コミットの行数はちょうど `limit` (上限で切り詰める)。"""
    scope = _arg_str(request.arguments, "path", "src/app")
    limit = max(1, _arg_int(request.arguments, "limit", 10))
    shown = min(limit, _MAX_RESULT_LINES)
    rng = request.rng
    lines = [f"# commits touching {scope}"]
    for _ in range(shown):
        sha = f"{rng.randrange(0x10000000, 0xFFFFFFFF):08x}"
        lines.append(
            f"{sha} {_fake_date(rng)} {rng.choice(_AUTHORS)} {rng.choice(_COMMIT_SUBJECTS)}"
        )
    if shown < limit:
        lines.append(f"... {limit - shown} older commits not shown (output limit) ...")
    return "\n".join(lines) + "\n", ContentKind.LOG


def _result_run_command(request: _ResultRequest) -> tuple[str, ContentKind]:
    """コマンドの出力。1 行目はコマンドの反響で、出力の形は動詞に合わせる。"""
    command = _arg_str(request.arguments, "command", "wc -l src/app/model_001.py")
    target = command.rsplit(" ", 1)[-1]
    rng = request.rng
    lines = [f"$ {command}"]
    if command.startswith("du -sh"):
        lines.append(f"{rng.randint(4, 980)}K\t{target}")
    elif command.startswith("wc -l"):
        lines.append(f"{rng.randint(10, 4000):>8} {target}")
    elif command.startswith("grep -rn"):
        budget = request.budget_chars(ContentKind.LOG)
        total = 0
        while total < budget:
            hit = f"{target}:{rng.randint(1, 4000)}:    # {rng.choice(_TODO_NOTES)}"
            lines.append(hit)
            total += len(hit) + 1
    else:  # find
        directory = _dirname(target)
        budget = request.budget_chars(ContentKind.LOG)
        lines.append(target)
        total = len(target)
        while total < budget:
            found = _under(rng, directory)
            lines.append(found)
            total += len(found) + 1
    return "\n".join(lines) + "\n", ContentKind.LOG


def _result_run_tests(request: _ResultRequest) -> tuple[str, ContentKind]:
    """試験のログ。行の切れ目で改行してから、まとめの 1 行で終わる。"""
    target = _arg_str(request.arguments, "target", "src/app/model_001.py")
    log_lines = _split_lines(request.corpus.log(request.size_tokens, request.seed))
    passed = request.rng.randint(10, 400)
    seconds = request.rng.randint(1, 900) / 10
    lines = [f"# tests for {target}", *log_lines, f"{passed} passed in {seconds:.1f}s"]
    return "\n".join(lines) + "\n", ContentKind.LOG


def _result_write_file(request: _ResultRequest) -> tuple[str, ContentKind]:
    """書き込みの確認。バイト数は `content` の実際の長さに合う。"""
    path = _arg_str(request.arguments, "path", "src/app/model_001.py")
    content = _arg_str(request.arguments, "content", "")
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}\n", ContentKind.LOG


def _result_edit_file(request: _ResultRequest) -> tuple[str, ContentKind]:
    """置き換えの差分。見出しは `path`、消えた行は `old_text`、足された行は `new_text`。"""
    path = _arg_str(request.arguments, "path", "src/app/model_001.py")
    old_text = _arg_str(request.arguments, "old_text", "old")
    new_text = _arg_str(request.arguments, "new_text", "new")
    start = request.rng.randint(1, 900)
    symbol = request.rng.choice(_SYMBOLS)
    lines = [
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -{start},3 +{start},3 @@",
        f"     {symbol} = {request.rng.randint(1, 900)}",
        f"-{old_text}",
        f"+{new_text}",
        f"     return {symbol}",
    ]
    return "\n".join(lines) + "\n", ContentKind.CODE


def _result_delete_file(request: _ResultRequest) -> tuple[str, ContentKind]:
    path = _arg_str(request.arguments, "path", "src/app/model_001.py")
    return f"deleted {path}\n", ContentKind.LOG


def _result_move_file(request: _ResultRequest) -> tuple[str, ContentKind]:
    source = _arg_str(request.arguments, "source", "src/app/a.py")
    destination = _arg_str(request.arguments, "destination", "src/app/b.py")
    return f"moved {source} -> {destination}\n", ContentKind.LOG


def _result_create_directory(request: _ResultRequest) -> tuple[str, ContentKind]:
    path = _arg_str(request.arguments, "path", "src/app/new_001")
    return f"created directory {path}\n", ContentKind.LOG


def _result_fetch_url(request: _ResultRequest) -> tuple[str, ContentKind]:
    """HTTP の往復。1 行目は要求の反響で、状態と本文は `method` に合わせる。"""
    url = _arg_str(request.arguments, "url", "https://api.internal.example/v1/health")
    method = _arg_str(request.arguments, "method", "GET")
    rng = request.rng
    lines = [f"> {method} {url}", "> accept: application/json", ""]
    if method == "DELETE":
        lines += ["< HTTP/1.1 204 No Content", "< connection: keep-alive"]
        return "\n".join(lines) + "\n", ContentKind.CODE
    status = "201 Created" if method == "POST" else "200 OK"
    lines += [f"< HTTP/1.1 {status}", "< content-type: application/json", ""]
    if method in ("POST", "PUT"):
        lines.append(
            f'{{"accepted":true,"id":"{rng.randrange(0x100000, 0xFFFFFF):06x}",'
            f'"queued":{rng.randint(1, 90)}}}'
        )
        return "\n".join(lines) + "\n", ContentKind.CODE
    budget = request.budget_chars(ContentKind.CODE)
    records: list[str] = []
    total = 0
    while total < budget:
        record = (
            f'  {{"id":"{rng.randrange(0x100000, 0xFFFFFF):06x}",'
            f'"{rng.choice(_SYMBOLS)}":{rng.randint(1, 900)},'
            f'"updated":"{_fake_date(rng)}"}}'
        )
        records.append(record)
        total += len(record) + 1
    lines.append("[")
    lines.append(",\n".join(records))
    lines.append("]")
    return "\n".join(lines) + "\n", ContentKind.CODE


def _result_list_directory(request: _ResultRequest) -> tuple[str, ContentKind]:
    """直下の項目だけ (名前に区切りを含まない)。見出しが `path` を名指しする。"""
    path = _arg_str(request.arguments, "path", "src/app")
    rng = request.rng

    def line() -> str:
        if rng.random() < 0.2:
            name = f"{rng.choice(_COMPONENTS)}/"
            return f"{name:<28} {'-':>8}        {_fake_date(rng)}"
        return f"{_entry_name(rng):<28} {rng.randint(100, 900_000):>8} bytes  {_fake_date(rng)}"

    body = _fill_lines(request, ContentKind.LOG, line)
    return f"# {path}\n{body}", ContentKind.LOG


_RESULT_BUILDERS: Final[dict[str, _ResultBuilder]] = {
    "read_file": _result_read_file,
    "read_file_range": _result_read_file_range,
    "read_files": _result_read_files,
    "search_text": _result_search_text,
    "search_files": _result_search_files,
    "git_status": _result_git_status,
    "git_diff": _result_git_diff,
    "git_log": _result_git_log,
    "run_command": _result_run_command,
    "run_tests": _result_run_tests,
    "write_file": _result_write_file,
    "edit_file": _result_edit_file,
    "delete_file": _result_delete_file,
    "move_file": _result_move_file,
    "create_directory": _result_create_directory,
    "fetch_url": _result_fetch_url,
    "list_directory": _result_list_directory,
}
"""ツールごとの、合成の結果の作り方 (目録のすべてのツールに 1 つずつ)。"""


# --- まとめの 1 文 (呼んだツールと引数に触れる) -------------------------------

_SUMMARY_TEMPLATES: Final[dict[str, tuple[str, ...]]] = {
    "read_file": (
        "Read {path} from top to bottom",
        "{path} is open in front of me now",
        "Went through the whole of {path}",
    ),
    "read_file_range": (
        "Looked at lines {start_line}-{end_line} of {path}",
        "That slice of {path} is in front of me now",
        "Read the requested window of {path}",
    ),
    "read_files": (
        "Read all of the files you listed, starting with {first_path}",
        "Have the contents of {count} files, {first_path} among them",
        "Went through each file in turn, {first_path} first",
    ),
    "search_text": (
        "Collected every hit for {query}",
        "The search for {query} came back with matches across the tree",
        "Have the list of places where {query} shows up",
    ),
    "search_files": (
        "Listed the files under {path} matching {pattern}",
        "The name search under {path} came back",
        "Have every {pattern} file under {path}",
    ),
    "git_status": (
        "Checked what is uncommitted under {path}",
        "The working tree under {path} has pending changes",
        "Have the list of touched files under {path}",
    ),
    "git_diff": (
        "Read the diff under {path} with {context_lines} lines of context",
        "The pending changes under {path} are in front of me",
        "Went through the hunks under {path}",
    ),
    "git_log": (
        "Read the last {limit} commits touching {path}",
        "Have the recent history for {path}",
        "The commit list for {path} came back",
    ),
    "run_command": (
        "Ran {command} and kept the output",
        "The command finished inside {timeout_s} seconds",
        "Have the output of {command}",
    ),
    "run_tests": (
        "Ran the tests for {target}; the summary is at the end",
        "The suite for {target} finished",
        "Have the test log for {target}",
    ),
    "write_file": (
        "Wrote the new file at {path}",
        "{path} now holds exactly what you gave me",
        "Created {path} with that content",
    ),
    "edit_file": (
        "Replaced that text in {path}",
        "The edit in {path} is applied",
        "{path} now reads with the new text in place",
    ),
    "delete_file": (
        "Removed {path}",
        "{path} is gone from the tree",
        "Deleted {path} as asked",
    ),
    "move_file": (
        "Moved {source} to {destination}",
        "{source} now lives at {destination}",
        "The move to {destination} is done",
    ),
    "create_directory": (
        "Created the directory {path}",
        "{path} exists and is empty",
        "The new directory at {path} is in place",
    ),
    "fetch_url": (
        "Fetched {url} and kept the response",
        "The {method} request to {url} came back",
        "Have the response body from {url}",
    ),
    "list_directory": (
        "Listed what sits directly inside {path}",
        "The entries under {path} are in front of me",
        "Have the directory listing for {path}",
    ),
}
"""ツールごとのまとめの型紙。引数を埋めるので、手番ごとに文が変わる。"""

_SUMMARY_TAILS: Final[tuple[str, ...]] = (
    "",
    ", and nothing looked out of place",
    ", so I am ready for the next step",
    " before the next instruction",
)
"""型紙のうしろに足す節。文の終わりの「.」は `_summary` が 1 つだけ付ける。"""


def _summary_fields(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
    """型紙に埋める値 (足りない鍵は、当たり障りのない言い方で埋める)。"""
    fields = {
        key: (str(value) if not isinstance(value, list) else ", ".join(str(item) for item in value))
        for key, value in arguments.items()
    }
    paths = _arg_strs(arguments, "paths")
    fields["first_path"] = paths[0] if paths else fields.get("path", "the file")
    fields["count"] = str(len(paths)) if paths else "the"
    return fields


def _summary(rng: random.Random, task: ToolTask) -> str:
    """呼んだツールと引数に触れる、1 文のまとめ。

    文は必ず 1 つ (途中に「. 」を作らない)。型紙は句点を持たず、節を足したあとに
    ここで「.」を 1 つだけ付ける。
    """
    templates = _SUMMARY_TEMPLATES[task.expected_tool]
    template = rng.choice(templates)
    tail = rng.choice(_SUMMARY_TAILS)
    fields = _summary_fields(task.expected_input)
    try:
        sentence = template.format(**fields)
    except KeyError:  # pragma: no cover - 型紙と引数が食い違ったときの当て木
        sentence = "Recorded the result"
    return f"{sentence}{tail}."


# --- 1 手番の組み立て ---------------------------------------------------------


@dataclass(frozen=True)
class _Round:
    """1 手番の発話と、そのおよそのトークン数。"""

    messages: tuple[InputMessage, ...]
    approx_tokens: float


def _estimate_tokens(text: str, kind: ContentKind, ratios: Mapping[ContentKind, float]) -> float:
    return len(text) / ratios[kind]


def _arguments_json(arguments: Mapping[str, JsonValue]) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _build_round(
    corpus: TemplateCorpus,
    ratios: Mapping[ContentKind, float],
    conversation_seed: int,
    round_index: int,
) -> _Round:
    """`k` 番目の手番を作る。`(conversation_seed, round_index)` だけで決まる。

    狙った長さは一切見ない (見ると、段階をまたいだ先頭の一致が壊れる)。
    """
    task = history_tool_task(conversation_seed, round_index)
    rng = random.Random(_derive_seed(conversation_seed, round_index, _ROUND_PURPOSE))
    call_id = _tool_use_id(conversation_seed, round_index)

    size_tokens = rng.choices(_RESULT_SIZE_TOKENS, weights=_RESULT_SIZE_WEIGHTS)[0]
    arguments = dict(task.expected_input)
    result_text, result_kind = _RESULT_BUILDERS[task.expected_tool](
        _ResultRequest(
            corpus=corpus,
            rng=rng,
            size_tokens=size_tokens,
            arguments=arguments,
            seed=_derive_seed(conversation_seed, round_index, _CONTENT_PURPOSE),
            chars_per_token=ratios,
        )
    )
    summary = _summary(rng, task)

    messages = (
        InputMessage(role="user", content=[TextBlockParam(text=task.prompt)]),
        InputMessage(
            role="assistant",
            content=[ToolUseBlockParam(id=call_id, name=task.expected_tool, input=arguments)],
        ),
        InputMessage(
            role="user",
            content=[ToolResultBlockParam(tool_use_id=call_id, content=result_text)],
        ),
        InputMessage(role="assistant", content=[TextBlockParam(text=summary)]),
    )
    approx = (
        _estimate_tokens(task.prompt, ContentKind.PROSE_EN, ratios)
        + _estimate_tokens(task.expected_tool, ContentKind.PROSE_EN, ratios)
        + _estimate_tokens(_arguments_json(arguments), ContentKind.CODE, ratios)
        + _estimate_tokens(result_text, result_kind, ratios)
        + _estimate_tokens(summary, ContentKind.PROSE_EN, ratios)
    )
    return _Round(messages=messages, approx_tokens=approx)


def _preamble_tokens(ratios: Mapping[ContentKind, float]) -> float:
    """システムプロンプトとツールの目録の、およそのトークン数 (どの会話でも同じ)。"""
    tools_json = json.dumps(
        [tool.model_dump(mode="json") for tool in TOOL_CATALOG],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _estimate_tokens(SYSTEM_PROMPT, ContentKind.PROSE_EN, ratios) + _estimate_tokens(
        tools_json, ContentKind.CODE, ratios
    )


# --- 会話の組み立て -----------------------------------------------------------


def build_conversation(
    target_tokens: int,
    conversation_seed: int,
    *,
    chars_per_token: Mapping[ContentKind, float] | None = None,
    fixed_tokens: float | None = None,
    history_scale: float = 1.0,
) -> ConversationPrefix:
    """takt の作業を模した会話を、狙った長さまで組み立てる (design.md corpus、6.1、6.2)。

    `conversation_seed` は `derive_conversation_seed(Profile.seed, 会話の番号)`
    で作る。`chars_per_token` は `Profile.chars_per_token` を渡す (省くと
    `Profile` の初期値)。同じ `conversation_seed` と同じ比なら、狙いが長い会話は
    狙いが短い会話を、そのまま先頭に含む (module docstring)。

    `fixed_tokens` は、手番以外に要求へ入る決まった分量の実測値 (issue #9)。
    渡すと、前置きの見積もり (`_preamble_tokens`) の代わりにこの値を使う。7.2 は
    `system` + `tools` + 最後の 1 手 + チャットテンプレートの包みを対象サーバーに
    数えさせた値を渡す。返り値の `approx_tokens` は、この値 + 履歴の長さになる
    (最後の 1 手まで含む)。省くと、これまでどおり前置き + 履歴の見積もりを返し、
    最後の 1 手は含まない。

    `history_scale` は、履歴の見積もりに対する対象サーバーの実測の比 (issue #24)。
    手番を足すたびの見積もりに掛ける。7.2 が `(数えた長さ − 包み) ÷ 履歴の
    見積もり` で決めた値を渡す。`approx_tokens` は
    `round(fixed_tokens + history_scale × Σ手番の見積もり)` になる。出力そのもの
    (system、tools、手番の中身) は比によって変わらないので、`GENERATOR_VERSION` は
    上げない。

    同じ引数なら出力は決定的で、`fixed_tokens` と `history_scale` を変えると
    止める位置だけが変わる。
    """
    if target_tokens <= 0:
        raise ValueError(
            f"build_conversation: target_tokens は正の整数である必要がある "
            f"(target_tokens={target_tokens})"
        )
    if target_tokens > _MAX_TARGET_TOKENS:
        raise ValueError(
            f"build_conversation: target_tokens が大きすぎる "
            f"(target_tokens={target_tokens}, 上限={_MAX_TARGET_TOKENS})"
        )
    if fixed_tokens is not None and (not math.isfinite(fixed_tokens) or fixed_tokens < 0):
        raise ValueError(
            f"build_conversation: fixed_tokens は 0 以上の有限の数 (fixed_tokens={fixed_tokens})"
        )
    if not math.isfinite(history_scale) or history_scale <= 0:
        raise ValueError(
            f"build_conversation: history_scale は 0 より大きい有限の数 "
            f"(history_scale={history_scale})"
        )

    ratios = (
        dict(chars_per_token)
        if chars_per_token is not None
        else dict(_DEFAULT_PROFILE.chars_per_token)
    )
    # 足りない種類、0 以下、nan / inf は、生成を始める前にここで弾かれる
    corpus = TemplateCorpus(ratios)

    messages: list[InputMessage] = []
    total = _preamble_tokens(ratios) if fixed_tokens is None else fixed_tokens
    round_index = 0
    while round_index < _MAX_ROUNDS:
        current = _build_round(corpus, ratios, conversation_seed, round_index)
        candidate = total + history_scale * current.approx_tokens
        if messages and abs(candidate - target_tokens) > abs(total - target_tokens):
            break  # 足すと狙いから遠ざかる。手番の境界で止める
        messages.extend(current.messages)
        total = candidate
        round_index += 1
    else:  # pragma: no cover - 異常な chars_per_token でだけ通る
        raise ValueError(
            f"build_conversation: 手番が {_MAX_ROUNDS} を超えた。chars_per_token が "
            f"大きすぎる可能性がある (target_tokens={target_tokens})"
        )

    return ConversationPrefix(
        system=SYSTEM_PROMPT,
        tools=list(TOOL_CATALOG),
        messages=messages,
        approx_tokens=max(0, round(total)),
        conversation_seed=conversation_seed,
    )
