"""モデルの応答からコードを取り出し、課題の検査と組み合わせて採点する (task 6.6、5.2)。

3 つの純粋な関数からなる。コンテナには触れない (動かすのは
`bench_harness.scoring.sandbox`)。依存するのは標準ライブラリ、pydantic、
`bench_harness.types` だけである (design.md の依存の向き: 採点は corpus /
client / suites / runner / analysis を読み込まない)。

- `extract_code(response_text, entry_point)`: 応答の本文から、候補のコードを
  取り出す。使えるものがなければ `None`
- `build_checked_program(problem, code)` → `CheckedProgram`: 指示 + 候補のコード +
  課題の検査 + `check(<entry_point>)` + **検査を通ったことの印** を組み立てる
- `score_checked(result, program)`: 印まで届いたかどうかも見て採点する
- `build_program(problem, code)` / `score_code(result)`: 印のない、弱いほうの
  道すじ (下記)。残してあるが、6.7 は上の 2 つを使うこと

## コードの取り出しの順序 (design.md が沈黙しているので、ここで決める)

上から順に、最初に当てはまったものを採る。

1. 本文を `MAX_RESPONSE_CHARS` 文字までに切ってから、正規化する。CRLF と CR を
   LF に直し、`<think>…</think>` の塊を落とし (考えの塊そのものは別のブロックと
   して届くが、本文に紛れ込む形もありうる)、**行頭の**タブを 4 個の空白に開く
   (行頭だけを直すので、文字列の中のタブは触らない)。行頭のタブを直すのは、
   指示の側が 4 個の空白で字下げしているため。混ざると `TabError` で、正直な
   正解まで落ちる
2. 囲みのブロック (``` または ~~~) を全部探す。言語の名前が空、`python`、
   `py`、`python3` で、かつ中身が空でないものだけを候補にする (言語の名前の
   大小文字は区別しない)。閉じ忘れた囲みは、本文の終わりまでを中身とする
   (出力の上限で切れた応答のため)
3. 候補のブロックが 1 つ以上あるとき:
   1. **入口の関数 (`def <entry_point>(`) を定義しているブロック**のうち、
      最初のものを採る。「試験を先に書いてから解答を書く」応答と、「解答の
      あとに使用例を書く」応答の、どちらでも解答のほうを採るための規則である
   2. どのブロックも定義していなければ、**いちばん長いブロック** (同じ長さなら
      先のもの) を採る。この場合に当てはまるのは、本体だけを書いた応答や、
      補助の関数だけを書いた応答である。あとのブロックを採ると、末尾の短い
      使用例を拾ってしまうので、長いほうを採る
4. 候補のブロックが 1 つもないとき (囲みがない応答、囲みが全部ほかの言語だった
   応答) は、囲みの中身を取り除いた残りの本文を見る:
   1. `def <entry_point>(` があれば、その行から (直前の空行・import・装飾子・
      注釈の行を巻き込んで) 末尾までを採り、`ast.parse` が通るまで末尾の行を
      1 行ずつ落とす (コードのあとに説明を続ける応答のため)
   2. すべての行が字下げされていれば、本体だけの応答とみなして、そのまま採る
   3. どちらでもなければ `None` (説明だけの応答、ほかの言語だけの応答)
5. 採ったコードから、`if __name__ == "__main__":` の塊を落とす (下記)
6. 中身が空になったら `None`

`None` は「モデルは答えたが、使えるコードがなかった」であり、呼び出し側は
**不正解** (`no_code_verdict()`) にする。「採点できなかった」(5.7) にしてよいのは、
要求そのものが失敗した場合と、隔離の実行環境が使えない場合だけである
(`not_scored_verdict()`)。

`extract_code` は、どんな本文に対しても例外を投げない。

## `if __name__ == "__main__":` を落とす理由

組み立てたプログラムは、**標準入力で**コンテナに渡す (`python -I -`)。そのため、
候補のコードの中の `input()` は、すでに終わりまで読まれた標準入力から読むことに
なり、`EOFError` になる (止まりはしない。実機で確かめた)。ところが `check(...)`
の呼び出しはプログラムの末尾にあるので、その手前で `EOFError` が出ると、正しい
解答でも不正解になってしまう。実演のコードは `if __name__ == "__main__":` の下に
書かれるのが普通なので、この塊 (と、その下の字下げされた行) だけを落とす。
それ以外の実演のコード (`print(...)` を直に書くなど) は落とさない。

## 検査を通ったことの印 (sentinel) — 終了コードだけでは足りない

隔離の側が見られるのは終了コードだけなので、候補のコードが末尾で
`sys.exit(0)` / `os._exit(0)` / `raise SystemExit(0)` を呼ぶと、`check(...)` に
届かないまま終了コード 0 になり、**検査を受けずに正解**になってしまう。
ふつうの応答はこれをしない (実演のコードは `if __name__ == "__main__":` の塊
ごと落としている) が、抜け道は塞いでおく。

`build_checked_program` は `check(<entry_point>)` の**あと**に、プログラムから
決まる印 (`SENTINEL_PREFIX` + SHA-256 の先頭 32 桁) を**記述子 2 に直接**書く
数行を足す。`score_checked` は、終了コードが 0 で、**かつ**標準エラーの最後の
行がその印であることを求める。

- 印はプログラムの中身から決まるので、同じ答えなら毎回同じ (11.4 の再現性)。
  候補のコードは、課題の検査も、この module の作り方も読めない (プログラムは
  標準入力で渡され、読み終わってから実行が始まる) ので、言い当てられない
- 記述子 2 に直接書くのは、`sys.stderr` を差し替える候補のコードに負けない
  ため。書く前に `sys.stderr` と `sys.__stderr__` を流して、書きかけの行が
  通訳器の終了時に印の**あと**へ出るのを防ぐ
- 標準エラーの末尾だけを覚えておく仕組み (`stderr_tail`) と噛み合う。印は
  いちばん最後に書かれるので、途中でどれだけ吐いても末尾には残る
- これは「うっかり / 素朴な抜け道」を塞ぐもので、同じ通訳器の中で `os.write`
  を差し替えるような本気の細工までは防げない (防ぎようがない)

`build_program` + `score_code` は、この印を持たない**弱いほうの**道すじである。
既存の呼び出しのために残してあるが、新しく採点をつなぐ側は
`build_checked_program` + `score_checked` を使うこと。

## 指示を前に置く理由 (note 6.4: 正直な正解を不正解にしない)

HumanEval+ の指示は `from typing import List` のような import と、関数の署名と、
説明の文字列で終わる。会話の形で答えるモデルは、関数をまるごと書き直すが、
import までは書かないことが多い。候補のコードだけを動かすと、`List[float]` の
注釈が `NameError` になり、**正しい解答が不正解になる**。

そこで `build_program` は、候補のコードの前に**必ず指示を丸ごと置く**。指示の
末尾の関数は「説明の文字列だけを本体に持つ関数」として構文的に完結しているので、

- 本体だけを書いた応答 (completion 形式) は、そのまま指示の続きになる
- 関数をまるごと書いた応答 (chat 形式) は、指示の関数を上書きする

のどちらも、同じ組み立て方で動く。場合分けをしないので、取り出したコードが
どちらの形かを当てる必要がない。

## 理由の文字列

`QualityVerdict.detail` は `MAX_DETAIL_CHARS` 文字までに収める。入れるのは
終了コードと、標準エラーの**最後の 1 行**だけである。標準エラーをまるごと
持ち回らない。最後の 1 行を採るのは、`SyntaxError` のときに手前の行へ
候補のコードそのものが現れるためでもある (最後の行は `SyntaxError: ...` の
ような型と説明になる)。
"""

from __future__ import annotations

import ast
import hashlib
import keyword
import re
from dataclasses import dataclass
from typing import Final

from bench_harness.types import CodeProblem, QualityOutcome, QualityVerdict, SandboxResult

__all__ = [
    "MAX_DETAIL_CHARS",
    "MAX_RESPONSE_CHARS",
    "SENTINEL_PREFIX",
    "TAB_WIDTH",
    "CheckedProgram",
    "CodeTaskError",
    "build_checked_program",
    "build_program",
    "extract_code",
    "no_code_verdict",
    "not_scored_verdict",
    "score_checked",
    "score_code",
]

MAX_DETAIL_CHARS: Final[int] = 200
"""`QualityVerdict.detail` の長さの上限。"""

MAX_RESPONSE_CHARS: Final[int] = 500_000
"""取り出しで見る本文の長さの上限。これを超えた分は見ない。"""

TAB_WIDTH: Final[int] = 4
"""行頭のタブを開く幅。HumanEval+ の指示の字下げに合わせる。"""

_MAX_STDERR_LINE_CHARS: Final[int] = 120
"""理由に入れる、標準エラーの最後の 1 行の長さの上限。"""

_MAX_TRIM_LINES: Final[int] = 40
"""囲みのない応答で、末尾から落としてみる行の数の上限。"""

SENTINEL_PREFIX: Final[str] = "__bench_ok_"
"""検査を通ったことの印の頭。うしろに、プログラムから決まる 32 桁が付く。"""

_SENTINEL_DIGITS: Final[int] = 32
"""印に使う、プログラムの SHA-256 の先頭の桁数。"""

_SIGKILL_EXIT_CODE: Final[int] = 137
"""128 + SIGKILL。資源の上限 (メモリ) に当たったコードが、これで落ちる。"""

_SIGKILL_DETAIL: Final[str] = "killed_sigkill_or_oom"
"""137 のときに理由へ添える印 (標準エラーが空になることが多いため)。"""

_PYTHON_LANGUAGES: Final[frozenset[str]] = frozenset({"", "python", "py", "python3"})
"""囲みの言語の名前のうち、Python とみなすもの (小文字にして比べる)。"""

_THINK_TAG_RE: Final[re.Pattern[str]] = re.compile(r"</?think(?:ing)?>", re.IGNORECASE)
"""考えの塊の開きと閉じの印。

以前は `<think>.*?</think>` (DOTALL) で塊ごと消していたが、閉じ忘れた印が
並ぶ応答 (壊れたモデルの繰り返し) では、開始位置ごとに末尾まで走査するので
**二次**になった (0.5 MB の `"<think>"` の繰り返しで 68 秒)。`MAX_RESPONSE_CHARS`
での切り詰めが、閉じ印を落としてこの形を作ってしまうこともある。いまは印の
位置だけを 1 回の `finditer` で集め、対応付けは Python 側で行う (直線)。
この module の正規表現は、どれも `re.DOTALL` を立てない。"""

_FENCE_OPEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$"
)
_FENCE_CLOSE_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")

_PREAMBLE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:from[ \t]+\S|import[ \t]+\S|@\w|#)")
"""入口の関数の手前で、巻き込んでよい行 (import、装飾子、注釈)。"""

_MAIN_GUARD_RE: Final[re.Pattern[str]] = re.compile(r"^if[ \t]+__name__[ \t]*==")
"""実演のコードの入り口。字下げなし (列 0) のものだけを見る。"""

_CHECK_DEF_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*def[ \t]+check[ \t]*\(", re.MULTILINE)
"""課題の検査のプログラムが持っているはずの、`check(candidate)` の定義。"""


class CodeTaskError(Exception):
    """課題 (`CodeProblem`) の側の誤り。モデルの応答のせいにせず、ここで止める。"""


# --- 1. コードの取り出し -----------------------------------------------------


@dataclass(frozen=True)
class _Block:
    """囲みのブロック 1 つ。"""

    info: str
    body: str
    start: int
    end: int


def _expand_leading_tabs(line: str) -> str:
    """行頭のタブだけを空白に開く (文字列の中のタブは触らない)。"""
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    return indent.replace("\t", " " * TAB_WIDTH) + stripped


def _strip_thinking(text: str) -> str:
    """考えの塊を落とす。印の位置を 1 回だけ走査する (直線の時間)。

    - 対応が取れた `<think>…</think>` は、**塊ごと**落とす (入れ子の開き印は、
      外側の塊の一部として一緒に消える。以前の非貪欲な正規表現と同じ結果)
    - 閉じ忘れた `<think>`、対応のない `</think>` は、**印だけ**落として本文は
      残す。閉じ忘れの後ろを全部「考え」として捨てると、答えを書いたのに
      閉じ忘れただけの応答をまるごと失う。印だけ落とせば、考えの文が本文に
      混ざるだけで、囲みのブロックや `def` の探索で弾かれる。誤って不正解に
      しない側を採る (note 6.4)
    """
    pieces: list[str] = []
    cursor = 0
    inside = False
    for match in _THINK_TAG_RE.finditer(text):
        is_open = not match.group(0).startswith("</")
        if not inside:
            # 印の手前までは残す。開き印なら、ここから塊が始まる。対応のない
            # 閉じ印なら、印だけを落として先へ進む
            pieces.append(text[cursor : match.start()])
            cursor = match.end()
            inside = is_open
            continue
        if is_open:
            continue  # 塊の中の開き印は、塊の一部として消える
        # 対応が取れた。前後の行がくっつかないように、塊は改行 1 つに置き換える
        pieces.append("\n")
        cursor = match.end()
        inside = False
    # 残り。閉じ忘れなら、開き印より後ろの本文がそのまま残る
    pieces.append(text[cursor:])
    return "".join(pieces)


def _normalize(text: str) -> str:
    """改行、考えの塊、行頭のタブをそろえる (module の docstring の 1)。"""
    flat = text.replace("\r\n", "\n").replace("\r", "\n")
    flat = _strip_thinking(flat)
    return "\n".join(_expand_leading_tabs(line) for line in flat.split("\n"))


def _fenced_blocks(text: str) -> list[_Block]:
    """囲みのブロックを、現れた順に返す (閉じ忘れは本文の終わりまで)。"""
    lines = text.split("\n")
    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        opened = _FENCE_OPEN_RE.match(lines[index])
        if opened is None:
            index += 1
            continue
        marker = opened.group("fence")
        indent = opened.group("indent")
        info = opened.group("info").strip()
        start = index
        index += 1
        body: list[str] = []
        while index < len(lines):
            closed = _FENCE_CLOSE_RE.match(lines[index])
            if (
                closed is not None
                and closed.group("fence")[0] == marker[0]
                and len(closed.group("fence")) >= len(marker)
            ):
                index += 1
                break
            body.append(lines[index])
            index += 1
        blocks.append(_Block(info=info, body=_strip_indent(body, indent), start=start, end=index))
    return blocks


def _strip_indent(body: list[str], indent: str) -> str:
    """囲みが字下げされていて、中身も全部その字下げを持つなら、剥がす。"""
    if indent and all(not line.strip() or line.startswith(indent) for line in body):
        body = [line[len(indent) :] if line.strip() else line for line in body]
    text = "\n".join(body)
    return text + "\n" if text and not text.endswith("\n") else text


def _without_blocks(text: str, blocks: list[_Block]) -> str:
    """囲みのブロックを取り除いた、残りの本文。"""
    if not blocks:
        return text
    lines = text.split("\n")
    covered = {index for block in blocks for index in range(block.start, block.end)}
    return "\n".join(line for index, line in enumerate(lines) if index not in covered)


def _definition_re(entry_point: str) -> re.Pattern[str] | None:
    """`def <entry_point>(` を探す正規表現。名前が識別子でなければ `None`。"""
    name = entry_point.strip()
    if not name.isidentifier() or keyword.iskeyword(name):
        return None
    return re.compile(rf"^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(name)}[ \t]*\(", re.MULTILINE)


def _defines(text: str, entry_point: str) -> bool:
    pattern = _definition_re(entry_point)
    return pattern is not None and pattern.search(text) is not None


def _is_indented_body(text: str) -> bool:
    """すべての中身のある行が字下げされている (本体だけの応答) かどうか。"""
    lines = [line for line in text.split("\n") if line.strip()]
    return bool(lines) and all(line[:1] in (" ", "\t") for line in lines)


def _parses(source: str) -> bool:
    """Python の構文として読めるかどうか。外から来る本文なので、投げない。"""
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def _trim_to_parsable(source: str) -> str:
    """末尾の行を 1 行ずつ落として、構文として読める形を探す。"""
    if _parses(source):
        return source
    current = source
    for _ in range(_MAX_TRIM_LINES):
        lines = current.rstrip("\n").split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            break
        lines.pop()
        current = "\n".join(lines)
        if not current.strip():
            break
        if _parses(current):
            return current
    return source


def _code_region(text: str, entry_point: str) -> str:
    """入口の関数の定義を含む、コードらしい範囲を切り出す。"""
    pattern = _definition_re(entry_point)
    if pattern is None:  # pragma: no cover - 呼び出し元が `_defines` で確かめている
        return text
    lines = text.split("\n")
    start = next(index for index, line in enumerate(lines) if pattern.match(line))
    while start > 0:
        previous = lines[start - 1]
        if previous.strip() and not _PREAMBLE_RE.match(previous):
            break
        start -= 1
    return "\n".join(lines[start:])


def _strip_main_guard(code: str) -> str:
    """`if __name__ == "__main__":` の塊 (と、その下の字下げされた行) を落とす。"""
    lines = code.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if _MAIN_GUARD_RE.match(lines[index]):
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index][:1] in (" ", "\t")
            ):
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept)


def _finish(code: str) -> str | None:
    """最後の仕上げ (module の docstring の 5、6)。"""
    trimmed = _strip_main_guard(code).rstrip()
    return f"{trimmed}\n" if trimmed.strip() else None


def _from_plain_text(text: str, entry_point: str) -> str | None:
    """囲みのない本文から、コードらしい部分を切り出す (docstring の 4)。"""
    if _defines(text, entry_point):
        return _trim_to_parsable(_code_region(text, entry_point))
    if _is_indented_body(text):
        return text
    return None


def extract_code(response_text: str, entry_point: str) -> str | None:
    """応答の本文から、候補のコードを取り出す (module の docstring の順序)。

    使えるコードがなければ `None`。どんな本文でも例外を投げない。
    """
    if not response_text:
        return None
    text = _normalize(response_text[:MAX_RESPONSE_CHARS])
    blocks = _fenced_blocks(text)
    accepted = [block for block in blocks if _is_python_language(block.info) and block.body.strip()]
    if accepted:
        named = [block for block in accepted if _defines(block.body, entry_point)]
        chosen = named[0] if named else max(accepted, key=lambda block: len(block.body.strip()))
        return _finish(chosen.body)
    candidate = _from_plain_text(_without_blocks(text, blocks), entry_point)
    return _finish(candidate) if candidate is not None else None


def _is_python_language(info: str) -> bool:
    """囲みの言語の名前が、Python とみなせるかどうか。"""
    parts = info.split()
    return (parts[0].lower() if parts else "") in _PYTHON_LANGUAGES


# --- 2. プログラムの組み立て -------------------------------------------------


def _normalize_source(source: str) -> str:
    """改行と行頭のタブをそろえる (指示と候補のコードで字下げを混ぜない)。"""
    flat = source.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(_expand_leading_tabs(line) for line in flat.split("\n"))


def _with_newline(source: str) -> str:
    return source if source.endswith("\n") else f"{source}\n"


def build_program(problem: CodeProblem, code: str) -> str:
    """指示 + 候補のコード + 課題の検査 + `check(<entry_point>)` を組み立てる。

    `DatasetRef.scoring_method` (`corpus/humaneval.py`) が言うとおりの形にする。
    指示を先頭に置く理由は、module の docstring を参照。課題の側が壊れている
    ときは `CodeTaskError` で止まる (モデルのせいにしない。fail closed)。
    """
    entry = problem.entry_point.strip()
    if not entry.isidentifier() or keyword.iskeyword(entry):
        raise CodeTaskError(
            f"課題 {problem.task_id} の entry_point が識別子でない: {problem.entry_point!r}"
        )
    test = _normalize_source(problem.test)
    if not _CHECK_DEF_RE.search(test):
        raise CodeTaskError(
            f"課題 {problem.task_id} の test に `def check(` がない (採点の方法と合わない)"
        )
    prompt = _with_newline(_normalize_source(problem.prompt))
    candidate = _with_newline(_normalize_source(code))
    return f"{prompt}{candidate}\n{_with_newline(test)}\ncheck({entry})\n"


# --- 2-b. 検査を通ったことの印 (sentinel) ------------------------------------


@dataclass(frozen=True)
class CheckedProgram:
    """印つきのプログラムと、その印。採点の側が、同じ印を持っている必要がある。"""

    source: str
    sentinel: str


def _sentinel_for(source: str) -> str:
    """プログラムから決まる印。同じ課題と同じ答えなら、同じ印になる。

    候補のコードは、課題の検査のプログラムも、この module の作り方も読めない
    (プログラムは標準入力で渡され、読み終わったあとに実行が始まる) ので、
    実際には言い当てられない。
    """
    digest = hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()
    return f"{SENTINEL_PREFIX}{digest[:_SENTINEL_DIGITS]}__"


def _sentinel_tail(sentinel: str) -> str:
    """`check(...)` のあとに足す、印の書き出し。

    - 記述子 2 に直接書く (`sys.stderr` を差し替える候補のコードにも負けない)
    - 先に `sys.stderr` を流す。書きかけの行が残っていると、通訳器の終了時に
      印の**あと**へ流れ出て、末尾の一致が崩れる
    - 名前は、候補のコードとぶつかりにくいものにし、`import ... as` で必ず
      束ね直す (候補が同じ名前を作っていても上書きされる)
    """
    return (
        "\nimport os as _bench_done_os, sys as _bench_done_sys\n"
        "for _bench_done_stream in (_bench_done_sys.stderr, _bench_done_sys.__stderr__):\n"
        "    try:\n"
        "        _bench_done_stream.flush()\n"
        "    except Exception:\n"
        "        pass\n"
        f'_bench_done_os.write(2, b"\\n{sentinel}\\n")\n'
    )


def build_checked_program(problem: CodeProblem, code: str) -> CheckedProgram:
    """`build_program` の組み立てに、検査を通ったことの印を足す (6.7 はこちらを使う)。

    `check(<entry_point>)` の**あと**でだけ印が書かれるので、候補のコードが
    `sys.exit(0)` や `os._exit(0)` で先に終わると、終了コードが 0 でも印が
    出ない。`score_checked` は、終了コードが 0 で**かつ**標準エラーの最後の行が
    この印であることを求める。

    これは「うっかり / 素朴な抜け道」を塞ぐもので、同じ通訳器の中で `os.write`
    を差し替えるような、本気の細工までは防げない (防ぎようがない)。
    """
    source = build_program(problem, code)
    sentinel = _sentinel_for(source)
    return CheckedProgram(source=source + _sentinel_tail(sentinel), sentinel=sentinel)


# --- 3. 採点 -----------------------------------------------------------------


def _bounded(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= MAX_DETAIL_CHARS:
        return flat
    return flat[: MAX_DETAIL_CHARS - 1] + "…"


def _last_stderr_line(stderr_tail: str) -> str:
    """標準エラーの最後の 1 行だけを、上限つきで返す。"""
    for line in reversed(stderr_tail.splitlines()):
        if line.strip():
            flat = " ".join(line.split())
            if len(flat) > _MAX_STDERR_LINE_CHARS:
                return flat[: _MAX_STDERR_LINE_CHARS - 1] + "…"
            return flat
    return ""


def score_code(result: SandboxResult) -> QualityVerdict:
    """隔離して動かした結果を、終了コードだけで採点する (5.2)。**弱いほうの道すじ**。

    例外なく終われば正解。落ちた場合と時間切れは不正解で、短い理由を添える。
    要求そのものの失敗と、隔離が使えない場合を「採点できなかった」にするのは
    呼び出し側 (6.7) の責任である (`not_scored_verdict`)。

    終了コードしか見ないので、候補のコードが `check(...)` の手前で
    `sys.exit(0)` を呼ぶと、検査を受けずに正解になる。新しく採点をつなぐ側は、
    `build_checked_program` + `score_checked` を使うこと (module の docstring
    「検査を通ったことの印」)。この関数は、既存の呼び出しのために残してある。
    """
    if result.passed:
        return QualityVerdict(
            task="code", outcome=QualityOutcome.CORRECT, detail="tests_passed", sandbox=result
        )
    if result.timed_out:
        return QualityVerdict(
            task="code", outcome=QualityOutcome.INCORRECT, detail="timed_out", sandbox=result
        )
    return _failed_verdict(result)


def _failed_verdict(result: SandboxResult) -> QualityVerdict:
    """落ちた結果を、短い理由つきの不正解にする (印つきの道すじと共通)。"""
    detail = f"exit_code={result.exit_code}"
    if result.exit_code == _SIGKILL_EXIT_CODE:
        # 128 + SIGKILL。メモリの上限に当たった場合が多く、標準エラーは空になる
        detail = f"{detail} {_SIGKILL_DETAIL}"
    last_line = _last_stderr_line(result.stderr_tail)
    if last_line:
        detail = f"{detail} {last_line}"
    return QualityVerdict(
        task="code", outcome=QualityOutcome.INCORRECT, detail=_bounded(detail), sandbox=result
    )


def score_checked(result: SandboxResult, program: CheckedProgram) -> QualityVerdict:
    """印つきのプログラムの結果を採点する (5.2。6.7 はこちらを使う)。

    正解にするのは、終了コードが 0 で、**かつ**標準エラーの最後の行が
    `program.sentinel` のときだけ。終了コードが 0 なのに印がない応答は、
    `check(...)` に届く前に終わった (`sys.exit(0)` など) ということなので、
    `exited_before_check` を理由に不正解にする。
    """
    if result.timed_out:
        return QualityVerdict(
            task="code", outcome=QualityOutcome.INCORRECT, detail="timed_out", sandbox=result
        )
    if not result.passed:
        return _failed_verdict(result)
    if _last_stderr_line(result.stderr_tail) != program.sentinel:
        return QualityVerdict(
            task="code",
            outcome=QualityOutcome.INCORRECT,
            detail="exited_before_check",
            sandbox=result,
        )
    return QualityVerdict(
        task="code", outcome=QualityOutcome.CORRECT, detail="tests_passed", sandbox=result
    )


def no_code_verdict() -> QualityVerdict:
    """応答に使えるコードがなかったときの判定。

    モデルは答えているので、**不正解**であって「採点できなかった」ではない (5.7)。
    """
    return QualityVerdict(task="code", outcome=QualityOutcome.INCORRECT, detail="no_code")


def not_scored_verdict(reason: str) -> QualityVerdict:
    """採点できなかったときの判定 (要求の失敗、隔離が使えない) (5.7)。"""
    return QualityVerdict(task="code", outcome=QualityOutcome.NOT_SCORED, detail=_bounded(reason))
