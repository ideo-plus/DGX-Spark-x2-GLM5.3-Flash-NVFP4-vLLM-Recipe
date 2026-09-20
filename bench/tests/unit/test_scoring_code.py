"""応答からコードを取り出して組み立てる部分の試験 (task 6.6: scoring/code、5.2)。

コンテナは使わない。`bench_harness.corpus` も読み込まず、課題 (`CodeProblem`) は
手で書く (design.md の依存の向き: 採点は corpus を読み込まない)。

この試験の主眼は「正直な正解を不正解にしないこと」である (note 6.4)。現実に
ありそうな応答の形を並べ、どれからもコードが取り出せて、組み立てた
プログラムが Python の構文として通ることを確かめる。実際に動かすのは
`tests/integration/test_sandbox_container.py`。
"""

from __future__ import annotations

import ast
import pathlib
import re
import time

import pytest

from bench_harness.scoring import code as code_module
from bench_harness.scoring.code import (
    MAX_DETAIL_CHARS,
    CheckedProgram,
    CodeTaskError,
    build_checked_program,
    build_program,
    extract_code,
    no_code_verdict,
    not_scored_verdict,
    score_checked,
    score_code,
)
from bench_harness.types import CodeProblem, QualityOutcome, SandboxResult

# --- 手で書いた課題 (HumanEval+ の実物と同じ形) -----------------------------

CLOSE_ELEMENTS = CodeProblem(
    task_id="HumanEval/0",
    prompt=(
        "from typing import List\n"
        "\n"
        "\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer to each\n'
        "    other than given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        '    """\n'
    ),
    entry_point="has_close_elements",
    test=(
        "def check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n"
        "    assert not candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)\n"
        "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95)\n"
        "    assert not candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8)\n"
    ),
    canonical_solution=None,
)
"""`from typing import List` が**指示の側にだけ**ある課題。応答が `List[float]` の
注釈を書いても動くこと (note 6.4: 正直な正解を落とさない) を、これで確かめる。"""

SUM_PRODUCT = CodeProblem(
    task_id="HumanEval/8",
    prompt=(
        "from typing import List, Tuple\n"
        "\n"
        "\n"
        "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        '    """ For a given list of integers, return a tuple consisting of a sum and a\n'
        "    product of all the integers in a list.\n"
        "    >>> sum_product([])\n"
        "    (0, 1)\n"
        '    """\n'
    ),
    entry_point="sum_product",
    test=(
        "def check(candidate):\n"
        "    assert candidate([]) == (0, 1)\n"
        "    assert candidate([1, 1, 1]) == (3, 1)\n"
        "    assert candidate([100, 0]) == (100, 0)\n"
        "    assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)\n"
    ),
    canonical_solution=None,
)

SOLUTION = (
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    for i, a in enumerate(numbers):\n"
    "        for j, b in enumerate(numbers):\n"
    "            if i != j and abs(a - b) < threshold:\n"
    "                return True\n"
    "    return False\n"
)
"""会話の形 (chat-style) の応答。関数をまるごと書き、注釈に `List` を使う。"""

BODY = (
    "    for i, a in enumerate(numbers):\n"
    "        for j, b in enumerate(numbers):\n"
    "            if i != j and abs(a - b) < threshold:\n"
    "                return True\n"
    "    return False\n"
)
"""続きを書く形 (completion-style) の応答。指示の末尾の署名に続く本体だけ。"""

FENCE = "```"


def _assert_parses(problem: CodeProblem, code: str | None) -> str:
    """取り出したコードから組み立てたプログラムが、構文として通ることを確かめる。"""
    assert code is not None
    program = build_program(problem, code)
    ast.parse(program)
    return program


# --- 1. 現実にありそうな応答の形 (15 以上) ----------------------------------


RESPONSE_SHAPES: list[tuple[str, str]] = [
    (
        "fenced_python",
        f"{FENCE}python\n{SOLUTION}{FENCE}\n",
    ),
    (
        "fenced_py",
        f"{FENCE}py\n{SOLUTION}{FENCE}\n",
    ),
    (
        "fenced_no_language",
        f"{FENCE}\n{SOLUTION}{FENCE}\n",
    ),
    (
        "fenced_uppercase_language",
        f"{FENCE}Python\n{SOLUTION}{FENCE}\n",
    ),
    (
        "japanese_explanation_around_block",
        "2 重ループで、すべての組の差を見ます。計算量は O(n^2) ですが、\n"
        "この課題の入力では十分です。\n"
        "\n"
        f"{FENCE}python\n{SOLUTION}{FENCE}\n"
        "\n"
        "閾値より近い組が 1 つでもあれば `True` を返します。\n",
    ),
    (
        "english_explanation_before_unfenced_code",
        f"Here is a straightforward solution that compares every pair.\n\n{SOLUTION}",
    ),
    (
        "trailing_prose_after_unfenced_code",
        f"{SOLUTION}\n"
        "この実装は、入力が空のときも安全に False を返します。\n"
        "計算量が気になる場合は、先に並べ替えてください。\n",
    ),
    (
        "thinking_preamble",
        "<think>\nまず素朴な二重ループを考える。ソートしてもよいが、まずは単純に。\n"
        "</think>\n"
        f"{FENCE}python\n{SOLUTION}{FENCE}\n",
    ),
    (
        "crlf_line_endings",
        (f"{FENCE}python\n{SOLUTION}{FENCE}\n").replace("\n", "\r\n"),
    ),
    (
        "tab_indented_solution",
        f"{FENCE}python\n"
        + SOLUTION.replace("                ", "\t\t\t\t")
        .replace("            ", "\t\t\t")
        .replace("        ", "\t\t")
        .replace("    ", "\t")
        + f"{FENCE}\n",
    ),
    (
        "body_only_unfenced",
        BODY,
    ),
    (
        "body_only_in_fence",
        f"{FENCE}python\n{BODY}{FENCE}\n",
    ),
    (
        "tests_block_then_solution_block",
        "まず期待する振る舞いを書きます。\n"
        "\n"
        f"{FENCE}python\n"
        "def test_has_close_elements():\n"
        "    assert True\n"
        f"{FENCE}\n"
        "\n"
        "つぎに解答です。\n"
        "\n"
        f"{FENCE}python\n{SOLUTION}{FENCE}\n",
    ),
    (
        "solution_block_then_usage_block",
        f"{FENCE}python\n{SOLUTION}{FENCE}\n"
        "\n"
        "使い方:\n"
        "\n"
        f"{FENCE}python\n"
        "print(has_close_elements([1.0, 2.0], 0.5))\n"
        f"{FENCE}\n",
    ),
    (
        "main_guard_with_input",
        f"{FENCE}python\n"
        f"{SOLUTION}"
        "\n"
        'if __name__ == "__main__":\n'
        "    raw = input()\n"
        "    print(has_close_elements([float(x) for x in raw.split()], 0.5))\n"
        f"{FENCE}\n",
    ),
    (
        "unterminated_fence",
        f"解答です。\n\n{FENCE}python\n{SOLUTION}",
    ),
    (
        "four_backtick_fence",
        f"````python\n{SOLUTION}````\n",
    ),
    (
        "indented_fence_in_list_item",
        f"1. つぎのように実装します。\n\n   {FENCE}python\n{SOLUTION}   {FENCE}\n",
    ),
    (
        "json_block_then_unfenced_solution",
        f'{FENCE}json\n{{"approach": "brute force"}}\n{FENCE}\n\n{SOLUTION}',
    ),
    (
        "solution_after_think_without_close_tag",
        f"<think>いろいろ考えた\n{FENCE}python\n{SOLUTION}{FENCE}\n",
    ),
]


@pytest.mark.parametrize(
    ("shape", "response"), RESPONSE_SHAPES, ids=[shape for shape, _ in RESPONSE_SHAPES]
)
def test_realistic_responses_yield_a_runnable_program(shape: str, response: str) -> None:
    """20 通りの応答の形すべてから、構文として通るプログラムが組み立てられる。"""
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    program = _assert_parses(CLOSE_ELEMENTS, code)
    # 取り出したコードが、解答の中身を持っていること (空の取り出しで通らない)
    assert "abs(a - b) < threshold" in program


def test_at_least_fifteen_response_shapes_are_covered() -> None:
    """応答の形の網羅が、あとから減らされないようにする。"""
    assert len(RESPONSE_SHAPES) >= 15
    assert len({shape for shape, _ in RESPONSE_SHAPES}) == len(RESPONSE_SHAPES)


# --- 1-b. 考えの塊の走査が、直線の時間で終わること ---------------------------


PATHOLOGICAL_THINKING: list[tuple[str, str]] = [
    ("repeated_open_tags", "<think>" * 70_000),
    ("open_tag_with_prose", "<think>ok let me reconsider. " * 18_000),
    ("truncated_block", "<think>" + "考え直す。" * 60_000),
    ("only_close_tags", "</think>" * 60_000),
]
"""壊れたモデルが出す、閉じていない `<think>` だらけの応答 (どれも約 0.4〜0.5 MB)。"""

SLOW_SCAN_CEILING_S = 2.0
"""`extract_code` 1 回の上限。

**なぜ絶対値にしたか (note 1.4 の決まりの読み替え)**: 「n と 8n の経過時間の比」
での判定も考えたが、直線の走査だと小さいほうが 1 ミリ秒未満になり、この Mac
(ふだんからロードアベレージが 10 前後) では取り込みの遅れが比を支配して、
偽の失敗を出す。いまの実装は 0.5 MB で約 5 ミリ秒、後戻り (`.*?` + DOTALL) の
実装は同じ入力で 68 秒だったので、2 秒は「直線なら 400 倍の余裕、二次なら
34 倍の超過」になる。2 桁の機械差では揺らがない。
構造の側の歯止めは `test_thinking_scan_uses_no_dotall_pattern` が受け持つ。
"""


@pytest.mark.parametrize(
    ("shape", "text"), PATHOLOGICAL_THINKING, ids=[shape for shape, _ in PATHOLOGICAL_THINKING]
)
def test_pathological_thinking_tags_do_not_blow_up(shape: str, text: str) -> None:
    """閉じていない `<think>` が並ぶ応答でも、直線の時間で終わる。"""
    started = time.perf_counter()
    extract_code(text, CLOSE_ELEMENTS.entry_point)
    assert time.perf_counter() - started < SLOW_SCAN_CEILING_S


def test_thinking_scan_uses_no_dotall_pattern() -> None:
    """走査に「任意の長さのワイルドカード」を使わない (時間で測らない歯止め)。

    `<think>.*?</think>` (DOTALL) は、閉じ括弧のない入力で開始位置ごとに
    末尾まで走査するので二次になる。この module の正規表現はどれも
    `re.DOTALL` を立てない、という形で禁じる。
    """
    patterns = [
        (name, value) for name, value in vars(code_module).items() if isinstance(value, re.Pattern)
    ]
    assert patterns, "正規表現が 1 つも見つからない (試験の前提が壊れている)"
    dotall = [name for name, value in patterns if value.flags & re.DOTALL]
    assert dotall == []

    # module の中で組み立てる正規表現 (関数の中の局所変数も) を、構文木で見る。
    # module の最上位の定数だけを見ていると、関数の中に書き戻す変更を見逃す
    tree = ast.parse(pathlib.Path(str(code_module.__file__)).read_text(encoding="utf-8"))
    used = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "DOTALL"
    ]
    assert used == []


def test_unclosed_think_tag_keeps_the_answer() -> None:
    """閉じ忘れた `<think>` は、印だけを落として本文は残す。

    閉じ忘れの後ろを全部「考え」として捨てると、答えを書いたのに閉じ忘れた
    応答を、まるごと失う。逆に印だけ落とすと、考えの文が本文に混ざるが、
    囲みのブロックや `def` の探索で弾かれるだけで済む。誤って不正解にしない
    側を採る (note 6.4)。
    """
    response = f"<think>まだ迷っているが、答えはこれだ。\n{FENCE}python\n{SOLUTION}{FENCE}\n"
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "def has_close_elements" in code


def test_closed_think_block_is_removed_whole() -> None:
    response = (
        "<think>\nここは考え。def has_close_elements(x): return 'WRONG'\n</think>\n"
        f"{FENCE}python\n{SOLUTION}{FENCE}\n"
    )
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "WRONG" not in code


# --- 2. どのブロックを選ぶか ------------------------------------------------


def test_block_that_defines_the_entry_point_wins_over_the_first_block() -> None:
    """最初のブロックではなく、入口の関数を定義したブロックを選ぶ。"""
    response = (
        f"{FENCE}python\n"
        "def test_has_close_elements():\n"
        "    assert True\n"
        f"{FENCE}\n"
        "\n"
        f"{FENCE}python\n{SOLUTION}{FENCE}\n"
    )
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "def has_close_elements" in code
    assert "def test_has_close_elements" not in code


def test_usage_block_after_the_solution_is_not_chosen() -> None:
    """あとに来る使用例のブロックではなく、定義のあるブロックを選ぶ。"""
    response = (
        f"{FENCE}python\n{SOLUTION}{FENCE}\n\n"
        f"{FENCE}python\nprint(has_close_elements([1.0], 0.5))\n{FENCE}\n"
    )
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "def has_close_elements" in code
    assert "print(has_close_elements" not in code


def test_longest_block_wins_when_no_block_defines_the_entry_point() -> None:
    """どのブロックも入口を定義しないときは、いちばん長いブロックを選ぶ。"""
    response = f"{FENCE}python\n# 覚え書き\nx = 1\n{FENCE}\n\n{FENCE}python\n{BODY}{FENCE}\n"
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "return False" in code
    assert "覚え書き" not in code


def test_javascript_only_response_yields_nothing() -> None:
    """Python ではないブロックだけの応答からは、コードを取り出さない。"""
    response = (
        f"{FENCE}javascript\n"
        "function hasCloseElements(numbers, threshold) { return false; }\n"
        f"{FENCE}\n"
    )
    assert extract_code(response, CLOSE_ELEMENTS.entry_point) is None


def test_prose_only_response_yields_nothing() -> None:
    """説明だけの応答からは、コードを取り出さない。"""
    response = "申し訳ありませんが、この課題には答えられません。別の課題をください。"
    assert extract_code(response, CLOSE_ELEMENTS.entry_point) is None


@pytest.mark.parametrize("response", ["", "   \n\n  \t ", "\r\n\r\n"])
def test_empty_response_yields_nothing(response: str) -> None:
    """空の応答からは、コードを取り出さない (例外も投げない)。"""
    assert extract_code(response, CLOSE_ELEMENTS.entry_point) is None


def test_main_guard_is_removed_so_input_cannot_break_the_program() -> None:
    """`if __name__ == "__main__":` の塊を落とす。

    プログラムそのものを標準入力で渡すので、候補のコードの中の `input()` は
    EOF を読む (止まりはしないが `EOFError` になる)。`check(...)` の前で
    落ちると、正しい解答まで不正解になってしまうため、この塊は取り除く。
    """
    response = (
        f"{FENCE}python\n{SOLUTION}\n"
        'if __name__ == "__main__":\n'
        "    data = input()\n"
        "    print(data)\n"
        f"{FENCE}\n"
    )
    code = extract_code(response, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    assert "input()" not in code
    assert "__main__" not in code
    assert "def has_close_elements" in code


def test_extract_code_never_raises_on_hostile_text() -> None:
    """どんな本文でも例外を投げない。"""
    hostile = [
        "```python\n" * 200,
        "\x00\x01�" * 100,
        "def " * 5000,
        "(" * 2000,
        f"{FENCE}python\n" + "a" * 100_000,
        "\t\t\t\n" * 1000,
    ]
    for text in hostile:
        extract_code(text, CLOSE_ELEMENTS.entry_point)
    extract_code("def f(): pass", "")
    extract_code("def f(): pass", "not an identifier")


# --- 3. プログラムの組み立て ------------------------------------------------


def test_prompt_is_prepended_so_imports_from_the_prompt_are_available() -> None:
    """指示の側にしかない import を、候補のコードの前に置く (note 6.4)。

    会話の形の応答は `List[float]` と注釈を書くが、`from typing import List` は
    指示の側にしかない。指示を前に置かないと、正直な正解が `NameError` で
    落ちる。
    """
    program = build_program(CLOSE_ELEMENTS, SOLUTION)
    assert "from typing import List" in program
    assert program.index("from typing import List") < program.index("def has_close_elements")
    ast.parse(program)


def test_prompt_prepend_also_covers_tuple_annotations() -> None:
    """`Tuple` のような、指示の側にしかない別の名前でも同じ。"""
    answer = (
        "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        "    total = 0\n"
        "    product = 1\n"
        "    for value in numbers:\n"
        "        total += value\n"
        "        product *= value\n"
        "    return total, product\n"
    )
    program = build_program(SUM_PRODUCT, answer)
    assert "from typing import List, Tuple" in program
    ast.parse(program)


def test_body_only_answer_continues_the_prompt() -> None:
    """本体だけの応答は、指示の末尾の署名に続けて 1 つの関数になる。"""
    program = build_program(CLOSE_ELEMENTS, BODY)
    tree = ast.parse(program)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions.count("has_close_elements") == 1


def test_program_ends_with_the_check_call() -> None:
    """検査のプログラムと `check(<entry_point>)` の呼び出しが、末尾に来る。"""
    program = build_program(CLOSE_ELEMENTS, SOLUTION)
    assert "def check(candidate):" in program
    assert program.rstrip().endswith("check(has_close_elements)")
    assert program.index("def check(candidate):") < program.rindex("check(has_close_elements)")


def test_build_program_normalizes_newlines_and_leading_tabs() -> None:
    """CRLF と、行頭のタブを正規化する (指示の 4 個の空白と混ぜない)。"""
    program = build_program(CLOSE_ELEMENTS, BODY.replace("    ", "\t").replace("\n", "\r\n"))
    assert "\r" not in program
    assert "\t" not in program
    ast.parse(program)


def test_build_program_rejects_a_broken_problem() -> None:
    """課題の側が壊れていたら、モデルのせいにせずに止まる (fail closed)。"""
    broken_entry = CodeProblem(
        task_id="broken/1",
        prompt="def f():\n    pass\n",
        entry_point="not an identifier",
        test="def check(candidate):\n    pass\n",
    )
    with pytest.raises(CodeTaskError):
        build_program(broken_entry, SOLUTION)

    broken_test = CodeProblem(
        task_id="broken/2",
        prompt="def f():\n    pass\n",
        entry_point="f",
        test="assert True\n",
    )
    with pytest.raises(CodeTaskError):
        build_program(broken_test, SOLUTION)


# --- 3-b. 検査を通ったことの印 (sentinel) ------------------------------------


def test_checked_program_writes_the_sentinel_after_the_check_call() -> None:
    """印の書き出しは、必ず `check(...)` の**あと**に来る。"""
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    assert isinstance(program, CheckedProgram)
    ast.parse(program.source)
    assert program.sentinel in program.source
    assert program.source.index("check(has_close_elements)") < program.source.index(
        f'b"\\n{program.sentinel}\\n"'
    )


def test_checked_program_writes_the_sentinel_through_the_file_descriptor() -> None:
    """`sys.stderr` を差し替えられても届くように、記述子 2 に直接書く。"""
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    assert "os.write(2," in program.source.replace("_bench_done_os", "os")
    # 標準エラーの書きかけの行が、印のあとに流れ出さないように先に流す
    assert ".flush()" in program.source


def test_checked_program_starts_from_the_plain_program() -> None:
    """`build_program` の中身を、そのまま前半に持つ (組み立ての決まりは 1 つ)。"""
    plain = build_program(CLOSE_ELEMENTS, SOLUTION)
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    assert program.source.startswith(plain)


def test_sentinel_is_deterministic_and_answer_specific() -> None:
    """同じ課題と答えなら同じ印。答えが違えば違う印 (当てずっぽうで書けない)。"""
    first = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    again = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    other = build_checked_program(CLOSE_ELEMENTS, SOLUTION + "# 違う答え\n")
    assert first.sentinel == again.sentinel
    assert first.sentinel != other.sentinel
    assert len(first.sentinel) >= 32


def test_build_checked_program_rejects_a_broken_problem() -> None:
    broken = CodeProblem(
        task_id="broken/3",
        prompt="def f():\n    pass\n",
        entry_point="f",
        test="assert True\n",
    )
    with pytest.raises(CodeTaskError):
        build_checked_program(broken, SOLUTION)


def _result(*, exit_code: int, stderr_tail: str = "", timed_out: bool = False) -> SandboxResult:
    return SandboxResult(
        passed=exit_code == 0 and not timed_out,
        timed_out=timed_out,
        exit_code=None if timed_out else exit_code,
        stderr_tail=stderr_tail,
    )


def test_score_checked_is_correct_only_with_the_sentinel() -> None:
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    verdict = score_checked(_result(exit_code=0, stderr_tail=f"\n{program.sentinel}\n"), program)
    assert verdict.outcome is QualityOutcome.CORRECT


def test_score_checked_marks_an_exit_before_the_check_as_incorrect() -> None:
    """`sys.exit(0)` などで `check(...)` に届かなかった応答を、正解にしない。"""
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    verdict = score_checked(_result(exit_code=0, stderr_tail=""), program)
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert "exited_before_check" in verdict.detail


def test_score_checked_rejects_a_forged_sentinel() -> None:
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    forged = "__bench_ok_" + "0" * 32 + "__"
    assert forged != program.sentinel
    verdict = score_checked(_result(exit_code=0, stderr_tail=f"\n{forged}\n"), program)
    assert verdict.outcome is QualityOutcome.INCORRECT


def test_score_checked_ignores_noise_before_the_sentinel() -> None:
    """印より前にある警告や大量の出力は、判定に響かない (末尾だけを見る)。"""
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    noisy = "DeprecationWarning: ...\n" + "x" * 1900 + f"\n{program.sentinel}\n"
    verdict = score_checked(_result(exit_code=0, stderr_tail=noisy), program)
    assert verdict.outcome is QualityOutcome.CORRECT


def test_score_checked_requires_the_sentinel_to_be_last() -> None:
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    tail = f"{program.sentinel}\nあとから書かれた行\n"
    verdict = score_checked(_result(exit_code=0, stderr_tail=tail), program)
    assert verdict.outcome is QualityOutcome.INCORRECT


def test_score_checked_reports_failures_and_timeouts_like_score_code() -> None:
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    failed = score_checked(_result(exit_code=1, stderr_tail="AssertionError\n"), program)
    assert failed.outcome is QualityOutcome.INCORRECT
    assert "exit_code=1" in failed.detail
    assert "AssertionError" in failed.detail

    timed_out = score_checked(_result(exit_code=0, timed_out=True), program)
    assert timed_out.outcome is QualityOutcome.INCORRECT
    assert "timed_out" in timed_out.detail


def test_score_checked_detail_is_bounded() -> None:
    program = build_checked_program(CLOSE_ELEMENTS, SOLUTION)
    verdict = score_checked(_result(exit_code=1, stderr_tail="ValueError: " + "x" * 5000), program)
    assert len(verdict.detail) <= MAX_DETAIL_CHARS


# --- 4. 採点 ----------------------------------------------------------------


def test_passed_result_is_correct() -> None:
    verdict = score_code(SandboxResult(passed=True, timed_out=False, exit_code=0))
    assert verdict.task == "code"
    assert verdict.outcome is QualityOutcome.CORRECT


def test_failed_result_is_incorrect_with_the_exit_code() -> None:
    result = SandboxResult(
        passed=False,
        timed_out=False,
        exit_code=1,
        stderr_tail=(
            'Traceback (most recent call last):\n  File "<stdin>", line 9\nAssertionError\n'
        ),
    )
    verdict = score_code(result)
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert "exit_code=1" in verdict.detail
    assert "AssertionError" in verdict.detail
    # stderr をまるごと持ち回らない
    assert "Traceback" not in verdict.detail
    assert verdict.sandbox == result


def test_killed_by_the_memory_limit_says_so() -> None:
    """137 (SIGKILL) は標準エラーが空になりやすいので、印を添える。"""
    verdict = score_code(
        SandboxResult(passed=False, timed_out=False, exit_code=137, stderr_tail="")
    )
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert "exit_code=137" in verdict.detail
    assert "killed_sigkill_or_oom" in verdict.detail


def test_timed_out_result_is_incorrect_and_says_so() -> None:
    verdict = score_code(SandboxResult(passed=False, timed_out=True, exit_code=None))
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert "timed_out" in verdict.detail


def test_detail_is_bounded() -> None:
    """理由の文字列は、必ず上限の中に収まる。"""
    result = SandboxResult(
        passed=False, timed_out=False, exit_code=1, stderr_tail="ValueError: " + "x" * 5000
    )
    verdict = score_code(result)
    assert len(verdict.detail) <= MAX_DETAIL_CHARS


def test_no_code_is_incorrect_not_unscored() -> None:
    """コードがない応答は「採点できなかった」ではなく、不正解 (5.7)。"""
    verdict = no_code_verdict()
    assert verdict.task == "code"
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert verdict.detail


def test_not_scored_verdict_carries_a_bounded_reason() -> None:
    """要求の失敗や、隔離が使えないときの受け皿 (5.7)。"""
    verdict = not_scored_verdict("x" * 5000)
    assert verdict.outcome is QualityOutcome.NOT_SCORED
    assert len(verdict.detail) <= MAX_DETAIL_CHARS
