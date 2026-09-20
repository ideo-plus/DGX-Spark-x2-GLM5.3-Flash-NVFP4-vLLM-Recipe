"""要約だけを公開の場所に置く (task 4.4)。

`bench publish <run>` から呼ばれる。計測ランのディレクトリから `summary.json`
と `summary.md` の 2 つだけを、`docs/results/<run_id>/` に写す。生データ
(`manifest.json`、`trials.jsonl`、`bodies/`、`metrics/` の生の指標) は、
どんな経路でも公開の場所に届かない (8.4、11.1)。

安全のための決まり (fail closed。確かめのどれか 1 つでも落ちれば、何も写さない):

- 写す前に、`summary.json` と `summary.md` の両方がそろっていることを確かめる
  (なければ、先に `bench summarize` を促す日本語のエラーにする)
- `summary.json` は `types.Summary` として検証し直す。埋め込まれた
  `conditions.run_id` が、この計測ランの `manifest.json` の `run_id` と
  違うものは、よその計測ランの要約として拒む
- `summary.md` は UTF-8 のテキストとして読み直す
- `summary.json` と `summary.md` は、どちらもシンボリックリンクなら拒む
  (`trials.jsonl` などをリンクで持ち出す経路を塞ぐ)。**ハードリンク**
  (`st_nlink > 1`) も同じ理由で拒む。シンボリックリンクの確認だけでは、
  `os.link(trials.jsonl, summary.md)` のように、名前を変えて同じ中身を
  指すファイルを通す経路が残る (独立レビュー 1 回目の指摘 3)
- 公開先に使う `run_id` は、単一の安全な経路の要素であることを確かめてから使う
  (空文字、長さの上限を超える値、NUL・改行 (`\n`、`\r`)、経路の区切り
  (`/`、`\\`)、`.`・`..`、先頭の `.` を拒む)。解決した公開先が、解決した
  公開の置き場所の中にあることも確かめる。これは、`<公開の場所>/<run_id>` に、
  公開の場所の外を指すシンボリックリンクがあらかじめ置かれていた場合の、
  唯一の守りである (`run_id` の確認は文字列しか見ないので、リンクそのものは
  防げない。独立レビュー 1 回目の指摘、必須 1)
- 公開の場所の既定値 (`default_docs_results_root`) は git を呼んで決める。
  git が見つからない、または 0 以外の終了コードを返したときは、作業ディレクトリ
  (`Path.cwd()`) へ黙って倒れたりせず `PublishError` にする (独立レビュー
  1 回目の指摘、必須 2)

**保証すること**: 上のすべての確認を通った `summary.json` は `types.Summary`
の形をしていて、この計測ランのものだと確かめられている (本文や送った内容を
入れる項目が型にないので、8.3 により決して混ざらない)。写す先には、この 2 つ
のファイルしか作らない・触れない (許可リスト)。写す前の確認がすべて済んだ
あとに限って、`dest_dir` の下に書く。

**保証しないこと**: `summary.md` は自由形式のテキストである。UTF-8 として
読めること、シンボリックリンクやハードリンクでないことは確かめるが、
**中身が実際に `bench summarize` の出力であることまでは検証しない**。計測
ランのディレクトリに書き込める人が、`summary.md` を普通のファイルとして
手で書き換えれば、その中身がそのまま公開される。この module が塞ぐのは、
ファイルシステムの別名づけ (シンボリックリンク・ハードリンク) を使って生
データを漏らす経路であって、正規のファイルの中身そのものの真正性ではない。
リンクの確認と、ファイルの読み取りの間に、ファイルを差し替える競合 (計測ランの
ディレクトリに書き込める人が、時機を合わせて行う) も、防がない。

写す先の 2 つのファイルは、決めうちの許可リスト (allow-list) で選ぶ。計測ランの
ディレクトリに何があっても、この 2 つ以外は絶対に読まないし写さない。
`summary.json` の書き出しは、読み込んだバイト列をそのまま使う (作り直さない)。
2 回公開しても同じ内容で上書きするだけで、一時ファイルを残さない。公開先に
無関係なファイルがすでにあっても、それには触れない。`_atomic_write` の
`os.replace` そのものが失敗した場合 (稀。ディスクの空き容量や権限など) は、
ここでは捕まえず、生の `OSError` のまま外に出す。`summary.json` と
`summary.md` を順に書くので、1 つ目の後始末に成功したあとに 2 つ目が失敗
すると、1 つ目は公開済みのまま、2 つ目は公開されずに終わる (どちらのファイル
にも、壊れた中身が残ることはない)。呼び出し側のコマンドの入口が、これを
外に出た `OSError` として終了の値 2 に直す決まりになっている (tasks.md
Implementation Notes 3.5) ので、ここでは変えない。

依存するのは標準ライブラリ、pydantic、`bench_harness.types`、
`bench_harness.store` だけ (design.md 「依存の向き」)。ネットワークには
触れない。純粋なファイルの操作だけを行う。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from bench_harness.store import RunStore
from bench_harness.types import Summary

__all__ = [
    "PublishError",
    "PublishedSummary",
    "default_docs_results_root",
    "publish_run",
]

_SUMMARY_JSON: Final[str] = "summary.json"
_SUMMARY_MD: Final[str] = "summary.md"
"""公開の対象のファイル名。この 2 つだけを許可リストとして扱う (8.4)。"""

_MAX_RUN_ID_LEN: Final[int] = 200
"""`run_id` の長さの上限。長すぎる名前は `mkdir` の生の `OSError`
(ファイル名が長すぎる) になるので、ここで先に拒む (store/rawstore.py の
`_MAX_SANITIZED_TARGET_NAME_LEN` と同じ理由。独立レビュー 1 回目の指摘 5)。"""


class PublishError(Exception):
    """計測ランの要約を公開できないこと。原因を示す。何も写さない (fail closed)。"""


class PublishedSummary(BaseModel):
    """`publish_run` が写した先 (`bench publish` の出力に使う)。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dest_dir: Path
    summary_json_path: Path
    summary_md_path: Path
    incomplete: bool


# --- run_id を経路の要素として使ってよいかの確認 -----------------------------


def _validate_run_id_component(run_id: str) -> None:
    """`run_id` が、公開先のディレクトリ名として安全な、単一の経路の要素かを確かめる。

    `RunStore` (`new_run_id`) が払い出す `run_id` は、すでにこの形を満たす
    (英数字と `-` だけを整えた対象サーバーの名前 + タイムスタンプ + 乱数、
    長さも `_MAX_SANITIZED_TARGET_NAME_LEN` で抑えてある)。それでも、
    `manifest.json` を直接書き換えられていた場合に備えて、ここでも確かめる
    (fail closed)。拒むもの: 空文字、`_MAX_RUN_ID_LEN` 文字を超える値、
    NUL・改行 (`\\n`、`\\r`)、経路の区切り (`/`、`\\`)、`.`・`..`、先頭が `.`。
    """
    if not run_id:
        raise PublishError("run_id が空である")
    if len(run_id) > _MAX_RUN_ID_LEN:
        raise PublishError(f"run_id が長すぎる ({len(run_id)} 文字 > {_MAX_RUN_ID_LEN})")
    if any(ch in run_id for ch in ("\x00", "\n", "\r")):
        raise PublishError(f"run_id に扱えない制御文字が含まれている: {run_id!r}")
    if run_id in {".", ".."}:
        raise PublishError(f"run_id が経路の特殊な要素である: {run_id!r}")
    if "/" in run_id or "\\" in run_id:
        raise PublishError(f"run_id に経路の区切りが含まれている: {run_id!r}")
    if run_id.startswith("."):
        raise PublishError(f"run_id が '.' から始まっている: {run_id!r}")


# --- 公開の場所の既定値 (git のリポジトリの直下の docs/results) -------------------


def _git_toplevel(start: Path) -> Path:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PublishError(f"git を実行できない (見つからないか、実行の権限がない): {exc}") from exc
    if proc.returncode != 0:
        raise PublishError(
            "公開の場所の既定値を決められない (git のリポジトリの外にいる)。"
            "docs_results_root を明示して渡すこと"
        )
    return Path(proc.stdout.strip())


def default_docs_results_root() -> Path:
    """公開の場所の既定値 (`bench_harness` を含む git のリポジトリの直下の `docs/results`)。"""
    top = _git_toplevel(Path(__file__).resolve().parent)
    return top / "docs" / "results"


# --- 別名で書いてから置き換える (atomic write) -------------------------------


def _fsync_dir(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """同じディレクトリに別名で書いてから `os.replace` し、親ディレクトリも `fsync` する
    (store/rawstore.py の `_atomic_write_bytes` と同じ理由。独立レビュー 1 回目の指摘 4)。

    失敗しても一時ファイルを残さない。`os.replace` そのものが失敗した場合は、
    ここでは捕まえず、生の `OSError` のまま外に出す (モジュールの docstring を参照)。
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


# --- 写す前の確認 -------------------------------------------------------------


def _reject_if_symlink_or_hardlinked(path: Path, filename: str) -> None:
    """シンボリックリンクと、ハードリンク (`st_nlink > 1`) のどちらも拒む。

    どちらも、`trials.jsonl` のような生データのファイルを、公開する名前の下に
    別名で持ち出す経路になりうる (独立レビュー 1 回目の指摘 3)。経路が存在
    しなければ、ここでは何もしない (「そろっているか」のあとの確認に任せる)。
    """
    if path.is_symlink():
        raise PublishError(
            f"{filename} がシンボリックリンクである ({path})。"
            "公開できない (ほかのファイルを持ち出す経路になりうる)"
        )
    try:
        st = path.stat()
    except FileNotFoundError:
        return
    if st.st_nlink > 1:
        raise PublishError(
            f"{filename} がハードリンクである ({path}、リンクの数={st.st_nlink})。"
            "公開できない (もとのファイルの中身と同じである保証がなく、"
            "ほかのファイルを持ち出す経路になりうる)"
        )


def _load_and_check_summary(raw: bytes, run_id: str) -> Summary:
    """`summary.json` を `types.Summary` として検証し、`run_id` が揃うかを確かめる。"""
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"{_SUMMARY_JSON} の JSON を読み取れない: {exc}") from exc
    try:
        summary = Summary.model_validate(parsed)
    except ValidationError as exc:
        raise PublishError(f"{_SUMMARY_JSON} が types.Summary として検証できない: {exc}") from exc
    if summary.conditions.run_id != run_id:
        raise PublishError(
            f"{_SUMMARY_JSON} の conditions.run_id ({summary.conditions.run_id!r}) が、"
            f"この計測ランの run_id ({run_id!r}) と違う。よその計測ランの要約は公開しない"
        )
    return summary


def _check_utf8_text(raw: bytes, filename: str) -> None:
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublishError(f"{filename} が UTF-8 のテキストとして読めない: {exc}") from exc


def _resolve_docs_results_root(explicit: Path | None) -> Path:
    root = explicit if explicit is not None else default_docs_results_root()
    return root.resolve()


def _safe_dest_dir(resolved_root: Path, run_id: str) -> Path:
    """`resolved_root / run_id` を作り、それが確かに `resolved_root` の中にあることを確かめる。"""
    dest_dir = resolved_root / run_id
    resolved_dest_dir = dest_dir.resolve()
    if not resolved_dest_dir.is_relative_to(resolved_root):
        raise PublishError(
            f"公開先 '{resolved_dest_dir}' が公開の置き場所 '{resolved_root}' の外になる"
        )
    return resolved_dest_dir


# --- 公開 ---------------------------------------------------------------


def publish_run(run_dir: Path, docs_results_root: Path | None = None) -> PublishedSummary:
    """`run_dir` の計測ランの要約だけを、公開の場所に写す (8.4)。

    写すのは `summary.json` と `summary.md` の 2 つだけ (許可リスト)。安全の
    確認が 1 つでも落ちれば `PublishError` になり、何も写さない。`run_dir` の
    `bodies/` と `trials.jsonl` は読まない。
    """
    manifest = RunStore.open(run_dir).manifest()
    run_id = manifest.run_id
    _validate_run_id_component(run_id)

    summary_json_src = run_dir / _SUMMARY_JSON
    summary_md_src = run_dir / _SUMMARY_MD
    sources = ((_SUMMARY_JSON, summary_json_src), (_SUMMARY_MD, summary_md_src))

    for filename, path in sources:
        _reject_if_symlink_or_hardlinked(path, filename)

    missing = [filename for filename, path in sources if not path.is_file()]
    if missing:
        raise PublishError(
            f"{', '.join(missing)} が見つからない ({run_dir})。"
            "先に 'bench summarize' を実行すること"
        )

    summary_json_bytes = summary_json_src.read_bytes()
    summary = _load_and_check_summary(summary_json_bytes, run_id)

    summary_md_bytes = summary_md_src.read_bytes()
    _check_utf8_text(summary_md_bytes, _SUMMARY_MD)

    resolved_root = _resolve_docs_results_root(docs_results_root)
    dest_dir = _safe_dest_dir(resolved_root, run_id)

    dest_dir.mkdir(parents=True, exist_ok=True)
    summary_json_dest = dest_dir / _SUMMARY_JSON
    summary_md_dest = dest_dir / _SUMMARY_MD
    _atomic_write(summary_json_dest, summary_json_bytes)
    _atomic_write(summary_md_dest, summary_md_bytes)

    return PublishedSummary(
        dest_dir=dest_dir,
        summary_json_path=summary_json_dest,
        summary_md_path=summary_md_dest,
        incomplete=summary.incomplete,
    )
