"""公開のコードの課題 (HumanEval+) を、版を固定して取得し、確かめて読む (task 6.5)。

計測に使う唯一の公開の課題。リポジトリには同梱せず、git の管理の対象でない
場所 (既定は `bench/data-cache/`) に置いて、使うときに取得する (11.1、11.2)。

## 固定している入手先

- 名前: HumanEval+ (EvalPlus)
- 版: `v0.1.10` (`evalplus/humanevalplus_release` のリリースのタグ)
- 入手先: リリースの資産 `HumanEvalPlus-OriginFmt.jsonl.gz`。`latest` のような
  動く名前ではなく、タグを URL に含めた、動かない場所を指す
- 大きさ: 1,350,689 バイト、SHA-256: `daa7661c…` (2026-09-20 に実測)
- ライセンス: EvalPlus は Apache-2.0、元になった OpenAI の HumanEval は MIT。
  どちらも公式のリポジトリの `LICENSE` で確かめた (`LICENSES.md` への記入は 8.5)

`-OriginFmt` の資産を選ぶ理由は、元の HumanEval と同じ 5 つの項目
(`task_id`、`prompt`、`canonical_solution`、`entry_point`、`test`) を持ち、
`test` が `check(candidate)` を定義した、それだけで完結した検査の
プログラムになっているためである。EvalPlus の既定の資産
(`HumanEvalPlus.jsonl.gz`) は、入力の一覧 (`base_input`、`plus_input`) と
期待値を出す契約を持つ形で、採点には EvalPlus のコードが要る。この道具は
クリーンルームで作る (11.5) ので、`evalplus` も `human-eval` も、コードは
一行も写さず、依存にも加えない。文書化された入れ物の形 (gzip の JSONL) だけ
を頼りに、ここで読み取りを書いている。

## 採点の方法 (5.2、5.4。実際に動かすのは 6.6)

応答から取り出したコードのうしろに、その課題の `test` (= `check(candidate)`
の定義) と `check(<entry_point>)` の呼び出しを続けて、隔離したコンテナの中で
動かす。例外なく終われば正解、それ以外 (表明の失敗、例外、時間切れ) は
不正解。正解の割合を、問題の数とともに示す。この文言は `DatasetRef.
scoring_method` として計測ランと要約に残る (5.5)。

## 安全の決まり (fail closed)

- 読む前に、必ず SHA-256 と大きさを確かめる。取得したてでも、キャッシュに
  あったものでも同じ。合わないファイルは 1 バイトも解釈しない
- キャッシュのファイルが合わないときは、**黙って取り直さない**。期待した値と
  実際の値、ファイルの場所、直し方 (消してからやり直す / 版を確かめる) を
  添えて失敗する。取り直しを自動でやると、入れ替えられたファイルが「直った」
  ように見えてしまうため
- 取得は HTTPS だけ。転送 (redirect) の行き先も、要求を出す前に毎回確かめる
  (GitHub のリリースの資産は、`release-assets.githubusercontent.com` のような
  CDN のホストへ 1 回転送される)。認証の情報は送らないし、環境の proxy や
  `.netrc` も使わない (`trust_env=False`)。proxy が要る環境では、呼び出し側が
  `client` を渡す
- 本文の大きさに上限を置き、読みながら超えたところで止める (固定した大きさの
  `_MAX_SIZE_FACTOR` 倍)。`Content-Length` があれば、読む前にも見る
- 同じディレクトリの一時ファイルに書いてから、確かめて、`os.replace` で
  置き換える。失敗したときは一時ファイルを消し、キャッシュのファイルは作らない
- 置き場所が git の作業木の中で、しかも無視されていないときは断る。判定は
  `config` の確かめ (外部のコマンドの失敗を「問題なし」と見なさない) を使い回す

## 取得できないとき (6.7 への渡し方)

キャッシュがなく、ネットワークにも届かないときは `DatasetUnavailable` を投げる。
`reason` に理由が入っているので、6.7 はこれを捕まえて
`SkippedCondition(suite="quality", key="quality/code/humaneval+", reason=…)`
にできる (httpx の例外がそのまま外へ出ることはない)。一方、ハッシュが合わない
(`DatasetVerificationError`) と、中身が期待の形でない (`DatasetFormatError`) は
飛ばす理由にしない。取り違えや壊れたファイルを黙って見逃さないためである。

依存するのは標準ライブラリと httpx、`bench_harness.types`、`bench_harness.config`
だけ (design.md 「依存の向き」: types → config → corpus)。
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

import httpx

from bench_harness.config import ConfigError, _ensure_not_tracked
from bench_harness.types import CodeProblem, DatasetRef

__all__ = [
    "DatasetPin",
    "HUMANEVAL_PLUS",
    "HumanEvalError",
    "UNSCORABLE_PROBLEMS",
    "DatasetCacheError",
    "DatasetUnavailable",
    "DatasetVerificationError",
    "DatasetFormatError",
    "default_cache_dir",
    "check_cache_dir",
    "dataset_ref",
    "ensure_humaneval_plus",
    "read_problems",
    "load_humaneval_plus",
    "select_problems",
]


# --- 固定した入手先 ---------------------------------------------------------


@dataclass(frozen=True)
class DatasetPin:
    """公開の課題の、動かない入手先と、期待する中身。

    すべての関数がこれを引数に取る。試験は、本物のデータセットを埋め込まずに、
    同じ形の小さな合成のファイルに合わせた固定を渡せる。
    """

    name: str
    version: str
    url: str
    sha256: str
    size_bytes: int
    problem_count: int
    file_name: str
    license: str
    scoring_method: str


_SCORING_METHOD: Final = (
    "応答から取り出したコードに、課題の `test` (`check(candidate)` の定義) と"
    " `check(<entry_point>)` の呼び出しを続けて、隔離したコンテナの中で動かす。"
    "例外なく終われば正解。正解の割合を、問題の数とともに示す (5.2、5.4)"
)

HUMANEVAL_PLUS: Final = DatasetPin(
    name="HumanEval+",
    version="v0.1.10",
    url=(
        "https://github.com/evalplus/humanevalplus_release/releases/download/"
        "v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz"
    ),
    sha256="daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5",
    size_bytes=1_350_689,
    problem_count=164,
    file_name="HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz",
    license="Apache-2.0 (EvalPlus)、MIT (OpenAI HumanEval)",
    scoring_method=_SCORING_METHOD,
)

_TASK_ID_RE: Final = re.compile(r"^HumanEval/(\d+)$")
_REQUIRED_KEYS: Final = ("task_id", "prompt", "entry_point", "test")
_MAX_REDIRECTS: Final = 5
_MAX_SIZE_FACTOR: Final = 2
_CHUNK_BYTES: Final = 64 * 1024
_DEFAULT_TIMEOUT_S: Final = 60.0


# --- エラー -----------------------------------------------------------------


class HumanEvalError(Exception):
    """公開のコードの課題の取得または読み取りで見つかった誤り。"""


class DatasetCacheError(HumanEvalError):
    """置き場所そのものが使えない (git の管理の対象、シンボリックリンクなど)。"""


class DatasetUnavailable(HumanEvalError):
    """課題を取得できなかった。6.7 は、これを条件を飛ばす理由にしてよい。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DatasetVerificationError(HumanEvalError):
    """SHA-256 または大きさが、固定した値と合わない。飛ばす理由にはしない。"""


class DatasetFormatError(HumanEvalError):
    """確かめたファイルの中身が、期待の形または範囲に収まらない (note 2.7)。"""


# --- 置き場所 ---------------------------------------------------------------


def default_cache_dir() -> Path:
    """既定の置き場所 `bench/data-cache/` (`.gitignore` に入っている)。

    この module (`bench/src/bench_harness/corpus/humaneval.py`) から辿る。
    """
    return Path(__file__).resolve().parents[3] / "data-cache"


def check_cache_dir(cache_dir: Path | None = None) -> Path:
    """置き場所が使えるかを、何も読まず、何も作らずに確かめる (入口が、計測の前に呼ぶ)。

    使えなければ `DatasetCacheError`。`None` は既定の置き場所。
    """
    return _check_cache_dir(cache_dir if cache_dir is not None else default_cache_dir())


def _check_cache_dir(cache_dir: Path) -> Path:
    """置き場所が git の管理の対象になり得ないことを確かめる (11.1、8.5 と同じ決まり)。

    判定は `config` の `_ensure_not_tracked` を使い回す。外部のコマンドの失敗を
    「問題なし」と見なさず、確かめられた場合だけを許可する (note 1.3)。ここでは
    ディレクトリを作らない。
    """
    resolved = cache_dir if cache_dir.is_absolute() else cache_dir.resolve()
    try:
        _ensure_not_tracked(resolved)
    except ConfigError as exc:
        raise DatasetCacheError(
            f"公開の課題の置き場所 '{resolved}' は使えない: {exc}。"
            f"git の管理の対象でない場所 (既定は {default_cache_dir()}) を指定すること (11.1)"
        ) from exc
    return resolved


def _check_cache_file(path: Path) -> None:
    """キャッシュのファイルが、普通のファイルであることを確かめる (note 4.4)。"""
    if path.is_symlink():
        raise DatasetCacheError(
            f"'{path}' はシンボリックリンク。公開の課題の置き場所には、"
            "普通のファイルだけを置くこと (リンクは消してからやり直す)"
        )
    if path.exists() and not path.is_file():
        raise DatasetCacheError(f"'{path}' は普通のファイルではない")


# --- ハッシュの確認 ---------------------------------------------------------


def _digest(path: Path) -> tuple[str, int]:
    """ファイルの SHA-256 と大きさを、少しずつ読んで求める。"""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _verify(path: Path, pin: DatasetPin, *, remedy: str) -> None:
    """固定した SHA-256 と大きさに合うことを確かめる。合わなければ投げる。"""
    actual_sha, actual_size = _digest(path)
    if actual_sha == pin.sha256 and actual_size == pin.size_bytes:
        return
    raise DatasetVerificationError(
        f"公開の課題のファイル '{path.name}' が、固定した版 ({pin.version}) と合わない。"
        f" 場所: {path}。"
        f" 期待した SHA-256: {pin.sha256} ({pin.size_bytes} バイト)。"
        f" 実際の SHA-256: {actual_sha} ({actual_size} バイト)。"
        f" {remedy}"
    )


# --- 取得 -------------------------------------------------------------------


def _require_https(url: httpx.URL) -> None:
    if url.scheme != "https":
        raise DatasetUnavailable(f"公開の課題の取得は https だけを許す (受け取った行き先: {url})")
    if url.userinfo:
        raise DatasetUnavailable(f"取得の URL に認証の情報を含めない ({url.host})")


def _download(url: str, dest: BinaryIO, pin: DatasetPin, client: httpx.Client) -> None:
    """`url` の本文を `dest` に書く。転送は HTTPS だけを辿り、大きさに上限を置く。"""
    max_bytes = pin.size_bytes * _MAX_SIZE_FACTOR
    current = httpx.URL(url)
    for _ in range(_MAX_REDIRECTS + 1):
        _require_https(current)
        try:
            with client.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DatasetUnavailable(
                            f"{current} が転送を返したが、行き先 (location) がない"
                        )
                    current = current.join(location)
                    continue
                if response.status_code != httpx.codes.OK:
                    raise DatasetUnavailable(
                        f"公開の課題を取得できない (HTTP {response.status_code}): {current}"
                    )
                _check_declared_size(response, max_bytes)
                _stream_to(response, dest, max_bytes)
                return
        except httpx.HTTPError as exc:
            raise DatasetUnavailable(
                f"公開の課題を取得できない ({type(exc).__name__}: {exc}): {pin.url}"
            ) from exc
    raise DatasetUnavailable(f"転送 (redirect) が {_MAX_REDIRECTS} 回を超えた: {pin.url}")


def _check_declared_size(response: httpx.Response, max_bytes: int) -> None:
    declared = response.headers.get("content-length")
    if declared is None:
        return
    try:
        length = int(declared)
    except ValueError:
        return  # 読めない Content-Length は、読みながらの上限にまかせる
    if length > max_bytes:
        raise DatasetUnavailable(
            f"本文が大きすぎる ({length} バイト。上限は {max_bytes} バイト): {response.url}"
        )


def _stream_to(response: httpx.Response, dest: BinaryIO, max_bytes: int) -> None:
    total = 0
    for chunk in response.iter_bytes(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise DatasetUnavailable(
                f"本文が大きすぎる ({max_bytes} バイトを超えた): {response.url}"
            )
        dest.write(chunk)
    dest.flush()
    os.fsync(dest.fileno())


def ensure_humaneval_plus(
    cache_dir: Path | None = None,
    *,
    pin: DatasetPin = HUMANEVAL_PLUS,
    client: httpx.Client | None = None,
    allow_download: bool = True,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> Path:
    """課題のファイルが、確かめられた状態で置き場所にあるようにし、その場所を返す。

    すでにあれば、SHA-256 と大きさを確かめて、そのまま返す (ネットワークには
    一切触れない)。なければ取得して、確かめてから置く。合わないファイルが
    あるときは `DatasetVerificationError` で、取り直しはしない。
    """
    resolved_dir = _check_cache_dir(cache_dir if cache_dir is not None else default_cache_dir())
    path = resolved_dir / pin.file_name
    _check_cache_file(path)
    if path.exists():
        _verify(path, pin, remedy="このファイルを消してからやり直すか、固定した版を確かめること")
        return path

    if not allow_download:
        raise DatasetUnavailable(
            f"公開の課題 {pin.name} {pin.version} が '{path}' になく、取得も許されていない"
        )

    resolved_dir.mkdir(parents=True, exist_ok=True)
    http = client if client is not None else _new_client(timeout_s)
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=resolved_dir, prefix=f".{pin.file_name}.", suffix=".part"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                _download(pin.url, fh, pin, http)
            _verify(
                tmp_path,
                pin,
                remedy=(
                    "取得の途中で切れたか、入手先の中身が変わっている。"
                    "やり直すか、固定した版を確かめること"
                ),
            )
            os.replace(tmp_path, path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise DatasetCacheError(
                f"公開の課題を '{resolved_dir}' に書けない ({type(exc).__name__}: {exc})。"
                "空きの容量と、置き場所の権限を確かめること"
            ) from exc
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    finally:
        if client is None:  # 自分で作ったクライアントだけを閉じる
            http.close()
    return path


def _new_client(timeout_s: float) -> httpx.Client:
    """取得に使う HTTP クライアント。認証の情報も、環境の proxy や `.netrc` も使わない。"""
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
        follow_redirects=False,
        trust_env=False,
    )


# --- 読み取り ---------------------------------------------------------------


def read_problems(path: Path, *, pin: DatasetPin = HUMANEVAL_PLUS) -> list[CodeProblem]:
    """確かめたファイルを読み、番号の順に並べた課題を返す。

    確かめ (SHA-256 と大きさ) は、この関数の中でもう一度行う。`ensure_*` を
    通らずに呼ばれても、確かめていないファイルを解釈しないようにするため。
    中身は固定したハッシュで押さえられているので、展開しながらの爆弾を気に
    する必要はない。

    ファイルは 1 回だけ読み、**確かめたのと同じバイト列**を解釈する。確かめた
    あとに開き直すと、その間にファイルが入れ替わっても気付けない。読む量は、
    固定した大きさ + 1 バイトまで (それより大きければ、合わないと分かる)。
    """
    try:
        with path.open("rb") as fh:
            data = fh.read(pin.size_bytes + 1)
    except OSError as exc:
        raise DatasetFormatError(f"'{path}' を読めない ({type(exc).__name__}: {exc})") from exc
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != pin.sha256 or len(data) != pin.size_bytes:
        # 文面は、ファイル全体のハッシュと大きさで作る (切り詰めた読み取りの値を出さない)
        _verify(path, pin, remedy="このファイルを消してからやり直すか、固定した版を確かめること")
        raise DatasetVerificationError(
            f"公開の課題のファイル '{path}' が、読んでいる間に変わった。やり直すこと"
        )
    return _parse(data, path, pin)


def _parse(data: bytes, path: Path, pin: DatasetPin) -> list[CodeProblem]:
    problems: list[tuple[int, CodeProblem]] = []
    seen: set[str] = set()
    try:
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                problems.append(_parse_line(line, line_no, seen))
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise DatasetFormatError(
            f"'{path}' を gzip の JSONL として読めない ({type(exc).__name__}: {exc})"
        ) from exc

    if len(problems) != pin.problem_count:
        raise DatasetFormatError(
            f"'{path.name}' の問題の数が合わない"
            f" (期待 {pin.problem_count} 問、実際 {len(problems)} 問)。"
            "固定した版と URL を確かめること"
        )
    problems.sort(key=lambda item: item[0])
    return [problem for _, problem in problems]


def _parse_line(line: str, line_no: int, seen: set[str]) -> tuple[int, CodeProblem]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{line_no} 行目を JSON として読めない: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"{line_no} 行目が JSON のオブジェクトでない")

    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise DatasetFormatError(f"{line_no} 行目に項目 '{key}' がない")
    values = {key: _as_text(raw, key, line_no) for key in _REQUIRED_KEYS}
    solution = _as_optional_text(raw, "canonical_solution", line_no)

    task_id = values["task_id"]
    match = _TASK_ID_RE.match(task_id)
    if match is None:
        raise DatasetFormatError(
            f"{line_no} 行目の task_id '{task_id}' が 'HumanEval/<番号>' の形でない"
        )
    if task_id in seen:
        raise DatasetFormatError(f"task_id '{task_id}' が 2 回以上ある ({line_no} 行目)")
    seen.add(task_id)

    _check_problem(values, line_no)
    return int(match.group(1)), CodeProblem(
        task_id=task_id,
        prompt=values["prompt"],
        entry_point=values["entry_point"],
        test=values["test"],
        canonical_solution=solution,
    )


def _check_problem(values: dict[str, str], line_no: int) -> None:
    """範囲の確認 (note 2.7)。型だけでなく、採点に使える中身かどうかまで見る。"""
    where = f"{line_no} 行目 ({values['task_id']})"
    for key in ("prompt", "entry_point", "test"):
        if not values[key].strip():
            raise DatasetFormatError(f"{where} の項目 '{key}' が空")
    entry_point = values["entry_point"]
    if not entry_point.isidentifier():
        raise DatasetFormatError(f"{where} の entry_point '{entry_point}' が Python の識別子でない")
    if entry_point not in values["prompt"]:
        raise DatasetFormatError(f"{where} の entry_point '{entry_point}' が prompt に現れない")
    if "def check(" not in values["test"]:
        raise DatasetFormatError(
            f"{where} の test に 'def check(' がない。採点は `check(candidate)` を呼ぶ"
        )


def _as_text(raw: dict[str, Any], key: str, line_no: int) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise DatasetFormatError(
            f"{line_no} 行目の項目 '{key}' が文字列でない ({type(value).__name__})"
        )
    return value


def _as_optional_text(raw: dict[str, Any], key: str, line_no: int) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return _as_text(raw, key, line_no)


# --- 出どころと部分集合 -----------------------------------------------------


def dataset_ref(pin: DatasetPin = HUMANEVAL_PLUS) -> DatasetRef:
    """計測ランと要約に残す出どころ (5.5、11.2)。名前、版、入手先、ライセンス、採点の方法。"""
    return DatasetRef(
        name=pin.name,
        version=pin.version,
        source_url=pin.url,
        license=pin.license,
        scoring_method=pin.scoring_method,
        sha256=pin.sha256,
    )


def load_humaneval_plus(
    cache_dir: Path | None = None,
    *,
    pin: DatasetPin = HUMANEVAL_PLUS,
    client: httpx.Client | None = None,
    allow_download: bool = True,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[DatasetRef, list[CodeProblem]]:
    """出どころと、番号の順に並べた課題の全体を返す (design.md 「corpus」)。"""
    path = ensure_humaneval_plus(
        cache_dir,
        pin=pin,
        client=client,
        allow_download=allow_download,
        timeout_s=timeout_s,
    )
    return dataset_ref(pin), read_problems(path, pin=pin)


UNSCORABLE_PROBLEMS: Final[dict[str, str]] = {
    "HumanEval/32": (
        "v0.1.10 の -OriginFmt の検査のプログラムが `_poly(*candidate(*inp), inp)` と書かれて"
        "いて (引数の順が逆)、データセット自身の正解の解でも `TypeError` で落ちる。"
        "どんな応答も合格にならないので、採点の対象から外す"
    ),
}
"""採点の対象にしない問題と、その理由。

固定した版の 164 問に、正解の解 (`canonical_solution`) + 検査のプログラム +
`check(<entry_point>)` を、隔離のイメージ (numpy 入り) の中で流して確かめた
(タスク 6.1、2026-09-20)。163 問は合格し、ここに挙げた問題だけが落ちた。
残しておくと、モデルの出来と関係なく、正解の割合の上限が下がる。固定した版を
変えるときは、同じ確認をやり直して、この表を見直すこと。
"""


def select_problems(problems: Sequence[CodeProblem], limit: int | None) -> list[CodeProblem]:
    """採点できる問題から、`Profile.quality.code_problem_limit` のぶんだけ、先頭から取る。

    先に `UNSCORABLE_PROBLEMS` を外す (外してから数えるので、限りが 40 なら 40 問になる)。
    並びは番号の順に決まっているので、同じ限りなら、いつでも同じ部分集合になる
    (11.4)。`None` は、採点できる問題の全部。
    """
    if limit is not None and limit < 1:
        raise ValueError(f"code_problem_limit は 1 以上である必要がある (受け取った値: {limit})")
    scorable = [problem for problem in problems if problem.task_id not in UNSCORABLE_PROBLEMS]
    return scorable if limit is None else scorable[:limit]
