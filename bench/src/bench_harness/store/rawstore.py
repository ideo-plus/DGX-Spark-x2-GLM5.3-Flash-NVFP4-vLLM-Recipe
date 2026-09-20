"""計測ランの生データを保存して読み戻す (task 2.3)。計測と分析の間の契約。

`results/<run_id>/` の下に、次の形で置く (design.md `store/rawstore`):

    manifest.json               # RunManifest。書き換えるたびに別名で書いてから置き換える
    trials.jsonl                # TrialRecord を 1 行ずつ。書くたびにディスクに同期する
    bodies/<sha256>.json.gz     # 送った要求の本文。内容のハッシュで重複を除く
    metrics/<condition>.before.prom / .after.prom   # 加工する前の /metrics
    metrics/deltas.jsonl        # 条件ごとの内部の指標の増分 (MetricDeltaRecord)

設計上の決めごと:

- `RunStore.create()` は、まず `config.resolve_results_root()` で置き場所が git の
  管理の対象でないことを確かめる (8.5)。確かめが失敗したら、ディレクトリを含めて
  何も作らない (fail closed)
- `manifest.json` の書き換えは、同じディレクトリに別名で書いて `flush` + `fsync`
  してから `os.replace` し、最後にディレクトリ自体も `fsync` する。既存の値は、
  置き換えが失敗しても壊れない
- `trials.jsonl` と `metrics/deltas.jsonl` は、どちらも 1 レコードにつき 1 回の
  `write` 呼び出し + `flush` + `fsync` で書き足す (8.7)。書いている途中で
  プロセスが死んでも、それより前のレコードは残る。読み戻すとき (`_read_jsonl`、
  2 つのファイルで共通) は、次の決まりで壊れた行を扱う: 「途中で切れた」と見な
  して捨ててよいのは、**最後の行が JSON として読めない場合だけ**。JSON として
  完全に読めるのに型の検証に落ちる行は、最後の行でも途中の行でも、本物の破損か
  `schema_version` の食い違いなので、行番号つきの `StoreError` にする
  (独立レビュー 1 回目の指摘、必須 1・必須 2)
- 分析 (4.1〜4.3) は `read_trials()` / `read_metric_deltas()` を主な読み取りの口
  として使う。どちらもレコードと警告を一緒に返すので、警告を見忘れない。
  `iter_trials()` / `iter_metric_deltas()` は、大きな計測ランを少しずつ読みたい
  ときのために残してある。使うときは、読み切ったあとに必ず `.warnings` を見ること
  (見忘れると、捨てた行の警告が黙って消える)
- 送った本文 (`put_body`) は、鍵を並べ替えた正準形の JSON を、決まった mtime で
  gzip 圧縮して保存する。同じ本文は、同じバイト列になり、同じファイルに落ちる
  (重複を除く)。読み戻すときはハッシュを確かめ直す
- `design.md` の `RunStore` の型紙は `write_metrics` の `before`/`after` を
  `MetricSnapshot` としているが、7.4 によりどちらの側も `/metrics` を読めない
  ことがある (`MetricsUnavailable`)。この実装では、両方とも
  `MetricSnapshot | MetricsUnavailable` を受け付ける形に広げてある
  (tasks.md 2.3 の指示に沿った、承認された修正。得られなかった側は、生の
  テキストを書かずに理由だけを `metrics/deltas.jsonl` の行に残す)
- 条件ごとの内部の指標の増分の行の型 (`MetricDeltaRecord`) は、共有の型
  (`types.py`) にはない。ここが定義と読み戻し (`read_metric_deltas`) の両方を
  持つ (tasks.md の Implementation Notes 1.2)
- 認証の情報は、この module のどの関数にも渡らない。`RunManifest.target` が持つ
  のは `TargetDef.api_key_env` (環境変数の名前) だけで、値そのものは含まない
  (1.8)

依存するのは標準ライブラリと pydantic、`bench_harness.types`、
`bench_harness.config` だけ。ほかの `bench_harness` の module は読み込まない。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import secrets
import string
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Self, cast
from urllib.parse import quote, unquote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from bench_harness import config
from bench_harness.types import (
    SCHEMA_VERSION,
    DerivedMetrics,
    MetricSnapshot,
    MetricsUnavailable,
    RunManifest,
    RunStatus,
    TrialRecord,
)

__all__ = [
    "MetricDeltaRecord",
    "RecordIterator",
    "RunStore",
    "StoreError",
    "decode_condition_filename",
    "encode_condition_filename",
    "list_run_dirs",
    "new_run_id",
]


class StoreError(Exception):
    """保存または読み取りで見つかった誤り。原因を、ファイルや行番号つきで示す。"""


# --- 計測ランの識別子 (1.5) -------------------------------------------------

_RUN_ID_SUFFIX_ALPHABET: Final[str] = string.ascii_lowercase + string.digits
_RUN_ID_SUFFIX_LEN: Final[int] = 6
_UNSAFE_NAME_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SANITIZED_TARGET_NAME_LEN: Final[int] = 64
"""整えたあとの対象サーバーの名前の長さの上限。上限がないと、長い名前で
`RunStore.create()` の `mkdir` が生の `OSError` (ファイル名が長すぎる) になる
(独立レビュー 1 回目の指摘 6)。切り詰めても、`new_run_id` の乱数の部分で
一意性は保たれる。"""


def _sanitize_target_name(name: str) -> str:
    """対象サーバーの名前を、ファイル名として安全で、長さも扱いやすい形にする。"""
    sanitized = _UNSAFE_NAME_CHARS.sub("-", name).strip("-")
    sanitized = sanitized[:_MAX_SANITIZED_TARGET_NAME_LEN].strip("-")
    return sanitized or "target"


def new_run_id(target_name: str, now: datetime) -> str:
    """`YYYYMMDDTHHMMSSZ-<対象サーバーの名前>-<乱数 6 文字>` を作る (1.5)。

    同じ秒に同じ対象サーバーへ 2 回呼んでも、乱数の部分で区別が付く。
    """
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(secrets.choice(_RUN_ID_SUFFIX_ALPHABET) for _ in range(_RUN_ID_SUFFIX_LEN))
    return f"{stamp}-{_sanitize_target_name(target_name)}-{suffix}"


# --- 条件の鍵とファイル名の対応 (metrics/<condition>.before.prom など) -----------


def encode_condition_filename(condition: str) -> str:
    """条件の鍵 (`prefill/cold/32k` のように `/` を含む) を、ファイル名として
    安全で、可逆な形にする。パーセントエンコーディングなので衝突しない。
    """
    return quote(condition, safe="")


def decode_condition_filename(encoded: str) -> str:
    """`encode_condition_filename` の逆変換。"""
    return unquote(encoded)


# --- 別名で書いてから置き換える (atomic write) -------------------------------


def _fsync_dir(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """同じディレクトリに別名で書いて `fsync` し、`os.replace` してから、
    ディレクトリ自体も `fsync` する。置き換えが失敗しても、元のファイルは残る。
    """
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
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
    _fsync_dir(directory)


def _append_line(path: Path, data: bytes) -> None:
    """1 行を、単一の `write` 呼び出しで書き足し、`flush` + `fsync` する。"""
    line = data if data.endswith(b"\n") else data + b"\n"
    with path.open("ab") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _canonical_json(body: dict[str, JsonValue]) -> bytes:
    """鍵を並べ替え、ASCII への逃がしをしない、決まった形の JSON にする。"""
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


# --- JSON Lines の読み戻し (trials.jsonl と metrics/deltas.jsonl で共通) --------


def _parse_jsonl_line[T: BaseModel](
    model: type[T], raw: bytes, line_no: int, filename: str, *, is_last: bool
) -> tuple[T | None, str | None]:
    """1 行を `model` にする。空行は無視する。

    「途中で切れた」と見なして捨ててよいのは、**最後の行が JSON として読めない
    場合だけ** (プロセスが書いている途中で落ちた跡)。JSON としては読めるのに
    `model` の検証に落ちる行は、最後の行でも途中の行でも、本物の破損か
    `schema_version` の食い違いなので、行番号つきの `StoreError` にする
    (独立レビュー 1 回目の指摘、必須 2)。
    """
    if raw.strip() == b"":
        return None, None
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if is_last:
            return (
                None,
                f"{filename} の最後の行 ({line_no} 行目) が途中で切れているため捨てた: {exc}",
            )
        raise StoreError(
            f"{filename} の {line_no} 行目が壊れている (JSON を読み取れない): {exc}"
        ) from exc
    try:
        record = model.model_validate(parsed)
    except ValidationError as exc:
        raise StoreError(
            f"{filename} の {line_no} 行目が壊れている (検証を通らない): {exc}"
        ) from exc
    return record, None


def _read_jsonl[T: BaseModel](
    model: type[T], path: Path, filename: str, warnings: list[str]
) -> Iterator[T]:
    """`path` を 1 行ずつ `model` に直す。末尾破損の扱いは `_parse_jsonl_line` を参照。"""
    if not path.exists():
        return
    with path.open("rb") as fh:
        pending: bytes | None = None
        pending_line_no = 0
        for line_no, raw_line in enumerate(fh, start=1):
            if pending is not None:
                record, warning = _parse_jsonl_line(
                    model, pending, pending_line_no, filename, is_last=False
                )
                if warning is not None:  # is_last=False では起きない。念のため反映する
                    warnings.append(warning)
                if record is not None:
                    yield record
            pending = raw_line
            pending_line_no = line_no
        if pending is not None:
            record, warning = _parse_jsonl_line(
                model, pending, pending_line_no, filename, is_last=True
            )
            if warning is not None:
                warnings.append(warning)
            if record is not None:
                yield record


class RecordIterator[T](Iterator[T]):
    """`RunStore.iter_trials()` / `iter_metric_deltas()` の返り値。

    たどり終えたあとに `.warnings` で読める (末尾の壊れた行を捨てたときの文。
    最後まで `next()` を呼び切るまでは確定しない。末尾かどうかは、ファイルを
    読み切って初めて分かるため)。大きな計測ランを少しずつ読みたいときに使う。
    分析は、警告を見忘れないよう `read_trials()` / `read_metric_deltas()` を
    主に使うこと。
    """

    def __init__(self, records: Iterator[T], warnings: list[str]) -> None:
        self._records = records
        self._warnings = warnings

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        return next(self._records)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)


# --- 条件ごとの内部の指標の増分 (types.py にはない。ここが定義と読み戻しを持つ) ----


class MetricDeltaRecord(BaseModel):
    """`metrics/deltas.jsonl` の 1 行。条件と、2 つの時点の読み取りの可否、導出値。

    共有の型 (`types.py`) にはこの行の型がない (tasks.md Implementation Notes 1.2)。
    分析 (task 4.1) は、この型と `RunStore.read_metric_deltas()` を読む。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    condition: str
    before_taken_at_utc: datetime | None = None
    after_taken_at_utc: datetime | None = None
    before_available: bool
    before_unavailable_reason: str | None = None
    after_available: bool
    after_unavailable_reason: str | None = None
    derived: DerivedMetrics


# --- 計測ランの状態の遷移 (running -> completed | aborted | interrupted) --------

_ALLOWED_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.ABORTED, RunStatus.INTERRUPTED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.ABORTED: frozenset(),
    RunStatus.INTERRUPTED: frozenset(),
}


# --- RunStore ---------------------------------------------------------------


class RunStore:
    """`results/<run_id>/` の生データを読み書きする (design.md `store/rawstore`)。

    直接インスタンス化せず、新しい計測ランは `create()`、既存の計測ランは
    `open()` で作る。
    """

    def __init__(self, run_dir: Path, manifest: RunManifest) -> None:
        self.run_dir = run_dir
        self._manifest = manifest

    def __repr__(self) -> str:
        return f"{type(self).__name__}(run_dir={str(self.run_dir)!r})"

    @property
    def _bodies_dir(self) -> Path:
        return self.run_dir / "bodies"

    @property
    def _metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    # --- 作成と読み込み ---

    @staticmethod
    def create(results_root: Path, manifest: RunManifest) -> RunStore:
        """新しい計測ランを作る。

        `results_root` が git の管理の対象なら、`config.resolve_results_root` が
        `ConfigError` を投げ、ディレクトリを含めて何も作らない (8.5, fail closed)。
        計測ランのディレクトリがすでにあれば `StoreError`。
        """
        if manifest.status is not RunStatus.RUNNING:
            raise StoreError(
                f"新しい計測ランの状態は running である必要がある (渡された値: "
                f"{manifest.status.value})"
            )
        resolved_root = config.resolve_results_root(results_root)  # 何も作る前に、まず確かめる

        run_dir = resolved_root / manifest.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise StoreError(f"計測ランのディレクトリがすでにある: {run_dir}") from exc

        bodies_dir = run_dir / "bodies"
        metrics_dir = run_dir / "metrics"
        bodies_dir.mkdir()
        metrics_dir.mkdir()
        (run_dir / "trials.jsonl").touch()
        (metrics_dir / "deltas.jsonl").touch()
        # ディレクトリの項目 (新しく作ったファイルとサブディレクトリ) を一度だけ同期する。
        # 以後の書き足しは、ファイルそのものの fsync だけで済む (dentry は変わらない)
        _fsync_dir(metrics_dir)
        _fsync_dir(bodies_dir)
        _fsync_dir(run_dir)

        store = RunStore(run_dir, manifest)
        store._write_manifest(manifest)
        return store

    @staticmethod
    def open(run_dir: Path) -> RunStore:
        """既存の計測ランを、読み取りと書き足しのために開く。

        `manifest.json` の `schema_version` が、この道具の `SCHEMA_VERSION` と
        違えば、はっきりした理由つきで `StoreError`。
        """
        path = run_dir / "manifest.json"
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StoreError(f"計測ランのディレクトリに manifest.json がない: {run_dir}") from exc
        try:
            raw: Any = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise StoreError(f"manifest.json の JSON を読み取れない: {exc}") from exc

        found_version = raw.get("schema_version") if isinstance(raw, dict) else None
        if found_version != SCHEMA_VERSION:
            raise StoreError(
                f"manifest.json の schema_version が {found_version!r} で、"
                f"この道具が読める版 ({SCHEMA_VERSION}) と違う: {path}"
            )
        try:
            manifest = RunManifest.model_validate(raw)
        except ValidationError as exc:
            raise StoreError(f"manifest.json を検証できない: {exc}") from exc
        return RunStore(run_dir, manifest)

    # --- manifest ---

    def manifest(self) -> RunManifest:
        return self._manifest

    def set_status(self, status: RunStatus, finished_at: datetime) -> RunManifest:
        """計測ランの状態を進める。`running -> completed | aborted | interrupted`
        以外の遷移は `StoreError`。

        design.md の型紙は `set_status` を `-> None` としているが、呼び出し側が
        更新後の値をその場で使えるよう (`manifest()` を呼び直さずに済むよう)、
        検証し直した `RunManifest` を返す形にしてある。
        """
        current = self._manifest.status
        allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
        if status not in allowed:
            allowed_names = ", ".join(sorted(s.value for s in allowed)) or "(なし)"
            raise StoreError(
                f"計測ランの状態を '{current.value}' から '{status.value}' へ変えられない"
                f" (許される先: {allowed_names})"
            )
        return self.update_manifest(status=status, finished_at=finished_at)

    def update_manifest(self, **changes: Any) -> RunManifest:
        """`warnings`、`skipped`、`datasets` などを書き換え、検証し直してから保存する。

        `RunManifest` は凍結 (frozen) なので `model_copy(update=...)` は検証を
        通らない (tasks.md Implementation Notes 1.2)。`model_dump()` した辞書を
        書き換えてから `model_validate()` に通し、下限などの検証を必ず経由させる。
        `set_status` と同じ理由で、更新後の `RunManifest` を返す (design.md には
        このメソッドの型紙がない)。
        """
        data = self._manifest.model_dump()
        data.update(changes)
        try:
            updated = RunManifest.model_validate(data)
        except ValidationError as exc:
            raise StoreError(f"manifest の更新が検証を通らない: {exc}") from exc
        self._write_manifest(updated)
        self._manifest = updated
        return updated

    def _write_manifest(self, manifest: RunManifest) -> None:
        path = self.run_dir / "manifest.json"
        data = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
        _atomic_write_bytes(path, data)

    # --- 試行 ---

    def append_trial(self, record: TrialRecord) -> None:
        """1 行を書き足す。単一の `write` 呼び出しのあとにディスクへ同期する (8.7)。"""
        line = record.model_dump_json().encode("utf-8") + b"\n"
        _append_line(self.run_dir / "trials.jsonl", line)

    def iter_trials(self) -> RecordIterator[TrialRecord]:
        """`trials.jsonl` を読み戻す。最後の行だけが JSON として壊れていれば捨てて
        警告にする (それ以外の壊れ方は `StoreError`)。分析は `read_trials()` を使う
        こと (`RecordIterator` の docstring を参照)。
        """
        warnings: list[str] = []
        path = self.run_dir / "trials.jsonl"
        return RecordIterator(_read_jsonl(TrialRecord, path, "trials.jsonl", warnings), warnings)

    def read_trials(self) -> tuple[list[TrialRecord], list[str]]:
        """`trials.jsonl` を最後まで読み切り、レコードと警告を一緒に返す。

        分析 (4.1〜4.3) は、警告を見忘れないよう、これを主な読み取りの口として使う。
        """
        iterator = self.iter_trials()
        records = list(iterator)
        return records, iterator.warnings

    # --- 送った要求の本文 ---

    def put_body(self, body: dict[str, JsonValue]) -> str:
        """実際に送った本文 (`build_request_body` の返り値) を、内容のハッシュで
        重複を除いて保存する。sha256 の ref を返す。
        """
        canonical = _canonical_json(body)
        digest = hashlib.sha256(canonical).hexdigest()
        path = self._bodies_dir / f"{digest}.json.gz"
        compressed = gzip.compress(canonical, compresslevel=9, mtime=0)
        _atomic_write_bytes(path, compressed)
        return digest

    def get_body(self, ref: str) -> dict[str, JsonValue]:
        """`put_body` が保存した本文を読み戻し、ハッシュを確かめ直す。"""
        path = self._bodies_dir / f"{ref}.json.gz"
        try:
            compressed = path.read_bytes()
        except FileNotFoundError as exc:
            raise StoreError(f"本文が見つからない: {ref}") from exc
        raw = gzip.decompress(compressed)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ref:
            raise StoreError(f"本文のハッシュが一致しない (期待 {ref}、実際 {digest})")
        parsed: Any = json.loads(raw)
        if not isinstance(parsed, dict):
            raise StoreError(f"本文が JSON の写像でない: {ref}")
        return cast(dict[str, JsonValue], parsed)

    # --- 内部の指標 ---

    def write_metrics(
        self,
        condition: str,
        before: MetricSnapshot | MetricsUnavailable,
        after: MetricSnapshot | MetricsUnavailable,
        derived: DerivedMetrics,
    ) -> None:
        """加工する前の `/metrics` を両側とも書き、導出値の行を 1 つ書き足す (7.5)。

        `before`/`after` のどちらかが `MetricsUnavailable` なら、その側のテキストは
        書かずに理由だけを `metrics/deltas.jsonl` の行に残す (7.4)。
        """
        encoded = encode_condition_filename(condition)
        before_available, before_reason, before_taken_at = self._write_metric_side(
            encoded, "before", before
        )
        after_available, after_reason, after_taken_at = self._write_metric_side(
            encoded, "after", after
        )
        row = MetricDeltaRecord(
            condition=condition,
            before_taken_at_utc=before_taken_at,
            after_taken_at_utc=after_taken_at,
            before_available=before_available,
            before_unavailable_reason=before_reason,
            after_available=after_available,
            after_unavailable_reason=after_reason,
            derived=derived,
        )
        _append_line(self._metrics_dir / "deltas.jsonl", row.model_dump_json().encode("utf-8"))

    def _write_metric_side(
        self,
        encoded_condition: str,
        side: Literal["before", "after"],
        snapshot: MetricSnapshot | MetricsUnavailable,
    ) -> tuple[bool, str | None, datetime | None]:
        if isinstance(snapshot, MetricsUnavailable):
            return False, snapshot.reason, None
        path = self._metrics_dir / f"{encoded_condition}.{side}.prom"
        _atomic_write_bytes(path, snapshot.raw_text.encode("utf-8"))
        return True, None, snapshot.taken_at_utc

    def iter_metric_deltas(self) -> RecordIterator[MetricDeltaRecord]:
        """`metrics/deltas.jsonl` を読み戻す。`write_metrics` も `append_trial` と
        同じ `_append_line` で書くので、書いている途中でプロセスが落ちる危険は
        `trials.jsonl` と同じ。末尾破損の扱いも `iter_trials()` と共通
        (`_read_jsonl`)。分析は `read_metric_deltas()` を使うこと。
        """
        warnings: list[str] = []
        path = self._metrics_dir / "deltas.jsonl"
        return RecordIterator(
            _read_jsonl(MetricDeltaRecord, path, "metrics/deltas.jsonl", warnings), warnings
        )

    def read_metric_deltas(self) -> tuple[list[MetricDeltaRecord], list[str]]:
        """`metrics/deltas.jsonl` を最後まで読み切り、レコードと警告を一緒に返す。

        分析 (4.1) は、警告を見忘れないよう、これを主な読み取りの口として使う。
        """
        iterator = self.iter_metric_deltas()
        records = list(iterator)
        return records, iterator.warnings


def list_run_dirs(results_root: Path) -> list[Path]:
    """`results_root` の下の計測ランのディレクトリを、名前の順で返す。

    `manifest.json` を持つ直下のディレクトリだけを数える。
    """
    if not results_root.is_dir():
        return []
    return sorted(
        entry
        for entry in results_root.iterdir()
        if entry.is_dir() and (entry / "manifest.json").is_file()
    )
