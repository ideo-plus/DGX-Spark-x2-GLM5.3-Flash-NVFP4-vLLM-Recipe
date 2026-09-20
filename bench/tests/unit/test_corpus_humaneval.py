"""`corpus/humaneval.py` (公開のコードの課題の取得と読み取り) の試験。

確かめること (タスク 6.5 の完了の状態):

- ハッシュが合わないファイルで失敗し、1 問も読まない。メッセージに、期待した
  ハッシュと実際のハッシュ、ファイルの名前、直し方が入っている
- 取得のあと、課題が読める (偽の転送に、本物と同じ形の小さな gz JSONL を出させる)。
  本物の定数に対しては、形 (16 進 64 文字、版の入った https の URL、164 問) を確かめ、
  実際の取得は `BENCH_NETWORK_TESTS=1` のときだけ行う
- 返した出どころ (`DatasetRef`) に、名前、版、入手先、ライセンス、採点の方法が入っている
- 途中で切れた取得、接続の失敗、HTTP 404、大きすぎる本文、HTTPS でない行き先への
  転送が、どれもはっきりしたエラーになる。一時ファイルも、キャッシュのファイルも残らない
- キャッシュのファイルが確かめられたときは、ネットワークに一切触れない
- 範囲の確認 (問題の数、識別子の形と重複、空の項目、入口の名前、`check(` の有無) を
  破るレコードが、はっきりしたエラーになる

ネットワークには触れない (`httpx.MockTransport` を使う)。`bench/data-cache/` にも
書かない (すべて `tmp_path`)。
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bench_harness.corpus import humaneval as h

# --- 試験用の小さなデータセット ---------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """本物と同じ 5 つの項目を持つ、合成のレコードを 1 つ作る。"""
    name = f"solve_{index}"
    record: dict[str, Any] = {
        "task_id": f"HumanEval/{index}",
        "prompt": f'def {name}(x: int) -> int:\n    """{index} を足して返す。"""\n',
        "canonical_solution": f"    return x + {index}\n",
        "entry_point": name,
        "test": f"def check(candidate):\n    assert candidate(0) == {index}\n",
    }
    record.update(overrides)
    return record


def _gz(records: list[dict[str, Any]]) -> bytes:
    """レコードの列を gzip の JSONL にする (mtime を 0 に固定して、内容だけで決まるようにする)。"""
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    return gzip.compress(body.encode("utf-8"), mtime=0)


def _pin(
    data: bytes, count: int, *, url: str = "https://example.invalid/he/v9.9.9.jsonl.gz"
) -> h.DatasetPin:
    """渡したバイト列そのものに合わせた、試験用の固定 (pin)。"""
    return h.DatasetPin(
        name="HumanEval+ (試験)",
        version="v9.9.9",
        url=url,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        problem_count=count,
        file_name="HumanEvalPlus-test-v9.9.9.jsonl.gz",
        license="Apache-2.0 (EvalPlus)、MIT (OpenAI HumanEval)",
        scoring_method="試験用",
    )


_SMALL = [_record(2), _record(0), _record(1)]  # わざと番号の順に並べていない
_SMALL_DATA = _gz(_SMALL)
_SMALL_PIN = _pin(_SMALL_DATA, 3)


class _Transport:
    """呼ばれた要求を記録する `httpx.MockTransport` の包み。"""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._dispatch))


def _serving(data: bytes) -> _Transport:
    return _Transport(lambda request: httpx.Response(200, content=data))


def _never_called() -> _Transport:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 呼ばれたら失敗
        raise AssertionError(f"ネットワークに触れてはいけない: {request.url}")

    return _Transport(handler)


# --- 本物の定数と出どころ ---------------------------------------------------


def test_pinned_constants_have_expected_shape() -> None:
    pin = h.HUMANEVAL_PLUS
    assert len(pin.sha256) == 64
    assert pin.sha256 == pin.sha256.lower()
    assert all(c in "0123456789abcdef" for c in pin.sha256)
    assert pin.url.startswith("https://")
    assert pin.version in pin.url  # 版が URL に明示されている (可変の latest を指さない)
    assert pin.problem_count == 164
    assert pin.size_bytes > 0
    assert pin.version in pin.file_name


def test_dataset_ref_carries_provenance() -> None:
    ref = h.dataset_ref()
    assert ref.name
    assert ref.version == h.HUMANEVAL_PLUS.version
    assert ref.source_url == h.HUMANEVAL_PLUS.url
    assert "Apache-2.0" in ref.license  # EvalPlus
    assert "MIT" in ref.license  # OpenAI HumanEval
    assert ref.sha256 == h.HUMANEVAL_PLUS.sha256
    assert "check(" in ref.scoring_method  # 採点の方法が、課題の `test` の使い方まで書いてある


def test_default_cache_dir_is_the_git_ignored_data_cache() -> None:
    assert h.default_cache_dir().name == "data-cache"


# --- 取得と読み取り ---------------------------------------------------------


def test_download_then_load_returns_problems_in_numeric_id_order(tmp_path: Path) -> None:
    transport = _serving(_SMALL_DATA)
    with transport.client() as client:
        ref, problems = h.load_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert len(transport.requests) == 1
    assert [p.task_id for p in problems] == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    assert problems[1].entry_point == "solve_1"
    assert "solve_1" in problems[1].prompt
    assert "def check(candidate)" in problems[1].test
    assert problems[1].canonical_solution == "    return x + 1\n"
    assert ref.sha256 == _SMALL_PIN.sha256
    assert (tmp_path / _SMALL_PIN.file_name).is_file()


def test_problems_are_ordered_by_number_and_not_as_strings(tmp_path: Path) -> None:
    """`HumanEval/10` は `HumanEval/2` のあと。文字列で並べると、先頭の N 問が変わる (11.4)。"""
    records = [_record(10), _record(2), _record(0)]
    path, pin = _write(tmp_path, _gz(records), count=3)

    problems = h.read_problems(path, pin=pin)

    assert [p.task_id for p in problems] == ["HumanEval/0", "HumanEval/2", "HumanEval/10"]
    assert [p.task_id for p in h.select_problems(problems, 2)] == ["HumanEval/0", "HumanEval/2"]


def test_a_corrupt_file_can_never_be_skipped_as_unavailable() -> None:
    """6.7 は `DatasetUnavailable` だけを「飛ばす理由」にする。壊れや取り違えは、飛ばさせない。"""
    assert not issubclass(h.DatasetVerificationError, h.DatasetUnavailable)
    assert not issubclass(h.DatasetFormatError, h.DatasetUnavailable)
    assert not issubclass(h.DatasetCacheError, h.DatasetUnavailable)


def test_the_bytes_that_were_verified_are_the_bytes_that_are_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """確かめたあとに開き直さない。ファイルを開くのは 1 回だけ。"""
    path, pin = _write(tmp_path, _SMALL_DATA, count=3)
    opened: list[str] = []
    real_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    real_gzip_open = gzip.open

    def guarded_gzip_open(target: Any, *args: Any, **kwargs: Any) -> Any:
        assert not isinstance(target, (str, Path)), "確かめたあとに、経路から開き直している"
        return real_gzip_open(target, *args, **kwargs)

    monkeypatch.setattr(gzip, "open", guarded_gzip_open)

    problems = h.read_problems(path, pin=pin)

    assert len(problems) == 3
    assert len(opened) == 1


def test_a_write_failure_is_reported_as_a_cache_error_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ディスクがいっぱいでも、素の `OSError` を外に出さず、残骸も残さない。"""

    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", full_disk)
    transport = _serving(_SMALL_DATA)
    with transport.client() as client, pytest.raises(h.DatasetCacheError, match="書けない"):
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert list(tmp_path.iterdir()) == []


def test_cached_verified_file_makes_no_network_call(tmp_path: Path) -> None:
    (tmp_path / _SMALL_PIN.file_name).write_bytes(_SMALL_DATA)
    transport = _never_called()
    with transport.client() as client:
        _, problems = h.load_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert transport.requests == []
    assert len(problems) == 3


def test_second_call_does_not_download_again(tmp_path: Path) -> None:
    transport = _serving(_SMALL_DATA)
    with transport.client() as client:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert len(transport.requests) == 1


# --- ハッシュの確認 ---------------------------------------------------------


def test_cached_file_with_wrong_hash_fails_naming_expected_and_actual(tmp_path: Path) -> None:
    path = tmp_path / _SMALL_PIN.file_name
    other = _gz([_record(7)])
    path.write_bytes(other)
    transport = _never_called()

    with transport.client() as client, pytest.raises(h.DatasetVerificationError) as exc_info:
        h.load_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    message = str(exc_info.value)
    assert _SMALL_PIN.file_name in message
    assert _SMALL_PIN.sha256 in message  # 期待したハッシュ
    assert hashlib.sha256(other).hexdigest() in message  # 実際のハッシュ
    assert "消して" in message and "版" in message  # 直し方 (ファイルを消す / 版を確かめる)
    assert transport.requests == []  # 黙って取り直さない (fail closed)
    assert path.read_bytes() == other  # 勝手に消さない


def test_ensure_refuses_a_cached_file_that_does_not_match(tmp_path: Path) -> None:
    """`ensure_*` そのものが確かめる (読み取りの側の確かめに頼らない)。"""
    (tmp_path / _SMALL_PIN.file_name).write_bytes(_gz([_record(7)]))
    transport = _never_called()

    with transport.client() as client, pytest.raises(h.DatasetVerificationError):
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)


def test_read_problems_refuses_a_file_that_does_not_match_the_pin(tmp_path: Path) -> None:
    path = tmp_path / _SMALL_PIN.file_name
    path.write_bytes(_gz([_record(7)]))

    with pytest.raises(h.DatasetVerificationError):
        h.read_problems(path, pin=_SMALL_PIN)


def test_file_of_the_right_size_but_wrong_content_fails(tmp_path: Path) -> None:
    path = tmp_path / _SMALL_PIN.file_name
    path.write_bytes(b"\x00" * _SMALL_PIN.size_bytes)

    with pytest.raises(h.DatasetVerificationError) as exc_info:
        h.read_problems(path, pin=_SMALL_PIN)
    assert _SMALL_PIN.sha256 in str(exc_info.value)


def test_partial_download_leaves_nothing_behind(tmp_path: Path) -> None:
    truncated = _SMALL_DATA[: len(_SMALL_DATA) // 2]
    transport = _serving(truncated)

    with transport.client() as client, pytest.raises(h.DatasetVerificationError):
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert list(tmp_path.iterdir()) == []


# --- 取得の失敗 -------------------------------------------------------------


def test_connection_error_becomes_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("接続できない", request=request)

    transport = _Transport(handler)
    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert exc_info.value.reason
    assert _SMALL_PIN.url in str(exc_info.value)
    assert list(tmp_path.iterdir()) == []


def test_http_404_becomes_unavailable(tmp_path: Path) -> None:
    transport = _Transport(lambda request: httpx.Response(404, content=b"nope"))
    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert "404" in str(exc_info.value)
    assert list(tmp_path.iterdir()) == []


def test_missing_cache_without_download_is_unavailable(tmp_path: Path) -> None:
    transport = _never_called()
    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client, allow_download=False)

    assert exc_info.value.reason
    assert transport.requests == []


def test_oversized_body_with_content_length_is_refused(tmp_path: Path) -> None:
    huge = b"x" * (_SMALL_PIN.size_bytes * 8)
    transport = _Transport(lambda request: httpx.Response(200, content=huge))

    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    # 本文を読む前に、宣言された大きさ (Content-Length) だけで断る
    assert str(len(huge)) in str(exc_info.value)
    assert list(tmp_path.iterdir()) == []


def test_oversized_body_without_content_length_is_refused(tmp_path: Path) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(64):
            yield b"x" * _SMALL_PIN.size_bytes

    # 反復子を渡すと、httpx は Content-Length を付けない (chunked)。
    # 事前の確認ではなく、読みながらの上限で止まることを確かめる
    transport = _Transport(lambda request: httpx.Response(200, content=chunks()))
    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    # 読みながらの上限で止まる (宣言された大きさは使えない)
    assert "超えた" in str(exc_info.value)
    assert list(tmp_path.iterdir()) == []


# --- 転送 (redirect) --------------------------------------------------------


def test_https_redirect_is_followed(tmp_path: Path) -> None:
    """本物の GitHub のリリースの資産は、CDN の別のホストへ転送される。"""
    cdn = "https://release-assets.example.invalid/asset"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _SMALL_PIN.url:
            return httpx.Response(302, headers={"location": cdn})
        assert str(request.url) == cdn
        return httpx.Response(200, content=_SMALL_DATA)

    transport = _Transport(handler)
    with transport.client() as client:
        path = h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert path.is_file()
    assert [str(r.url) for r in transport.requests] == [_SMALL_PIN.url, cdn]


def test_redirect_to_non_https_is_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://evil.invalid/asset"})

    transport = _Transport(handler)
    with transport.client() as client, pytest.raises(h.DatasetUnavailable) as exc_info:
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert "https" in str(exc_info.value).lower()
    assert len(transport.requests) == 1  # 転送先へは要求を出さない
    assert list(tmp_path.iterdir()) == []


def test_redirect_loop_is_bounded(tmp_path: Path) -> None:
    transport = _Transport(
        lambda request: httpx.Response(302, headers={"location": _SMALL_PIN.url})
    )
    with transport.client() as client, pytest.raises(h.DatasetUnavailable):
        h.ensure_humaneval_plus(tmp_path, pin=_SMALL_PIN, client=client)

    assert len(transport.requests) <= 8
    assert list(tmp_path.iterdir()) == []


def test_non_https_pin_is_refused_before_any_request(tmp_path: Path) -> None:
    pin = _pin(_SMALL_DATA, 3, url="http://example.invalid/he.jsonl.gz")
    transport = _never_called()
    with transport.client() as client, pytest.raises(h.DatasetUnavailable):
        h.ensure_humaneval_plus(tmp_path, pin=pin, client=client)

    assert transport.requests == []


# --- 置き場所の安全 ---------------------------------------------------------


def _init_git_repo(root: Path, *, gitignore: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(gitignore, encoding="utf-8")


def test_cache_dir_under_version_control_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="")  # data-cache/ は無視されていない
    tracked = repo / "data-cache"
    transport = _never_called()

    with transport.client() as client, pytest.raises(h.DatasetCacheError):
        h.ensure_humaneval_plus(tracked, pin=_SMALL_PIN, client=client)

    assert not tracked.exists()  # ディレクトリも作らない
    assert transport.requests == []


def test_git_ignored_cache_dir_passes_the_placement_check(tmp_path: Path) -> None:
    """git の作業木の中でも、無視されている場所なら通る (既定の `bench/data-cache/` と同じ形)。

    通ったことは、次の段階 (取得) のエラーになることで分かる。ここでは取得しない。
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo, gitignore="data-cache/\n")

    with pytest.raises(h.DatasetUnavailable):
        h.ensure_humaneval_plus(repo / "data-cache", pin=_SMALL_PIN, allow_download=False)


def test_cache_dir_outside_any_repo_passes_the_placement_check(tmp_path: Path) -> None:
    with pytest.raises(h.DatasetUnavailable):
        h.ensure_humaneval_plus(tmp_path / "elsewhere", pin=_SMALL_PIN, allow_download=False)


def test_symlinked_cache_file_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.gz"
    real.write_bytes(_SMALL_DATA)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / _SMALL_PIN.file_name).symlink_to(real)

    with pytest.raises(h.DatasetCacheError):
        h.ensure_humaneval_plus(cache_dir, pin=_SMALL_PIN, allow_download=False)


# --- 範囲の確認 (note 2.7) --------------------------------------------------


def _write(tmp_path: Path, data: bytes, count: int) -> tuple[Path, h.DatasetPin]:
    pin = _pin(data, count)
    path = tmp_path / pin.file_name
    path.write_bytes(data)
    return path, pin


def test_wrong_problem_count_is_refused(tmp_path: Path) -> None:
    data = _gz([_record(0), _record(1)])
    path, pin = _write(tmp_path, data, count=2)
    broken = dataclasses.replace(pin, problem_count=3)

    with pytest.raises(h.DatasetFormatError) as exc_info:
        h.read_problems(path, pin=broken)
    assert "3" in str(exc_info.value) and "2" in str(exc_info.value)


@pytest.mark.parametrize(
    ("records", "needle"),
    [
        pytest.param([_record(0), _record(0)], "HumanEval/0", id="duplicate_id"),
        pytest.param([_record(0, task_id="HumanEval/x")], "HumanEval/x", id="bad_id_form"),
        pytest.param([_record(0, task_id="MBPP/0")], "MBPP/0", id="wrong_prefix"),
        pytest.param([_record(0, prompt="   ")], "prompt", id="empty_prompt"),
        pytest.param([_record(0, test="")], "test", id="empty_test"),
        pytest.param([_record(0, entry_point="")], "entry_point", id="empty_entry_point"),
        pytest.param(
            [_record(0, entry_point="absent_name")], "prompt に現れない", id="entry_point_absent"
        ),
        pytest.param([_record(0, entry_point="1bad")], "entry_point", id="entry_point_not_ident"),
        pytest.param([_record(0, test="assert True\n")], "check", id="test_without_check"),
        pytest.param([_record(0, prompt=42)], "prompt", id="prompt_not_a_string"),
        pytest.param([_record(0, canonical_solution=5)], "canonical_solution", id="solution_type"),
    ],
)
def test_records_outside_the_expected_range_are_refused(
    tmp_path: Path, records: list[dict[str, Any]], needle: str
) -> None:
    path, pin = _write(tmp_path, _gz(records), count=len(records))

    with pytest.raises(h.DatasetFormatError) as exc_info:
        h.read_problems(path, pin=pin)
    assert needle in str(exc_info.value)


def test_missing_key_is_refused(tmp_path: Path) -> None:
    record = _record(0)
    del record["entry_point"]
    path, pin = _write(tmp_path, _gz([record]), count=1)

    with pytest.raises(h.DatasetFormatError) as exc_info:
        h.read_problems(path, pin=pin)
    assert "entry_point" in str(exc_info.value)


def test_line_that_is_not_json_is_refused(tmp_path: Path) -> None:
    data = gzip.compress(b'{"task_id": "HumanEval/0"\n', mtime=0)
    path, pin = _write(tmp_path, data, count=1)

    with pytest.raises(h.DatasetFormatError) as exc_info:
        h.read_problems(path, pin=pin)
    assert "1" in str(exc_info.value)  # 何行目かを示す


def test_line_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    data = gzip.compress(b"[1, 2, 3]\n", mtime=0)
    path, pin = _write(tmp_path, data, count=1)

    with pytest.raises(h.DatasetFormatError):
        h.read_problems(path, pin=pin)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    body = "\n".join(json.dumps(r) for r in [_record(0), _record(1)]) + "\n\n"
    data = gzip.compress(body.encode("utf-8"), mtime=0)
    path, pin = _write(tmp_path, data, count=2)

    assert len(h.read_problems(path, pin=pin)) == 2


def test_not_gzip_is_refused(tmp_path: Path) -> None:
    data = b'{"task_id": "HumanEval/0"}\n'
    path, pin = _write(tmp_path, data, count=1)

    with pytest.raises(h.DatasetFormatError):
        h.read_problems(path, pin=pin)


# --- 部分集合 (Profile.quality.code_problem_limit) --------------------------


def test_select_problems_takes_the_first_n_in_order(tmp_path: Path) -> None:
    path, pin = _write(tmp_path, _SMALL_DATA, count=3)
    problems = h.read_problems(path, pin=pin)

    assert [p.task_id for p in h.select_problems(problems, 2)] == ["HumanEval/0", "HumanEval/1"]
    assert h.select_problems(problems, None) == problems
    assert h.select_problems(problems, 99) == problems


def test_select_problems_leaves_out_the_problem_whose_own_test_is_broken(tmp_path: Path) -> None:
    """`HumanEval/32` は、正解の解でも検査に通らない (6.1 で確かめた)。外してから数える。"""
    records = [_record(31), _record(32), _record(33), _record(34)]
    path, pin = _write(tmp_path, _gz(records), count=4)
    problems = h.read_problems(path, pin=pin)
    assert "HumanEval/32" in h.UNSCORABLE_PROBLEMS

    assert [p.task_id for p in h.select_problems(problems, None)] == [
        "HumanEval/31",
        "HumanEval/33",
        "HumanEval/34",
    ]
    # 限りは、外したあとの数で数える (32 のぶん、1 問少なくならない)
    assert [p.task_id for p in h.select_problems(problems, 2)] == ["HumanEval/31", "HumanEval/33"]
    # 読み取りの側は、164 問をそのまま返す (外すのは、採点に回すときだけ)
    assert len(problems) == 4


def test_select_problems_refuses_a_limit_below_one(tmp_path: Path) -> None:
    path, pin = _write(tmp_path, _SMALL_DATA, count=3)
    problems = h.read_problems(path, pin=pin)

    with pytest.raises(ValueError):
        h.select_problems(problems, 0)


# --- 本物の取得 (`BENCH_NETWORK_TESTS=1` のときだけ) -------------------------


@pytest.mark.skipif(
    os.environ.get("BENCH_NETWORK_TESTS") != "1",
    reason="ネットワークに触れるので、BENCH_NETWORK_TESTS=1 のときだけ動かす",
)
def test_real_dataset_downloads_and_parses(tmp_path: Path) -> None:
    ref, problems = h.load_humaneval_plus(tmp_path)

    assert len(problems) == 164
    assert problems[0].task_id == "HumanEval/0"
    assert problems[-1].task_id == "HumanEval/163"
    assert ref.sha256 == h.HUMANEVAL_PLUS.sha256
    assert (tmp_path / h.HUMANEVAL_PLUS.file_name).stat().st_size == h.HUMANEVAL_PLUS.size_bytes
