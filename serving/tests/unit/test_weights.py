"""重みのマニフェストの生成の試験 (tasks.md 3.2)。

確かめること (design.md 「イメージと重み › weights」、tasks.md 3.2 の完了の状態。レビューの
指摘を受けて足したものを含む):

- 偽の `tree` API の応答 (ページ送り、サブディレクトリを含む) から、`path` の順のマニフェスト
  ができる
- モデルカードらしい名前 (`readme` で始まる、`model_card` / `modelcard` / `model-card` で
  始まる。大小文字を区別しない) と `.gitattributes` に対して、中身を取りに行く要求が 1 つも
  出ない。マニフェストにも載らない。除いた道筋の一覧が、結果と一緒に返る
- 同じ入力 (同じ `generated_at`) から、同じバイト列のファイルができる
- どの要求にも `Authorization` ヘッダがなく、環境に `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` を
  置いても、どの要求にも (ヘッダにも URL にも) 現れない
- LFS のファイルは取得されず、一覧の値がそのまま使われる (`size == lfs.size` を確かめる)。
  LFS でないファイルだけが取得され、`sha256` が計算される
- 取得した大きさが一覧と合わないと断る。LFS でないのに大きすぎるファイルは、取得せずに断る
- 401 / 403 / 404 / 500、壊れた JSON、配列でない応答、接続の失敗、時間切れを、日本語の文の
  例外で断る
- 版が 40 桁の 16 進でない、`repo` が空か前後に空白があると、要求を 1 つも出さずに断る
- **一覧のページ送り (`Link` ヘッダ) は、スキームが https、ホストが Hub のホスト、道筋が
  いまの一覧の道筋と一致するときだけ辿る**。よそのホストや、別の道筋には、要求を 1 つも
  出さない (指摘 1)
- **一覧の `path` は、中身を取りに行くどの要求よりも前にすべて確かめる**。先頭の `/`、
  バックスラッシュ、`.` / `..` / 空の要素、制御文字、長さの上限、重複を、要求を 1 つも
  出さずに断る (指摘 2)
- **中身の取得の転送は自分でたどり**、上限 (5 回) を超えない。行き先は、スキームが https で、
  ホストが `huggingface.co` そのものか `.huggingface.co` / `.hf.co` で終わるものだけを許す。
  それ以外への転送は、要求を出さずに断る (指摘 3)
- `new_client()` は `trust_env=False` で作る

ネットワークには一切つながない (`httpx.MockTransport`)。実物の Hugging Face Hub には、
どの試験からもつながない。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from serving_kit import types as kit_types
from serving_kit import weights as w

REPO = "test-org/test-weights"
REVISION = "a" * 40
GENERATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
CONFIG_CONTENT = b'{"hello": "world"}\n'
VOCAB_CONTENT = b'{"vocab": ["a", "b"]}\n'
TREE_URL = f"{w.HUB_BASE_URL}/api/models/{REPO}/tree/{REVISION}?recursive=1"


def _resolve_url(path: str) -> str:
    """`_resolve_url` と同じ形 (試験の道筋は、記号を含まないので、quote の結果と一致する)。"""
    return f"{w.HUB_BASE_URL}/{REPO}/resolve/{REVISION}/{path}"


def _entry(
    path: str, size: int = 0, *, is_dir: bool = False, lfs_sha256: str | None = None
) -> dict[str, Any]:
    """`tree` API の一覧の 1 行を、偽物として組み立てる (2026-09-22 に確かめた実物の形)。"""
    raw: dict[str, Any] = {"type": "directory" if is_dir else "file", "path": path, "oid": "0" * 40}
    if not is_dir:
        raw["size"] = size
        if lfs_sha256 is not None:
            raw["lfs"] = {"oid": lfs_sha256, "size": size, "pointerSize": 136}
            raw["xetHash"] = "9" * 64  # 想定の外の鍵 (読まない。断らないことを確かめる)
    return raw


def _json_response(
    entries: list[dict[str, Any]], *, link: str | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    headers = {"Link": link} if link is not None else {}

    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=entries, headers=headers)

    return make


def _status_response(status: int) -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "for-test"})

    return make


def _content_response(content: bytes) -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    return make


def _redirect_response(
    location: str, *, status: int = 302
) -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"location": location})

    return make


def _broken_json_response() -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {{{")

    return make


def _non_array_json_response() -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not": "a list"})

    return make


def _connect_error() -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("接続できない (試験)", request=request)

    return make


def _read_timeout() -> Callable[[httpx.Request], httpx.Response]:
    def make(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("時間切れ (試験)", request=request)

    return make


class _Transport:
    """呼ばれた要求を記録する `httpx.MockTransport` の包み (bench の試験と同じ形)。"""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._dispatch), trust_env=False)


def _router(mapping: dict[str, Callable[[httpx.Request], httpx.Response]]) -> _Transport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url not in mapping:
            raise AssertionError(f"想定していない要求: {request.method} {url}")
        return mapping[url](request)

    return _Transport(handler)


def _never_called() -> _Transport:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 呼ばれたら失敗
        raise AssertionError(f"ネットワークに触れてはいけない: {request.url}")

    return _Transport(handler)


# --- HTTP クライアント ----------------------------------------------------


def test_new_client_disables_trust_env_and_does_not_auto_follow_redirects() -> None:
    client = w.new_client(5.0)
    try:
        assert client.trust_env is False
        assert client.follow_redirects is False
    finally:
        client.close()


# --- マニフェストの組み立て ------------------------------------------------


def test_the_manifest_is_sorted_paginated_recursive_and_excludes_the_model_card() -> None:
    page2_url = f"{TREE_URL}&cursor=p2"
    excluded_paths = ["README.md", ".gitattributes", "docs/README.md", "readme.md"]
    page1_entries = [
        _entry("README.md", 10),
        _entry(".gitattributes", 5),
        _entry("docs/README.md", 20),
        _entry("readme.md", 8),
        _entry("tokenizer", is_dir=True),
        _entry("config.json", len(CONFIG_CONTENT)),
        _entry("model-00001-of-00002.safetensors", 10_000, lfs_sha256="1" * 64),
    ]
    page2_entries = [
        _entry("tokenizer/vocab.json", len(VOCAB_CONTENT)),
        _entry("model-00002-of-00002.safetensors", 20_000, lfs_sha256="2" * 64),
    ]
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response(page1_entries, link=f'<{page2_url}>; rel="next"'),
        page2_url: _json_response(page2_entries),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
        _resolve_url("tokenizer/vocab.json"): _content_response(VOCAB_CONTENT),
    }
    transport = _router(mapping)

    result = w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    manifest = result.manifest

    assert [entry.path for entry in manifest.files] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer/vocab.json",
    ]
    assert sorted(result.excluded_paths) == sorted(excluded_paths)
    by_path = {entry.path: entry for entry in manifest.files}
    assert by_path["config.json"].sha256 == hashlib.sha256(CONFIG_CONTENT).hexdigest()
    assert by_path["config.json"].size == len(CONFIG_CONTENT)
    assert by_path["tokenizer/vocab.json"].sha256 == hashlib.sha256(VOCAB_CONTENT).hexdigest()
    assert by_path["model-00001-of-00002.safetensors"].sha256 == "1" * 64
    assert by_path["model-00001-of-00002.safetensors"].size == 10_000
    assert by_path["model-00002-of-00002.safetensors"].sha256 == "2" * 64
    assert by_path["model-00002-of-00002.safetensors"].size == 20_000
    assert manifest.total_bytes == sum(entry.size for entry in manifest.files)
    assert manifest.repo == REPO
    assert manifest.revision == REVISION

    requested_urls = [str(request.url) for request in transport.requests]
    for excluded_path in excluded_paths:
        assert _resolve_url(excluded_path) not in requested_urls
    # LFS のファイルにも、中身を取りに行く要求が出ない (一覧の値をそのまま使う)
    assert _resolve_url("model-00001-of-00002.safetensors") not in requested_urls
    assert _resolve_url("model-00002-of-00002.safetensors") not in requested_urls
    for request in transport.requests:
        assert "authorization" not in request.headers


def test_broadened_exclusion_patterns_are_excluded_and_not_fetched() -> None:
    excluded_entries = [
        _entry("README.txt", 5),
        _entry("README.md.bak", 5),
        _entry("readme_ja.md", 5),
        _entry("docs/README_ja.md", 5),
        _entry("model_card.json", 5),
        _entry("modelcard.yaml", 5),
        _entry("model-card.txt", 5),
        _entry("MODEL_CARD.MD", 5),
    ]
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([*excluded_entries, _entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    transport = _router(mapping)

    result = w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())

    assert [entry.path for entry in result.manifest.files] == ["config.json"]
    assert sorted(result.excluded_paths) == sorted(entry["path"] for entry in excluded_entries)
    requested = [str(request.url) for request in transport.requests]
    for entry in excluded_entries:
        assert _resolve_url(entry["path"]) not in requested


def test_no_authorization_header_appears_even_with_an_hf_token_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret-abc")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub-secret-xyz")
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    transport = _router(mapping)

    w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())

    assert transport.requests, "要求が 1 つも出ていない"
    for request in transport.requests:
        assert "authorization" not in request.headers
        assert "hf-secret-abc" not in str(request.url)
        assert "hub-secret-xyz" not in str(request.url)
        for value in request.headers.values():
            assert "hf-secret-abc" not in value
            assert "hub-secret-xyz" not in value


def test_the_same_input_produces_byte_identical_manifests() -> None:
    def make_mapping() -> dict[str, Callable[[httpx.Request], httpx.Response]]:
        return {
            TREE_URL: _json_response(
                [
                    _entry("config.json", len(CONFIG_CONTENT)),
                    _entry("model.safetensors", 999, lfs_sha256="3" * 64),
                ]
            ),
            _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
        }

    manifest1 = w.build_manifest(
        REPO, REVISION, generated_at=GENERATED_AT, client=_router(make_mapping()).client()
    ).manifest
    manifest2 = w.build_manifest(
        REPO, REVISION, generated_at=GENERATED_AT, client=_router(make_mapping()).client()
    ).manifest

    bytes1 = w.to_json_bytes(manifest1)
    bytes2 = w.to_json_bytes(manifest2)
    assert bytes1 == bytes2
    assert bytes1.endswith(b"\n")
    assert not bytes1.endswith(b"\n\n")
    payload = json.loads(bytes1)
    assert list(payload.keys()) == sorted(payload.keys())
    assert list(payload["files"][0].keys()) == sorted(payload["files"][0].keys())


def test_writing_and_loading_a_manifest_round_trips(tmp_path: Path) -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    manifest = w.build_manifest(
        REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
    ).manifest
    path = w.manifest_path(REPO, weights_dir=tmp_path)
    assert path == tmp_path / "test-org__test-weights.manifest.json"

    w.write_manifest(manifest, path)
    loaded = w.load_manifest(path)

    assert loaded == manifest
    assert w.to_json_bytes(loaded) == w.to_json_bytes(manifest)
    assert path.read_bytes() == w.to_json_bytes(manifest)


# --- 派生の重み (手元で変換した重み) のマニフェスト -------------------------

ORIGIN_REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
ORIGIN_REVISION = "18d55bfd" + "0" * 32
DERIVED_TOOL = "experiments/k2-quant/convert.py"
DERIVED_COMMIT = "b" * 40
DERIVED_TARGET = r"^model\.layers\.\d+\.self_attn\..*$"


def _derived_payload() -> dict[str, Any]:
    """#56 の道具が書く、派生のマニフェストの JSON (契約の形。`serving_kit` を通さず手で書く)。"""
    return {
        "kind": "derived",
        "derivation": {
            "name": "k2s1",
            "origin": {"repo": ORIGIN_REPO, "revision": ORIGIN_REVISION},
            "conversion": {
                "tool": DERIVED_TOOL,
                "commit": DERIVED_COMMIT,
                "args": ["--dtype", "fp8", "--note", "日本語"],
                "target_pattern": DERIVED_TARGET,
            },
        },
        "generated_at": "2026-09-25T00:00:00Z",
        "total_bytes": 5096,
        "files": [
            {"path": "config.json", "sha256": "1" * 64, "size": 96},
            {"path": "model-00001-of-00001.safetensors", "sha256": "2" * 64, "size": 5000},
        ],
    }


def _hub_payload() -> dict[str, Any]:
    return {
        "repo": ORIGIN_REPO,
        "revision": ORIGIN_REVISION,
        "generated_at": "2026-09-25T00:00:00Z",
        "total_bytes": 96,
        "files": [{"path": "config.json", "sha256": "1" * 64, "size": 96}],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _derived_manifest() -> kit_types.DerivedWeightsManifest:
    return kit_types.DerivedWeightsManifest.model_validate(_derived_payload())


def test_a_derived_manifest_file_is_loaded_as_a_derived_manifest(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "k2s1.manifest.json", _derived_payload())

    loaded = w.load_manifest(path)

    assert isinstance(loaded, kit_types.DerivedWeightsManifest)
    assert loaded.derivation.name == "k2s1"
    assert loaded.derivation.origin.repo == ORIGIN_REPO
    assert loaded.derivation.origin.revision == ORIGIN_REVISION
    assert loaded.derivation.conversion.commit == DERIVED_COMMIT
    assert loaded.derivation.conversion.args == ("--dtype", "fp8", "--note", "日本語")
    assert loaded.derivation.conversion.target_pattern == DERIVED_TARGET
    assert loaded.total_bytes == 5096
    assert [entry.path for entry in loaded.files] == [
        "config.json",
        "model-00001-of-00001.safetensors",
    ]


def test_a_hub_manifest_file_is_still_loaded_as_a_hub_manifest(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "hub.manifest.json", _hub_payload())

    loaded = w.load_manifest(path)

    assert isinstance(loaded, kit_types.WeightsManifest)
    assert (loaded.repo, loaded.revision) == (ORIGIN_REPO, ORIGIN_REVISION)


def test_a_derived_looking_file_without_the_kind_is_not_read_as_derived(tmp_path: Path) -> None:
    # 読み分けは、明示した `kind` だけで行う。項目の有無から、派生だと推測して読まない
    payload = _derived_payload()
    del payload["kind"]
    path = _write_json(tmp_path / "k2s1.manifest.json", payload)

    with pytest.raises(w.WeightsError) as caught:
        w.load_manifest(path)

    assert str(path) in str(caught.value)


def test_a_derived_manifest_file_with_a_broken_body_refuses_with_a_weights_error(
    tmp_path: Path,
) -> None:
    payload = _derived_payload()
    del payload["derivation"]["conversion"]["commit"]
    path = _write_json(tmp_path / "k2s1.manifest.json", payload)

    with pytest.raises(w.WeightsError) as caught:
        w.load_manifest(path)

    assert str(path) in str(caught.value)


def test_writing_and_loading_a_derived_manifest_round_trips(tmp_path: Path) -> None:
    manifest = _derived_manifest()
    path = tmp_path / "weights" / "k2s1.manifest.json"

    w.write_manifest(manifest, path)
    loaded = w.load_manifest(path)

    assert loaded == manifest
    assert w.to_json_bytes(loaded) == w.to_json_bytes(manifest)
    assert path.read_bytes() == w.to_json_bytes(manifest)


def test_the_same_derived_input_produces_byte_identical_manifests() -> None:
    bytes1 = w.to_json_bytes(_derived_manifest())
    bytes2 = w.to_json_bytes(_derived_manifest())

    assert bytes1 == bytes2
    assert bytes1.endswith(b"\n")
    assert not bytes1.endswith(b"\n\n")
    payload = json.loads(bytes1)
    assert payload["kind"] == "derived"
    assert list(payload.keys()) == sorted(payload.keys())
    assert list(payload["derivation"].keys()) == sorted(payload["derivation"].keys())
    assert list(payload["derivation"]["conversion"].keys()) == sorted(
        payload["derivation"]["conversion"].keys()
    )
    assert list(payload["files"][0].keys()) == sorted(payload["files"][0].keys())


# --- 誤り: 大きさと sha256 --------------------------------------------------


def test_a_size_mismatch_after_the_download_refuses() -> None:
    declared_size = len(CONFIG_CONTENT) + 5  # わざと食い違わせる
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", declared_size)]),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    with pytest.raises(w.WeightsFetchError, match="大きさ"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_content_larger_than_declared_is_truncated_and_refused() -> None:
    # 一覧の大きさより、実際に届いた本文のほうが大きい場合 (declared < actual)。読みながら
    # 打ち切るので、`_size_mismatch` の試験 (declared > actual) とは別の道筋を通る
    declared_size = 4
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", declared_size)]),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    with pytest.raises(w.WeightsFetchError, match="超えた"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_large_non_lfs_file_is_refused_without_being_fetched() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("weird.bin", 20)]),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsRefError, match="大きすぎる"):
        w.build_manifest(
            REPO,
            REVISION,
            generated_at=GENERATED_AT,
            client=transport.client(),
            small_file_max_bytes=10,
        )
    assert _resolve_url("weird.bin") not in [str(request.url) for request in transport.requests]


def test_a_malformed_lfs_oid_refuses() -> None:
    entry = _entry("model.safetensors", 100)
    entry["lfs"] = {"oid": "not-a-sha256", "size": 100}
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([entry]),
    }

    with pytest.raises(w.WeightsFetchError, match="lfs.oid"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_lfs_size_mismatched_with_the_top_level_size_refuses() -> None:
    entry = _entry("model.safetensors", 100, lfs_sha256="4" * 64)
    entry["lfs"]["size"] = 200  # size (100) と lfs.size (200) を、わざと食い違わせる
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([entry]),
    }

    with pytest.raises(w.WeightsFetchError, match="size"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


# --- 誤り: 一覧の応答の形 ---------------------------------------------------


def test_a_server_error_on_the_listing_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _status_response(500),
    }
    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_malformed_json_in_the_listing_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _broken_json_response(),
    }
    with pytest.raises(w.WeightsFetchError, match="JSON"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_non_array_listing_response_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _non_array_json_response(),
    }
    with pytest.raises(w.WeightsFetchError, match="配列"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_connection_failure_on_the_listing_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _connect_error(),
    }
    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_timeout_on_the_listing_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _read_timeout(),
    }
    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_string_size_refuses() -> None:
    entry = _entry("config.json", 10)
    entry["size"] = "10"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([entry]),
    }
    with pytest.raises(w.WeightsFetchError, match="size"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_missing_path_refuses() -> None:
    entry = _entry("config.json", 10)
    del entry["path"]
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([entry]),
    }
    with pytest.raises(w.WeightsFetchError, match="path"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_a_non_object_row_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response(["not-an-object"]),  # type: ignore[list-item]
    }
    with pytest.raises(w.WeightsFetchError, match="オブジェクト"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


def test_an_unknown_type_value_refuses() -> None:
    entry = _entry("config.json", 10)
    entry["type"] = "symlink"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([entry]),
    }
    with pytest.raises(w.WeightsFetchError, match="type"):
        w.build_manifest(
            REPO, REVISION, generated_at=GENERATED_AT, client=_router(mapping).client()
        )


@pytest.mark.parametrize(
    ("status", "expect_auth_note"),
    [(401, True), (403, True), (404, False)],
)
def test_401_403_and_404_refuse_without_asking_for_a_token(
    status: int, expect_auth_note: bool
) -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _status_response(status),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsRefError) as excinfo:
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    message = str(excinfo.value)
    if expect_auth_note:
        # 認証が要ることは言うが、トークンを求めずに断る (メッセージにその旨がある)
        assert "認証" in message
        assert "トークンを渡さずに断る" in message
    else:
        assert "見つからない" in message
    assert len(transport.requests) == 1  # 一覧の 1 回だけ。取り直しも、ほかの道筋への要求もない


@pytest.mark.parametrize("revision", ["short", "A" * 40, "g" * 40, ("a" * 39) + "!"])
def test_an_invalid_revision_refuses_before_any_request(revision: str) -> None:
    transport = _never_called()

    with pytest.raises(w.WeightsRefError):
        w.build_manifest(REPO, revision, generated_at=GENERATED_AT, client=transport.client())
    assert transport.requests == []


@pytest.mark.parametrize("repo", ["", "  x  ", " x"])
def test_an_invalid_repo_refuses_before_any_request(repo: str) -> None:
    transport = _never_called()

    with pytest.raises(w.WeightsRefError):
        w.build_manifest(repo, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert transport.requests == []


def test_a_naive_generated_at_is_rejected() -> None:
    transport = _never_called()

    with pytest.raises(ValueError, match="tz-aware"):
        w.build_manifest(
            REPO, REVISION, generated_at=datetime(2026, 1, 1), client=transport.client()
        )
    assert transport.requests == []


# --- 誤り: 一覧の path の安全 (指摘 2) --------------------------------------


def test_a_path_traversal_entry_refuses_before_any_content_request() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response(
            [
                _entry("config.json", len(CONFIG_CONTENT)),
                _entry("../../../../etc/passwd", 10),
            ]
        ),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    # 一覧の 1 回だけで、中身を取りに行く要求 (config.json を含む) が 1 つも出ていない
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


@pytest.mark.parametrize(
    "bad_path",
    [
        pytest.param("/etc/passwd", id="leading-slash"),
        pytest.param("a/../b", id="dotdot-segment"),
        pytest.param("a/./b", id="dot-segment"),
        pytest.param("a//b", id="empty-segment"),
        pytest.param("a\\b", id="backslash"),
        pytest.param("a\nb", id="control-newline"),
        pytest.param("a\tb", id="control-tab"),
        pytest.param("a\x00b", id="control-nul"),
        pytest.param("a" * 2000, id="too-long"),
    ],
)
def test_an_unsafe_path_refuses_before_any_content_request(bad_path: str) -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry(bad_path, 10)]),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


def test_a_duplicate_path_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", 1), _entry("config.json", 2)]),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError, match="2 度"):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


# --- 誤り: ページ送りの行き先 (指摘 1) ---------------------------------------


def test_a_pagination_link_to_a_different_host_refuses() -> None:
    evil_next = f"https://evil.example.invalid/api/models/{REPO}/tree/{REVISION}?recursive=1"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response(
            [_entry("config.json", len(CONFIG_CONTENT))], link=f'<{evil_next}>; rel="next"'
        ),
        _resolve_url("config.json"): _content_response(CONFIG_CONTENT),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    # 一覧の最初の 1 回だけ。よそのホストへの要求が出ていない
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


def test_a_pagination_link_with_a_different_path_refuses() -> None:
    wrong_path_next = (
        f"{w.HUB_BASE_URL}/api/models/other-org/other-repo/tree/{REVISION}?recursive=1"
    )
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([], link=f'<{wrong_path_next}>; rel="next"'),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


def test_a_pagination_link_downgrading_to_http_refuses() -> None:
    http_next = f"http://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1&cursor=x"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([], link=f'<{http_next}>; rel="next"'),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


# --- 転送 (redirect。指摘 3) -------------------------------------------------


def test_a_content_redirect_to_a_disallowed_host_refuses() -> None:
    evil_url = "https://evil.example.invalid/blob/abc"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _redirect_response(evil_url),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert evil_url not in [str(request.url) for request in transport.requests]


def test_a_content_redirect_downgrading_to_http_refuses() -> None:
    http_url = "http://huggingface.co/blob/abc"
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _redirect_response(http_url),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert http_url not in [str(request.url) for request in transport.requests]


@pytest.mark.parametrize(
    "location",
    [
        "data:text/plain;base64,QUJD",
        "javascript:alert(1)",
        "urn:example:blob",
        "mailto:x@example.invalid",
    ],
)
def test_a_content_redirect_to_an_opaque_scheme_refuses_without_a_traceback(location: str) -> None:
    """`//` を持たない形の `Location` は、httpx が `InvalidURL` を投げる (`HTTPError` ではない)。

    それも、traceback ではなく、この module の誤りとして断る (design.md 「Error Handling」)。
    """
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _redirect_response(location),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())


def test_a_listing_redirect_to_an_opaque_scheme_refuses_without_a_traceback() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _redirect_response("data:text/plain;base64,QUJD"),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())
    assert [str(request.url) for request in transport.requests] == [TREE_URL]


def test_too_many_content_redirects_refuses() -> None:
    mapping: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        TREE_URL: _json_response([_entry("config.json", len(CONFIG_CONTENT))]),
        _resolve_url("config.json"): _redirect_response(_resolve_url("config.json")),
    }
    transport = _router(mapping)

    with pytest.raises(w.WeightsFetchError, match="転送"):
        w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())


@pytest.mark.parametrize("cdn_host", ["cdn-lfs.huggingface.co", "cdn-lfs.hf.co", "huggingface.co"])
def test_a_content_redirect_to_an_allowed_hub_host_is_followed_without_leaking_headers(
    monkeypatch: pytest.MonkeyPatch, cdn_host: str
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret-redirect")
    cdn_url = f"https://{cdn_host}/blob/abc"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == TREE_URL:
            return httpx.Response(200, json=[_entry("config.json", len(CONFIG_CONTENT))])
        if url == _resolve_url("config.json"):
            return httpx.Response(302, headers={"location": cdn_url})
        if url == cdn_url:
            return httpx.Response(200, content=CONFIG_CONTENT)
        raise AssertionError(f"想定していない要求: {url}")

    transport = _Transport(handler)

    result = w.build_manifest(REPO, REVISION, generated_at=GENERATED_AT, client=transport.client())

    assert result.manifest.files[0].sha256 == hashlib.sha256(CONFIG_CONTENT).hexdigest()
    assert len(transport.requests) == 3  # 一覧 + 中身の要求 (302) + 転送先 (CDN)
    for request in transport.requests:
        assert "authorization" not in request.headers
        assert "hf-secret-redirect" not in str(request.url)
