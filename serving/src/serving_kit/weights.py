"""重みのマニフェストの生成 (design.md 「イメージと重み › weights」、tasks.md 3.2)。

Mac で、Hugging Face Hub の**公開の API を匿名で**読み、固定した版 (40 桁の commit) の
ファイルの一覧、大きさ、sha256 から `types.WeightsManifest` を組み立てる。**取得のコンテナを
2 台に起こすこと、2 台での照合 (`serve fetch` / `serve verify`) は 3.3 の仕事なので、ここには
置かない** (tasks.md 3.2 の `_Boundary: weights_`)。

## Hub の API の形

2026-09-22 に、実物の公開の API を匿名で 1 度読んで確かめた (レビューの指摘を受けて)。

- 一覧は `GET https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1`
  (research.md b-4、205 行。OpenAPI の説明: "List the content of a repository tree,
  with pagination support.")
- 一覧の 1 行は `type` (`"file"` / `"directory"`)、`path`、`size`、`oid` (git の sha1、
  40 桁) を持つ。LFS のファイルには、加えて `lfs: {"oid": <64 桁の sha256>, "size": <バイト>,
  "pointerSize": <バイト>}` と `xetHash` (64 桁) が付き、**`size` と `lfs.size` が一致する**
  (一致しなければ断る)
- ページ送りは、応答のヘッダ `link: <https://huggingface.co/api/models/…?…&cursor=…>;
  rel="next"` (絶対の URL)
- 中身の取得は `.../resolve/{revision}/<path>` の形 (research.md 206 行が引く
  `HEAD .../resolve/main/<file>` と同じ道筋)。CDN への転送 (302) がありうる

`xetHash`、`pointerSize`、`oid` (git sha1) は読まない。想定の外の鍵があっても断らない。

## 決めごと

- **認証の情報を一切使わない**: `Authorization` ヘッダを付けない。環境変数
  (`HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN` など。`os.environ` を読まない)、
  `~/.cache/huggingface/token`、`huggingface_hub` のどれも使わない。依存は `httpx` だけ
  (requirements 2.6)。`httpx.Client` は `trust_env=False` で作り、環境の `.netrc` やプロキシの
  認証も拾わない (`bench/src/bench_harness/corpus/humaneval.py` と同じ決めごと)
- **モデルカードらしい名前を取得も記載もしない** (requirements 8.8、11.3): モデルカードは、
  DGX Spark の起動のレシピそのものでありうるので開かない。道筋のどの深さにあっても
  (basename で見る、大小文字を区別しない)、`readme` で始まる名前 (`README`、`README.md`、
  `README.txt`、`README.md.bak`、`readme_ja.md` など)、`model_card` / `modelcard` /
  `model-card` で始まる名前、`.gitattributes` は、除く側に倒す。除いた行には、中身を取りに
  行く要求を 1 つも出さない。除いた道筋の一覧は `ManifestResult.excluded_paths` で返す
  (名前だけ。中身は取らない)。マニフェストの合計の大きさにも入らない
- **一覧のページ送り (`Link` ヘッダ) の行き先を確かめる** (指摘 1、SSRF の防止): スキームが
  `https`、ホストが Hub のホスト (`huggingface.co`) と完全に一致、道筋が、いまの一覧の API の
  道筋 (`/api/models/<repo>/tree/<revision>`) と一致するときだけ辿る。合わなければ、要求を
  出さずに断る。同じ道筋の繰り返しと、ページ数の上限 (`_MAX_PAGES`) で、終わらないページ送りも
  止める
- **一覧のすべての `path` を、中身を取りに行くどの要求よりも前に確かめる** (指摘 2、道筋の
  脱出の防止): 空でない、先頭が `/` でない、バックスラッシュを含まない、`/` で割った要素に
  空・`.`・`..` がない、制御文字を含まない、長さの上限に収まる。同じ `path` が 2 度あることも
  断る。1 つでも
  合わなければ、どの行かを示して、中身の取得の要求を 1 つも出さずに断る
- **LFS のファイルは取得しない**。大きさと sha256 は、一覧の応答の値をそのまま使う
  (`size == lfs.size` を確かめる)。LFS でないファイルだけを取得し、sha256 を自分で計算する。
  LFS でないのに大きい (既定 64 MiB 超) ものは、取得せずに断る (LFS の扱いの誤りを疑う)
- **中身の取得の転送 (redirect) は自分でたどる** (指摘 3、なりすましの防止):
  `follow_redirects=False` で受け、行き先を自分で確かめてから、次の要求を出す。上限
  (`_MAX_REDIRECTS`) を超えたら断る。行き先は、スキームが `https` で、ホストが
  `huggingface.co` そのものか、`.huggingface.co` / `.hf.co` で終わる部分ドメインのものだけを
  許す (プロトコルの格下げも、よそのホストへの転送も断る)。各段は新しい要求として組み立てる
  ので、前の段のヘッダを持ち越さない (この module は、そもそも `Authorization` などの秘密の
  ヘッダを 1 つも持たない)
- **同じ入力から、同じファイル (バイト列) ができる**: JSON の鍵の順 (アルファベット順)、
  2 字の字下げ、末尾の改行を固定する (`logs.py` の `collect.json` と同じ流儀)。作った時刻
  (`generated_at`) は、呼び出し側が明示して渡す (内部で `datetime.now()` を呼ばない)。これで、
  同じ `generated_at` を渡せば、何度組み立てても同じバイト列になる。**design.md のマニフェスト
  の JSON の形と `types.WeightsManifest` は `generated_at` を必須の項目として持つので、消さずに
  持ち回す**。実際にコミットする版を作るとき (6.1) は、その時点の UTC の時刻を渡す
- **マニフェストを読む関数は、ここに置く**。`guards.py` にまだ無いためである
  (tasks.md 3.2 の Implementation Notes を参照)

## 誤りの扱いと終了コード (design.md 「Error Handling」)

- `WeightsRefError`: 進める前に分かる不備。終了コード 1 (前提の不足、断った) に写す。版の形が
  誤り (要求を 1 つも出す前に上げる)、リポジトリまたは版が見つからない (404)、認証が要る
  (401 / 403。公開でないリポジトリはこの道具の対象の外なので、トークンを求めずに断る)、LFS で
  ないのに大きすぎるファイル、が、ここに乗る
- `WeightsFetchError`: 要求を出したあとに見つかった、実行しての失敗。終了コード 2 (実行して
  失敗した) に写す。一覧または中身の応答の形が想定と違う、取得した大きさが一覧の大きさと合わ
  ない、ページ送りや転送の行き先が信頼できない、ページ送りや転送が終わらない、一覧の `path`
  が安全でない、ネットワークの失敗 (時間切れ、接続できない) が、ここに乗る
- `WeightsError`: 上の 2 つの基底。マニフェストのファイルの読み書きの誤りにも使う

依存の向きにより、この module が読み込む `serving_kit` は `types` と `guards` (`weights_slug`
だけ) である。`image`、`logs` (同じ層)、`lifecycle` 以降は読み込まない。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from serving_kit.guards import weights_slug
from serving_kit.types import MANIFEST_EXCLUDED_PATHS, ManifestFile, WeightsManifest

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "HUB_BASE_URL",
    "SMALL_FILE_MAX_BYTES",
    "ManifestResult",
    "WeightsError",
    "WeightsFetchError",
    "WeightsRefError",
    "build_manifest",
    "default_weights_dir",
    "load_manifest",
    "manifest_path",
    "new_client",
    "to_json_bytes",
    "write_manifest",
]


# --- 決まった値 -------------------------------------------------------------

HUB_BASE_URL: Final[str] = "https://huggingface.co"
"""Hugging Face Hub の、匿名で読む公開の API と、ファイルの中身の入口。"""

_HUB_HOST: Final[str] = "huggingface.co"
"""一覧のページ送りの行き先を確かめる、Hub のホスト (完全一致だけを許す。指摘 1)。"""

_ALLOWED_CONTENT_HOST_SUFFIXES: Final[tuple[str, ...]] = (".huggingface.co", ".hf.co")
"""中身の取得の転送先として許す、部分ドメインの終わり方 (`_HUB_HOST` そのものは別に許す)。

指摘 3。
"""

SMALL_FILE_MAX_BYTES: Final[int] = 64 * 1024 * 1024
"""LFS でないファイルの、取得してよい大きさの上限 (64 MiB)。これを超えるものは断る。"""

DEFAULT_TIMEOUT_S: Final[float] = 60.0
"""一覧と中身の、1 回の要求の時間切れ。"""

_CHUNK_BYTES: Final[int] = 64 * 1024
"""中身を読むときの、1 回に読む大きさ。"""

_MAX_PAGES: Final[int] = 1000
"""一覧のページ送りの上限 (循環を止める安全弁。実物でこれだけの回数になる見込みはない)。"""

_MAX_REDIRECTS: Final[int] = 5
"""中身の取得の転送 (redirect) を、自分でたどる上限の回数 (指摘 3)。"""

_MAX_PATH_LENGTH: Final[int] = 1024
"""一覧の `path` の、許す長さの上限 (指摘 2)。"""

_REVISION_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
"""`types.WeightsRef.revision` と同じ形 (40 桁の 16 進の commit sha)。"""

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_EXCLUDED_EXACT_NAMES: Final[frozenset[str]] = frozenset(
    name.lower() for name in MANIFEST_EXCLUDED_PATHS
)
"""道筋の basename が、これと完全に一致すれば除く (大小無視)。"""

_EXCLUDED_NAME_PREFIXES: Final[tuple[str, ...]] = (
    "readme",
    "model_card",
    "modelcard",
    "model-card",
)
"""道筋の basename が、これで始まれば除く (大小無視。指摘 4)。"""

_LINK_ENTRY_RE: Final[re.Pattern[str]] = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')
"""`Link` ヘッダ (RFC 8288 風、GitHub の一覧の API と同じ書式) の 1 つの行。"""


# --- 誤り ---------------------------------------------------------------


class WeightsError(Exception):
    """重みのマニフェストまわりの誤りの基底 (module の docstring を見よ)。"""


class WeightsRefError(WeightsError):
    """進める前に分かる不備 (design.md「Error Handling」の終了コード 1)。"""


class WeightsFetchError(WeightsError):
    """要求を出したあとに見つかった、実行しての失敗 (design.md「Error Handling」の終了コード 2)。"""


# --- HTTP クライアント ----------------------------------------------------


def new_client(timeout_s: float = DEFAULT_TIMEOUT_S) -> httpx.Client:
    """Hub の匿名の読み取りに使う HTTP クライアント。

    認証の情報を一切使わない: `Authorization` ヘッダを付けず、環境変数や
    `~/.cache/huggingface/token`、`huggingface_hub` のどれも読まない。`trust_env=False` で、
    環境の `.netrc` やプロキシの認証も拾わない。既定では転送 (redirect) を辿らない (一覧も
    中身も、この module が自分で行き先を確かめてから辿る。指摘 1、指摘 3)。
    """
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
        follow_redirects=False,
        trust_env=False,
    )


# --- 一覧の行の読み取り ---------------------------------------------------


@dataclass(frozen=True)
class _RawEntry:
    """一覧の 1 行から読み取った、この module が要る項目だけ。"""

    path: str
    is_file: bool
    size: int
    lfs_sha256: str | None


def _parse_json(response: httpx.Response, *, source: str) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise WeightsFetchError(f"{source} の応答を JSON として読めない: {exc}") from exc


def _parse_entries(payload: object, *, source: str) -> list[_RawEntry]:
    if not isinstance(payload, list):
        raise WeightsFetchError(
            f"{source} の応答が一覧 (JSON の配列) でない: {type(payload).__name__}"
        )
    return [_parse_entry(raw, source=source, index=index) for index, raw in enumerate(payload)]


def _parse_entry(raw: object, *, source: str, index: int) -> _RawEntry:
    if not isinstance(raw, dict):
        raise WeightsFetchError(f"{source} の {index} 番目の行がオブジェクトでない")
    raw_path = raw.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise WeightsFetchError(f"{source} の {index} 番目の行に 'path' がない、または文字列でない")
    raw_type = raw.get("type")
    if raw_type not in ("file", "directory"):
        raise WeightsFetchError(f"{source} の '{raw_path}' の 'type' が想定外の値: {raw_type!r}")
    if raw_type == "directory":
        return _RawEntry(path=raw_path, is_file=False, size=0, lfs_sha256=None)
    raw_size = raw.get("size")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
        raise WeightsFetchError(
            f"{source} の '{raw_path}' の 'size' が非負の整数でない: {raw_size!r}"
        )
    lfs_sha256: str | None = None
    raw_lfs = raw.get("lfs")
    if raw_lfs is not None:
        if not isinstance(raw_lfs, dict):
            raise WeightsFetchError(f"{source} の '{raw_path}' の 'lfs' がオブジェクトでない")
        raw_oid = raw_lfs.get("oid")
        if not isinstance(raw_oid, str) or not _SHA256_RE.fullmatch(raw_oid):
            raise WeightsFetchError(
                f"{source} の '{raw_path}' の lfs.oid が sha256 の形 (16 進 64 文字) でない:"
                f" {raw_oid!r} (2026-09-22 に確かめた実物の形: lfs.oid が sha256 である)"
            )
        raw_lfs_size = raw_lfs.get("size")
        if not isinstance(raw_lfs_size, int) or isinstance(raw_lfs_size, bool):
            raise WeightsFetchError(
                f"{source} の '{raw_path}' の lfs.size が整数でない: {raw_lfs_size!r}"
            )
        if raw_lfs_size != raw_size:
            raise WeightsFetchError(
                f"{source} の '{raw_path}' の size ({raw_size}) と lfs.size ({raw_lfs_size})"
                " が合わない (2026-09-22 に確かめた実物では、size == lfs.size のはず)"
            )
        lfs_sha256 = raw_oid
    return _RawEntry(path=raw_path, is_file=True, size=raw_size, lfs_sha256=lfs_sha256)


def _exclusion_reason(path: str) -> str | None:
    """モデルカードらしい名前と `.gitattributes` を除く理由 (指摘 4)。

    requirements 8.8 / 11.3: モデルカードは、DGX Spark の起動のレシピそのものでありうるので
    開かない。写しらしい名前 (`README.txt`、`readme_ja.md`、`model_card.json` など) も、同じ
    理由で開かない側に倒す。大小文字を区別せず、道筋のどの深さにあっても (basename で見る)
    除く。
    """
    name = PurePosixPath(path).name.lower()
    if name in _EXCLUDED_EXACT_NAMES:
        return f"'{name}' という名前"
    if name.startswith(_EXCLUDED_NAME_PREFIXES):
        return f"'{name}' が、モデルカードらしい名前で始まる"
    return None


# --- 一覧の path の安全 (指摘 2) --------------------------------------------


def _unsafe_path_reason(path: str) -> str | None:
    """`path` が、中身を取りに行く道筋として安全かを確かめる (安全なら `None`)。"""
    if not path:
        return "空である"
    if len(path) > _MAX_PATH_LENGTH:
        return f"長さの上限 ({_MAX_PATH_LENGTH} 文字) を超えている"
    if path.startswith("/"):
        return "先頭が '/' である"
    if "\\" in path:
        return "バックスラッシュを含む"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        return "制御文字を含む"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "道筋の要素に、空、'.'、'..' のいずれかがある"
    return None


def _check_entry_paths(entries: Sequence[_RawEntry], *, repo: str, revision: str) -> None:
    """一覧のすべての行の `path` を、中身を取りに行くどの要求よりも前に確かめる (指摘 2)。

    1 つでも合わなければ、どの行かを示して、中身の取得の要求を 1 つも出さずに断る。
    """
    seen: set[str] = set()
    for entry in entries:
        reason = _unsafe_path_reason(entry.path)
        if reason is not None:
            raise WeightsFetchError(
                f"{repo}@{revision} の一覧の path '{entry.path}' が使えない ({reason})"
            )
        if entry.path in seen:
            raise WeightsFetchError(
                f"{repo}@{revision} の一覧に、同じ path が 2 度ある: '{entry.path}'"
            )
        seen.add(entry.path)


# --- 一覧の取得 (ページ送り) ----------------------------------------------


def _tree_url(repo: str, revision: str) -> str:
    return f"{HUB_BASE_URL}/api/models/{repo}/tree/{revision}?recursive=1"


def _tree_path(repo: str, revision: str) -> str:
    return f"/api/models/{repo}/tree/{revision}"


def _next_page_url(
    current: str, response: httpx.Response, *, repo: str, revision: str
) -> str | None:
    """`Link` ヘッダの次のページを、行き先を確かめたうえで返す (指摘 1、SSRF の防止)。

    スキームが `https`、ホストが Hub のホストと完全一致、道筋がいまの一覧の API の道筋と
    完全一致するときだけ辿る。合わなければ、要求を出さずに断る。
    """
    link_header = response.headers.get("link")
    if not link_header:
        return None
    candidate: str | None = None
    for raw_url, rel in _LINK_ENTRY_RE.findall(link_header):
        if rel == "next":
            candidate = raw_url
            break
    if candidate is None:
        return None
    resolved = httpx.URL(current).join(candidate)
    expected_path = _tree_path(repo, revision)
    if resolved.scheme != "https" or resolved.host != _HUB_HOST or resolved.path != expected_path:
        raise WeightsFetchError(
            f"{repo}@{revision} の一覧のページ送り (Link ヘッダ) の行き先が想定と違う:"
            f" {resolved} (期待: https://{_HUB_HOST}{expected_path} の道筋)"
        )
    return str(resolved)


def _raise_for_listing_status(response: httpx.Response, *, repo: str, revision: str) -> None:
    if response.status_code == httpx.codes.OK:
        return
    if response.status_code == httpx.codes.NOT_FOUND:
        raise WeightsRefError(f"リポジトリまたは版が見つからない ({repo}@{revision}、HTTP 404)")
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        raise WeightsRefError(
            f"{repo} は認証が要る (HTTP {response.status_code})。この道具の対象は公開の"
            "リポジトリだけなので、トークンを渡さずに断る"
        )
    raise WeightsFetchError(f"{repo}@{revision} の一覧を取得できない (HTTP {response.status_code})")


def _get_listing(client: httpx.Client, url: str, *, timeout_s: float) -> httpx.Response:
    # `httpx.InvalidURL` は `httpx.HTTPError` の下にない。`follow_redirects=False` でも、httpx は
    # 応答の `Location` から次の要求を組み立てようとするので、`data:` のような `//` を持たない
    # 形の `Location` で投げられる。traceback にせず、この module の誤りとして断る
    try:
        return client.get(url, timeout=timeout_s, follow_redirects=False)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise WeightsFetchError(
            f"Hugging Face Hub への一覧の要求が失敗した ({type(exc).__name__}: {exc}): {url}"
        ) from exc


def _fetch_tree(
    client: httpx.Client, repo: str, revision: str, *, timeout_s: float
) -> list[_RawEntry]:
    """`GET /api/models/{repo}/tree/{revision}?recursive=1` を、ページ送りを辿って読む。"""
    entries: list[_RawEntry] = []
    seen_urls: set[str] = set()
    url: str | None = _tree_url(repo, revision)
    while url is not None:
        if len(seen_urls) >= _MAX_PAGES:
            raise WeightsFetchError(
                f"{repo}@{revision} の一覧のページ送りが {_MAX_PAGES} 回を超えた"
            )
        if url in seen_urls:
            raise WeightsFetchError(
                f"{repo}@{revision} の一覧のページ送りが同じ道筋を繰り返した: {url}"
            )
        seen_urls.add(url)
        response = _get_listing(client, url, timeout_s=timeout_s)
        _raise_for_listing_status(response, repo=repo, revision=revision)
        payload = _parse_json(response, source=f"{repo}@{revision} の一覧")
        entries.extend(_parse_entries(payload, source=f"{repo}@{revision} の一覧"))
        url = _next_page_url(url, response, repo=repo, revision=revision)
    return entries


# --- 中身の取得 (LFS でない小さなファイルだけ) -----------------------------


def _resolve_url(repo: str, revision: str, path: str) -> str:
    """LFS でない小さなファイルの中身を取る道筋 (research.md の `.../resolve/<rev>/<file>`)。"""
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{HUB_BASE_URL}/{repo}/resolve/{revision}/{encoded}"


def _is_allowed_content_host(host: str) -> bool:
    return host == _HUB_HOST or host.endswith(_ALLOWED_CONTENT_HOST_SUFFIXES)


def _require_safe_content_url(url: httpx.URL, *, path: str) -> None:
    """中身の取得 (最初の要求、または転送の行き先) が、信頼できる Hub のホストかを確かめる。

    指摘 3: スキームの格下げ (https → http) も、よそのホストへの転送も、ここで断つ。
    """
    if url.scheme != "https":
        raise WeightsFetchError(f"'{path}' の取得の行き先が https でない ({url.scheme}): {url}")
    if not _is_allowed_content_host(url.host):
        raise WeightsFetchError(
            f"'{path}' の取得の行き先のホストが、Hub のホストでない (許すのは"
            f" {_HUB_HOST} と、その部分ドメインだけ): {url}"
        )


def _raise_for_content_status(
    response: httpx.Response, *, repo: str, revision: str, path: str
) -> None:
    if response.status_code == httpx.codes.OK:
        return
    if response.status_code == httpx.codes.NOT_FOUND:
        raise WeightsRefError(f"'{path}' が見つからない ({repo}@{revision}、HTTP 404)")
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        raise WeightsRefError(
            f"'{path}' の取得に認証が要る (HTTP {response.status_code})。公開のリポジトリだけが"
            "この道具の対象なので、トークンを求めずに断る"
        )
    raise WeightsFetchError(
        f"'{path}' を取得できない (HTTP {response.status_code}): {response.url}"
    )


def _fetch_small_file(
    client: httpx.Client,
    repo: str,
    revision: str,
    path: str,
    declared_size: int,
    *,
    timeout_s: float,
) -> bytes:
    """LFS でない小さなファイルの中身を取る。

    転送 (redirect) は自分でたどる (`follow_redirects=False`。上限 `_MAX_REDIRECTS` 回)。
    行き先は、`_require_safe_content_url` で毎回確かめてから要求を出す (指摘 3)。各段は新しい
    要求として組み立てるので、前の段のヘッダを持ち越さない。一覧の `size` を 1 バイトでも
    超えたら、読み切らずに打ち切って断る。
    """
    url = httpx.URL(_resolve_url(repo, revision, path))
    max_bytes = declared_size + 1
    for _ in range(_MAX_REDIRECTS + 1):
        _require_safe_content_url(url, path=path)
        try:
            with client.stream("GET", url, timeout=timeout_s, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise WeightsFetchError(
                            f"'{path}' の取得が転送 (HTTP {response.status_code}) を返したが、"
                            "行き先 (location) がない"
                        )
                    url = url.join(location)
                    continue
                _raise_for_content_status(response, repo=repo, revision=revision, path=path)
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    total += len(chunk)
                    if total > max_bytes:
                        raise WeightsFetchError(
                            f"'{path}' の取得した大きさが、一覧の大きさ"
                            f" ({declared_size} バイト) を超えた (読みながら打ち切った)"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise WeightsFetchError(
                f"'{path}' を取得できない ({type(exc).__name__}: {exc}): {url}"
            ) from exc
    raise WeightsFetchError(
        f"'{path}' の取得の転送 (redirect) が {_MAX_REDIRECTS} 回を超えた: {url}"
    )


# --- マニフェストの組み立て ------------------------------------------------


def _check_repo(repo: str) -> None:
    if not repo or repo != repo.strip():
        raise WeightsRefError(f"repo が空、または前後に空白がある: '{repo}'")


def _check_revision(revision: str) -> None:
    if not _REVISION_RE.fullmatch(revision):
        raise WeightsRefError(
            f"revision は 40 桁の 16 進の commit sha にする (受け取った値: '{revision}')。"
            "短縮形は使えない (research.md b-4: 'When using the commit hash, it must be the"
            " full-length hash instead of a 7-character commit hash.')"
        )


def _manifest_file(path: str, size: int, sha256: str) -> ManifestFile:
    try:
        return ManifestFile(path=path, size=size, sha256=sha256)
    except ValidationError as exc:
        raise WeightsFetchError(f"'{path}' をマニフェストの行にできない: {exc}") from exc


def _build_files(
    client: httpx.Client,
    repo: str,
    revision: str,
    raw_entries: Sequence[_RawEntry],
    *,
    timeout_s: float,
    small_file_max_bytes: int,
) -> tuple[list[ManifestFile], list[str]]:
    """マニフェストのファイルと、除いた道筋の一覧を作る。

    `raw_entries` の `path` は、呼ぶ側 (`build_manifest`) が `_check_entry_paths` で、
    ここに来るより前にすべて確かめている前提である (指摘 2)。
    """
    files: list[ManifestFile] = []
    excluded: list[str] = []
    for entry in raw_entries:
        if not entry.is_file:
            continue
        if _exclusion_reason(entry.path) is not None:
            excluded.append(entry.path)
            continue
        if entry.lfs_sha256 is not None:
            # LFS のファイルは取得しない。一覧が返した大きさと sha256 をそのまま使う
            files.append(_manifest_file(entry.path, entry.size, entry.lfs_sha256))
            continue
        if entry.size > small_file_max_bytes:
            raise WeightsRefError(
                f"'{entry.path}' は LFS でないのに大きすぎる ({entry.size} バイト。上限は"
                f" {small_file_max_bytes} バイト)。LFS の扱いに誤りがないか確かめること"
            )
        content = _fetch_small_file(
            client, repo, revision, entry.path, entry.size, timeout_s=timeout_s
        )
        if len(content) != entry.size:
            raise WeightsFetchError(
                f"'{entry.path}' の取得した大きさ ({len(content)} バイト) が、一覧の大きさ"
                f" ({entry.size} バイト) と合わない"
            )
        files.append(_manifest_file(entry.path, entry.size, hashlib.sha256(content).hexdigest()))
    if not files:
        raise WeightsFetchError(
            f"{repo}@{revision} に、マニフェストに載せられるファイルが 1 つもない"
            " (モデルカードらしい名前と .gitattributes を除いたあと)"
        )
    return files, sorted(excluded)


@dataclass(frozen=True)
class ManifestResult:
    """`build_manifest` の結果 (指摘 4)。

    `excluded_paths` は、モデルカードらしい名前や `.gitattributes` として除いた行の `path`
    (`path` の順。名前だけで、中身は取っていない)。計測者が、何を除いたかを見られるように
    する。除いたファイルは、`manifest.total_bytes` に入らない。
    """

    manifest: WeightsManifest
    excluded_paths: tuple[str, ...]


def build_manifest(
    repo: str,
    revision: str,
    *,
    generated_at: datetime,
    client: httpx.Client | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    small_file_max_bytes: int = SMALL_FILE_MAX_BYTES,
) -> ManifestResult:
    """固定した版のマニフェストを、Hub の公開の `tree` API から組み立てる (`serve manifest`)。

    進む順: 版と repo の形の検査 (要求を 1 つも出さない) → 一覧の取得 (ページ送りは、行き先を
    確かめながら辿る) → 一覧のすべての `path` の安全を確かめる (中身の取得より前) → モデル
    カードらしい名前と `.gitattributes` を除く → LFS のファイルは一覧の値をそのまま使う
    (`size == lfs.size` を確かめる)、LFS でない小さなファイルは取得して sha256 を計算する →
    `types.WeightsManifest` を組み立てる (`path` の順)。

    引数:
        repo: Hugging Face Hub のリポジトリの名前 (例: `RedHatAI/GLM-5.3-Flash-NVFP4`)。
        revision: 40 桁の 16 進の commit sha (短縮形は不可)。
        generated_at: マニフェストに書く作成の時刻。**呼ぶ側が明示して渡す** (この関数の中で
            `datetime.now()` を呼ばない)。同じ値を渡せば、同じバイト列のマニフェストになる。
            UTC の tz-aware な値にすること (types.py の決まり)。
        client: 使う HTTP クライアント。`None` なら `new_client(timeout_s)` を使い、
            この関数の中で閉じる (試験は `httpx.MockTransport` を積んだクライアントを渡す)。
        timeout_s: `client` を自分で作るときの時間切れと、1 回の要求の時間切れ。
        small_file_max_bytes: LFS でないファイルの、取得してよい大きさの上限。

    返り値:
        `path` の順に並んだ `WeightsManifest` と、除いた道筋の一覧 (`ManifestResult`)。

    例外:
        WeightsRefError: 版の形が誤り (要求を 1 つも出さない)、リポジトリまたは版が見つから
            ない、認証が要る、LFS でないのに大きすぎるファイルがある (終了コード 1)。
        WeightsFetchError: 一覧または中身の応答の形が想定と違う、取得した大きさが一覧の大きさ
            と合わない、一覧の `path` が安全でない、ページ送りや転送の行き先が信頼できない、
            ページ送りや転送が終わらない、ネットワークの失敗 (終了コード 2)。
        ValueError: `generated_at` が UTC の tz-aware な `datetime` でないとき。
    """
    if generated_at.tzinfo is None:
        raise ValueError("generated_at は UTC の tz-aware な datetime にする (types.py の決まり)")
    _check_repo(repo)
    _check_revision(revision)

    http = client if client is not None else new_client(timeout_s)
    try:
        raw_entries = _fetch_tree(http, repo, revision, timeout_s=timeout_s)
        # 中身を取りに行くどの要求よりも前に、一覧のすべての path の安全を確かめる (指摘 2)
        _check_entry_paths(raw_entries, repo=repo, revision=revision)
        files, excluded_paths = _build_files(
            http,
            repo,
            revision,
            raw_entries,
            timeout_s=timeout_s,
            small_file_max_bytes=small_file_max_bytes,
        )
    finally:
        if client is None:
            http.close()

    sorted_files = tuple(sorted(files, key=lambda entry: entry.path))
    total_bytes = sum(entry.size for entry in sorted_files)
    try:
        manifest = WeightsManifest(
            repo=repo,
            revision=revision,
            generated_at=generated_at,
            total_bytes=total_bytes,
            files=sorted_files,
        )
    except ValidationError as exc:
        raise WeightsFetchError(
            f"{repo}@{revision} のマニフェストを組み立てられない: {exc}"
        ) from exc
    return ManifestResult(manifest=manifest, excluded_paths=tuple(excluded_paths))


# --- ファイルへの書き出しと読み込み ----------------------------------------


def default_weights_dir() -> Path:
    """マニフェストを置く既定の場所 (`serving/weights/`)。この module から辿る。"""
    return Path(__file__).resolve().parents[2] / "weights"


def manifest_path(repo: str, *, weights_dir: Path | None = None) -> Path:
    """`serving/weights/<slug>.manifest.json` の道筋 (`guards.weights_slug` と同じ作り方)。"""
    directory = weights_dir if weights_dir is not None else default_weights_dir()
    return directory / f"{weights_slug(repo)}.manifest.json"


def to_json_bytes(manifest: WeightsManifest) -> bytes:
    """`WeightsManifest` を、同じ入力なら同じバイト列になる形で書き出す。

    鍵の順 (アルファベット順)、2 字の字下げ、末尾の改行を固定する (`logs.py` の
    `collect.json` と同じ流儀。tasks.md 3.2 の完了の状態: 同じ入力から同じファイルができる)。
    """
    payload = manifest.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def write_manifest(manifest: WeightsManifest, path: Path) -> None:
    """マニフェストをファイルに書く (道筋は呼ぶ側が決める。既定は `manifest_path` を使うこと)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_json_bytes(manifest))


def load_manifest(path: Path) -> WeightsManifest:
    """マニフェストを読む (`guards` にまだ読む関数がないので、ここに置く)。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WeightsError(f"マニフェスト '{path}' を読めない: {exc}") from exc
    try:
        return WeightsManifest.model_validate_json(text)
    except ValidationError as exc:
        raise WeightsError(
            f"マニフェスト '{path}' の中身が WeightsManifest の形でない: {exc}"
        ) from exc
