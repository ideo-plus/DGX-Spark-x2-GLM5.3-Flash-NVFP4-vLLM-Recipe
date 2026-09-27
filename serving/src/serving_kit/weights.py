"""重みのマニフェストの生成と、2 台での取得と照合
(design.md 「イメージと重み › weights」、tasks.md 3.2 と 3.3)。

この module は、2 つの節に分かれる。

1. **マニフェストの生成** (3.2): Mac で、Hugging Face Hub の**公開の API を匿名で**読み、
   固定した版 (40 桁の commit) のファイルの一覧、大きさ、sha256 から
   `types.WeightsManifest` を組み立てる (`serve manifest`)
2. **取得と照合** (3.3): 取得の種類の構成から組み立てたコンテナを 2 台に起こして重みを
   取得し (`serve fetch`)、2 台で `sha256sum` を流して、**Mac の側で**マニフェストと
   突き合わせ、結果を記録として残す (`serve verify`)。module の後半 (「取得と照合」の節)
   にある

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

3.3 が足した `WeightsMismatchError` は `WeightsFetchError` の一種 (終了コード 2) で、合わな
かったファイルの名前を持つ。5.1 の写し方は、次のとおりである。

| 出る誤り | 終了コード | いつ |
|---|---|---|
| `config.ConfigError` | 1 | 種類の違う構成、重みを持たない構成、重みの置き場所を結び付けない構成 |
| `WeightsRefError` | 1 | 版の形、マニフェストと構成の食い違い、マニフェストの道筋が安全でない |
| `guards.ApprovalError` | 1 | 計測者が了承しなかった |
| `FetchOutcome.status == "refused"` | 1 | 関門が断った (例外ではない。`gates` を並べて示す) |
| `ValueError` | 1 | 役割のノードの定義がない、`verified_at` に時刻の帯がない (前提の不足) |
| `WeightsFetchError` | 2 | 取得の失敗、時間切れ、届かなかった読み取り、片付けの失敗 |
| `WeightsMismatchError` | 2 | 照合の不一致 (合わない、ない、読めない) |
| `KeyboardInterrupt` | 130 | 中断 (取得のコンテナは、**止めない**) |

依存の向きにより、この module が読み込む `serving_kit` は `types`、`config`、`remote`、
`plan`、`guards` である。`image`、`logs` (同じ層)、`lifecycle` 以降は読み込まない。
**記録の回収 (`logs.collect_logs`) は、同じ層なので、ここからは呼べない**。取得の失敗を
見せるための記録の末尾だけを `docker logs --tail` で読み、回収そのものは、呼ぶ側 (5.1) が
`logs.collect_logs` で行う。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TextIO
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from serving_kit.config import ConfigError
from serving_kit.guards import (
    GATE_OWN_STATE,
    READ_TIMEOUT_S,
    STOP_TIMEOUT_S,
    Confirmer,
    OwnContainer,
    build_approved_plan,
    gate_disk_space,
    gate_gpu_idle,
    gate_image_digest,
    gate_layout,
    gate_ports_free,
    gate_reachable,
    list_own_containers,
    match_running,
    request_approval,
    rollback_commands,
    weights_record_path,
    weights_ref_mismatch,
    weights_slug,
)
from serving_kit.plan import LABEL_KIND, LABEL_WEIGHTS, build_plans
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    MANIFEST_EXCLUDED_PATHS,
    AnyVerificationRecord,
    AnyWeightsManifest,
    AnyWeightsRef,
    ApprovedPlan,
    CommandResult,
    ConfigDef,
    ContainerPlan,
    DerivedVerificationRecord,
    DerivedWeightsManifest,
    DerivedWeightsRef,
    GateResult,
    ManifestFile,
    NodeDef,
    NodeRole,
    PlannedPush,
    PlannedRun,
    VerificationRecord,
    VerificationScope,
    WeightsManifest,
)

__all__ = [
    "CLEANUP_TIMEOUT_S",
    "DEFAULT_TIMEOUT_S",
    "FETCH_POLL_INTERVAL_S",
    "FETCH_START_TIMEOUT_S",
    "FETCH_TAIL_LINES",
    "HUB_BASE_URL",
    "RECORD_SUBDIR",
    "SHA256_BASE_TIMEOUT_S",
    "SHA256_BATCH_FILES",
    "SHA256_TIMEOUT_PER_GIB_S",
    "SMALL_FILE_MAX_BYTES",
    "FetchOutcome",
    "ManifestResult",
    "NodeFetch",
    "NodeVerification",
    "VerifyOutcome",
    "WeightsError",
    "WeightsFetchError",
    "WeightsMismatchError",
    "WeightsRefError",
    "build_manifest",
    "default_weights_dir",
    "exclusion_reason",
    "fetch_weights",
    "load_manifest",
    "manifest_path",
    "new_client",
    "parse_sha256_output",
    "scoped_files",
    "to_json_bytes",
    "verify_weights",
    "weights_dir_on_spark",
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


def exclusion_reason(path: str) -> str | None:
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
        if exclusion_reason(entry.path) is not None:
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


def to_json_bytes(manifest: AnyWeightsManifest) -> bytes:
    """マニフェストを、同じ入力なら同じバイト列になる形で書き出す。

    鍵の順 (アルファベット順)、2 字の字下げ、末尾の改行を固定する (`logs.py` の
    `collect.json` と同じ流儀。tasks.md 3.2 の完了の状態: 同じ入力から同じファイルができる)。
    バイト列の作り方の所有者は `types.ManifestFiles.canonical_bytes` で、照合の記録が結び付く
    SHA-256 (`content_sha256`) も、同じバイト列から作る。
    """
    return manifest.canonical_bytes()


def write_manifest(manifest: AnyWeightsManifest, path: Path) -> None:
    """マニフェストをファイルに書く (道筋は呼ぶ側が決める。既定は `manifest_path` を使うこと)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_json_bytes(manifest))


def load_manifest(path: Path) -> AnyWeightsManifest:
    """マニフェストを読む (`guards` にまだ読む関数がないので、ここに置く)。

    読み分けは、最上位の `kind` が `"derived"` かどうかだけで行う。Hub のマニフェストは
    `kind` を持たないので、いまと同じ `WeightsManifest` として読む (項目の有無から、
    派生だと推測しない)。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WeightsError(f"マニフェスト '{path}' を読めない: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WeightsError(f"マニフェスト '{path}' の JSON を読めない: {exc}") from exc
    derived = isinstance(raw, dict) and raw.get("kind") == "derived"
    model = DerivedWeightsManifest if derived else WeightsManifest
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        raise WeightsError(
            f"マニフェスト '{path}' の中身が {model.__name__} の形でない: {exc}"
        ) from exc


# =========================================================================
# 取得と照合 (tasks.md 3.3。design.md 「イメージと重み › weights」)
# =========================================================================
#
# ここから下が、`serve fetch <構成> [--probe-files]` と `serve verify <構成>` の中身である。
#
# 守る決まり:
#
# - **1 つの起動の仕組み** (design.md 「Architecture Integration」): 取得のコンテナも、ほかの
#   4 つの `kind` と同じ道を通る。`plan.build_plans` が組み立てた引数の列を、名前とラベルを
#   付けて `-d` で起こす。この module は、取得のコンテナの引数を、自分で足したり書き換えたり
#   しない (`--probe-files` も、専用の `fetch` の構成から起こす。**照合の範囲を選ぶだけ**)
# - **トークンを渡さない** (requirements 2.6): 環境変数は、構成の `env` に書いたものだけで
#   ある (`plan` が守る)。この module は、環境変数を足す経路を持たない
# - **ホストに何も入れない**: 遠隔で流すのは `docker`、`sha256sum`、`uname`、`df`、`ss`、
#   `nvidia-smi`、`test` の読み取りと、了承済みの計画にある `docker run` / `stop` / `rm` と
#   配布だけである (`hf` も `pip` も `apt` も、`remote` の許可の一覧にないので呼べない)
# - **コンテナを対象にする操作の不変条件** (requirements 2.3、2.4): 対象にできるのは、(a)
#   `guards.list_own_containers` が返した行の識別子か、(b) 了承済みの計画が自分で起こす名前
#   だけである。終わりの確認と記録の末尾の読み取りは (a)、片付けは (b) で、流す直前に
#   `guards.rollback_commands` が一覧で確かめる
# - **Spark の側で正解を作らない** (requirements 3.5): 照合は、Mac でコミットしたマニフェスト
#   との突き合わせである。Spark では `sha256sum` を流すだけで、`hf cache verify` は使わない
# - **黙って取り直さない**: 合わないファイル、ないファイル、読めない出力は、名前を並べて
#   `WeightsMismatchError` にする。そのあとに、取り直しの呼び出し (`docker run`、`docker
#   pull`、`docker rm`) を 1 つも出さない
#
# design が明示しない細部の、意図した決めごと:
#
# 1. **取得の待ちの途中では、取得のコンテナを止めない** (3.1 の片付けの道との、意図した違い)。
#    3.1 の `inspect` は数秒で終わる読み取りなので、`RemoteError` と中断のどちらでも片付けて
#    から終わる。取得は 184 GiB を何時間もかけて落とすので、同じようにすると、ssh が切れた
#    だけで、あるいは Ctrl-C を 1 度打っただけで、それを捨ててしまう。**切り離して起こして
#    あるので、Mac の側が終わっても取得は続く**。そこで、待ちの途中の `RemoteError`、時間
#    切れ、中断では、`docker stop` も `docker rm` も 1 つも出さずに終わり、「取得は続いて
#    いる。もう一度 `serve fetch` を打つと、待ちに戻る」と言う。止めるのは `serve stop` だけ
#    である。**終了した取得のコンテナ (成功でも失敗でも) は、片付ける**
# 2. **二重に取得せず、やり直しは台ごとに決める** (design.md 「weights」の `serve fetch` の
#    Idempotency:「もう一度流すと、動いている取得を見つけて待ち、**足りないものだけが対象に
#    なる**」)。起こす前に、自分のラベルで絞った一覧を台ごとに 1 度読み、次の 3 つに分ける。
#      (a) この構成と一致する (名前、イメージ、`config-sha256`)、動いている取得がある →
#          その台は**起こさずに、終わりを待つ**
#      (b) 自分のコンテナが 1 つもない → その台は**起こす対象**
#      (c) それ以外 (この構成と一致しない自分のコンテナが動いている / 終了した自分のコンテナが
#          残っている / 名前は同じで `config-sha256` やイメージが違う) → **断る**
#          (`guards.match_running` の `differences` を文に出し、終了したものは `serve logs` で
#          記録を回収してから `serve stop` で消すよう促す)
#    **1 台でも (c) なら、どの台でも起こさない**。関門を流すのは (b) の台だけで、了承を得る
#    計画に入る `docker run` も (b) の台のぶんだけである。これで、起こす途中で失敗したときの
#    案内 (「もう一度打つと、動いている台は待ち、起きていない台だけを起こす」) が、**本当に
#    なる** (片方だけ動いている状態から、もう一度打てる)
# 2b. **了承を得る計画は、`build_approved_plan` に全部の台の計画を渡して作り、(a) の台の
#    `docker run` だけを落とす** (落としたあと、`ApprovedPlan` を作り直して、検証を通す)。
#    `build_approved_plan` は、渡した計画のすべてに
#    `docker run` を作るので、こうしないと「(a) の台を起こす」と書いた計画を見せてしまう。
#    落とすのは前に進むコマンドだけで、**巻き戻し (終わったコンテナの片付けの権限) は残る**。
#    権限を足すのではなく減らすので、不変条件を迂回しない。むしろ、(a) の台の `docker run` が
#    計画に無くなるので、**合流する台を誤って起こそうとしても `remote` が断る** (構造で守る)
# 3. **`guards.run_gates` を使わない**。`run_gates` は `gate_weights_verified` を含むが、
#    取得の前には、まだ照合の記録がないので、必ず断られてしまう。そこで、`GATE_ORDER` から
#    その 1 つを除いた並びを、公開の `gate_*` で自分で流す (`guards` の口の使い方は同じ)。
#    `already_running` のときは、design.md のとおり関門より前に判定して、関門を飛ばす
# 4. **要るディスクの量は、照合の範囲から数える**。`guards.required_free_bytes` は構成の
#    `kind` で範囲を決めるので、`kind = "fetch"` の `--probe-files` (設定とトークナイザ
#    だけ) を区別できない。範囲は呼ぶ側が渡すものなので、ここで数えて `gate_disk_space` に
#    渡す
# 5. **重みの置き場所は、`--mount` の `target` から決める**。構成の `--mount` のうち、
#    `target` が `weights.mount_at` と一致するか、その親であるものの `source` に、残りの
#    道筋を継ぐ (`guards.mount_sources` は `source` しか返さないので、ここに置く。
#    `guards.py` は書き換えない)。**6.2 と 5.1 への申し送り**: `gate_layout` は `--mount` の
#    元を `test -d` で見るので、取得の構成が `models/<slug>/` を直に結び付けると、**最初の
#    取得の前には、その置き場所がまだ無くて断られる** (`serve push` が作るのは、`models/` を
#    含む 6 つの置き場所までで、その下の名前は作らない)。取得の構成では、`models/` そのものを
#    結び付けて `--local-dir` で下の名前を指す書き方も採れる (親でも読めるようにしてある)。
#    どちらにするかは、6.2 (構成の値) と 5.1 (`serve push` の計画) で決める
# 6. **記録の道筋は `guards.weights_record_path` が決める**。範囲ごとに別のファイル
#    (`….verified.json` と `….probe.verified.json`。派生は `….derived.verified.json`) に
#    なる。Mac で作った 1 ファイルを、
#    `push(delete=False)` で `state/` に置く (`remote` が、宛先を `payload/` と `state/` に
#    絞っている)。**配る元は、この module だけが使う `<record_dir>/verified/<役割>/` にし、
#    配る前に空にする**。`remote.push` は、渡したディレクトリの中身を丸ごと送るので、呼ぶ側が
#    渡した `record_dir` (5.1 は `logs.var_dir` を渡しうる) をそのまま元にすると、回収した
#    コンテナの記録が Spark の `state/` に逆流する。配る元には、いつでも今回の 1 ファイルだけが
#    ある (前の回の、別の範囲の記録も残さない)
# 7. **合わなかったことも、記録に残して配る**。`gate_weights_verified` は、`mismatched` の
#    ある記録を断るので、古い「合っていた」記録を残すより、見つけた食い違いで上書きする
#    ほうが安全である (残すと、壊れていると分かっているのに `serve start` が通ってしまう)
# 8. **読めない出力や、渡していない道筋の行が 1 行でもあれば、その回 (batch) のファイルを、
#    すべて「確かめられなかった」に倒す**。1 行だけを捨てると、どのファイルの行だったかが
#    決められず、確かめていないものを確かめたことにしかねない。ないファイルは、標準エラーに
#    出て標準出力には行が出ないので、この決めごとに掛からない (名前だけが「ない」に入る)
# 8b. **`serve verify` は、その重みの取得が動いている台があれば、照合を始めずに断る**。
#    取得の途中のファイルの sha256 を 184 GiB ぶん計算するのは、無駄で、紛らわしい (結果は
#    「合わない」になり、状態としては正しいが、壊れているようにも読める)。`serve fetch` が、
#    取得の終わりに照合まで行う
# 9. **結果の型は、この module に置く** (`types.py` は、ほかのタスクが使うので変更しない。
#    tasks.md の Implementation Notes 1.2: weights の結果の型は、凍結の対象の外)


# --- 決まった値 (取得と照合) ----------------------------------------------

FETCH_START_TIMEOUT_S: Final[float] = 120.0
"""`docker run -d` の時間切れ。切り離して起こすので、すぐ返る。"""

FETCH_POLL_INTERVAL_S: Final[float] = 10.0
"""取得の終わりを見に行く間隔。184 GiB の取得は時間の単位が分なので、長めに取る。"""

FETCH_TAIL_LINES: Final[int] = 80
"""失敗のときに見せる、取得の記録の末尾の行数 (`logs.DEFAULT_TAIL_LINES` と同じ値)。"""

CLEANUP_TIMEOUT_S: Final[float] = float(STOP_TIMEOUT_S) + 30.0
"""片付け (`docker stop -t 90` と `docker rm`) の時間切れ。停止の猶予より長くする。"""

SHA256_BATCH_FILES: Final[int] = 8
"""1 回の `sha256sum` に渡す道筋の数。

遠隔のシェルに渡る文字列の長さと、時間切れの見積もりやすさで決めた。1 ファイルずつだと
ssh の往復が数十回になり、全部を 1 回に渡すと、時間切れの見積もりが粗くなる。
"""

SHA256_BASE_TIMEOUT_S: Final[float] = 120.0
"""1 回の `sha256sum` の、中身の大きさによらない時間切れ。"""

SHA256_TIMEOUT_PER_GIB_S: Final[float] = 60.0
"""1 回の `sha256sum` の、1 GiB あたりに足す時間切れ (20 GB で数十秒〜数分を見込む)。"""

RECORD_SUBDIR: Final[str] = "state"
"""照合の結果の記録を配る先 (`remote` が許す 2 つの宛先のうちの 1 つ)。"""

_RECORD_DIRNAME: Final[str] = "verified"
"""記録を配る元の、`record_dir` の下の名前 (**この module だけが使う**。決めごとの 6)。"""

_KIND_FETCH: Final[str] = "fetch"
"""取得の構成の種類 (design.md Data Models の `p1-fetch-nvfp4`)。"""

_STATE_FORMAT: Final[str] = "{{.State.Status}} {{.State.ExitCode}}"
"""`docker container inspect` に渡す書式 (状態と終了コードを、1 行で読む)。"""

_UNFINISHED_STATES: Final[frozenset[str]] = frozenset(
    {"created", "running", "restarting", "paused", "removing"}
)
"""まだ終わっていないコンテナの状態。ここにない状態 (`exited`、`dead`) は、終わりとみなす。"""

_NOT_EXECUTABLE_EXIT_CODES: Final[frozenset[int]] = frozenset({126, 127})
"""コンテナの中のプログラムを実行できなかったときの終了コード (`hf` がない場合を疑う)。"""

_REMOVE_SUBCOMMAND: Final[str] = "rm"
"""片付けのうち、中断が来たあとでも試みるもの (コンテナを残さないことを優先する)。"""

_CLEANUP_LABELS: Final[Mapping[str, str]] = {"stop": "止められなかった", "rm": "消せなかった"}
"""片付けの失敗を言うときの、docker のサブコマンドごとの言い方。"""

_MOUNT_FLAG: Final[str] = "--mount"
_MOUNT_SOURCE_KEYS: Final[frozenset[str]] = frozenset({"source", "src"})
_MOUNT_TARGET_KEYS: Final[frozenset[str]] = frozenset({"target", "destination", "dst"})

_SCOPE_PROBE_FILES: Final[VerificationScope] = "probe_files"
_SHA256_DIGEST_LENGTH: Final[int] = 64
_SHA256_LINE_MIN_LENGTH: Final[int] = _SHA256_DIGEST_LENGTH + 3
"""`sha256sum` の 1 行の、最も短い形 (`<64 桁><空白><印><1 文字の道筋>`)。"""

_BINARY_MARKS: Final[tuple[str, ...]] = (" ", "*")
"""`sha256sum` の 2 文字目の印 (テキストは空白、バイナリは `*`)。"""

_GIB: Final[int] = 1024**3

_KEEP_GOING: Final[str] = (
    "取得は続いている (切り離して起こしてあるので、Mac の側が終わっても止まらない)。"
    "もう一度 `serve fetch <構成>` を打つと、動いている台は、その終わりを待ち、"
    "起きていない台だけを起こす。止めたいときは `serve stop <構成>` を打つ"
)
"""待ちや起動の途中で終わるときに、必ず添える文 (184 GiB の取得を、誤って捨てないため)。

案内が本当になるように、やり直しは**台ごとに**判定する (module の上の決めごとの 2)。
"""


class WeightsMismatchError(WeightsFetchError):
    """2 台での照合が合わなかった (design.md 「Error Handling」の終了コード 2)。

    合わないファイル、ないファイル、読めない出力の名前を並べる。**黙って取り直さない**ので、
    この誤りのあとに、取り直しの呼び出しは 1 つも出ない。
    """

    def __init__(self, message: str, verifications: Sequence[NodeVerification] = ()) -> None:
        self.verifications: tuple[NodeVerification, ...] = tuple(verifications)
        super().__init__(message)


# --- 結果の型 -------------------------------------------------------------


@dataclass(frozen=True)
class NodeFetch:
    """1 台ぶんの、取得のコンテナの結末。

    `attached` は、**新しく起こしたのではなく、動いていた取得に合流した**ことを表す
    (二重に取得しない)。
    """

    node: NodeRole
    container_name: str
    attached: bool
    state: str
    exit_code: int | None = None
    log_tail: str = ""

    @property
    def ok(self) -> bool:
        """コンテナが 0 で終わったかどうか。"""
        return self.exit_code == 0


@dataclass(frozen=True)
class NodeVerification:
    """1 台ぶんの照合の結果。

    `record` は、Spark の `state/` に置いた記録そのもの (`remote_path` がその道筋)。
    `mismatched` は sha256 が合わなかったファイル、`missing` は行が出てこなかったファイル
    (ない、または、読めない出力と同じ回に入っていた)、`unreadable` は読めなかった出力の行
    である。記録の `mismatched` には、`mismatched` と `missing` の両方が、名前の順で入る。
    """

    node: NodeRole
    record: AnyVerificationRecord
    remote_path: str
    mismatched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """この台で、すべてのファイルがマニフェストと合ったかどうか。"""
        return not (self.mismatched or self.missing or self.unreadable)

    @property
    def text(self) -> str:
        """計測者に見せる文字列 (合わなかったものを、種類ごとに並べる)。"""
        if self.ok:
            return (
                f"[{self.node}] {self.record.file_count} ファイル"
                f" ({self.record.total_bytes:,} バイト) が、マニフェストと合った"
            )
        parts: list[str] = []
        if self.mismatched:
            parts.append(f"合わないファイル: {', '.join(self.mismatched)}")
        if self.missing:
            parts.append(f"確かめられなかったファイル: {', '.join(self.missing)}")
        if self.unreadable:
            parts.append(f"読めなかった sha256sum の出力: {' / '.join(self.unreadable)}")
        return f"[{self.node}] " + " / ".join(parts)


@dataclass(frozen=True)
class FetchOutcome:
    """`serve fetch` の結末。

    `refused` は、関門が断ったこと (呼ぶ側は `gates` の理由を並べて、終了コード 1)。
    取得や照合が失敗したときは、この型では返らず `WeightsFetchError` になる。
    """

    status: Literal["fetched", "refused"]
    config_name: str
    scope: VerificationScope
    gates: tuple[GateResult, ...] = ()
    fetches: tuple[NodeFetch, ...] = ()
    verifications: tuple[NodeVerification, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class VerifyOutcome:
    """`serve verify` の結末 (読み取りと、記録の配布だけ)。"""

    status: Literal["verified", "refused"]
    config_name: str
    scope: VerificationScope
    gates: tuple[GateResult, ...] = ()
    verifications: tuple[NodeVerification, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _Controls:
    """待ちの、差し替えられる口と時間切れ (試験は、実際に眠らない)。"""

    sleep: Callable[[float], None]
    clock: Callable[[], float]
    report: TextIO
    poll_interval_s: float
    timeout_s: float
    start_timeout_s: float
    read_timeout_s: float


@dataclass(frozen=True)
class _Target:
    """待ちと読み取りの相手 (識別子は、必ず自分のラベルで絞った一覧から来る)。"""

    node: NodeDef
    plan: ContainerPlan
    container_id: str
    attached: bool


_NodePhase = Literal["starting", "waiting"]
"""台ごとのやり直しの判定 (module の上の決めごとの 2)。

`starting` は「自分のコンテナが 1 つもないので、起こす」、`waiting` は「この構成の取得が
動いているので、起こさずに待つ」。断る台は、この値を持たない (`GateResult` で返す)。
"""


# --- 置き場所と、範囲 -----------------------------------------------------


def _mount_spec(value: str) -> tuple[str, str] | None:
    """`type=bind,source=…,target=…` から、元と先の組を取り出す。"""
    source: str | None = None
    target: str | None = None
    for part in value.split(","):
        key, _, found = part.partition("=")
        name = key.strip()
        if name in _MOUNT_SOURCE_KEYS:
            source = found.strip()
        elif name in _MOUNT_TARGET_KEYS:
            target = found.strip()
    return None if source is None or target is None else (source, target)


def _mount_specs(argv: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """組み立てた引数の列から、`--mount` の元と先の組を、書かれた順に取り出す。"""
    specs: list[tuple[str, str]] = []
    pending = False
    for item in argv:
        if pending:
            pending = False
            spec = _mount_spec(item)
            if spec is not None:
                specs.append(spec)
            continue
        if item == _MOUNT_FLAG:
            pending = True
        elif item.startswith(f"{_MOUNT_FLAG}="):
            spec = _mount_spec(item[len(_MOUNT_FLAG) + 1 :])
            if spec is not None:
                specs.append(spec)
    return tuple(specs)


def weights_dir_on_spark(argv: Sequence[str], mount_at: str) -> str:
    """重みが置かれる、**Spark の側の**ディレクトリを、組み立てた引数の列から決める。

    構成の `--mount` のうち、`target` が `mount_at` と一致するか、その親であるものを選び、
    その `source` に、残りの道筋を継ぐ。親でもよいのは、取得の構成が `models/` そのものを
    結び付けて、`--local-dir` で下の名前を指す書き方も採れるためである (どちらの書き方でも
    読める)。いちばん深く一致するものを選ぶ。

    引数:
        argv: `plan.build_plans` が組み立てた引数の列 (置き換えの印は埋めたあと)。
        mount_at: `types.WeightsRef.mount_at` (コンテナの中の、重みの置き場所)。

    例外:
        config.ConfigError: `mount_at` を含む `--mount` が、構成にないとき (終了コード 1)。
    """
    wanted = PurePosixPath(mount_at)
    best: tuple[int, str] | None = None
    for source, target in _mount_specs(argv):
        place = PurePosixPath(target)
        if wanted != place and not wanted.is_relative_to(place):
            continue
        rest = wanted.relative_to(place)
        found = str(PurePosixPath(source).joinpath(*rest.parts))
        depth = len(place.parts)
        if best is None or depth > best[0]:
            best = (depth, found)
    if best is None:
        mounts = ", ".join(f"{source} -> {target}" for source, target in _mount_specs(argv))
        raise ConfigError(
            f"構成の --mount に、重みの置き場所 ({mount_at}) を含むものがない"
            f" (取得したファイルが、Spark のどこに置かれるかを決められない)。いまの --mount:"
            f" {mounts or '(1 つもない)'}"
        )
    return best[1]


def scoped_files(
    manifest: AnyWeightsManifest, scope: VerificationScope
) -> tuple[ManifestFile, ...]:
    """照合の範囲のファイル (`probe_files` は、safetensors を除いたもの)。"""
    return manifest.probe_files if scope == _SCOPE_PROBE_FILES else manifest.files


def _check_manifest_paths(files: Sequence[ManifestFile]) -> None:
    """マニフェストの道筋を、遠隔に渡す前に確かめる (`sha256sum` の出力を読むため)。

    `sha256sum` は、名前に `\\` や改行があると、行の頭に `\\` を付けて逃がすので、その形の
    道筋は、読み違えないために、流す前に断る。
    """
    bad = [entry.path for entry in files if _unsafe_path_reason(entry.path) is not None]
    if bad:
        raise WeightsRefError(
            "マニフェストに、照合に使えない道筋がある"
            f" (sha256sum の出力を読み違えないために断る): {', '.join(bad)}"
        )


# --- sha256sum の出力の読み取り -------------------------------------------


def parse_sha256_output(text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """`sha256sum` の標準出力を読む。

    1 行は `<64 桁の 16 進><空白><空白か '*'><道筋>` である (2 文字目の印は、テキストか
    バイナリか)。道筋に空白が入っていてもよい (2 文字の印の後ろは、すべて道筋)。名前に
    `\\` や改行があるときに GNU coreutils が付ける、行頭の `\\` の形は、16 進で始まらない
    ので「読めない行」になる。同じ道筋が 2 度出てきた行も、読めない行として返す。

    返り値:
        道筋ごとの sha256 と、読めなかった行 (出てきた順)。
    """
    digests: dict[str, str] = {}
    unreadable: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        digest = line[:_SHA256_DIGEST_LENGTH]
        if (
            len(line) < _SHA256_LINE_MIN_LENGTH
            or not _SHA256_RE.fullmatch(digest)
            or line[_SHA256_DIGEST_LENGTH] != " "
            or line[_SHA256_DIGEST_LENGTH + 1] not in _BINARY_MARKS
        ):
            unreadable.append(line)
            continue
        path = line[_SHA256_DIGEST_LENGTH + 2 :]
        if path in digests:
            unreadable.append(line)
            continue
        digests[path] = digest
    return digests, tuple(unreadable)


# --- 小さな助け -----------------------------------------------------------


def _node_of(nodes: Mapping[NodeRole, NodeDef], config: ConfigDef, role: NodeRole) -> NodeDef:
    """構成が使う役割のノードの定義を取る (なければ、触る前に断る)。"""
    node = nodes.get(role)
    if node is None:
        raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
    return node


def _shown_failure(result: CommandResult) -> str:
    """失敗した呼び出しの、見せる文 (標準エラーがなければ標準出力)。"""
    return result.stderr.strip() or result.stdout.strip() or "(出力なし)"


def _refused(gates: Sequence[GateResult]) -> tuple[GateResult, ...]:
    return tuple(gate for gate in gates if not gate.passed)


def _refusal_detail(refused: Sequence[GateResult]) -> str:
    """断った関門を、台と関門の名前つきで並べる。"""
    return "関門が断った: " + " / ".join(
        f"{gate.node or '-'} の {gate.gate}: {gate.detail}" for gate in refused
    )


def _weights_of(config: ConfigDef, manifest: AnyWeightsManifest) -> AnyWeightsRef:
    """構成の重みの参照を取り、マニフェストと同じものを指しているかを確かめる。

    派生の重みは、名前と元の重みだけでなく、**変換の条件 (道具・コミット・引数・対象の
    正規表現)** まで一致していなければ、Spark に触る前に断る (`guards.weights_ref_mismatch`
    が関門と同じ比較をする)。
    """
    weights = config.weights
    if weights is None:
        raise ConfigError(f"構成 '{config.name}' は、重みを持たないので、取得も照合もできない")
    mismatch = weights_ref_mismatch(weights, manifest)
    if mismatch is not None:
        raise WeightsRefError(mismatch)
    return weights


def _check_verified_at(verified_at: datetime) -> None:
    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise ValueError("verified_at は、時差の付いた日時にする (types.py の決まり)")


def _plans_by_role(plans: Sequence[ContainerPlan]) -> Mapping[NodeRole, ContainerPlan]:
    return {plan.node: plan for plan in plans}


def _report(stream: TextIO, message: str) -> None:
    """知らせを 1 行で出す (標準出力は、結果を見せるのに使うので、混ぜない)。"""
    stream.write(f"{message}\n")
    stream.flush()


# --- 関門 -----------------------------------------------------------------


def _prelude(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    read_timeout_s: float,
) -> tuple[tuple[GateResult, ...], dict[NodeRole, tuple[OwnContainer, ...]]]:
    """入れるかを見て、自分のコンテナの一覧を、台ごとに 1 度だけ読む。

    この 1 回の読み取りで、台ごとのやり直しの判定 (`_classify_node`) を行う (2 度読んで
    食い違うことをなくす)。
    """
    gates: list[GateResult] = []
    containers: dict[NodeRole, tuple[OwnContainer, ...]] = {}
    for role in config.nodes:
        node = _node_of(nodes, config, role)
        reachable = gate_reachable(runner, node, timeout_s=read_timeout_s)
        gates.append(reachable)
        if not reachable.passed:
            continue
        try:
            containers[role] = list_own_containers(runner, node, timeout_s=read_timeout_s)
        except (RemoteError, ValueError) as exc:
            gates.append(
                GateResult(
                    gate=GATE_OWN_STATE,
                    node=role,
                    passed=False,
                    detail=f"自分のコンテナの一覧を読めなかった: {exc}",
                )
            )
    return tuple(gates), containers


def _shown_containers(containers: Iterable[OwnContainer]) -> str:
    """一覧の行を、名前と種類と状態で並べる (よその行は、そもそも一覧に来ない)。"""
    return ", ".join(
        f"{item.name} [{item.labels.get(LABEL_KIND) or '種類のラベルがない'}] {item.state}"
        for item in containers
    )


def _classify_node(
    role: NodeRole, plan: ContainerPlan, mine: Sequence[OwnContainer]
) -> tuple[_NodePhase | None, GateResult]:
    """1 台ぶんの、やり直しの判定 (module の上の決めごとの 2)。

    返り値の `None` は、(c)「断る」である (`GateResult.passed` が偽になる)。判定は、
    `guards.match_running` と、その台の自分の一覧だけを見る (読み取りをしない純粋な関数)。
    """
    match = match_running((plan,), {role: mine})
    same_name = [item for item in mine if item.name == plan.container_name]
    others = [item for item in mine if item.name != plan.container_name]
    reasons: list[str] = list(match.differences)
    running_others = [item for item in others if item.running]
    left_others = [item for item in others if not item.running]
    if running_others:
        reasons.append(
            f"{role}: ほかの自分のコンテナが動いている ({_shown_containers(running_others)})。"
            "同時に動かす自分のコンテナは、1 つの構成ぶんだけなので、先に `serve stop` で止める"
        )
    if left_others or any(not item.running for item in same_name):
        left = [*left_others, *(item for item in same_name if not item.running)]
        reasons.append(
            f"{role}: 終了した自分のコンテナが残っている ({_shown_containers(left)})。"
            "`serve logs` で記録を回収してから `serve stop` で消す (残すと名前が衝突する)"
        )
    if reasons:
        return None, GateResult(
            gate=GATE_OWN_STATE, node=role, passed=False, detail=" / ".join(reasons)
        )
    if match.already_running:
        return "waiting", GateResult(
            gate=GATE_OWN_STATE,
            node=role,
            passed=True,
            detail=(
                f"この構成の取得が動いているので、起こさずに終わりを待つ ({plan.container_name})"
            ),
        )
    return "starting", GateResult(
        gate=GATE_OWN_STATE,
        node=role,
        passed=True,
        detail=f"自分のコンテナは 1 つもない (起こす対象: {plan.container_name})",
    )


def _classify_nodes(
    config: ConfigDef,
    plans: Sequence[ContainerPlan],
    containers: Mapping[NodeRole, Sequence[OwnContainer]],
) -> tuple[dict[NodeRole, _NodePhase], tuple[GateResult, ...]]:
    """構成の台のそれぞれを、「起こす」「待つ」「断る」に分ける (module の上の決めごとの 2)。"""
    by_role = _plans_by_role(plans)
    phases: dict[NodeRole, _NodePhase] = {}
    results: list[GateResult] = []
    for role in config.nodes:
        mine = containers.get(role)
        if mine is None:
            continue  # 入れなかった台 (理由は `_prelude` が持っている)
        phase, result = _classify_node(role, by_role[role], mine)
        results.append(result)
        if phase is not None:
            phases[role] = phase
    return phases, tuple(results)


def _fetch_gates(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    starting: Sequence[ContainerPlan],
    required_bytes: int,
    *,
    read_timeout_s: float,
) -> tuple[GateResult, ...]:
    """取得の前の関門を、**起こす台についてだけ**流す。

    並びは `guards.GATE_ORDER` から 3 つを除いたものである。`gate_weights_verified` を除くのは
    **取得の前には、まだ照合の記録がないから** (決めごとの 3)、`gate_own_state` を除くのは
    台ごとのやり直しの判定 (`_classify_node`) が、同じ一覧で、より細かく見ているからである
    (決めごとの 2)。`gate_memory_free` を除くのは、取得 (`hf download`) が
    `--load-format instanttensor` で読み込まないから (issue #84)。取得は、むしろ大量の書き込みで
    ページキャッシュを埋める側であり、`memory_free` の下限で断る対象ではない。ほかは design.md
    「関門」の表のとおりで、GPU を使わない種類でも `gate_gpu_idle` を流す。
    """
    gates: list[GateResult] = []
    for plan in starting:
        node = _node_of(nodes, config, plan.node)
        gates.append(gate_gpu_idle(runner, node, timeout_s=read_timeout_s))
        gates.append(gate_layout(runner, node, plan, timeout_s=read_timeout_s))
        gates.append(gate_image_digest(runner, node, config, timeout_s=read_timeout_s))
        gates.append(gate_disk_space(runner, node, required_bytes, timeout_s=read_timeout_s))
        gates.append(gate_ports_free(runner, node, config, timeout_s=read_timeout_s))
    return tuple(gates)


def _fetching_now(mine: Sequence[OwnContainer], weights: AnyWeightsRef) -> tuple[OwnContainer, ...]:
    """この重みの取得が、いま動いている行 (照合を始めない理由。決めごとの 8b)。

    種類のラベルが `fetch` で、重みのラベルがこの重みのもの (または、読めない) 行を見る。
    """
    label = weights.identity
    return tuple(
        item
        for item in mine
        if item.running
        and item.labels.get(LABEL_KIND) == _KIND_FETCH
        and item.labels.get(LABEL_WEIGHTS) in (None, label)
    )


def _verify_gates(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    weights: AnyWeightsRef,
    *,
    read_timeout_s: float,
) -> tuple[GateResult, ...]:
    """照合の前の関門 (入れるか、取得が動いていないか、置き場所があるか)。

    照合はコンテナを起こさないので、GPU、イメージ、空き、ポートの関門は掛からない。
    置き場所がないことを、照合の不一致 (終了コード 2) ではなく、関門の断り (終了コード 1)
    にするために、`gate_layout` は流す。**この重みの取得が動いている台があれば、照合を
    始めずに断る** (決めごとの 8b)。
    """
    by_role = _plans_by_role(plans)
    gates: list[GateResult] = []
    for role in config.nodes:
        node = _node_of(nodes, config, role)
        reachable = gate_reachable(runner, node, timeout_s=read_timeout_s)
        gates.append(reachable)
        if not reachable.passed:
            continue
        try:
            mine = list_own_containers(runner, node, timeout_s=read_timeout_s)
        except (RemoteError, ValueError) as exc:
            gates.append(
                GateResult(
                    gate=GATE_OWN_STATE,
                    node=role,
                    passed=False,
                    detail=f"自分のコンテナの一覧を読めなかった: {exc}",
                )
            )
            continue
        fetching = _fetching_now(mine, weights)
        if fetching:
            gates.append(
                GateResult(
                    gate=GATE_OWN_STATE,
                    node=role,
                    passed=False,
                    detail=(
                        f"この重みの取得が動いている ({_shown_containers(fetching)})。"
                        "取得の途中のファイルを照合すると、合わない記録ができるので、照合を"
                        "始めない。`serve fetch` の終わりを待つ (`serve fetch` は、取得の"
                        "終わりに照合まで行う)"
                    ),
                )
            )
            continue
        gates.append(
            GateResult(
                gate=GATE_OWN_STATE,
                node=role,
                passed=True,
                detail="この重みの取得は動いていない",
            )
        )
        gates.append(gate_layout(runner, node, by_role[role], timeout_s=read_timeout_s))
    return tuple(gates)


# --- 了承を得る計画 --------------------------------------------------------


def _check_record_dir(record_dir: Path) -> None:
    """`record_dir` が絶対の道筋であることを確かめる (Spark に触る前に断る)。

    配る元 (`<record_dir>/verified/<役割>/`) は、配る前に空にする。消す操作なので、相対の
    道筋 (空の道筋を含む) を受けると、プロセスの作業ディレクトリに対して効いてしまう。
    `remote` が、回収の宛先を `var_root` の下に絞っているのと同じ考え方である。
    """
    if not record_dir.is_absolute():
        raise ValueError(
            f"record_dir は、絶対の道筋で渡す (配る元を、配る前に空にするため): '{record_dir}'"
        )


def _record_source_dir(record_dir: Path, role: NodeRole) -> Path:
    """照合の結果の記録を配る元 (**この module だけが使う場所**。決めごとの 6)。

    `remote.push` は、渡したディレクトリの中身を丸ごと送るので、呼ぶ側が渡した `record_dir`
    そのものを元にしない (5.1 が `logs.var_dir` を渡すと、回収したコンテナの記録が Spark の
    `state/` に逆流する)。
    """
    return record_dir / _RECORD_DIRNAME / role


def _record_pushes(
    config: ConfigDef, record_dir: Path, scope: VerificationScope
) -> tuple[PlannedPush, ...]:
    """照合の結果の記録を配る計画 (宛先は `state/`。`--delete` は付けない)。"""
    return tuple(
        PlannedPush(
            node=role,
            local_dir=_record_source_dir(record_dir, role),
            remote_subdir=RECORD_SUBDIR,
            delete=False,
            purpose=f"{role} に、照合の結果の記録 (対象 {scope}) を置く (取得と照合のあと)",
        )
        for role in config.nodes
    )


def _approved_plan(
    plans: Sequence[ContainerPlan],
    starting: Sequence[ContainerPlan],
    pushes: Sequence[PlannedPush],
) -> ApprovedPlan:
    """了承を得る計画を作る (**起こすのは、`starting` の台だけ**。決めごとの 2b)。

    `guards.build_approved_plan` は、渡した計画のすべてに `docker run` を作るので、全部の台の
    計画を渡して検査 (名前と所有のラベル、巻き戻しの相手) を通したうえで、動いている取得に
    合流する台の `docker run` だけを落とす。落ちるのは**前に進むコマンドだけ**で、巻き戻し
    (終わったコンテナを片付ける権限) は残る。権限を足すのではなく減らすので、不変条件を
    迂回しない。むしろ、合流する台の `docker run` が計画に無くなるので、誤って起こそうと
    しても `remote` が断る。
    """
    approved = build_approved_plan(plans, extra_forward=tuple(pushes))
    names = {plan.container_name for plan in starting}
    forward = tuple(
        command
        for command in approved.forward
        if not (
            isinstance(command, PlannedRun)
            and command.container is not None
            and command.container not in names
        )
    )
    # `model_copy(update=…)` は pydantic の検証を通らないので、作り直して、検証を通す
    # (巻き戻しの対象が、`own_container_names` の中にあること、前に進むコマンドが空でないこと)
    return ApprovedPlan(
        forward=forward,
        rollback=approved.rollback,
        own_container_names=approved.own_container_names,
    )


# --- 取得のコンテナ --------------------------------------------------------


def _running_note(names: Sequence[str]) -> str:
    """誤りの文に、いま動いている取得の名前を添える (止めないことも言う)。"""
    if not names:
        return "いま動いている取得はない。"
    return f"いま動いている取得は、止めない ({', '.join(names)})。"


def _start_all(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    starting: Sequence[ContainerPlan],
    already_running: Sequence[str],
    controls: _Controls,
) -> None:
    """**起こす台**のすべてで、続けて `docker run -d` を流す。

    切り離して起こすので、2 台の取得は、そのまま並行に進む (Mac の側でスレッドを使わない)。
    **途中で失敗しても、すでに動いている取得を止めない**: 起こす呼び出しが届かなかった場合でも
    遠隔の `docker run -d` は完了しうるし、先に起こした台の取得は続いている (決めごとの 1)。
    誤りの文には、(i) いま動いている取得の名前、(ii) 失敗した台と理由、(iii) もう一度打てば
    動いている台は待ち、起きていない台だけを起こすこと、(iv) 止めるなら `serve stop`、を入れる
    (この案内は、やり直しを台ごとに判定するので、本当になる。決めごとの 2)。

    引数:
        starting: 起こす台の計画 (動いている取得に合流する台は、ここに入らない)。
        already_running: すでに動いている取得のコンテナの名前 (合流する台のぶん)。
    """
    running = list(already_running)
    for plan in starting:
        node = _node_of(nodes, config, plan.node)
        try:
            started = runner.run(node, plan.argv, timeout_s=controls.start_timeout_s, mutating=True)
        except RemoteError as exc:
            raise WeightsFetchError(
                f"{plan.node} ({node.ssh_host}) で、取得のコンテナを起こす途中に、読み取りが"
                f"届かなかった ({plan.container_name}): {exc}。"
                f"この台の取得も、起き上がっているかもしれない。{_running_note(running)}"
                f"{_KEEP_GOING}"
            ) from exc
        if started.exit_code != 0:
            raise WeightsFetchError(
                f"{plan.node} ({node.ssh_host}) で、取得のコンテナを起こせなかった"
                f" ({plan.container_name}): {_shown_failure(started)}。"
                "その名前のコンテナが、自分のものかどうかは分からないので、止めも消しもしない。"
                f"{_running_note(running)}{_KEEP_GOING}"
            )
        running.append(plan.container_name)


def _started_target(
    runner: RemoteRunner,
    node: NodeDef,
    plan: ContainerPlan,
    controls: _Controls,
) -> _Target:
    """起こした取得のコンテナの識別子を、自分のラベルで絞った一覧から取る (不変条件の (a))。"""
    try:
        containers = list_own_containers(runner, node, timeout_s=controls.read_timeout_s)
    except (RemoteError, ValueError) as exc:
        raise WeightsFetchError(
            f"{plan.node} ({node.ssh_host}) で、起こした取得のコンテナを探せなかった"
            f" ({plan.container_name}): {exc}。{_KEEP_GOING}"
        ) from exc
    found = next((item for item in containers if item.name == plan.container_name), None)
    if found is None:
        raise WeightsFetchError(
            f"{plan.node} ({node.ssh_host}) で、起こしたはずの取得のコンテナが、自分の"
            f"一覧にない ({plan.container_name})。{_KEEP_GOING}"
        )
    return _Target(node=node, plan=plan, container_id=found.id, attached=False)


def _attached_target(
    node: NodeDef,
    plan: ContainerPlan,
    containers: Mapping[NodeRole, Sequence[OwnContainer]],
) -> _Target:
    """動いている取得に合流するときの相手 (判定に使った一覧の行から、そのまま取る)。"""
    found = next(
        (item for item in containers.get(plan.node, ()) if item.name == plan.container_name),
        None,
    )
    if found is None:  # pragma: no cover - `_classify_node` が先に見ている
        raise WeightsFetchError(
            f"{plan.node} で、動いているはずの取得のコンテナが見つからない ({plan.container_name})"
        )
    return _Target(node=node, plan=plan, container_id=found.id, attached=True)


def _container_state(
    runner: RemoteRunner, target: _Target, timeout_s: float
) -> tuple[str, int | None]:
    """取得のコンテナの状態と終了コードを読む (対象は、一覧から来た識別子だけ)。

    読めなかったことを「終わった」に倒さない。倒すと、取得が続いているのに片付けに進んで
    しまう (module の上の決めごとの 1)。
    """
    argv = ("docker", "container", "inspect", "--format", _STATE_FORMAT, target.container_id)
    role = target.plan.node
    try:
        result = runner.run(target.node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError as exc:
        raise WeightsFetchError(
            f"{role} ({target.node.ssh_host}) で、取得の終わりを見に行けなかった"
            f" ({target.plan.container_name}): {exc}。{_KEEP_GOING}"
        ) from exc
    if result.exit_code != 0:
        raise WeightsFetchError(
            f"{role} ({target.node.ssh_host}) で、待っていた取得のコンテナが見つからなくなった"
            f" ({target.plan.container_name}): {_shown_failure(result)}。"
            "自分のラベルで絞った一覧 (`serve status`) で、いまの様子を確かめ直すこと。"
            f"{_KEEP_GOING}"
        )
    fields = result.stdout.split()
    if not fields:
        raise WeightsFetchError(
            f"{role} ({target.node.ssh_host}) で、取得のコンテナの状態が空で返った"
            f" ({target.plan.container_name})。{_KEEP_GOING}"
        )
    code = fields[1] if len(fields) > 1 else ""
    return fields[0], int(code) if code.lstrip("-").isdecimal() else None


def _wait_for_all(
    runner: RemoteRunner, targets: Sequence[_Target], controls: _Controls
) -> dict[NodeRole, tuple[str, int | None]]:
    """**構成の台のすべて**で、取得のコンテナが終わるまで待つ。

    終わった台は、もう見に行かない。上限は、構成の `ready_timeout_s` である。時計と眠りは
    引数から来るので、試験は実際に眠らない。**時間切れでも、取得のコンテナを止めない**
    (module の上の決めごとの 1)。
    """
    finished: dict[NodeRole, tuple[str, int | None]] = {}
    deadline = controls.clock() + controls.timeout_s
    while True:
        for target in targets:
            if target.plan.node in finished:
                continue
            status, exit_code = _container_state(runner, target, controls.read_timeout_s)
            if status not in _UNFINISHED_STATES:
                finished[target.plan.node] = (status, exit_code)
        if len(finished) == len(targets):
            return finished
        if controls.clock() >= deadline:
            waiting = ", ".join(
                target.plan.container_name for target in targets if target.plan.node not in finished
            )
            raise WeightsFetchError(
                f"取得が、{controls.timeout_s:.0f} 秒のうちに終わらなかった ({waiting})。"
                f"{_KEEP_GOING}"
            )
        controls.sleep(controls.poll_interval_s)


def _log_tail(runner: RemoteRunner, target: _Target, *, lines: int, timeout_s: float) -> str:
    """取得の記録の末尾を読む (見せるためだけ。回収そのものは、呼ぶ側 (5.1) の仕事)。

    対象は、自分のラベルで絞った一覧から来た識別子だけである。ここで読めなかったことは、
    取得の成否を変えないので、理由の文を返して先に進む。
    """
    argv = ("docker", "logs", "--tail", str(lines), target.container_id)
    try:
        result = runner.run(target.node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError as exc:
        return f"(記録の末尾を読めなかった: {exc})"
    if result.exit_code != 0:
        return f"(記録の末尾を読めなかった: {_shown_failure(result)})"
    return (result.stdout + result.stderr).strip()


def _clean_up(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: _Controls,
) -> tuple[str, ...]:
    """**終了した**取得のコンテナを片付ける (3.1 の巻き戻しの決まりに従う)。

    相手を決めるのは `guards.rollback_commands` で、流す直前に、自分のラベルで絞った一覧に
    その名前の行があることを確かめる (`docker stop <名前>` は、その名前のコンテナが誰のもの
    でも止めるので、名前だけを頼りにしない。requirements 2.3)。引数の列も、その関数が
    `guards.stop_argv` / `guards.remove_argv` で作るので、了承を得た計画の巻き戻しと完全に
    一致する。

    **ここに来るのは、2 台とも終わったあとだけである** (待ちの途中では、片付けない)。
    片付けの途中で中断が来ても、消すところまでは試みる。
    """
    commands, unchecked = rollback_commands(runner, nodes, plans, timeout_s=controls.read_timeout_s)
    problems: list[str] = list(unchecked)
    interrupted: KeyboardInterrupt | None = None
    for command in commands:
        what = _CLEANUP_LABELS[command.argv[1]]
        if interrupted is not None and command.argv[1] != _REMOVE_SUBCOMMAND:
            continue  # 中断のあとは、消すことだけを試みる
        node = nodes[command.node]
        try:
            result = runner.run(node, command.argv, timeout_s=CLEANUP_TIMEOUT_S, mutating=True)
        except KeyboardInterrupt as exc:
            if interrupted is not None:
                raise  # 2 度目の中断は、そのまま伝える
            interrupted = exc
            problems.append(f"{command.container}: {what} (中断された。消すところまでは試みる)")
            continue
        except RemoteError as exc:
            problems.append(f"{command.container}: {what} (読み取りが届かなかった): {exc}")
            continue
        if result.exit_code != 0:
            problems.append(f"{command.container}: {what}: {_shown_failure(result)}")
    if interrupted is not None:
        if problems:
            _report(
                controls.report,
                "警告: 取得のコンテナの片付けが、すべては終わらなかった: " + " / ".join(problems),
            )
        raise interrupted
    return tuple(problems)


def _fetch_failure_detail(fetches: Sequence[NodeFetch], problems: Sequence[str]) -> str:
    """取得のコンテナが 0 以外で終わったときの、見せる文 (記録の末尾を添える)。"""
    parts: list[str] = []
    for fetched in fetches:
        if fetched.ok:
            continue
        code = "不明" if fetched.exit_code is None else str(fetched.exit_code)
        reason = f"{fetched.node} の {fetched.container_name} が、終了コード {code} で終わった"
        if fetched.exit_code in _NOT_EXECUTABLE_EXIT_CODES:
            reason += (
                " (イメージの中に `hf` がない、または実行できない可能性が高い。"
                "**ホストには何も入れない**ので、代わりの方法 (Mac に取得して rsync で配る) を"
                "採るかどうかを、計測者に尋ねる)"
            )
        parts.append(f"{reason}。記録の末尾:\n{fetched.log_tail}")
    detail = "重みの取得が失敗した: " + " / ".join(parts)
    if problems:
        detail += " 片付けも失敗した: " + " / ".join(problems)
    return detail


# --- 照合 ------------------------------------------------------------------


def _batches(files: Sequence[ManifestFile], size: int) -> tuple[tuple[ManifestFile, ...], ...]:
    """`sha256sum` の 1 回ぶんに分ける。"""
    step = max(1, size)
    return tuple(tuple(files[index : index + step]) for index in range(0, len(files), step))


def _sha256_timeout(files: Sequence[ManifestFile]) -> float:
    """この回の `sha256sum` の時間切れ (中身の大きさから見積もる)。"""
    total = sum(entry.size for entry in files)
    return SHA256_BASE_TIMEOUT_S + SHA256_TIMEOUT_PER_GIB_S * (total / _GIB)


def _read_digests(
    runner: RemoteRunner,
    node: NodeDef,
    directory: str,
    files: Sequence[ManifestFile],
    *,
    batch_files: int,
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    """1 台で `sha256sum` を流し、道筋ごとの sha256 を集める。

    **Spark の側で正解を作らない**: ここで流すのは `sha256sum` だけで、突き合わせは Mac の
    側で行う。読めない行が 1 行でもあった回は、その回のファイルの値を 1 つも採らない
    (module の上の決めごとの 8)。

    返り値:
        マニフェストの `path` ごとの sha256、値を採れなかった `path`、読めなかった行。
    """
    digests: dict[str, str] = {}
    unverified: list[str] = []
    unreadable: list[str] = []
    for batch in _batches(files, batch_files):
        paths = [f"{directory}/{entry.path}" for entry in batch]
        argv = ("sha256sum", "--", *paths)
        try:
            result = runner.run(node, argv, timeout_s=_sha256_timeout(batch), mutating=False)
        except RemoteError as exc:
            raise WeightsFetchError(
                f"{node.role} ({node.ssh_host}) で、重みの sha256 を読めなかった: {exc}"
            ) from exc
        found, bad = parse_sha256_output(result.stdout)
        unexpected = sorted(path for path in found if path not in set(paths))
        if bad or unexpected:
            # どの行がどのファイルのものか決められないので、この回は 1 つも採らない
            unreadable.extend(bad)
            if unexpected:
                unreadable.append("渡していない道筋の行が返った: " + ", ".join(unexpected))
            unverified.extend(entry.path for entry in batch)
            continue
        for entry, path in zip(batch, paths, strict=True):
            digest = found.get(path)
            if digest is None:
                unverified.append(entry.path)  # ない (標準エラーに出る)
            else:
                digests[entry.path] = digest
    return digests, tuple(unverified), tuple(unreadable)


def _weights_dirs(
    config: ConfigDef, plans: Sequence[ContainerPlan], weights: AnyWeightsRef
) -> Mapping[NodeRole, str]:
    """台ごとの、**Spark の側の**重みの置き場所を、先に決めておく。

    design.md 「Error Handling」の「早く断る」: 構成が重みの置き場所を結び付けていないことは、
    Spark に触る前に分かるので、ここで断つ (照合まで待つと、184 GiB を取得したあとに断る
    ことになる)。
    """
    by_role = _plans_by_role(plans)
    return {
        role: weights_dir_on_spark(by_role[role].argv, weights.mount_at) for role in config.nodes
    }


def _make_record(
    manifest: AnyWeightsManifest,
    *,
    scope: VerificationScope,
    node: NodeRole,
    verified_at: datetime,
    files: Sequence[ManifestFile],
    mismatched: tuple[str, ...],
) -> AnyVerificationRecord:
    """照合の結果の記録を、マニフェストの種類に合わせて組み立てる。

    Hub は `repo` と版、派生は `derivation` (名前・元の重み・変換の条件) を同一性の
    項目として持つ。件数と合計と `scope` と `node` は、どちらも同じである。構成の重みと
    マニフェストの種類が合っていることは、入口の `_weights_of` が確かめている。
    """
    total_bytes = sum(entry.size for entry in files)
    if isinstance(manifest, DerivedWeightsManifest):
        return DerivedVerificationRecord(
            kind="derived",
            derivation=manifest.derivation,
            manifest_sha256=manifest.content_sha256,
            scope=scope,
            node=node,
            verified_at=verified_at,
            file_count=len(files),
            total_bytes=total_bytes,
            mismatched=mismatched,
        )
    return VerificationRecord(
        repo=manifest.repo,
        revision=manifest.revision,
        scope=scope,
        node=node,
        verified_at=verified_at,
        file_count=len(files),
        total_bytes=total_bytes,
        mismatched=mismatched,
    )


def _verify_one(
    runner: RemoteRunner,
    node: NodeDef,
    directory: str,
    weights: AnyWeightsRef,
    manifest: AnyWeightsManifest,
    *,
    scope: VerificationScope,
    record_dir: Path,
    verified_at: datetime,
    batch_files: int,
) -> NodeVerification:
    """1 台ぶんの「sha256 を読む → Mac で突き合わせる → 記録を作って配る」。

    **合わなかったときも、記録を作って配る** (module の上の決めごとの 7)。`mismatched` の
    ある記録は `guards.gate_weights_verified` が断るので、古い「合っていた」記録を残すより
    安全である。
    """
    files = scoped_files(manifest, scope)
    digests, unverified, unreadable = _read_digests(
        runner, node, directory, files, batch_files=batch_files
    )
    mismatched = tuple(
        entry.path
        for entry in files
        if entry.path in digests and digests[entry.path] != entry.sha256
    )
    record = _make_record(
        manifest,
        scope=scope,
        node=node.role,
        verified_at=verified_at,
        files=files,
        mismatched=tuple(sorted({*mismatched, *unverified})),
    )
    remote_path = _push_record(runner, node, record, weights, record_dir=record_dir)
    return NodeVerification(
        node=node.role,
        record=record,
        remote_path=remote_path,
        mismatched=mismatched,
        missing=unverified,
        unreadable=unreadable,
    )


def _record_bytes(record: AnyVerificationRecord) -> bytes:
    """記録を、同じ入力なら同じバイト列になる形で書き出す (`to_json_bytes` と同じ流儀)。"""
    text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def _push_record(
    runner: RemoteRunner,
    node: NodeDef,
    record: AnyVerificationRecord,
    weights: AnyWeightsRef,
    *,
    record_dir: Path,
) -> str:
    """照合の結果の記録を Mac で作り、Spark の `state/` に置く。

    道筋は `guards.weights_record_path` が決める (Hub は `<slug>`、派生は `<名前>.derived`。
    範囲ごとに別のファイル)。配布は `push(delete=False)` で、**了承済みの計画に入っていなければ
    `remote` が断る**。
    """
    remote_path = weights_record_path(node, weights, record.scope)
    place = PurePosixPath(remote_path)
    expected = f"{node.remote_root}/{RECORD_SUBDIR}"
    if str(place.parent) != expected:
        raise WeightsError(
            f"照合の記録の道筋 ({remote_path}) が、配れる宛先 ({expected}/) の下にない"
            " (guards.weights_record_path と、この module の決まりが食い違っている)"
        )
    # 配る元は、この module だけが使う場所にして、配る前に空にする (決めごとの 6)。
    # `push` は、渡したディレクトリの中身を丸ごと送るので、よそのファイル (回収した記録や、
    # 前の回の別の範囲の記録) が Spark の `state/` に入らないようにする
    local_dir = _record_source_dir(record_dir, node.role)
    if local_dir.is_symlink():
        # `rmtree` はシンボリックリンクを消さないので、リンク先のよそのファイルが、配る元に
        # 現れて、Spark の `state/` に配られてしまう
        raise WeightsError(
            f"照合の記録を配る元 ({local_dir}) が、シンボリックリンクになっている"
            " (この場所は、この道具だけが使う。リンクを外してから、やり直す)"
        )
    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / place.name).write_bytes(_record_bytes(record))
    try:
        result = runner.push(node, local_dir, RECORD_SUBDIR, delete=False)
    except RemoteError as exc:
        raise WeightsFetchError(
            f"{node.role} ({node.ssh_host}) に、照合の結果の記録を置けなかった"
            f" ({remote_path}): {exc}"
        ) from exc
    if result.exit_code != 0:
        raise WeightsFetchError(
            f"{node.role} ({node.ssh_host}) に、照合の結果の記録を置けなかった"
            f" ({remote_path}): {_shown_failure(result)}"
        )
    return remote_path


def _verify_nodes(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    directories: Mapping[NodeRole, str],
    weights: AnyWeightsRef,
    manifest: AnyWeightsManifest,
    *,
    scope: VerificationScope,
    record_dir: Path,
    verified_at: datetime,
    batch_files: int,
) -> tuple[NodeVerification, ...]:
    """構成の台のそれぞれで照合し、合わなければ、名前を並べて断る。

    **黙って取り直さない**: この関数は、`docker run` も `docker pull` も `docker rm` も、
    1 つも出さない (requirements 3.5、design.md 「weights」)。
    """
    verifications = tuple(
        _verify_one(
            runner,
            _node_of(nodes, config, role),
            directories[role],
            weights,
            manifest,
            scope=scope,
            record_dir=record_dir,
            verified_at=verified_at,
            batch_files=batch_files,
        )
        for role in config.nodes
    )
    failed = tuple(item for item in verifications if not item.ok)
    if failed:
        # Hub の重みは `serve fetch` で取り直せる。派生の重みは、`serve fetch` が断るので、
        # 変換の道具で作り直す (`guards._how_to_verify` も、同じように案内を分けている)
        redo = (
            "作り直すときは、計測者が変換の道具で変換し直す (`serve fetch` は、手元で変換した"
            "派生の重みを取得できない。手順は `k2-derived-weights-procedure.md` の §1)"
            if isinstance(weights, DerivedWeightsRef)
            else "取り直すときは、計測者が `serve fetch` を打ち直す"
        )
        raise WeightsMismatchError(
            f"{weights.identity} の重みが、マニフェストと合わない (対象"
            f" {scope}): " + " / ".join(item.text for item in failed) + "。"
            "黙って取り直さない。" + redo,
            verifications,
        )
    return verifications


# --- 入口 ------------------------------------------------------------------


def fetch_weights(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    manifest: AnyWeightsManifest,
    started_at: datetime,
    *,
    confirmer: Confirmer,
    record_dir: Path,
    verified_at: datetime,
    scope: VerificationScope = "all",
    poll_interval_s: float = FETCH_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    start_timeout_s: float = FETCH_START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    batch_files: int = SHA256_BATCH_FILES,
    tail_lines: int = FETCH_TAIL_LINES,
) -> FetchOutcome:
    """固定した版の重みを、構成の台に取得して、照合する (`serve fetch <構成>`)。

    進む順 (design.md 「weights」): 一覧の読み取り → **台ごとのやり直しの判定** (起こす /
    待つ / 断る) → 起こす台の関門 → 了承 → **起こす台で、続けて `docker run -d`** (切り離して
    起こすので、取得は並行に進む) → 構成の台のすべてが終わるまで待つ → 記録の末尾を読む →
    終了したコンテナを片付ける → 照合。

    **やり直しは台ごとに決める** (design.md 「weights」の Idempotency:「足りないものだけが
    対象になる」)。この構成の取得が動いている台は起こさずに待ち、自分のコンテナが 1 つもない
    台だけを起こす。それ以外 (ほかの構成が動いている、終了したものが残っている、名前は同じで
    中身が違う) が 1 台でもあれば、**どの台でも起こさずに断る**。

    **待ちや起動の途中の `RemoteError`、時間切れ、中断 (Ctrl-C) では、取得のコンテナを
    止めない** (184 GiB の取得を、誤って捨てない。止めるのは `serve stop` だけ)。

    引数:
        runner: 遠隔の実行役。
        config: `kind = "fetch"` の構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        manifest: 照合の正解 (Mac でコミットしたもの)。構成の重みと同じ版であること。
        started_at: 起こす時刻 (ラベルに書く。時差の付いた日時)。
        confirmer: 計画を見せて、了承を得る口。
        record_dir: 照合の結果の記録を作る、Mac の側の置き場所 (`serving/var/` の下)。
            **配るのは、この下に、この module が作る `verified/<役割>/` だけ**で、配る前に
            空にする (`push` は、渡したディレクトリの中身を丸ごと送るので、呼ぶ側が同じ
            場所に置いた記録が、Spark の `state/` に入らないようにする)。
        verified_at: 記録に書く照合の時刻 (**呼ぶ側が渡す**。中で `datetime.now()` を
            呼ばない)。時差の付いた日時にすること。
        scope: 照合の範囲。`all` はマニフェストのすべてのファイル、`probe_files` は
            safetensors を除いたファイル (`serve fetch --probe-files`)。**範囲を選ぶだけで、
            取得のコンテナの引数は変えない** (縮小の確認用の取得は、専用の `fetch` の構成
            から起こす)。
        poll_interval_s: 取得の終わりを見に行く間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        report: 片付けが終わらなかったことを知らせる先。既定は `sys.stderr`。
        start_timeout_s: `docker run -d` の時間切れ。
        read_timeout_s: 関門と、状態と記録の読み取りの時間切れ。
        batch_files: 1 回の `sha256sum` に渡す道筋の数。
        tail_lines: 失敗のときに見せる、記録の末尾の行数。

    返り値:
        取得と照合ができたか、関門が断ったか (`gates` に、流した関門の結果を持つ)。

    例外:
        config.ConfigError: `kind` が `fetch` でない、重みを持たない、重みの置き場所を
            結び付けない構成 (終了コード 1)。
        WeightsRefError: マニフェストが構成の重みと違う、マニフェストの道筋が照合に使えない
            (終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        WeightsFetchError: 取得の失敗、時間切れ、了承のあとに届かなかった読み取り、片付けの
            失敗 (終了コード 2)。
        WeightsMismatchError: 照合が合わなかった (終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。**取得のコンテナは止めない**。
        ValueError: 構成が使う役割のノードの定義がない、`verified_at` が素の日時のとき。
    """
    if config.kind != _KIND_FETCH:
        raise ConfigError(
            f"重みの取得に使えるのは、kind = '{_KIND_FETCH}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if isinstance(config.weights, DerivedWeightsRef):
        # 手元で作った重みには、Hub の取得元がない (置くのは変換の道具、照合は `serve verify`)。
        # Spark に触る前に断る (module の docstring の「早く断る」)。
        raise ConfigError(
            f"構成 '{config.name}' の重みは、手元で変換した派生の重みである。"
            "Hub からは取得できない (置くのは変換の道具、照合は `serve verify`)"
        )
    _check_verified_at(verified_at)
    _check_record_dir(record_dir)
    weights = _weights_of(config, manifest)
    files = scoped_files(manifest, scope)
    _check_manifest_paths(files)
    plans = build_plans(config, nodes, started_at)
    # 照合の相手 (Spark の側の置き場所) を、Spark に触る前に決める (早く断る)
    directories = _weights_dirs(config, plans, weights)

    # やり直しの判定 (起こす / 待つ / 断る) は、**関門より前に、台ごとに**行う
    gates, containers = _prelude(runner, config, nodes, read_timeout_s=read_timeout_s)
    phases, decided = _classify_nodes(config, plans, containers)
    gates += decided
    refused = _refused(gates)
    if refused:
        # 1 台でも合わなければ、どの台でも起こさない
        return FetchOutcome(
            status="refused",
            config_name=config.name,
            scope=scope,
            gates=gates,
            detail=_refusal_detail(refused),
        )

    starting = tuple(plan for plan in plans if phases[plan.node] == "starting")
    waiting = tuple(plan for plan in plans if phases[plan.node] == "waiting")
    if starting:
        gates += _fetch_gates(
            runner,
            config,
            nodes,
            starting,
            sum(entry.size for entry in files),
            read_timeout_s=read_timeout_s,
        )
        refused = _refused(gates)
        if refused:
            return FetchOutcome(
                status="refused",
                config_name=config.name,
                scope=scope,
                gates=gates,
                detail=_refusal_detail(refused),
            )

    approved = _approved_plan(plans, starting, _record_pushes(config, record_dir, scope))
    request_approval(confirmer, runner, approved, nodes)
    controls = _Controls(
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
        report=sys.stderr if report is None else report,
        poll_interval_s=poll_interval_s,
        timeout_s=float(config.ready_timeout_s),
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
    )

    if waiting:
        _report(
            controls.report,
            "動いている取得を見つけたので、その台では起こさずに、終わりを待つ"
            f" ({', '.join(plan.container_name for plan in waiting)})",
        )
    _start_all(
        runner,
        config,
        nodes,
        starting,
        tuple(plan.container_name for plan in waiting),
        controls,
    )
    targets = tuple(
        _attached_target(_node_of(nodes, config, plan.node), plan, containers)
        if phases[plan.node] == "waiting"
        else _started_target(runner, _node_of(nodes, config, plan.node), plan, controls)
        for plan in plans
    )

    # 待ちの途中では、何があっても片付けに行かない (module の上の決めごとの 1)
    finished = _wait_for_all(runner, targets, controls)
    fetches = tuple(
        NodeFetch(
            node=target.plan.node,
            container_name=target.plan.container_name,
            attached=target.attached,
            state=finished[target.plan.node][0],
            exit_code=finished[target.plan.node][1],
            log_tail=_log_tail(runner, target, lines=tail_lines, timeout_s=controls.read_timeout_s),
        )
        for target in targets
    )
    problems = _clean_up(runner, nodes, plans, controls)
    if any(not fetched.ok for fetched in fetches):
        raise WeightsFetchError(_fetch_failure_detail(fetches, problems))
    if problems:
        raise WeightsFetchError("取得のコンテナを片付けられなかった: " + " / ".join(problems))

    verifications = _verify_nodes(
        runner,
        config,
        nodes,
        directories,
        weights,
        manifest,
        scope=scope,
        record_dir=record_dir,
        verified_at=verified_at,
        batch_files=batch_files,
    )
    how = (
        f"{len(starting)} 台で取得し、{len(waiting)} 台は動いていた取得に合流し"
        if waiting
        else f"{len(starting)} 台で取得し"
    )
    return FetchOutcome(
        status="fetched",
        config_name=config.name,
        scope=scope,
        gates=gates,
        fetches=fetches,
        verifications=verifications,
        detail=f"{how}、{len(verifications)} 台で照合した (対象 {scope}、{len(files)} ファイル)",
    )


def verify_weights(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    manifest: AnyWeightsManifest,
    started_at: datetime,
    *,
    confirmer: Confirmer,
    record_dir: Path,
    verified_at: datetime,
    scope: VerificationScope = "all",
    read_timeout_s: float = READ_TIMEOUT_S,
    batch_files: int = SHA256_BATCH_FILES,
) -> VerifyOutcome:
    """構成の台にある重みを、マニフェストと突き合わせる (`serve verify <構成>`)。

    進む順: 関門 (入れるか、この重みの取得が動いていないか、置き場所があるか) → 了承 (記録の
    配布のため) → 2 台で `sha256sum` → **Mac の側で**マニフェストと突き合わせ → 結果の記録を
    `state/` に置く。コンテナは 1 つも起こさない。合わないファイルは、名前を並べて断り、
    **黙って取り直さない**。

    取得が動いている間に打つと、途中のファイルで、合わない記録が書かれる (状態としては
    正しいが、紛らわしい)。そこで、**この重みの取得が動いている台があれば、照合を始めずに
    断る** (`serve fetch` が、取得の終わりに照合まで行う)。

    引数:
        runner: 遠隔の実行役。`sha256sum` などの読み取りと、記録の配布しか出さない。
        config: 重みを使う構成 (`serve`、`probe`、`fetch` のどれでもよい)。
        nodes: 役割ごとのノードの定義。
        manifest: 照合の正解。
        started_at: 引数の列を組み立てるときの時刻。**コンテナは起こさない**が、置き換えの
            印を埋めた `--mount` から重みの置き場所を読むために、計画を組み立てる。
        confirmer: 計画を見せて、了承を得る口 (前に進むコマンドは、記録の配布だけ)。
        record_dir: 記録を作る、Mac の側の置き場所。**配るのは、この下に、この module が
            作る `verified/<役割>/` だけ**で、配る前に空にする (`fetch_weights` と同じ)。
        verified_at: 記録に書く照合の時刻 (呼ぶ側が渡す)。
        scope: 照合の範囲 (`kind = "probe"` の構成と `--probe-files` は `probe_files`)。
        read_timeout_s: 関門の読み取りの時間切れ (`sha256sum` は、中身の大きさから見積もる)。
        batch_files: 1 回の `sha256sum` に渡す道筋の数。

    返り値:
        照合できたか、関門が断ったか。

    例外:
        config.ConfigError: 重みを持たない、重みの置き場所を結び付けない構成 (終了コード 1)。
        WeightsRefError: マニフェストが構成の重みと違う (終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        WeightsFetchError: 読み取りが届かなかった、記録を置けなかった (終了コード 2)。
        WeightsMismatchError: 照合が合わなかった (終了コード 2)。
        ValueError: 構成が使う役割のノードの定義がない、`verified_at` が素の日時のとき。
    """
    _check_verified_at(verified_at)
    _check_record_dir(record_dir)
    weights = _weights_of(config, manifest)
    files = scoped_files(manifest, scope)
    _check_manifest_paths(files)
    plans = build_plans(config, nodes, started_at)
    # 照合の相手 (Spark の側の置き場所) を、Spark に触る前に決める (早く断る)
    directories = _weights_dirs(config, plans, weights)

    gates = _verify_gates(runner, config, nodes, plans, weights, read_timeout_s=read_timeout_s)
    refused = _refused(gates)
    if refused:
        return VerifyOutcome(
            status="refused",
            config_name=config.name,
            scope=scope,
            gates=gates,
            detail=_refusal_detail(refused),
        )

    approved = build_approved_plan((), extra_forward=_record_pushes(config, record_dir, scope))
    request_approval(confirmer, runner, approved, nodes)
    verifications = _verify_nodes(
        runner,
        config,
        nodes,
        directories,
        weights,
        manifest,
        scope=scope,
        record_dir=record_dir,
        verified_at=verified_at,
        batch_files=batch_files,
    )
    return VerifyOutcome(
        status="verified",
        config_name=config.name,
        scope=scope,
        gates=gates,
        verifications=verifications,
        detail=(
            f"{len(verifications)} 台で、{len(files)} ファイル (対象 {scope}) が"
            "マニフェストと合った"
        ),
    )
