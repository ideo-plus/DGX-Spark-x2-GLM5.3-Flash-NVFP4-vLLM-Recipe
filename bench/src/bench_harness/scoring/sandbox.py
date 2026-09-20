"""モデルが書いたコードを、隔離したコンテナの中で動かす (task 6.6、5.4)。

`ContainerSandbox` が、設計の `SandboxRunner` の約束事を満たす。`available()`
で実行環境とイメージを確かめ、`run_python(source, timeout_s)` でコードを
1 回だけ動かす。コードはファイルを経由せず、標準入力で渡す。

依存するのは標準ライブラリと `bench_harness.types` だけである (design.md の
依存の向き: 採点は corpus / client / suites / runner / analysis を読み込まない)。

## 隔離の中身 (design.md scoring/sandbox、note 6.1 で実機で確かめた)

起動の引数は `build_run_argv` に固定してある (純粋な関数なので、試験がそのまま
中身を見られる)。

| 引数 | 効くこと |
|---|---|
| `--network none` | 外への接続が `Network is unreachable` になる |
| `--read-only` | `/` への書き込みが `Read-only file system` になる |
| `--tmpfs /tmp:rw,size=64m` | 検査のプログラムが使う一時領域だけを書けるようにする |
| `--cap-drop ALL` + `--security-opt no-new-privileges` | 権限の昇格を断つ |
| `--memory` / `--memory-swap` / `--cpus` / `--pids-limit` | 資源に上限を置く |
| `--user 65534:65534` | 権限のない利用者で動かす |
| `--rm` | 実行のたびに使い捨てる |
| `--pull never` | 手元にないイメージを、黙って取りに行かせない |

ホストのディレクトリのマウント (`-v`、`--mount`)、環境変数の受け渡し (`-e`)、
`--privileged`、`--device` は**いっさい付けない**。`FORBIDDEN_RUN_ARGS` が、
その一覧を持っている (試験が、引数に混ざっていないことを確かめる)。

## イメージは、いつも識別子で動かす

`available()` は、必ず `image inspect --format '{{.Id}}'` で手元のイメージの
識別子を調べ、**名前ではなくその識別子で**動かす (`image_ref()` が同じ値を
返すので、計測ランにはこの識別子が残る。design.md「識別子を計測ランに記録
する」)。名前を付け替えただけの別のイメージに、黙って入れ替わらないように
するためである。

`SandboxSettings.image_digest` があるときは、調べた識別子が**完全に**一致
することも確かめる。合わなければ「使えない」として扱い、何も動かさない。
docker の `{{.Id}}` は `sha256:` 付き、podman は付けずに返すので、どちらも
`sha256:<16 進 64 桁>` にそろえてから比べる (`_normalize_image_id`)。
識別子として読めない値が返ってきたら、使えないものとして扱う (fail closed)。
イメージを取りに行くことは、どの場面でもしない (`--pull never` も付ける)。

## 後始末 (note 6.1 で確かめた壊れ方)

**時間切れで `docker run` のプロセスを殺しても、コンテナは動き続ける。**
そのため、コンテナには一意の名前 (`bench-sbx-<pid>-<連番>-<乱数>`) を付けて
起動し、時間切れ・中断 (`KeyboardInterrupt`)・そのほかの例外のときは、まず
`<runtime> kill <name>` でコンテナを止め、それから `docker run` のプロセスを
回収する。後始末の失敗は握りつぶす (元の結果や例外の上に、別の失敗を投げない)。

## 出力の上限

標準出力は捨てる (`DEVNULL`)。合格か不合格かは終了コードで決まるので、中身は
要らない。標準エラーは、別の糸 (thread) で少しずつ読み、**末尾の
`STDERR_TAIL_BYTES` バイトだけ**を覚えておく。ギガバイトを吐くプログラムでも、
手元のメモリを使い切らない。標準入力への書き込みも別の糸で行う (いちばん長い
検査のプログラムは約 502 KB あり、パイプの緩衝に収まらないため。note 6.5)。
`wait()` がパイプで詰まらないのは、この読み手の糸がいるからである。

## 終了コードの読み分け

| 終了コード | 標準エラー | 意味 | 返すもの |
|---|---|---|---|
| 0 | — | 検査のプログラムが例外なく終わった | `passed=True` |
| 125 / 126 / 127 | 実行環境の印**あり** | **docker 自身**の失敗 | `SandboxError` |
| 125 / 126 / 127 | 印**なし** | モデルのコードが自分で終了した | `passed=False` |
| 137 | — | SIGKILL。メモリの上限に当たった場合を含む | `passed=False` |
| そのほか | — | モデルのコードが落ちた | `passed=False` |

125/126/127 を無条件に「不正解」として数えると、設備の不調がモデルの成績に
なってしまう。逆に無条件に `SandboxError` にすると、モデルのコードが自分で
`sys.exit(127)` を呼んだだけの**不正解**まで分母から外れ、正確さが水増しされる。

そこで、実行環境自身が出す印 (`_RUNTIME_ERROR_LINE_PREFIXES` と
`_RUNTIME_ERROR_SUBSTRINGS`。`docker: Error response from daemon: …` など) が
標準エラーにあるときだけ `SandboxError` を投げる。呼び出し側 (6.7) は、これを
`QualityOutcome.NOT_SCORED` として、不正解と分けて数えること (5.7)。印がない
125/126/127 は、ふつうの 0 以外の終了として扱う。

印の照合は控えめにしてある (行頭の `docker:` / `podman:` / `error:` と、
実行環境の決まり文句)。Python の traceback の最後の行は `ValueError: …` の
ように型の名前から始まるので、行頭の照合には当たらない。紛れたときは
「数えない」側に倒れるので、正確さを水増しする向きには外れない。なお、
検査を受けずに終了コード 0 で抜ける道は、`scoring/code.py` の印 (sentinel) が
別に塞いでいる。

## 投げるもの、投げないもの

- `SandboxError`: 隔離の仕組みの側の失敗 (実行環境が使えない、docker 自身の
  失敗、プロセスを回収できない)。モデルの答えについての判定ではない
- `ValueError`: 呼び出し方の誤り (`timeout_s` が 0 以下、または有限でない)
- モデルのコードがどう落ちても、例外にはしない (`SandboxResult` で返す)

## そのほか

- 複数の糸から同時に使うことは考えていない (6.7 はコードの課題を 1 つずつ
  動かす)。`available()` の結果を覚えておく処理にも、鍵は掛けていない
- `available()` は 1 つの `ContainerSandbox` につき 1 回だけ実際に確かめ、
  以後はその結果を返す (外部のコマンドを 3 回起こすので、毎回はやらない)
"""

from __future__ import annotations

import contextlib
import itertools
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from bench_harness.types import SandboxResult, SandboxSettings, SandboxUnavailable

__all__ = [
    "CONTAINER_NAME_PREFIX",
    "FORBIDDEN_RUN_ARGS",
    "KILL_TIMEOUT_S",
    "PROBE_TIMEOUT_S",
    "SANDBOX_USER",
    "SECURITY_OPT",
    "STDERR_TAIL_BYTES",
    "TMPFS_SIZE_MB",
    "ByteStream",
    "CommandOutput",
    "ContainerSandbox",
    "ProcessLike",
    "SandboxError",
    "SandboxRunner",
    "build_run_argv",
]

CONTAINER_NAME_PREFIX: Final[str] = "bench-sbx"
"""コンテナの名前の頭。後始末の確認で、この頭を手がかりに探せるようにする。"""

STDERR_TAIL_BYTES: Final[int] = 2048
"""標準エラーから覚えておく、末尾の大きさ。"""

TMPFS_SIZE_MB: Final[int] = 64
"""書き込みを許す一時領域 (`/tmp`) の大きさ。"""

PROBE_TIMEOUT_S: Final[float] = 30.0
"""実行環境とイメージの確認 1 回あたりの制限時間。"""

KILL_TIMEOUT_S: Final[float] = 20.0
"""コンテナを止めるコマンド 1 回あたりの制限時間。"""

SANDBOX_USER: Final[str] = "65534:65534"
"""権限のない利用者 (`nobody:nogroup`)。イメージの側でも同じ利用者にしてある。"""

FORBIDDEN_RUN_ARGS: Final[frozenset[str]] = frozenset(
    {
        # ホストのファイルを見せる
        "-v",
        "--volume",
        "--volumes-from",
        "--mount",
        # ホストの値を渡す
        "-e",
        "--env",
        "--env-file",
        # 権限と資源の囲いを外す
        "--privileged",
        "--device",
        "--device-cgroup-rule",
        "--cap-add",
        "--sysctl",
        "--gpus",
        # 名前空間をホストと共有する
        "--userns",
        "--pid",
        "--ipc",
        "--uts",
        "--cgroupns",
        # 外から届くようにする
        "--net",
        "--add-host",
        "-p",
        "--publish",
        "-P",
        "--publish-all",
        # 起動するものを差し替える
        "--entrypoint",
        "--tmpfs-mode",
    }
)
"""けっして付けない引数。隔離を開けてしまうものだけを挙げる。

`tests/unit/test_scoring_sandbox.py::test_forbidden_args_cover_the_ways_out_of_the_box`
が、この一覧から項目が減らされていないことを確かめる。"""

SECURITY_OPT: Final[str] = "no-new-privileges"
"""`--security-opt` に渡す唯一の値。seccomp や apparmor を緩める値は渡さない。"""

_RUNTIME_SEARCH_ORDER: Final[tuple[str, ...]] = ("podman", "docker")
"""`runtime = "auto"` のときに探す順 (design.md scoring/sandbox)。"""

_CLI_ERROR_EXIT_CODES: Final[frozenset[int]] = frozenset({125, 126, 127})
"""docker / podman 自身の失敗**になりうる**終了コード。

この値だけでは足りない。モデルのコードが自分で `sys.exit(127)` を呼んでも
同じ値になるので、`_RUNTIME_ERROR_LINE_PREFIXES` か
`_RUNTIME_ERROR_SUBSTRINGS` の印が標準エラーにあるときだけ、実行環境の失敗と
見なす (下記)。"""

_RUNTIME_ERROR_LINE_PREFIXES: Final[tuple[str, ...]] = ("docker:", "podman:", "error:")
"""実行環境自身の失敗が、標準エラーの**行頭**に書く印 (小文字にして比べる)。

Python の traceback の最後の行は `ValueError: ...` のように型の名前から
始まるので、行頭の一致では当たらない。"""

_RUNTIME_ERROR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "error response from daemon",
    "executable file not found",
    "unable to find image",
    "oci runtime create failed",
    "cannot be invoked",
    "cannot connect to the docker daemon",
)
"""実行環境自身の失敗に出てくる語句 (小文字にして、どこにあっても見る)。"""

_CONTAINER_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
"""コンテナの名前に使ってよい文字 (実行環境の決まりに合わせる)。"""

_IMAGE_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
"""イメージの識別子の、正規化したあとの形。小文字の 16 進 64 桁。"""

_BARE_IMAGE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
"""podman の `{{.Id}}` は `sha256:` を付けずに返す。この形も受け取る。"""

_IMAGE_REF_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]*$")
"""イメージの名前に許す形。引数に見える値 (`-` で始まる) を断つ。"""

_READ_CHUNK_BYTES: Final[int] = 64 * 1024
"""標準エラーを 1 回に読む大きさ。丸呑み (`read()`) はしない。"""

_REAP_TIMEOUT_S: Final[float] = 20.0
"""コンテナを止めたあと、`docker run` のプロセスを待つ上限。"""

_JOIN_TIMEOUT_S: Final[float] = 10.0
"""読み書きの糸の終わりを待つ上限。"""

_MAX_REASON_CHARS: Final[int] = 300
"""`SandboxUnavailable.reason` に入れる、外部のコマンドの出力の上限。"""


class SandboxError(Exception):
    """隔離の仕組みの側の失敗。モデルの答えについての判定ではない (6.7 は 5.7 の
    「採点できなかった」として、不正解と分けて数えること)。"""


# --- 外部のコマンドの層 (試験が差し替えられる継ぎ目) -------------------------


@dataclass(frozen=True)
class CommandOutput:
    """短い確認のコマンドの結果。"""

    returncode: int
    stdout: str
    stderr: str


class ByteStream(Protocol):
    """標準入力・標準エラーとして使う、最小限の口。"""

    def read(self, size: int = ..., /) -> bytes: ...

    def write(self, data: bytes, /) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class ProcessLike(Protocol):
    """`docker run` のプロセスとして使う、最小限の口 (`subprocess.Popen` が満たす)。"""

    @property
    def stdin(self) -> ByteStream | None: ...

    @property
    def stderr(self) -> ByteStream | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class SandboxRunner(Protocol):
    """design.md scoring/sandbox の約束事。"""

    def available(self) -> bool | SandboxUnavailable: ...

    def run_python(self, source: str, timeout_s: float) -> SandboxResult: ...


WhichFn = Callable[[str], str | None]
RunCommandFn = Callable[[Sequence[str], float], CommandOutput]
StartProcessFn = Callable[[Sequence[str]], ProcessLike]


def _default_run_command(argv: Sequence[str], timeout_s: float) -> CommandOutput:
    """短い確認のコマンドを 1 回動かす (出力は小さいので、そのまま受け取る)。"""
    completed = subprocess.run(  # noqa: S603 - 引数は build 側で組み立てた固定の並び
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return CommandOutput(
        returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
    )


def _default_start_process(argv: Sequence[str]) -> ProcessLike:
    """コンテナを起こす。標準出力は捨て、標準エラーだけをパイプで受ける。"""
    return subprocess.Popen(  # noqa: S603 - 引数は build_run_argv が組み立てる
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


# --- 起動の引数 --------------------------------------------------------------


SELF_DESTRUCT_MARGIN_S: Final[int] = 10
"""コンテナが自分で終わるまでの、こちらの時間切れへの上乗せ (秒)。

ふつうは、こちらの時間切れが先に働いて `kill` する。コンテナの中の上限は、
こちらのプロセスが死んだときのための、最後の守りである。
"""


def build_run_argv(
    runtime: str,
    image_ref: str,
    container_name: str,
    settings: SandboxSettings,
    *,
    self_destruct_s: int | None = None,
) -> list[str]:
    """コンテナを起こす引数を組み立てる (純粋な関数)。

    `image_ref` は、ダイジェストが決まっていれば識別子、決まっていなければ
    イメージの名前。`-` で始まる値 (引数に見える値) は断る (fail closed)。

    `self_destruct_s` を渡すと、コンテナの中の `timeout -s KILL <秒>` の下で動かす。
    こちらのプロセスが強制終了されると、時間切れの `kill` も後始末も走らないので、
    終わらないコードのコンテナが、いつまでも残る (実際に、試験の中断で残った)。
    中からも止まるようにしておけば、こちらが死んでも、コンテナは自分で終わる。
    """
    if self_destruct_s is not None and self_destruct_s < 1:
        raise SandboxError(f"self_destruct_s は 1 以上である必要がある: {self_destruct_s!r}")
    command = ["python", "-I", "-"]
    if self_destruct_s is not None:
        command = ["timeout", "-s", "KILL", str(self_destruct_s), *command]
    if not _IMAGE_REF_RE.match(image_ref):
        raise SandboxError(f"イメージの指定が使えない値である: {image_ref!r}")
    if not _CONTAINER_NAME_RE.match(container_name):
        raise SandboxError(f"コンテナの名前が使えない値である: {container_name!r}")
    memory = f"{settings.memory_mb}m"
    return [
        runtime,
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,size={TMPFS_SIZE_MB}m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        SECURITY_OPT,
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--cpus",
        f"{settings.cpus:g}",
        "--pids-limit",
        str(settings.pids_limit),
        "--user",
        SANDBOX_USER,
        image_ref,
        *command,
    ]


# --- 標準入力と標準エラーの世話 ----------------------------------------------


def _write_all(stream: ByteStream | None, payload: bytes) -> None:
    """プログラムを標準入力に流し込む。相手が先に落ちても、静かに終わる。"""
    if stream is None:  # pragma: no cover - `Popen` は必ず口を作る
        return
    try:
        stream.write(payload)
        stream.flush()
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


class _TailBuffer:
    """標準エラーを少しずつ読み、末尾の `limit` バイトだけを覚えておく。"""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = b""

    def drain(self, stream: ByteStream | None) -> None:
        if stream is None:  # pragma: no cover - `Popen` は必ず口を作る
            return
        try:
            while True:
                chunk = stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                merged = self._data + chunk
                self._data = merged[-self._limit :] if len(merged) > self._limit else merged
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    def text(self) -> str:
        """覚えておいた末尾を文字列にする (切れ目が文字の途中でも落ちない)。"""
        return self._data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class _Resolved:
    """確かめが済んだ実行環境と、実際に動かすイメージの指定。"""

    runtime_path: str
    image_ref: str


# --- 本体 --------------------------------------------------------------------


class ContainerSandbox:
    """コンテナの中でコードを動かす `SandboxRunner` の実装。"""

    def __init__(
        self,
        settings: SandboxSettings,
        *,
        which: WhichFn = shutil.which,
        run_command: RunCommandFn = _default_run_command,
        start_process: StartProcessFn = _default_start_process,
    ) -> None:
        self._settings = settings
        self._which = which
        self._run_command = run_command
        self._start_process = start_process
        self._resolved: _Resolved | SandboxUnavailable | None = None
        self._counter = itertools.count(1)

    # -- 使えるかどうか --

    def available(self) -> bool | SandboxUnavailable:
        """実行環境とイメージを確かめる。結果は、この個体の中で覚えておく。"""
        resolved = self._resolve()
        return True if isinstance(resolved, _Resolved) else resolved

    def image_ref(self) -> str | None:
        """実際に動かすイメージの指定 (計測ランに記録する。使えないときは `None`)。"""
        resolved = self._resolve()
        return resolved.image_ref if isinstance(resolved, _Resolved) else None

    def _resolve(self) -> _Resolved | SandboxUnavailable:
        if self._resolved is None:
            self._resolved = self._probe()
        return self._resolved

    def _probe(self) -> _Resolved | SandboxUnavailable:
        settings = self._settings

        image = settings.image.strip()
        if not _IMAGE_REF_RE.match(image):
            return SandboxUnavailable(
                reason=(
                    f"設定の sandbox.image が使えない値である ({settings.image!r})。"
                    "bench/config/profiles.toml の sandbox.image を直すこと"
                )
            )

        digest: str | None = None
        if settings.image_digest is not None:
            digest = _normalize_image_id(settings.image_digest.strip())
            if digest is None:
                return SandboxUnavailable(
                    reason=(
                        f"設定の sandbox.image_digest の形が違う ({settings.image_digest!r})。"
                        "`docker image inspect <image> --format '{{.Id}}'` の値"
                        " (sha256: と小文字 16 進 64 桁) を書くこと"
                    )
                )

        runtime_path = self._find_runtime()
        if runtime_path is None:
            wanted = (
                " / ".join(_RUNTIME_SEARCH_ORDER)
                if settings.runtime == "auto"
                else settings.runtime
            )
            return SandboxUnavailable(
                reason=(
                    f"コンテナの実行環境 ({wanted}) が見つからない。"
                    "Podman か Docker を入れて、PATH から見えるようにすること"
                )
            )

        version = self._command([runtime_path, "version", "--format", "{{.Server.Version}}"])
        if version.returncode != 0:
            return SandboxUnavailable(
                reason=(
                    f"{runtime_path} の守護プロセスが応答しない"
                    f" ({_short(version.stderr or version.stdout)})。"
                    "Docker Desktop を起動する、または `podman machine start` を実行すること"
                )
            )

        inspect = self._command([runtime_path, "image", "inspect", image, "--format", "{{.Id}}"])
        if inspect.returncode != 0:
            return SandboxUnavailable(
                reason=(
                    f"イメージ {image} が手元にない ({_short(inspect.stderr or inspect.stdout)})。"
                    "bench/sandbox/Dockerfile の手順で作ること (取りに行くことはしない)"
                )
            )

        actual_id = _normalize_image_id(inspect.stdout.strip())
        if actual_id is None:
            return SandboxUnavailable(
                reason=(
                    f"イメージ {image} の識別子を読み取れない"
                    f" ({_short(inspect.stdout)!r})。"
                    "`image inspect --format '{{.Id}}'` が sha256 の識別子を返すことを"
                    "確かめること"
                )
            )
        if digest is not None and actual_id != digest:
            return SandboxUnavailable(
                reason=(
                    f"イメージ {image} の識別子が、設定の image_digest と合わない"
                    f" (手元: {actual_id}、設定: {digest})。"
                    "bench/sandbox/Dockerfile で作り直すか、"
                    "bench/config/profiles.toml の sandbox.image_digest を書き換えること"
                )
            )
        # 固定していてもいなくても、名前ではなく識別子で動かし、それを記録する
        return _Resolved(runtime_path=runtime_path, image_ref=actual_id)

    def _find_runtime(self) -> str | None:
        settings = self._settings
        candidates = _RUNTIME_SEARCH_ORDER if settings.runtime == "auto" else (settings.runtime,)
        for name in candidates:
            found = self._which(name)
            if found:
                return found
        return None

    def _command(self, argv: Sequence[str]) -> CommandOutput:
        """確認のコマンドを 1 回動かす。失敗は、値として返す。"""
        try:
            return self._run_command(argv, PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return CommandOutput(
                returncode=-1, stdout="", stderr=f"{PROBE_TIMEOUT_S} 秒のあいだ応答がなかった"
            )
        except OSError as error:
            return CommandOutput(returncode=-1, stdout="", stderr=str(error))

    # -- 実行 --

    def run_python(self, source: str, timeout_s: float) -> SandboxResult:
        """コードを 1 回だけ、隔離の中で動かす。

        `timeout_s` を過ぎたら、コンテナを止めて `timed_out=True` を返す。
        実行環境が使えないとき、または docker 自身が失敗したときは
        `SandboxError` を投げる (隔離なしで動かす道はない)。
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError(
                f"timeout_s は有限で 0 より大きい必要がある (受け取った値: {timeout_s})"
            )

        resolved = self._resolve()
        if isinstance(resolved, SandboxUnavailable):
            raise SandboxError(f"隔離の実行環境が使えないので、何も動かさない: {resolved.reason}")

        name = self._next_name()
        argv = build_run_argv(
            resolved.runtime_path,
            resolved.image_ref,
            name,
            self._settings,
            self_destruct_s=math.ceil(timeout_s) + SELF_DESTRUCT_MARGIN_S,
        )
        payload = source.encode("utf-8", errors="replace")

        try:
            process = self._start_process(argv)
        except OSError as error:
            raise SandboxError(f"{resolved.runtime_path} を起こせなかった: {error}") from error

        tail = _TailBuffer(STDERR_TAIL_BYTES)
        writer = threading.Thread(
            target=_write_all, args=(process.stdin, payload), name=f"{name}-stdin", daemon=True
        )
        reader = threading.Thread(
            target=tail.drain, args=(process.stderr,), name=f"{name}-stderr", daemon=True
        )
        writer.start()
        reader.start()

        timed_out = False
        returncode: int | None = None
        try:
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                # `docker run` のプロセスを殺してもコンテナは残る。先にコンテナを止める
                timed_out = True
                self._kill_container(resolved.runtime_path, name)
                returncode = self._reap(process)
        except BaseException:
            # `KeyboardInterrupt` を含む。コンテナを残したまま外に出ない
            self._kill_container(resolved.runtime_path, name)
            self._reap(process)
            raise
        finally:
            writer.join(_JOIN_TIMEOUT_S)
            reader.join(_JOIN_TIMEOUT_S)

        stderr_tail = tail.text()
        if timed_out:
            # 止めたのはこちらなので、終了コード (137) は結果に入れない
            return SandboxResult(
                passed=False, timed_out=True, exit_code=None, stderr_tail=stderr_tail
            )
        if returncode is None:
            raise SandboxError(
                f"{resolved.runtime_path} のプロセスを回収できなかった (コンテナ {name})"
            )
        if returncode in _CLI_ERROR_EXIT_CODES and _is_runtime_error(stderr_tail):
            raise SandboxError(
                f"{resolved.runtime_path} 自身が失敗した (終了コード {returncode}):"
                f" {_short(stderr_tail)}"
            )
        return SandboxResult(
            passed=returncode == 0,
            timed_out=False,
            exit_code=returncode,
            stderr_tail=stderr_tail,
        )

    # -- 後始末 --

    def _kill_container(self, runtime_path: str, name: str) -> None:
        """コンテナを止める。失敗しても、元の結果や例外の上に投げない。"""
        with contextlib.suppress(Exception):
            self._run_command([runtime_path, "kill", name], KILL_TIMEOUT_S)

    def _reap(self, process: ProcessLike) -> int | None:
        """`docker run` のプロセスを回収する。戻らなければ `None`。"""
        try:
            return process.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            try:
                return process.wait(timeout=_REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover - 実機では起きなかった
                return None
        except Exception:  # pragma: no cover - 後始末の途中の失敗は握りつぶす
            return None

    def _next_name(self) -> str:
        """ほかの計測や、同じ計測の別の試行と重ならない名前。"""
        name = f"{CONTAINER_NAME_PREFIX}-{os.getpid()}-{next(self._counter)}-{uuid.uuid4().hex[:8]}"
        if not _CONTAINER_NAME_RE.match(name):  # pragma: no cover - 組み立てから起きない
            raise SandboxError(f"コンテナの名前が使えない値になった: {name!r}")
        return name


def _normalize_image_id(value: str) -> str | None:
    """イメージの識別子を `sha256:<16 進 64 桁>` にそろえる。違う形なら `None`。

    docker の `{{.Id}}` は `sha256:` 付き、podman は付けずに返す。どちらも
    同じ識別子として扱えるように、ここで 1 つの形にする。
    """
    flat = value.strip()
    if _IMAGE_DIGEST_RE.match(flat):
        return flat
    if _BARE_IMAGE_ID_RE.match(flat):
        return f"sha256:{flat}"
    return None


def _is_runtime_error(stderr_tail: str) -> bool:
    """標準エラーに、実行環境自身の失敗の印があるかどうか (module の docstring)。"""
    lowered = stderr_tail.lower()
    if any(marker in lowered for marker in _RUNTIME_ERROR_SUBSTRINGS):
        return True
    return any(
        line.lstrip().startswith(_RUNTIME_ERROR_LINE_PREFIXES) for line in lowered.splitlines()
    )


def _short(text: str) -> str:
    """外部のコマンドの出力を、理由に入れられる長さに詰める。"""
    flat = " ".join(text.split())
    if len(flat) <= _MAX_REASON_CHARS:
        return flat
    return flat[: _MAX_REASON_CHARS - 1] + "…"
