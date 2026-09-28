"""README の使い方に出てくる構成名が、実物の構成の定義に実在することを確かめる試験 (tasks.md 6.5、
issue #103)。

`serving/README.md` の「使い方 (段の順)」は、計測者が読んで打つ、`serve` の呼び出しの並びである。
ここに書いた構成名が、`serving/config/configs.toml` に実在しない名前だと、書いてあるとおりに
打っても断られる。この試験は、README のコード塊 (```bash ... ```) に現れる
`serve <サブコマンド> <構成名>` の構成名が、`config.load_configs` でそのまま読み込める実物の
構成の定義に実在することを、実際に読み込んで固定する。対象は `serving/README.md`・`README.md`・
`README.ja.md` の 3 本である (issue #103: ルートの 2 本も、起動の手順のコード塊で構成名を使う)。

対象にするのは、構成名を第一の引数に取ると `serving/README.md` の「サブコマンド」の表が
定める並び (`check`、`pull-image`、`image-licenses`、`fetch`、`verify`、`start`、`smoke`、
`logs`、`probe`、`watch`、`thinking`、`netcheck bandwidth`、`netcheck sanity`、`netcheck ab`)
だけである。`push`、`status`、`stop`、`netcheck links`、`manifest <repo> <revision>` は構成名を
取らないので対象にしない。`<構成>` のような山括弧の置き場所は、実在しない名前として扱わない
(除く)。

この試験は、Spark に 1 度も触らない (README を読み、`configs.toml` を読み込むだけ)。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from serving_kit import config as c

# --- コミットしたファイルの場所 ----------------------------------------

SERVING_DIR = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT = SERVING_DIR.parent
CONFIGS_PATH = SERVING_DIR / "config" / "configs.toml"

README_PATHS = (
    SERVING_DIR / "README.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.ja.md",
)
"""構成名の実在を確かめる対象の 3 本 (issue #103 で、ルートの 2 本も対象に加えた)。"""

# --- README のコード塊からの抽出 ----------------------------------------

_BASH_FENCE = re.compile(r"```bash\n(.*?)```", re.DOTALL)
"""README の ```bash ... ``` の塊 (中身だけを取る)。"""

_CONFIG_TAKING_SUBCOMMANDS = frozenset(
    {
        "check",
        "pull-image",
        "image-licenses",
        "fetch",
        "verify",
        "start",
        "smoke",
        "logs",
        "probe",
        "watch",
        "thinking",
    }
)
"""`serve <サブコマンド> <構成名>` の形で、直後の引数が構成名になるサブコマンド。

`serving/README.md` の「サブコマンド」の表 (`serve <cmd> <構成>` の形の行) から取った。
"""

_NETCHECK_CONFIG_TAKING_SUBSUBCOMMANDS = frozenset({"bandwidth", "sanity", "ab"})
"""`serve netcheck <この 1 つ> <構成名>` の形になる、netcheck の子のサブコマンド。

`netcheck links` は構成名を取らないので、ここに入れない。
"""


def _is_placeholder(token: str) -> bool:
    """`<構成>` のような、山括弧で書いた置き場所かどうか。"""
    return token.startswith("<") and token.endswith(">")


def _bash_blocks(readme_text: str) -> list[str]:
    return _BASH_FENCE.findall(readme_text)


def _config_names_referenced(readme_text: str) -> list[str]:
    """README のコード塊に現れる、`serve <サブコマンド> <構成名>` の構成名を、順に並べて返す。"""
    names: list[str] = []
    for block in _bash_blocks(readme_text):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if "serve" not in tokens:
                continue
            rest = tokens[tokens.index("serve") + 1 :]
            if not rest:
                continue
            subcommand = rest[0]
            if subcommand == "netcheck":
                if len(rest) < 3 or rest[1] not in _NETCHECK_CONFIG_TAKING_SUBSUBCOMMANDS:
                    continue
                candidate = rest[2]
            elif subcommand in _CONFIG_TAKING_SUBCOMMANDS:
                if len(rest) < 2:
                    continue
                candidate = rest[1]
            else:
                continue
            if candidate.startswith("-") or _is_placeholder(candidate):
                continue
            names.append(candidate)
    return names


@pytest.mark.parametrize(
    "readme_path", README_PATHS, ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_readme_bash_blocks_reference_real_config_names(readme_path: Path) -> None:
    """README のコード塊の構成名は、すべて `config.load_configs` で読める実物に実在する。"""
    readme_text = readme_path.read_text(encoding="utf-8")
    referenced = _config_names_referenced(readme_text)
    assert referenced, (
        f"{readme_path} のコード塊から、serve <サブコマンド> <構成名> の形を 1 つも取れなかった"
    )

    configs = c.load_configs(CONFIGS_PATH, REPO_ROOT)
    missing = sorted({name for name in referenced if name not in configs})
    assert not missing, f"{readme_path} に、実在しない構成名がある: {missing}"


def test_the_root_readmes_start_the_marlin_config() -> None:
    """ルートの README 2 本は、起動の手順のコード塊で `glm53-tp2-mtp3-marlin` を使う

    (issue #103: 45 tok/s に届いた構成)。
    """
    for readme_path in (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md"):
        referenced = _config_names_referenced(readme_path.read_text(encoding="utf-8"))
        assert "glm53-tp2-mtp3-marlin" in referenced, (
            f"{readme_path} の起動の手順のコード塊に glm53-tp2-mtp3-marlin がない"
        )
