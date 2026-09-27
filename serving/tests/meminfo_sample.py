"""`/proc/meminfo` の合成の本文と、台本の規則 (関門 `memory_free` の試験が共有する)。

**これは実機から採った見本ではない**。Spark の `/proc/meminfo` は、まだ採っていない
(`tests/fixtures/spark/` は実機の見本の置き場なので、ここには置かない)。行の形
(`<鍵>:  <数> kB`。単位のない行もある) は、Linux の `/proc/meminfo` の書式に沿って書いた。

台本の規則は、`cat /proc/meminfo` だけに答える。ほかの `cat` の規則 (照合の記録の読み取りなど) は、
`prefix=("cat",)` で何にでも当たるので、**それらより前**に置くこと (`fake_runner.py` の決まり:
上から順に見て、最初に当たった規則を使う)。
"""

from __future__ import annotations

from fake_runner import Reply, Rule
from serving_kit.types import NodeRole

__all__ = [
    "EXHAUSTED_AVAILABLE_KB",
    "EXHAUSTED_CACHED_KB",
    "EXHAUSTED_FREE_KB",
    "EXHAUSTED_MEMINFO",
    "HEALTHY_AVAILABLE_KB",
    "HEALTHY_CACHED_KB",
    "HEALTHY_FREE_KB",
    "HEALTHY_MEMINFO",
    "MEMINFO_PATH",
    "kb_in_bytes_text",
    "meminfo_rule",
    "meminfo_text",
]

MEMINFO_PATH = "/proc/meminfo"

_MEM_TOTAL_KB = 131_072_000
_SWAP_CACHED_KB = 24_680
"""`Cached` と別の値にしておく (`SwapCached` の行を、`Cached` と読み違える実装を見つけるため)。"""


def meminfo_text(*, mem_free_kb: int, mem_available_kb: int, cached_kb: int) -> str:
    """3 つの値 (kB) を持つ `/proc/meminfo` の本文。"""
    rows: tuple[tuple[str, int | None], ...] = (
        ("MemTotal", _MEM_TOTAL_KB),
        ("MemFree", mem_free_kb),
        ("MemAvailable", mem_available_kb),
        ("Buffers", 524_288),
        ("Cached", cached_kb),
        ("SwapCached", _SWAP_CACHED_KB),
        ("Active", 8_388_608),
        ("Inactive", 40_960_000),
        ("HugePages_Total", None),
        ("Hugepagesize", 2_048),
    )
    lines = []
    for key, value in rows:
        if value is None:
            lines.append(f"{key + ':':<16}{0:>9}")
        else:
            lines.append(f"{key + ':':<16}{value:>9} kB")
    return "\n".join(lines) + "\n"


HEALTHY_FREE_KB = 62_914_560
HEALTHY_AVAILABLE_KB = 100_663_296
HEALTHY_CACHED_KB = 31_457_280
HEALTHY_MEMINFO = meminfo_text(
    mem_free_kb=HEALTHY_FREE_KB,
    mem_available_kb=HEALTHY_AVAILABLE_KB,
    cached_kb=HEALTHY_CACHED_KB,
)
"""空きが十分ある台 (`MemFree` 60 GiB)。"""

EXHAUSTED_FREE_KB = 1_310_720
EXHAUSTED_AVAILABLE_KB = 116_654_080
EXHAUSTED_CACHED_KB = 115_343_360
EXHAUSTED_MEMINFO = meminfo_text(
    mem_free_kb=EXHAUSTED_FREE_KB,
    mem_available_kb=EXHAUSTED_AVAILABLE_KB,
    cached_kb=EXHAUSTED_CACHED_KB,
)
"""ページキャッシュが埋まった台 (`MemFree` 1.25 GiB、`Cached` 110 GiB)。`MemAvailable` は、
キャッシュが取り戻せるので大きい。"""


def kb_in_bytes_text(kilobytes: int) -> str:
    """kB の値を、`guards._bytes_text` が書く「バイト数の桁区切り」の形にする。

    関門の `detail` は、値を人が読める形と、そのままのバイト数 (桁区切り) の両方で示す
    (`test_guards.py` の `disk_space` の試験と同じ確かめ方)。
    """
    return f"{kilobytes * 1024:,}"


def meminfo_rule(text: str = HEALTHY_MEMINFO, *, node: NodeRole | None = None) -> Rule:
    """`cat /proc/meminfo` に、`text` を返す規則 (`node` を書かなければ、両方の台)。"""
    return Rule(prefix=("cat", MEMINFO_PATH), node=node, replies=(Reply(stdout=text),))
