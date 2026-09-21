"""2 台の Spark から採った見本 (`tests/fixtures/spark/`) の決まりを固定する (task 1.6)。

見本は、実機の出力を、機械を識別できる値だけ置き換えたもの。ここでは、あとで見本を足す
とき (7.1、7.2) にも、置き換えの漏れと、別の構成の中身の紛れ込みが起きないことを確かめる。
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spark"
ROLES = ("head", "worker")

# tasks.md 1.6 が「採るもの」として挙げる、読み取りのコマンドの見本
REQUIRED = (
    "uname-n.txt",
    "docker-version.txt",
    "docker-ps-own.txt",
    "nvidia-smi-compute-apps.txt",
    "nvidia-smi-gpu-util.txt",
    "df-avail.txt",
    "ip-br-link.txt",
    "ip-br-addr.txt",
    "ss-ltnH.txt",
)

_MAC_RE = re.compile(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", re.IGNORECASE)
_ALLOWED_MAC_RE = re.compile(r"^(00:00:00:00:00:00|02:00:00:00:00:[0-9a-f]{2})$")
# `::` の省略 (空の群) を含む形も拾う。MAC や時刻も当たるが、あとで ipaddress に通してふるう
_IPV6_RE = re.compile(
    r"(?<![0-9a-f:.])[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}(?![0-9a-f:.])", re.IGNORECASE
)
_TAILSCALE_V4_RE = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b")

# 見本に現れてよい IPv6 の範囲 (置き換えに使った文書用の範囲、ループバック、`[::]` の待ち受け)
_ALLOWED_V6 = tuple(
    ipaddress.ip_network(net) for net in ("2001:db8::/32", "fd00:db8::/32", "::1/128", "::/128")
)
# リンクローカルは、合成の値 `fe80::<16 進で 4 桁まで>` だけ。実物のリンクローカルは、
# modified EUI-64 (`…ff:fe…`) なら MAC をそのまま含み、そうでなくても機械に固有の値である
_LINK_LOCAL = ipaddress.ip_network("fe80::/10")
_SYNTHETIC_LINK_LOCAL_MAX = 0xFFFF

# 別の構成の中身、認証の情報が紛れ込んでいないことを見る語
_FORBIDDEN_WORDS = ("exl3", "hf_", "token", "password", "secret", "authorization", "bearer")


def _files() -> list[Path]:
    return sorted(p for p in FIXTURES.rglob("*.txt"))


def test_both_nodes_have_every_required_sample() -> None:
    for role in ROLES:
        for name in REQUIRED:
            assert (FIXTURES / role / name).is_file(), f"{role}/{name} がない"
        assert list((FIXTURES / role).glob("ethtool-*.txt")), f"{role} に ethtool の見本がない"


def test_both_nodes_have_the_same_set_of_samples() -> None:
    head, worker = ({p.name for p in (FIXTURES / role).glob("*.txt")} for role in ROLES)
    assert head == worker


def test_the_container_list_was_taken_with_the_label_filter_and_is_empty() -> None:
    # 1.6 の時点では、この道具のコンテナはない。絞らない一覧を採っていれば、ここが空にならない
    for role in ROLES:
        assert (FIXTURES / role / "docker-ps-own.txt").read_text() == ""


@pytest.mark.parametrize("path", _files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_sample_has_no_machine_identifiers(path: Path) -> None:
    text = path.read_text()
    for mac in _MAC_RE.findall(text):
        assert _ALLOWED_MAC_RE.match(mac.lower()), f"置き換えていない MAC アドレス: {mac}"
    for found in _IPV6_RE.findall(text):
        if _MAC_RE.fullmatch(found):
            continue
        try:
            addr = ipaddress.ip_address(found)
        except ValueError:
            continue  # 時刻や版の番号など、アドレスでないもの
        if addr in _LINK_LOCAL:
            host = int(addr) - int(_LINK_LOCAL.network_address)
            assert host <= _SYNTHETIC_LINK_LOCAL_MAX, f"置き換えていないリンクローカル: {found}"
            continue
        assert any(addr in net for net in _ALLOWED_V6), f"置き換えていない IPv6 アドレス: {found}"
    for found in _TAILSCALE_V4_RE.findall(text):
        assert found.startswith("100.64.0."), f"置き換えていない Tailscale のアドレス: {found}"
    assert not re.search(r"br-(?!0{12})[0-9a-f]{12}", text), "置き換えていないブリッジの名前"


@pytest.mark.parametrize("path", _files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_sample_has_no_foreign_content_or_credentials(path: Path) -> None:
    lowered = path.read_text().lower()
    for word in _FORBIDDEN_WORDS:
        assert word not in lowered, f"{word} が含まれている"


def test_every_address_in_the_address_list_is_a_valid_address() -> None:
    # 置き換えで、アドレスの形を壊していないこと (読み取りの部品が ipaddress で読めること)
    for role in ROLES:
        for token in (FIXTURES / role / "ip-br-addr.txt").read_text().split():
            if "/" in token:
                ipaddress.ip_interface(token)


def test_the_port_list_keeps_its_columns() -> None:
    # ss は、アドレスの列を右に寄せて出す。置き換えで桁をずらしていないこと
    for role in ROLES:
        for line in (FIXTURES / role / "ss-ltnH.txt").read_text().splitlines():
            fields = line.split()
            assert fields[0] == "LISTEN"
            assert len(fields) == 5
            assert fields[3].rsplit(":", 1)[1].isdigit()


@pytest.mark.parametrize(
    "leaked",
    [
        "fe80::a8ec:75ff:fe32:d464",  # modified EUI-64 (MAC を含む)
        "fe80::a931:dd36:3b08:9a33",  # stable-privacy (機械に固有)
        "240f:71:e460:3133:eef0:71f5:1bb7:c421",  # グローバル
        "4c:bb:47:e9:15:3d",  # MAC
        "100.109.104.27",  # Tailscale
    ],
)
def test_the_identifier_check_catches_a_real_looking_value(tmp_path: Path, leaked: str) -> None:
    # 検査そのものが空振りでないこと (置き換えていない値を足した見本は、落ちる)
    sample = tmp_path / "ip-br-addr.txt"
    sample.write_text(f"eth0             UP             10.0.0.1/24 {leaked}/64 \n")
    with pytest.raises(AssertionError, match="置き換えていない"):
        test_a_sample_has_no_machine_identifiers(sample)
