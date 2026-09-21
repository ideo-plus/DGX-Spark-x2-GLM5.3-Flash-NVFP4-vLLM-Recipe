"""直結のインターフェースの読み取り (design.md 「確認 › netcheck」の `serve netcheck links`)。

受け持つのは、`serve netcheck links` の中身 (tasks.md 4.3) だけである。帯域の計測、事前の
確認、A/B の比較 (`serve netcheck bandwidth` / `sanity`) は 4.4 の仕事で、同じ `netcheck.py`
にあとで足されるが、ここでは書かない (`_Boundary: netcheck_` のうち、`links` の節だけ)。

## すること

2 台の Spark それぞれで、**読み取りだけ**を行う (`runner.run(..., mutating=False)`。状態を
変える呼び出しを 1 つも出さない。了承も要らない):

1. `ip -br link` でインターフェースの名前と状態、`ip -br addr` でアドレスを読む
2. 管理の側 (ノードの定義の `lan_addr` が付いているインターフェース)、ループバック、
   `docker0` / `br-*` / `veth*` / `tailscale0` / 無線 (`wl*`) を除いた残りを、直結の候補に
   する (下の「候補の決め方」)
3. 候補それぞれについて、MTU (`cat /sys/class/net/<if>/mtu`)、速さと接続 (`ethtool <if>`)
   を読む
4. `ibdev2netdev` で、候補と RoCE のデバイスの対応を読む。あれば `ibv_devinfo` も読み、
   `ibdev2netdev` が挙げたデバイスが、そこにも現れることを確かめる (裏付けの読み取り。
   `ibv_devinfo` の GUID や `node_guid` は、機械を識別できる値なので、結果には入れない)
5. つながっている候補の数から、ケーブルの本数を判断する (下の「ケーブルの本数の判断」)

**入っていない道具** (`ibdev2netdev` / `ibv_devinfo` / `ethtool` が `command not found` =
終了コード 127) は、入れずに `LinkReport.tools_missing` に記録し、読み取りと判断を続ける
(requirements 4.1、design.md 「netcheck」)。**`ethtool` の標準エラーの
`netlink error: Operation not permitted` は、誤りとして扱わない** (終了コードは 0 のまま。
tasks.md の Implementation Notes 「1.6 の見本の追加」: root でなくても `ethtool` は 0 で
終わる)。片方の台に入れない (`remote.RemoteError`)、コマンドが 0 以外で終わる、出力が
読めないのそれぞれでも、落ちずに、その台のその項目を「読めなかった」に倒す。

## 本数を確定する条件 (安全の側に倒す)

**本数を確定するのは、2 台それぞれで、(i) インターフェースの一覧、(ii) アドレス (管理の側の
見分け)、(iii) つながりの判定、が、すべて読めたときだけである。読めなかったものがあれば、
読めたところまでを `detail` に示して、判断しない。**

とくに (ii) は、単に読めれば足りるのではなく、**`node.lan_addr` に一致するインターフェースを
実際に 1 つ以上見つけられたときだけ**「見分けられた」とする。`ip -br addr` が読めない
(`RemoteError`、0 以外の終了、空の出力) と、この module の管理の側の見分け方 (`_management_names`。
名前を決め打ちせず、`lan_addr` に一致するアドレスだけで見分ける) が働かず、管理の側のインター
フェースが、直結の候補に紛れ込みかねない (差し戻しの原因になった不具合: 直結の側が実際には
1 つしかつながっていない状況で、管理の側が「つながっている候補」に数えられ、`cable_count=1`
という、誤った確定の結果が返っていた)。`lan_addr` が、どのインターフェースにも付いていない
台 (ノードの定義の誤り、または別の経路で入っている) も、同じ理由で見分けの前提が崩れている
ので、同じ扱いにする。このいずれかに当たる台は、`LinkReport.interfaces` を空にし、
`cable_count` も「埋める候補」も出さない (`_unconfirmed_report`)。

## 候補の決め方

`ibdev2netdev` に現れるものを直結の候補にするのが素直だが、**`ibdev2netdev` が入っていない
台でも判断できるように**、候補は `ibdev2netdev` の有無に関わらず、構造 (名前と、管理の側の
アドレス) だけで決める。読み取れた `ibdev2netdev` は、決まった候補に RoCE のデバイスを
対応づけるためだけに使う (tasks.md 4.3 の要点)。

DGX Spark の物理のインターフェースは、NVIDIA の DGX Spark ユーザーガイド (ConnectX-7
Networking) の対応表 (research.md §e-1) のとおり、管理の 1 つ (10GbE。ノードの定義の
`lan_addr`) と、直結の側の 4 つ (`en{p1s0f0,P2p1s0f0,p1s0f1,P2p1s0f1}np{0,1}`) である。
仮想のインターフェース (`docker0`、`br-*`、`veth*`、`tailscale0`) と無線 (`wl*`) は、直結に
使わない。管理の側は、名前ではなく `lan_addr` に一致するアドレスで見分ける (実機のインター
フェースの名前 (`enP7s7`) を、この module が決め打ちしないため)。これらを除いた残りを候補と
する。

## ケーブルの本数の判断

NVIDIA の DGX Spark ユーザーガイドの原文 (research.md §e-1): **"Each QSFP port appears as
two independent Linux Ethernet interfaces. As a result, plugging in two cables shows a total
of four Linux Ethernet interfaces."** ここから、つながっている直結の候補が **2 つならケーブル
1 本、4 つなら 2 本**と判断する (tasks.md 4.3)。0、1、3、それ以外の数のときは、本数を決め
つけず、`cable_count` を空にして、`detail` に「判断できない」旨と、数えたインターフェースを
書く。2 台の判断が食い違うとき (`read_link_reports` が両方を読んだとき) は、両台の `detail`
に、そのことを書き添える。

**つながっているかどうかの判定は、`lifecycle.parse_link_state` を使い回す** (二重に持たない。
tasks.md の Implementation Notes 3.5)。`ip -br link` の 2 列目 (`UP` / `DOWN`) を主に使い、
`ethtool` の `Link detected` で裏付ける。**この 2 つの出どころが、両方読めて食い違うとき
(例: `ip` は `UP`、`ethtool` は `Link detected: no`) は、その候補のつながりが確定できない
ので、台全体のケーブルの本数を判断しない** (`connectivity_uncertain`)。`ethtool` が読めず、
`ip` の状態だけで判断した候補は、そのことを `detail` に書く (どちらで判断したかを、つねに
たどれるようにする)。`lifecycle.read_fabric_link` は、1 つの名前ごとに
`ip -br link show dev <名前>` を遠隔で流すので、ここでは使わない (直結の候補は複数あるので、
1 回だけ読んだ `ip -br link` の出力を、`parse_link_state` で名前ごとに読み直すほうが、遠隔の
呼び出しが少ない)。

## ノードの定義との突き合わせ

`node.fabric_ifname` / `fabric_addr` が埋まっていれば、読み取った結果と突き合わせ、名前が
見つからない・つながっていない・アドレスが違う、のそれぞれを `detail` に書く。**`fabric_ifname`
が、管理の側のインターフェースの名前を指しているとき (設定の誤り) は、「候補の一覧にない」
という汎用の文ではなく、「管理の側のインターフェースである」と言う。** `fabric_ifname` が
空であれば、つながっている候補の名前とアドレスを、埋める候補として `detail` に示す。
**この module は、ノードの定義のファイルを書き換えない** (7.3 が、この結果を見て、人が
`nodes.toml` を書く)。「本数を確定する条件」が崩れている台 (`_unconfirmed_report`) では、
`fabric_ifname` / `fabric_addr` の突き合わせそのものができないので、**「合っている」とは
言わず、「確かめられなかった」と言う**。

## 依存の向きと、書かないもの

`netcheck` は `types`、`config`、`remote`、`plan`、`observe`、`guards`、`logs`、`lifecycle`
まで読み込める。`probe`、`watch`、`thinking`、`cli` (同じ層、または入口) は読み込まない。
帯域の計測 (`serve netcheck bandwidth`)、事前の確認 (`serve netcheck sanity`)、A/B の比較は
4.4 の仕事で、ここには書かない。`cli.py` へのつなぎ込みは 5.1 の仕事である。

## 見せてよい値

要約に出すのは、名前・状態・MTU・速さ・アドレス・RoCE のデバイス名・ケーブルの本数だけ。
MAC アドレスや GUID など、アドレス以外の機械を識別できる値は、そもそも `InterfaceLink` に
持たせていないので、`format_link_report` にも出てこない (design.md 「netcheck」)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from serving_kit.guards import READ_TIMEOUT_S
from serving_kit.lifecycle import parse_link_state
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import CommandResult, InterfaceLink, LinkReport, NodeDef, NodeRole

__all__ = ["format_link_report", "read_link_report", "read_link_reports"]


# --- 候補の決め方 -----------------------------------------------------------

_EXCLUDED_NAMES: Final[frozenset[str]] = frozenset({"lo", "docker0", "tailscale0"})
"""名前がちょうど一致すれば、直結の候補から除くもの。"""

_EXCLUDED_PREFIXES: Final[tuple[str, ...]] = ("br-", "veth", "wl")
"""名前がこの接頭辞で始まれば、直結の候補から除くもの (ブリッジ、veth の対、無線)。"""

_MISSING_TOOL_EXIT_CODE: Final[int] = 127
"""`command not found` の終了コード (道具が入っていないことの印。tasks.md 4.3 の要点)。"""

_STATE_UP: Final[str] = "UP"
_STATE_DOWN: Final[str] = "DOWN"
_STATE_UNKNOWN: Final[str] = "UNKNOWN"
"""`InterfaceLink.state` に書く文字列 (`ip -br link` の 2 列目と同じ語)。"""

_CABLE_COUNT_BY_CONNECTED: Final[Mapping[int, int]] = {2: 1, 4: 2}
"""つながっている候補の数から、ケーブルの本数へ (research.md §e-1。module docstring の
「ケーブルの本数の判断」)。"""

_ETHTOOL_SPEED_RE: Final[re.Pattern[str]] = re.compile(r"^\s*Speed:\s*(\d+)Mb/s\s*$", re.MULTILINE)
_ETHTOOL_LINK_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Link detected:\s*(yes|no)\b", re.MULTILINE
)
_IBDEV2NETDEV_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\S+)\s+port\s+\d+\s+==>\s+(\S+)\s+\((?:Up|Down)\)\s*$", re.MULTILINE
)
_HCA_ID_PREFIX: Final[str] = "hca_id:"


def _is_excluded_by_name(name: str) -> bool:
    """名前だけで、直結の候補から除けるかどうか (管理の側は、アドレスで別に見る)。"""
    return name in _EXCLUDED_NAMES or name.startswith(_EXCLUDED_PREFIXES)


def _management_names(node: NodeDef, addr_by_name: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """ノードの定義の `lan_addr` が付いているインターフェースの名前。"""
    lan_addr = str(node.lan_addr)
    return frozenset(
        name
        for name, addrs in addr_by_name.items()
        if any(addr.split("/", 1)[0] == lan_addr for addr in addrs)
    )


def _candidate_names(
    names: Sequence[str], node: NodeDef, addr_by_name: Mapping[str, tuple[str, ...]]
) -> list[str]:
    """直結の候補の名前を、読み取った順のまま選ぶ (module docstring の「候補の決め方」)。"""
    management = _management_names(node, addr_by_name)
    return [name for name in names if name not in management and not _is_excluded_by_name(name)]


# --- `ip -br` の出力の読み取り -----------------------------------------------


def _link_names(text: str) -> list[str]:
    """`ip -br link` の出力から、インターフェースの名前を、現れた順のまま拾う (重複なし)。

    `veth0000000@if2` のような対の印は、`lifecycle.parse_link_state` と同じく `@` の前まで
    で見る。
    """
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("@", 1)[0]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _addr_rows(text: str) -> dict[str, tuple[str, ...]]:
    """`ip -br addr` の出力から、名前ごとのアドレスの列を読む (アドレスがなければ空)。"""
    rows: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("@", 1)[0]
        rows[name] = tuple(fields[2:])
    return rows


def _link_state_label(linked: bool | None) -> str:
    """`lifecycle.parse_link_state` の判定を、`InterfaceLink.state` の文字列にする。"""
    if linked is True:
        return _STATE_UP
    if linked is False:
        return _STATE_DOWN
    return _STATE_UNKNOWN


# --- `ethtool` / `ibdev2netdev` / `ibv_devinfo` の出力の読み取り -----------------


def _parse_ethtool(text: str) -> tuple[int | None, bool | None]:
    """`ethtool <if>` から、ネゴシエートした速さ (Mb/s) と、接続の有無を読む。

    つながっていないと `Speed: Unknown!` になるので、その場合は空を返す (見本
    `tests/fixtures/spark/*/ethtool-enp1s0f1np1.txt`)。
    """
    speed_match = _ETHTOOL_SPEED_RE.search(text)
    speed = int(speed_match.group(1)) if speed_match else None
    link_match = _ETHTOOL_LINK_RE.search(text)
    link_detected = link_match.group(1) == "yes" if link_match else None
    return speed, link_detected


def _parse_ibdev2netdev(text: str) -> dict[str, str]:
    """`ibdev2netdev` の出力から、インターフェースの名前と RoCE のデバイスの対応を読む。

    1 行の形は `<RoCE のデバイス> port <番号> ==> <インターフェース> (Up|Down)`
    (見本 `tests/fixtures/spark/*/ibdev2netdev.txt`)。`(Up|Down)` は、ここでは使わない
    (つながっているかどうかは `ip -br link` で決める。module docstring の「ケーブルの本数の
    判断」)。
    """
    return {ifname: device for device, ifname in _IBDEV2NETDEV_RE.findall(text)}


def _parse_ibv_devinfo_hca_ids(text: str) -> frozenset[str]:
    """`ibv_devinfo` の出力から、`hca_id` の名前の集まりを読む (裏付けの読み取りに使う)。"""
    ids: set[str] = set()
    for line in text.splitlines():
        if line.startswith(_HCA_ID_PREFIX):
            name = line[len(_HCA_ID_PREFIX) :].strip()
            if name:
                ids.add(name)
    return frozenset(ids)


# --- 遠隔の読み取りの小さな助け ----------------------------------------------


def _shown(result: CommandResult) -> str:
    """失敗した呼び出しの、見せる文 (標準エラーがなければ標準出力)。"""
    return result.stderr.strip() or result.stdout.strip() or "(出力なし)"


def _run_read(
    runner: RemoteRunner, node: NodeDef, argv: tuple[str, ...], timeout_s: float
) -> tuple[CommandResult | None, str]:
    """読み取りを 1 つ流す。届かなかったこと (`RemoteError`) は、投げずに理由の文字列で返す。"""
    try:
        return runner.run(node, argv, timeout_s=timeout_s, mutating=False), ""
    except RemoteError as exc:
        return None, str(exc)


def _read_tool(
    runner: RemoteRunner,
    node: NodeDef,
    argv: tuple[str, ...],
    timeout_s: float,
    *,
    tool_label: str,
    problems: list[str],
    tools_missing: set[str],
) -> CommandResult | None:
    """入っていないかもしれない道具 (`ibdev2netdev` / `ibv_devinfo` / `ethtool`) を読む。

    終了コード 127 (`command not found`) は「入っていない」として `tools_missing` に記録し、
    誤りにしない (requirements 4.1)。それ以外の失敗は「読めなかった」として `problems` に
    書く。
    """
    result, err = _run_read(runner, node, argv, timeout_s)
    if result is None:
        problems.append(f"{tool_label} に届かない ({err})")
        return None
    if result.exit_code == _MISSING_TOOL_EXIT_CODE:
        tools_missing.add(tool_label)
        return None
    if result.exit_code != 0:
        problems.append(f"{tool_label} を読めなかった ({_shown(result)})")
        return None
    return result


def _read_mtu(
    runner: RemoteRunner, node: NodeDef, name: str, timeout_s: float, problems: list[str]
) -> int | None:
    """`cat /sys/class/net/<if>/mtu` で MTU を読む (`remote` の許可の形を通る)。"""
    result, err = _run_read(runner, node, ("cat", f"/sys/class/net/{name}/mtu"), timeout_s)
    if result is None:
        problems.append(f"{name}: MTU に届かない ({err})")
        return None
    if result.exit_code != 0:
        problems.append(f"{name}: MTU を読めなかった ({_shown(result)})")
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        problems.append(f"{name}: MTU の出力を数として読めない ({result.stdout.strip()!r})")
        return None


def _read_ethtool(
    runner: RemoteRunner,
    node: NodeDef,
    name: str,
    timeout_s: float,
    *,
    problems: list[str],
    tools_missing: set[str],
) -> tuple[int | None, bool | None]:
    """`ethtool <if>` で速さと接続を読む。

    標準エラーの `netlink error: Operation not permitted` は誤りにしない (`_read_tool` は
    終了コードしか見ず、標準エラーの中身を検査しない。tasks.md の Implementation Notes
    「1.6 の見本の追加」)。
    """
    result = _read_tool(
        runner,
        node,
        ("ethtool", name),
        timeout_s,
        tool_label="ethtool",
        problems=problems,
        tools_missing=tools_missing,
    )
    if result is None:
        return None, None
    return _parse_ethtool(result.stdout)


# --- ケーブルの本数の判断、ノードの定義との突き合わせ -----------------------------


def _judge_cable_count(connected_names: Sequence[str]) -> tuple[int | None, str]:
    """つながっている候補の数から、ケーブルの本数を判断する (module docstring の該当節)。"""
    count = len(connected_names)
    named = "、".join(connected_names) if connected_names else "なし"
    basis = (
        f"つながっている直結の候補 {count} 個 ({named})。NVIDIA の DGX Spark ユーザーガイド "
        "(ConnectX-7 Networking): 1 つの QSFP ポートが、2 つの Linux のインターフェースとして"
        "見える (research.md §e-1)"
    )
    cables = _CABLE_COUNT_BY_CONNECTED.get(count)
    if cables is None:
        return None, (
            f"ケーブルの本数を判断できない ({basis}。2 個なら 1 本、4 個なら 2 本と判断できるが、"
            f"{count} 個では決めつけない)"
        )
    return cables, f"ケーブルは {cables} 本と判断した ({basis})"


def _fabric_note(
    node: NodeDef, interfaces: Sequence[InterfaceLink], management_names: frozenset[str]
) -> str:
    """ノードの定義の `fabric_ifname` / `fabric_addr` と、読み取った結果を突き合わせる。

    埋まっていれば食い違いを、空であれば埋める候補を返す (module docstring の「ノードの定義
    との突き合わせ」。この関数はノードの定義を書き換えない)。`fabric_ifname` が管理の側の
    名前を指しているとき (設定の誤り) は、「候補の一覧にない」という汎用の文ではなく、
    「管理の側のインターフェースである」と言う (レビューの指摘 3)。
    """
    by_name = {link.name: link for link in interfaces}
    if node.fabric_ifname is not None:
        if node.fabric_ifname in management_names:
            return (
                f"fabric_ifname ('{node.fabric_ifname}') は、管理の側のインターフェースで"
                "ある (直結には使えない)"
            )
        found = by_name.get(node.fabric_ifname)
        if found is None:
            return (
                f"ノードの定義の fabric_ifname ('{node.fabric_ifname}') に当たる直結の候補が、"
                "読み取った一覧にない"
            )
        mismatches: list[str] = []
        if found.state != _STATE_UP:
            mismatches.append(f"つながっていない (状態: {found.state})")
        if node.fabric_addr is not None:
            expected = str(node.fabric_addr)
            if not any(addr.split("/", 1)[0] == expected for addr in found.addrs):
                mismatches.append(f"アドレスが fabric_addr ('{expected}') と違う")
        if mismatches:
            return (
                f"fabric_ifname ('{node.fabric_ifname}') が、読み取った結果と食い違う: "
                + "、".join(mismatches)
            )
        return f"fabric_ifname ('{node.fabric_ifname}') は、読み取った結果と合っている"
    connected = [link for link in interfaces if link.state == _STATE_UP]
    if not connected:
        return "fabric_ifname が空で、つながっている直結の候補もない (埋める候補を示せない)"
    candidates_text = "、".join(
        f"{link.name} ({link.addrs[0]})" if link.addrs else link.name for link in connected
    )
    return f"fabric_ifname が空。埋める候補: {candidates_text}"


def _unconfirmed_report(node: NodeDef, reason: str) -> LinkReport:
    """管理の側を見分けられなかったときの結果 (レビューの指摘 1。安全の側に倒す)。

    `ip -br addr` が読めない、または `lan_addr` に一致するインターフェースが 1 つも見つから
    ないと、管理の側を、名前ではなくアドレスで見分けるこの module の仕組みが働かない。その
    状態で候補を決めると、管理の側のインターフェースが、直結の候補に紛れ込みかねない
    (レビュー担当の再現: 直結の側が実際には 1 つしかつながっていないのに、管理の側が候補に
    数えられて `cable_count=1` という、誤った確定の結果が返っていた)。そこで、直結の候補
    (`interfaces`)、「埋める候補」、ケーブルの本数のどれも出さない。`fabric_ifname` /
    `fabric_addr` の突き合わせも、確かめられない (レビューの指摘 2: 「合っている」とは
    言わない)。
    """
    parts = [reason]
    if node.fabric_ifname is not None:
        parts.append(f"fabric_ifname ('{node.fabric_ifname}') との突き合わせも確かめられなかった")
    return LinkReport(node=node.role, detail=" / ".join(parts))


# --- 公開の口 ----------------------------------------------------------------


def read_link_report(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> LinkReport:
    """1 台ぶんの直結のリンクを読み取る (`serve netcheck links` の中身。requirements 4.1、4.2)。

    読み取りだけである (状態を変える呼び出しを 1 つも出さない。了承も要らない)。**本数を確定
    するのは、(i) インターフェースの一覧、(ii) アドレス (管理の側の見分け)、(iii) つながりの
    判定、の 3 つが、すべて読めたときだけである** (module docstring の「本数を確定する条件」。
    レビューの指摘 1 を受けた決まり)。どれか 1 つでも読めなければ、読めたところまでを
    `detail` に示して、判断しない。
    """
    problems: list[str] = []
    tools_missing: set[str] = set()

    # (i) インターフェースの一覧
    link_result, link_err = _run_read(runner, node, ("ip", "-br", "link"), timeout_s)
    if link_result is None:
        return LinkReport(
            node=node.role,
            detail=f"{node.role} ({node.ssh_host}) の ip -br link に届かない ({link_err})",
        )
    if link_result.exit_code != 0:
        return LinkReport(
            node=node.role, detail=f"ip -br link を読めなかった ({_shown(link_result)})"
        )

    # (ii) アドレス (管理の側の見分け)。読めない、または管理の側が 1 つも見つからないと、
    # 直結の候補を安全に決められない (レビュー担当の再現: 管理の側が候補に紛れ込み、誤った
    # cable_count が確定していた)。そこで、直結の候補もケーブルの本数も出さずに終える
    addr_result, addr_err = _run_read(runner, node, ("ip", "-br", "addr"), timeout_s)
    if addr_result is None:
        return _unconfirmed_report(
            node,
            "ip -br addr に届かないので、管理の側のインターフェースを見分けられず、"
            f"直結の候補もケーブルの本数も判断しない ({addr_err})",
        )
    if addr_result.exit_code != 0:
        return _unconfirmed_report(
            node,
            "ip -br addr を読めなかったので、管理の側のインターフェースを見分けられず、"
            f"直結の候補もケーブルの本数も判断しない ({_shown(addr_result)})",
        )

    addr_by_name = _addr_rows(addr_result.stdout)
    management_names = _management_names(node, addr_by_name)
    if not management_names:
        return _unconfirmed_report(
            node,
            f"ノードの定義の lan_addr ('{node.lan_addr}') に一致するインターフェースが"
            "見つからないので、管理の側を見分けられず、直結の候補もケーブルの本数も判断しない"
            " (見分けの前提が崩れている)",
        )

    names = _link_names(link_result.stdout)
    candidates = _candidate_names(names, node, addr_by_name)

    roce_by_name: dict[str, str] = {}
    ibdev_result = _read_tool(
        runner,
        node,
        ("ibdev2netdev",),
        timeout_s,
        tool_label="ibdev2netdev",
        problems=problems,
        tools_missing=tools_missing,
    )
    if ibdev_result is not None:
        roce_by_name = _parse_ibdev2netdev(ibdev_result.stdout)

    hca_ids: frozenset[str] = frozenset()
    ibv_result = _read_tool(
        runner,
        node,
        ("ibv_devinfo",),
        timeout_s,
        tool_label="ibv_devinfo",
        problems=problems,
        tools_missing=tools_missing,
    )
    if ibv_result is not None:
        hca_ids = _parse_ibv_devinfo_hca_ids(ibv_result.stdout)

    interfaces: list[InterfaceLink] = []
    connected_names: list[str] = []
    # (iii) つながりの判定。`ip` の状態と `ethtool` の `Link detected` の、どちらで判断したか
    # を `detail` に残す。両方読めて食い違うときは、この台のケーブルの本数を確定しない
    connectivity_uncertain = False
    for name in candidates:
        linked = parse_link_state(link_result.stdout, name)
        state = _link_state_label(linked)
        mtu = _read_mtu(runner, node, name, timeout_s, problems)
        speed, link_detected = _read_ethtool(
            runner, node, name, timeout_s, problems=problems, tools_missing=tools_missing
        )
        if link_detected is None:
            problems.append(
                f"{name}: ethtool の Link detected を読めなかったので、ip の状態 ({state}) "
                "だけでつながりを判断した"
            )
        elif linked is not None and link_detected != linked:
            connectivity_uncertain = True
            problems.append(
                f"{name}: ip の状態 ({state}) と ethtool の Link detected "
                f"({'yes' if link_detected else 'no'}) が食い違う"
            )
        roce_device = roce_by_name.get(name)
        if roce_device is not None and hca_ids and roce_device not in hca_ids:
            problems.append(
                f"{name}: ibdev2netdev が挙げる RoCE のデバイス ('{roce_device}') が、"
                "ibv_devinfo の一覧にない"
            )
        interfaces.append(
            InterfaceLink(
                name=name,
                state=state,
                mtu=mtu,
                speed_mbps=speed,
                addrs=addr_by_name.get(name, ()),
                roce_device=roce_device,
            )
        )
        if linked is True:
            connected_names.append(name)

    cable_count: int | None
    if connectivity_uncertain:
        cable_count = None
        problems.append(
            "つながりの判定 (ip の状態と ethtool の Link detected) が食い違うインターフェース"
            "があるので、ケーブルの本数を判断しない"
        )
    else:
        cable_count, cable_note = _judge_cable_count(connected_names)
        problems.append(cable_note)
    problems.append(_fabric_note(node, interfaces, management_names))

    return LinkReport(
        node=node.role,
        interfaces=tuple(interfaces),
        cable_count=cable_count,
        tools_missing=tuple(sorted(tools_missing)),
        detail=" / ".join(text for text in problems if text),
    )


def read_link_reports(
    runner: RemoteRunner, nodes: Mapping[NodeRole, NodeDef], *, timeout_s: float = READ_TIMEOUT_S
) -> dict[NodeRole, LinkReport]:
    """2 台ぶんの直結のリンクを読み取る (`read_link_report` を役割ごとに呼ぶ)。

    2 台の判断 (`cable_count`) が食い違うときは、両台の `detail` に、そのことを書き添える
    (tasks.md 4.3 の要点)。
    """
    reports = {
        role: read_link_report(runner, node, timeout_s=timeout_s) for role, node in nodes.items()
    }
    return _note_cable_count_mismatch(reports)


def _note_cable_count_mismatch(
    reports: Mapping[NodeRole, LinkReport],
) -> dict[NodeRole, LinkReport]:
    """2 台の `cable_count` が食い違うとき、両台の `detail` にそのことを書き添える。"""
    counted = {
        role: report.cable_count
        for role, report in reports.items()
        if report.cable_count is not None
    }
    if len({*counted.values()}) <= 1:
        return dict(reports)
    updated: dict[NodeRole, LinkReport] = {}
    for role, report in reports.items():
        others = "、".join(
            f"{other_role}: {other_count} 本"
            for other_role, other_count in sorted(counted.items())
            if other_role != role
        )
        mine = f"{report.cable_count} 本" if report.cable_count is not None else "判断できない"
        note = f"2 台の判断が食い違っている ({role}: {mine}、{others})"
        updated[role] = report.model_copy(
            update={"detail": " / ".join(text for text in (report.detail, note) if text)}
        )
    return updated


def format_link_report(report: LinkReport) -> str:
    """`LinkReport` を、人が読める形にする (5.1 が画面に、7.3 が `docs/results/` の要約に使う
    助け)。アドレス以外の機械を識別できる値 (MAC、GUID) は、そもそも `InterfaceLink` に
    持たせていないので、ここにも出てこない。
    """
    lines = [f"{report.node}:"]
    if not report.interfaces:
        lines.append("  直結の候補のインターフェースを読めなかった")
    for link in report.interfaces:
        speed = f"{link.speed_mbps}Mb/s" if link.speed_mbps is not None else "不明"
        mtu = str(link.mtu) if link.mtu is not None else "不明"
        roce = link.roce_device or "対応するデバイスがない"
        addrs = ", ".join(link.addrs) if link.addrs else "(アドレスなし)"
        lines.append(f"  {link.name}: {link.state}, MTU {mtu}, {speed}, RoCE {roce}, {addrs}")
    cable = f"{report.cable_count} 本" if report.cable_count is not None else "判断できない"
    lines.append(f"  ケーブルの本数: {cable}")
    if report.tools_missing:
        lines.append(f"  入っていない道具: {', '.join(report.tools_missing)}")
    if report.detail:
        lines.append(f"  詳細: {report.detail}")
    return "\n".join(lines)
