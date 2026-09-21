"""直結のインターフェースの読み取りの試験 (tasks.md 4.3、`_Boundary: netcheck_` の `links` の節)。

確かめること (tasks.md 4.3 の完了の状態、要点):

- 見本の出力 (つながっている直結の候補が 2 つ) から、2 台とも、ケーブルの本数が 1 と判断される。
  名前、状態、MTU 9000、速さ 200000Mb/s、アドレス、RoCE のデバイスとの対応が結果に入る
- 見本の `DOWN` の 2 つを `UP` とつながった形に、試験の中で書き換えた台本 (見本のファイルは
  変えない) では、ケーブルの本数が 2 と判断される
- つながっている候補が 0、1、3 のときは、本数を決めつけない
- 読み取りだけで、状態を変える呼び出しが出ない (`mutating=True` が記録に現れない)
- `ibdev2netdev` がない台 (終了コード 127) で、入れずに `tools_missing` に記録し、判断は続く
- `ethtool` の標準エラー (`netlink error: Operation not permitted`) を誤りとして扱わない
- 片方の台に入れない (`RemoteError`) ときも、落ちずに「読めなかった」を返す
- 流すコマンドは、すべて `remote.CallGuard` の許可の形を通る (実物の `CallGuard` を共有する
  `FakeRunner` で断られない)
- 管理の側、`docker0`、`br-*`、`veth*`、`tailscale0`、無線が、直結の候補に入らない
- ノードの定義の `fabric_ifname` / `fabric_addr` との突き合わせ (合っている、食い違う、
  つながっていない、名前がない、空なら埋める候補を示す)
- 2 台の判断が食い違うとき、そのことを示す

**レビューの指摘 (task 4.3、差し戻し) を受けて足したもの**: 本数を確定するのは、(i) インター
フェースの一覧、(ii) アドレス (管理の側の見分け)、(iii) つながりの判定、の 3 つが、すべて
読めたときだけである。

- `ip -br addr` が読めない (`RemoteError`、0 以外の終了、空の出力) と、管理の側 (見本では
  `enP7s7`) を見分けられないので、直結の候補にも「埋める候補」にも出さず、`cable_count` を
  `None` にする (指摘 1、Critical。レビュー担当の再現: 直結の側が実際には 1 つしかつながって
  いない状況で、`enP7s7` が「つながっている候補」に紛れ込み、`cable_count=1` という誤った
  確定の結果が返っていた)
- 同じ考え方で、`lan_addr` に一致するインターフェースが 1 つも見つからない台でも、本数を
  確定しない
- `fabric_ifname` / `fabric_addr` が埋まっているのに `ip -br addr` が読めないときは、「合って
  いる」と言わず、「確かめられなかった」と言う (指摘 2)
- `fabric_ifname` が、管理の側のインターフェースの名前を指しているときは、「候補の一覧に
  ない」ではなく、「管理の側のインターフェースである」と言う (指摘 3)
- `ip` の状態と `ethtool` の `Link detected` が食い違うインターフェースがあれば、本数を
  確定しない

実物の ssh、docker は、どの段でも呼ばない (`FakeRunner` と、見本の実物の出力を使う)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address
from pathlib import Path

import pytest

from fake_runner import FakeRunner, Reply, Rule
from serving_kit import netcheck as nc
from serving_kit.remote import RemoteError
from serving_kit.types import NodeDef, NodeRole

# --- 見本の値 ---------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spark"

REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"

HEAD = NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("10.0.1.60"),
    remote_root=REMOTE_ROOT,
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}

CANDIDATES: tuple[str, ...] = ("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1")
"""直結の候補の、見本での現れ順 (README.md: 直結の側の 4 つのインターフェース)。"""

CONNECTED: tuple[str, ...] = ("enp1s0f0np0", "enP2p1s0f0np0")
DISCONNECTED: tuple[str, ...] = ("enp1s0f1np1", "enP2p1s0f1np1")
"""見本の時点で、つながっている側と、つながっていない側 (README.md)。"""


def _fixture(role: str, name: str) -> str:
    return (FIXTURES / role / name).read_text()


def _mtu_texts(role: str) -> dict[str, str]:
    """見本にある 3 つと、見本にない 1 つ (`enP2p1s0f1np1`)。

    README.md: 「管理の側と、つながっていないものは 1500」。見本にない分は、同じ DOWN 側の
    見本 (`mtu-enp1s0f1np1.txt` = 1500) を使い回す (見本のファイルは変えない)。
    """
    return {
        "enp1s0f0np0": _fixture(role, "mtu-enp1s0f0np0.txt"),
        "enp1s0f1np1": _fixture(role, "mtu-enp1s0f1np1.txt"),
        "enP2p1s0f0np0": _fixture(role, "mtu-enP2p1s0f0np0.txt"),
        "enP2p1s0f1np1": _fixture(role, "mtu-enp1s0f1np1.txt"),
    }


def _ethtool_texts(role: str) -> dict[str, tuple[str, str]]:
    """見本にある 3 つ (`enp1s0f0np0`、`enp1s0f1np1`、`enP2p1s0f0np0`) と、見本にない 1 つ
    (`enP2p1s0f1np1`)。README.md: `ethtool` の見本は、直結の側では、つながっている 2 つと、
    つながっていない 1 つ (`enp1s0f1np1`) しか採っていない。見本にない分は、同じ
    「つながっていない」側の見本を使い回す (見本のファイルは変えない)。
    """
    disconnected = (
        _fixture(role, "ethtool-enp1s0f1np1.txt"),
        _fixture(role, "ethtool-enp1s0f1np1.stderr.txt"),
    )
    return {
        "enp1s0f0np0": (
            _fixture(role, "ethtool-enp1s0f0np0.txt"),
            _fixture(role, "ethtool-enp1s0f0np0.stderr.txt"),
        ),
        "enp1s0f1np1": disconnected,
        "enP2p1s0f0np0": (
            _fixture(role, "ethtool-enP2p1s0f0np0.txt"),
            _fixture(role, "ethtool-enP2p1s0f0np0.stderr.txt"),
        ),
        "enP2p1s0f1np1": disconnected,
    }


def _rules(
    role: NodeRole,
    *,
    link_text: str,
    addr_text: str,
    ibdev_text: str,
    ibv_text: str,
    mtu_texts: Mapping[str, str],
    ethtool_texts: Mapping[str, tuple[str, str]],
) -> list[Rule]:
    """1 台ぶんの台本 (`netcheck.read_link_report` が流す、すべての読み取りに答える)。"""
    rules = [
        Rule(node=role, prefix=("ip", "-br", "link"), replies=(Reply(stdout=link_text),)),
        Rule(node=role, prefix=("ip", "-br", "addr"), replies=(Reply(stdout=addr_text),)),
        Rule(node=role, prefix=("ibdev2netdev",), replies=(Reply(stdout=ibdev_text),)),
        Rule(node=role, prefix=("ibv_devinfo",), replies=(Reply(stdout=ibv_text),)),
    ]
    for name, mtu_text in mtu_texts.items():
        rules.append(
            Rule(
                node=role,
                prefix=("cat", f"/sys/class/net/{name}/mtu"),
                replies=(Reply(stdout=mtu_text),),
            )
        )
    for name, (stdout, stderr) in ethtool_texts.items():
        rules.append(
            Rule(
                node=role,
                prefix=("ethtool", name),
                replies=(Reply(stdout=stdout, stderr=stderr),),
            )
        )
    return rules


def _base_rules(role: NodeRole) -> list[Rule]:
    """見本のまま (つながっている候補が 2 つ) の台本。"""
    return _rules(
        role,
        link_text=_fixture(role, "ip-br-link.txt"),
        addr_text=_fixture(role, "ip-br-addr.txt"),
        ibdev_text=_fixture(role, "ibdev2netdev.txt"),
        ibv_text=_fixture(role, "ibv_devinfo.txt"),
        mtu_texts=_mtu_texts(role),
        ethtool_texts=_ethtool_texts(role),
    )


def _flip_link_state(text: str, name: str, *, up: bool) -> str:
    """見本の `ip -br link` の 1 行の状態を書き換える。

    試験の中で、見本の文字列から作る (tasks.md 4.3 の完了の状態: 「見本の DOWN の 2 つを、
    UP とつながっている形に変えた台本。試験の中で、見本の文字列から作る」)。見本のファイルは
    変えない。
    """
    target_state = "UP" if up else "DOWN"
    out: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        matches = fields and fields[0].split("@", 1)[0] == name
        if matches and len(fields) >= 2 and fields[1] != target_state:
            fields[1] = target_state
            if fields[-1].startswith("<") and fields[-1].endswith(">"):
                flags = [flag for flag in fields[-1].strip("<>").split(",") if flag != "NO-CARRIER"]
                if up:
                    if "LOWER_UP" not in flags:
                        flags.append("LOWER_UP")
                else:
                    flags = [flag for flag in flags if flag != "LOWER_UP"]
                    if "NO-CARRIER" not in flags:
                        flags.insert(0, "NO-CARRIER")
                fields[-1] = "<" + ",".join(flags) + ">"
            out.append(" ".join(fields))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _replace_rule(
    rules: list[Rule], role: NodeRole, prefix: tuple[str, ...], reply: Reply
) -> list[Rule]:
    """台本のうち、`prefix` に当たる規則だけを、1 つの返事の規則に差し替える。"""
    kept = [rule for rule in rules if rule.prefix != prefix]
    return [*kept, Rule(node=role, prefix=prefix, replies=(reply,))]


def _four_connected_rules(role: NodeRole) -> list[Rule]:
    """見本の `DOWN` の 2 つを、つながった形に変えた台本 (完了の状態の 2 つ目)。"""
    link_text = _fixture(role, "ip-br-link.txt")
    for name in DISCONNECTED:
        link_text = _flip_link_state(link_text, name, up=True)

    mtu_texts = dict(_mtu_texts(role))
    for name in DISCONNECTED:
        mtu_texts[name] = _fixture(role, "mtu-enp1s0f0np0.txt")  # つながった直結の側は 9000

    ethtool_texts = dict(_ethtool_texts(role))
    connected_sample = _fixture(role, "ethtool-enp1s0f0np0.txt")
    connected_stderr = _fixture(role, "ethtool-enp1s0f0np0.stderr.txt")
    for name in DISCONNECTED:
        ethtool_texts[name] = (
            connected_sample.replace("Settings for enp1s0f0np0:", f"Settings for {name}:", 1),
            connected_stderr,
        )

    return _rules(
        role,
        link_text=link_text,
        addr_text=_fixture(role, "ip-br-addr.txt"),
        ibdev_text=_fixture(role, "ibdev2netdev.txt"),
        ibv_text=_fixture(role, "ibv_devinfo.txt"),
        mtu_texts=mtu_texts,
        ethtool_texts=ethtool_texts,
    )


# --- 見本のまま (つながっている候補が 2 つ) ---------------------------------------


def test_two_connected_interfaces_are_judged_as_one_cable_for_both_nodes(tmp_path: Path) -> None:
    runner = FakeRunner(var_root=tmp_path, script=[*_base_rules("head"), *_base_rules("worker")])

    reports = nc.read_link_reports(runner, NODES)

    for role in ("head", "worker"):
        report = reports[role]
        assert report.node == role
        assert report.cable_count == 1
        names = {link.name for link in report.interfaces}
        assert names == set(CANDIDATES)
        up_names = {link.name for link in report.interfaces if link.state == "UP"}
        assert up_names == set(CONNECTED)
        assert not report.tools_missing
        assert "netlink" not in report.detail
        assert "Operation not permitted" not in report.detail
        for link in report.interfaces:
            if link.name in CONNECTED:
                assert link.mtu == 9000
                assert link.speed_mbps == 200000
            else:
                assert link.mtu == 1500
                assert link.speed_mbps is None
            assert link.roce_device is not None

    head_by_name = {link.name: link for link in reports["head"].interfaces}
    assert head_by_name["enp1s0f0np0"].roce_device == "rocep1s0f0"
    assert head_by_name["enP2p1s0f0np0"].roce_device == "roceP2p1s0f0"
    assert "192.168.100.10/24" in head_by_name["enp1s0f0np0"].addrs

    # 読み取りだけ: 状態を変える呼び出しが 1 つも出ない。docker への呼び出しも出ない
    assert not any(call.mutating for call in runner.calls)
    assert not any(call.argv[0] == "docker" for call in runner.runs)
    assert {call.argv[0] for call in runner.runs} <= {
        "ip",
        "cat",
        "ethtool",
        "ibdev2netdev",
        "ibv_devinfo",
    }


# --- 見本を書き換えた台本 (つながっている候補が 4 つ) -------------------------------


def test_four_connected_interfaces_are_judged_as_two_cables(tmp_path: Path) -> None:
    runner = FakeRunner(
        var_root=tmp_path,
        script=[*_four_connected_rules("head"), *_four_connected_rules("worker")],
    )

    reports = nc.read_link_reports(runner, NODES)

    for role in ("head", "worker"):
        report = reports[role]
        assert report.cable_count == 2
        assert {link.name for link in report.interfaces} == set(CANDIDATES)
        assert all(link.state == "UP" for link in report.interfaces)
        assert all(link.mtu == 9000 for link in report.interfaces)


# --- 入っていない道具 ---------------------------------------------------------


def test_a_missing_tool_is_recorded_and_the_judgement_continues(tmp_path: Path) -> None:
    rules = [rule for rule in _base_rules("head") if rule.prefix != ("ibdev2netdev",)]
    rules.append(
        Rule(
            node="head",
            prefix=("ibdev2netdev",),
            replies=(Reply(exit_code=127, stderr="bash: ibdev2netdev: command not found\n"),),
        )
    )
    runner = FakeRunner(var_root=tmp_path, script=rules)

    report = nc.read_link_report(runner, HEAD)

    assert report.cable_count == 1
    assert "ibdev2netdev" in report.tools_missing
    assert all(link.roce_device is None for link in report.interfaces)


# --- 片方の台に入れない -------------------------------------------------------


def test_an_unreachable_node_is_reported_without_crashing(tmp_path: Path) -> None:
    unreachable = [
        Rule(
            node="head",
            prefix=("ip", "-br", "link"),
            replies=(
                Reply(
                    raises=RemoteError(
                        "つながらなかった",
                        node="head",
                        ssh_host=HEAD.ssh_host,
                        argv=("ip", "-br", "link"),
                    )
                ),
            ),
        )
    ]
    runner = FakeRunner(var_root=tmp_path, script=[*unreachable, *_base_rules("worker")])

    reports = nc.read_link_reports(runner, NODES)

    assert reports["head"].interfaces == ()
    assert reports["head"].cable_count is None
    assert "届かない" in reports["head"].detail
    assert reports["worker"].cable_count == 1


# --- 0、1、3 のときは決めつけない (`_judge_cable_count` を直に試す) -------------------


@pytest.mark.parametrize(
    ("connected", "expected"),
    [
        ((), None),
        (("enp1s0f0np0",), None),
        (("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0"), None),
        (("enp1s0f0np0", "enP2p1s0f0np0"), 1),
        (("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1"), 2),
    ],
)
def test_cable_count_is_judged_only_from_two_or_four_connected(
    connected: Sequence[str], expected: int | None
) -> None:
    count, note = nc._judge_cable_count(connected)
    assert count == expected
    if expected is None:
        assert "判断できない" in note
    else:
        assert f"{expected} 本" in note


# --- 候補の決め方 -------------------------------------------------------------


def test_candidate_interfaces_exclude_management_and_virtual_ones() -> None:
    names = [
        "lo",
        "enP7s7",
        "enp1s0f0np0",
        "enp1s0f1np1",
        "enP2p1s0f0np0",
        "enP2p1s0f1np1",
        "wlP9s9",
        "tailscale0",
        "docker0",
        "br-000000000000",
        "veth0000000",
    ]
    addr_by_name = {
        "enP7s7": ("10.0.1.60/24",),
        "enp1s0f0np0": ("192.168.100.10/24",),
    }

    assert nc._candidate_names(names, HEAD, addr_by_name) == list(CANDIDATES)


# --- ノードの定義との突き合わせ -----------------------------------------------


def test_fabric_ifname_already_filled_matches_the_reading(tmp_path: Path) -> None:
    node = HEAD.model_copy(
        update={"fabric_ifname": "enp1s0f0np0", "fabric_addr": IPv4Address("192.168.100.10")}
    )
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert "合っている" in report.detail


def test_fabric_addr_mismatch_is_flagged(tmp_path: Path) -> None:
    node = HEAD.model_copy(
        update={"fabric_ifname": "enp1s0f0np0", "fabric_addr": IPv4Address("192.168.100.99")}
    )
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert "アドレスが fabric_addr" in report.detail


def test_fabric_ifname_pointing_to_a_down_interface_is_flagged(tmp_path: Path) -> None:
    node = HEAD.model_copy(update={"fabric_ifname": "enp1s0f1np1"})
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert "つながっていない" in report.detail


def test_fabric_ifname_pointing_to_an_unknown_interface_is_flagged(tmp_path: Path) -> None:
    node = HEAD.model_copy(update={"fabric_ifname": "enp99s99"})
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert "読み取った一覧にない" in report.detail


def test_fabric_ifname_empty_suggests_connected_candidates(tmp_path: Path) -> None:
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, HEAD)

    assert "埋める候補" in report.detail
    assert "enp1s0f0np0" in report.detail


# --- 2 台の判断の食い違い -----------------------------------------------------


def test_the_two_nodes_disagreeing_on_cable_count_is_noted(tmp_path: Path) -> None:
    runner = FakeRunner(
        var_root=tmp_path,
        script=[*_base_rules("head"), *_four_connected_rules("worker")],
    )

    reports = nc.read_link_reports(runner, NODES)

    assert reports["head"].cable_count == 1
    assert reports["worker"].cable_count == 2
    assert "食い違っている" in reports["head"].detail
    assert "食い違っている" in reports["worker"].detail


# --- 人が読める形 (機械を識別できる値を出さない) -----------------------------------


def test_format_link_report_does_not_include_mac_or_guid(tmp_path: Path) -> None:
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))
    report = nc.read_link_report(runner, HEAD)

    text = nc.format_link_report(report)

    assert "enp1s0f0np0" in text
    assert "ケーブルの本数" in text
    assert re.search(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", text) is None


# --- レビューの指摘 1 (Critical): ip -br addr が読めないと、管理の側が紛れ込む -------


@pytest.mark.parametrize(
    ("label", "addr_reply"),
    [
        (
            "remote_error",
            Reply(
                raises=RemoteError(
                    "つながらなかった",
                    node="head",
                    ssh_host=HEAD.ssh_host,
                    argv=("ip", "-br", "addr"),
                )
            ),
        ),
        ("nonzero_exit", Reply(exit_code=1, stderr="ip: 何か失敗した\n")),
        ("empty_output", Reply(stdout="")),
    ],
    ids=["remote_error", "nonzero_exit", "empty_output"],
)
def test_the_management_interface_never_leaks_in_when_addr_is_unreadable(
    tmp_path: Path, label: str, addr_reply: Reply
) -> None:
    rules = _replace_rule(_base_rules("head"), "head", ("ip", "-br", "addr"), addr_reply)
    runner = FakeRunner(var_root=tmp_path, script=rules)

    report = nc.read_link_report(runner, HEAD)

    # 管理の側の名前が、直結の候補にも「埋める候補」にも出ない。本数も確定しない
    assert report.interfaces == ()
    assert report.cable_count is None
    assert "enP7s7" not in report.detail
    assert report.detail  # 読めなかった理由が書いてある


def test_the_reviewers_repro_with_addr_broken_and_only_one_real_link_up(tmp_path: Path) -> None:
    """レビュー担当の再現: 直結の側が実際には 1 つしかつながっていないのに、`ip -br addr` が
    読めないと、管理の側が候補に紛れ込んで `cable_count=1` という誤った確定の結果が返って
    いた (指摘 1)。修正後は、`cable_count` が確定しない。
    """
    link_text = _flip_link_state(_fixture("head", "ip-br-link.txt"), "enP2p1s0f0np0", up=False)
    rules = _replace_rule(
        _base_rules("head"), "head", ("ip", "-br", "link"), Reply(stdout=link_text)
    )
    rules = _replace_rule(
        rules,
        "head",
        ("ip", "-br", "addr"),
        Reply(
            raises=RemoteError(
                "つながらなかった", node="head", ssh_host=HEAD.ssh_host, argv=("ip", "-br", "addr")
            )
        ),
    )
    runner = FakeRunner(var_root=tmp_path, script=rules)

    report = nc.read_link_report(runner, HEAD)

    assert report.cable_count is None
    assert report.cable_count != 1


def test_no_interface_matching_lan_addr_does_not_determine_a_count(tmp_path: Path) -> None:
    """`ip -br addr` は読めたが、`lan_addr` に一致するインターフェースが 1 つもない
    (ノードの定義の誤り)。見分けの前提が崩れているので、本数を確定しない。
    """
    node = HEAD.model_copy(update={"lan_addr": IPv4Address("10.9.9.9")})
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert report.interfaces == ()
    assert report.cable_count is None
    assert report.detail


# --- レビューの指摘 2: fabric_ifname の突き合わせも「確かめられなかった」と言う -------


def test_the_fabric_ifname_check_is_unconfirmed_when_addr_is_unreadable(tmp_path: Path) -> None:
    node = HEAD.model_copy(
        update={"fabric_ifname": "enp1s0f0np0", "fabric_addr": IPv4Address("192.168.100.10")}
    )
    rules = _replace_rule(
        _base_rules("head"),
        "head",
        ("ip", "-br", "addr"),
        Reply(exit_code=1, stderr="ip: エラー\n"),
    )
    runner = FakeRunner(var_root=tmp_path, script=rules)

    report = nc.read_link_report(runner, node)

    assert "確かめられなかった" in report.detail
    assert "合っている" not in report.detail


# --- レビューの指摘 3: fabric_ifname が管理の側の名前を指しているとき -----------------


def test_fabric_ifname_pointing_to_the_management_interface_is_flagged(tmp_path: Path) -> None:
    node = HEAD.model_copy(update={"fabric_ifname": "enP7s7"})
    runner = FakeRunner(var_root=tmp_path, script=_base_rules("head"))

    report = nc.read_link_report(runner, node)

    assert "管理の側のインターフェースである" in report.detail


# --- あわせて確かめること: ip と ethtool のつながりの判定が食い違うとき ---------------


def test_ip_and_ethtool_disagreement_prevents_a_cable_count(tmp_path: Path) -> None:
    """`ip -br link` は `UP` (つながっている) だが、`ethtool` は `Link detected: no`。
    どちらで判断したかを `detail` に残し、本数は確定しない。
    """
    rules = _replace_rule(
        _base_rules("head"),
        "head",
        ("ethtool", "enp1s0f0np0"),
        Reply(
            stdout=_fixture("head", "ethtool-enp1s0f1np1.txt"),
            stderr=_fixture("head", "ethtool-enp1s0f1np1.stderr.txt"),
        ),
    )
    runner = FakeRunner(var_root=tmp_path, script=rules)

    report = nc.read_link_report(runner, HEAD)

    assert report.cable_count is None
    assert "食い違う" in report.detail
