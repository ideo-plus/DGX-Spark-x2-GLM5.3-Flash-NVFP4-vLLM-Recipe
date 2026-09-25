"""開始時の発見の台本を組み立てる助け (issue #41)。

`serve watch` の開始時の発見 (`_discover_thermal_zones`、`_discover_hwmon`、
`_discover_cpufreq_cores`) を、種類ごとに 1 回の `cat` にまとめる実装に合わせて、偽の実行役
(`fake_runner.FakeRunner`) に渡す台本を組み立てる。

`fake_runner.Rule` は引数の列の**前方一致**で選ぶので、候補をすべて並べた発見の呼び出しは、
観察の規則 (`prefix=("cat", <1 つの道筋>)`) にも当たる。発見の規則を観察の規則より**先**に
置くことで、発見の呼び出しが発見の規則に当たるようにする。

上限 (32) は、実装の定数ではなく試験側のリテラルとして持つ (実装の定数を読むと、上限を
変えた変異を検出できないため)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from serving_kit import watch as w
from serving_kit.types import NodeRole

ZONE_LIMIT = 32
"""発見が試す、熱区域の番号の上限 (`watch._THERMAL_ZONE_LIMIT` と同じ値。試験側で固定する)。"""
HWMON_LIMIT = 32
"""発見が試す、hwmon の番号の上限 (`watch._HWMON_LIMIT` と同じ値。試験側で固定する)。"""
HWMON_TEMP_LIMIT = 32
"""発見が試す、hwmon の温度センサーの番号の上限 (`watch._HWMON_TEMP_LIMIT` と同じ値)。"""
CPU_CORE_LIMIT = 32
"""発見が試す、CPU コアの番号の上限 (`watch._CPU_CORE_LIMIT` と同じ値。試験側で固定する)。"""


def thermal_candidates() -> tuple[str, ...]:
    """熱区域の発見の候補 (0〜31 の `temp`)。"""
    return tuple(f"{w.THERMAL_ZONE_DIR}thermal_zone{number}/temp" for number in range(ZONE_LIMIT))


def hwmon_name_candidates() -> tuple[str, ...]:
    """hwmon の `name` の発見の候補 (0〜31)。"""
    return tuple(f"{w.HWMON_DIR}hwmon{number}/name" for number in range(HWMON_LIMIT))


def hwmon_input_candidates(chips: Sequence[str]) -> tuple[str, ...]:
    """採用した chip の温度センサーの候補 (chip の順 → `temp1..32_input` の順)。"""
    return tuple(
        f"{w.HWMON_DIR}{chip}/temp{number}_input"
        for chip in chips
        for number in range(1, HWMON_TEMP_LIMIT + 1)
    )


def label_candidates(sensor_ids: Sequence[str]) -> tuple[str, ...]:
    """発見したセンサーのラベルの候補 (識別子 `hwmon<N>/temp<M>` の順)。"""
    return tuple(f"{w.HWMON_DIR}{sensor_id}_label" for sensor_id in sensor_ids)


def cpufreq_candidates() -> tuple[str, ...]:
    """cpufreq のコアの発見の候補 (0〜31 の `scaling_max_freq`)。"""
    return tuple(
        f"{w.CPU_DIR}cpu{number}/cpufreq/scaling_max_freq" for number in range(CPU_CORE_LIMIT)
    )


def unreadable_lines(paths: Sequence[str]) -> str:
    """`cat` が読めなかった道筋を stderr に書くときの行 (GNU coreutils の書式)。"""
    return "".join(f"cat: {path}: No such file or directory\n" for path in paths)


def batch_reply(candidates: Sequence[str], present: Mapping[str, str]) -> Reply:
    """候補のうち `present` の鍵だけが存在する、1 回の `cat` の返事。

    stdout は候補の順に `present` の値を連ね、stderr は残りの道筋の「読めなかった」の行、
    `exit_code` は残りがあれば 1、なければ 0 にする。`present` の値は、`cat` が返す中身
    (末尾の改行を含む)。
    """
    present_paths = [path for path in candidates if path in present]
    absent_paths = [path for path in candidates if path not in present]
    return Reply(
        exit_code=1 if absent_paths else 0,
        stdout="".join(present[path] for path in present_paths),
        stderr=unreadable_lines(absent_paths),
    )


def discovery_rule(
    candidates: Sequence[str], present: Mapping[str, str], *, node: NodeRole | None = None
) -> Rule:
    """候補をすべて並べた 1 回の `cat` に当たる、発見の規則。"""
    return Rule(
        prefix=("cat", *candidates),
        node=node,
        replies=(batch_reply(candidates, present),),
    )


def discovery_calls(runner: FakeRunner) -> list[RecordedCall]:
    """最初の `nvidia-smi` より前の `run` の呼び出し (開始時の発見だけ)。"""
    calls: list[RecordedCall] = []
    for call in runner.calls:
        if call.kind != "run":
            continue
        if call.argv and call.argv[0] == "nvidia-smi":
            break
        calls.append(call)
    return calls


def discovery_calls_for_node(runner: FakeRunner, node: NodeRole) -> list[RecordedCall]:
    """1 台ぶんの、開始時の発見の `run` の呼び出し。"""
    return [call for call in discovery_calls(runner) if call.node == node]
