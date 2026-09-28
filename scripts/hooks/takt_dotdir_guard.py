#!/usr/bin/env python3
"""TAKT のエージェントが `.takt/` を読むのを断る、Claude Code の PreToolUse フック。

TAKT が起動した claude (`scripts/takt-claude.sh` が `TAKT_AGENT=1` を付ける) のときだけ効く。
対話のセッションでは何もしない。

通すのは、いま動いている実行のディレクトリ (`.takt/runs/` の下で、いちばん新しいもの) だけ。
各段階の指示が読ませる Report Directory と文脈は、ここにある。それ以外の `.takt/` (古い実行、
ワークフロー、指示の部品、設定) を指すツール呼び出しは断る。

限界: 道筋を書かずに広く探す呼び出し (`Grep` をリポジトリの根で流すなど) の結果に `.takt/` の
ファイルが混ざるのは止められない。道筋の文字列に `.takt` が現れる呼び出しだけを見る。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# 道筋の区切りか、語の始まりのあとに現れる `.takt` を、道筋の終わりまで拾う
_TAKT_PATH = re.compile(r"""(?:^|(?<=[\s"'=(:/]))((?:[^\s"'`;|&<>()]*/)?\.takt(?:/[^\s"'`;|&<>()]*)?)""")


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _current_run(project: Path) -> Path | None:
    runs = project / ".takt" / "runs"
    if not runs.is_dir():
        return None
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # TAKT が「実行中」と記録している実行を先に選ぶ。無ければ、いちばん新しいもの
    running = [p for p in candidates if _status(p) == "running"]
    return max(running or candidates, key=lambda p: p.stat().st_mtime).resolve()


def _status(run: Path) -> str | None:
    try:
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    status = meta.get("status") if isinstance(meta, dict) else None
    return status if isinstance(status, str) else None


def _deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
        ensure_ascii=False,
    )


def main() -> int:
    if os.environ.get("TAKT_AGENT") != "1":
        return 0
    payload = json.load(sys.stdin)
    cwd = Path(payload.get("cwd") or os.getcwd())
    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    current = _current_run(project)

    offending: list[str] = []
    for text in _strings(payload.get("tool_input", {})):
        for match in _TAKT_PATH.finditer(text):
            raw = match.group(1)
            path = Path(os.path.expanduser(raw))
            resolved = (path if path.is_absolute() else cwd / path).resolve()
            if current is not None and resolved.is_relative_to(current):
                continue
            offending.append(raw)

    if offending:
        allowed = str(current) if current is not None else "(なし)"
        _deny(
            "TAKT のエージェントは .takt/ を読まない (計測者の指示、2026-09-28)。"
            f"読んでよいのは、いまの実行のディレクトリ {allowed} の中の Report Directory と"
            f"文脈だけ。断った道筋: {', '.join(sorted(set(offending)))}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
