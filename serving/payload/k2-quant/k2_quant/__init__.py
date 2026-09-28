"""K2 第 1〜4 段: 選んだモジュールを FP8 (重みだけ、チャネルごと) か NVFP4A16 (重みだけ NVFP4。
Issue #95) にする変換の道具。第 4 段 (Issue #99) は、MTP の層 45 の MLA の射影と FP8 の専門家を
追加で NVFP4A16 に変換する (`--preset k2s4`)。"""

from __future__ import annotations

from typing import Final

TOOL_NAME: Final = "k2-quant"
TOOL_VERSION: Final = "0.2.0"

__all__ = ["TOOL_NAME", "TOOL_VERSION"]
