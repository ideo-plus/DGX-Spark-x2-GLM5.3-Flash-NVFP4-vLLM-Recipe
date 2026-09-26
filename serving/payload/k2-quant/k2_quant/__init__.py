"""K2 第 1 段: 共有の専門家・dense・lm_head を FP8 (重みだけ、チャネルごと) にする変換の道具。"""

from __future__ import annotations

from typing import Final

TOOL_NAME: Final = "k2-quant"
TOOL_VERSION: Final = "0.2.0"

__all__ = ["TOOL_NAME", "TOOL_VERSION"]
