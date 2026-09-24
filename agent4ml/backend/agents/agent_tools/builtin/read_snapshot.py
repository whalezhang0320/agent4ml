"""read_snapshot 工具：LLM 调用读压缩前快照内容。"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def read_snapshot(snapshot_path: str) -> str:
    """读取指定路径的压缩前快照，找回丢失的重要上下文（工具调用/核心思路）。

    参数：
        snapshot_path: 快照文件路径（从 <summary> 标签或上下文提示获取）
    """
    try:
        with open(snapshot_path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return f"读取快照失败：{snapshot_path}"
