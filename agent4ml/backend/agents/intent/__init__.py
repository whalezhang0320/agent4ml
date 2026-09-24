"""意图识别模块 — agent 核心能力。

属 agent core，与 CLI/API/IM 解耦。调用方（CLI 主循环等）通过
IntentTree.detect_and_dispatch 路由用户输入：True = 意图已处理（不进 graph），
False = 无匹配进 graph。

未来扩展：多层意图树 + LLM 识别只需加 IntentNode + IntentStrategy + IntentAction，
不改 IntentTree 遍历逻辑。
"""

from agent4ml.backend.agents.intent.engine import (
    AnyMatchStrategy,
    Intent,
    IntentAction,
    IntentNode,
    IntentStrategy,
    IntentTree,
    IntentType,
    MatchResult,
    ReportAction,
    ReportIntentStrategy,
    default_intent_tree,
)

__all__ = [
    "AnyMatchStrategy",
    "Intent",
    "IntentAction",
    "IntentNode",
    "IntentStrategy",
    "IntentTree",
    "IntentType",
    "MatchResult",
    "ReportAction",
    "ReportIntentStrategy",
    "default_intent_tree",
]
