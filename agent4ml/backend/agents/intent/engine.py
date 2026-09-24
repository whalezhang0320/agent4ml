"""意图识别引擎 — agent 核心能力（理解用户需求）。

三分离架构：IntentNode（节点）+ IntentStrategy（匹配策略）+ IntentAction（动作）。
MVP 单层树（root → ReportIntent leaf）。未来扩展多层 + LLM 识别只需加 Node + Strategy + Action。

属 agent 核心模块，与 CLI 解耦：CLI/API/IM channel 均可作为消费者调用
IntentTree.detect_and_dispatch。路由决策返回 bool，由调用方决定是否进 graph。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


class IntentType(Enum):
    REPORT = "report"
    # 未来: DEEP_RESEARCH, CLARIFY_TOPIC, CLARIFY_SCOPE ...


@dataclass
class Intent:
    """匹配成功的意图。"""

    type: IntentType
    confidence: float
    payload: dict


@dataclass
class MatchResult:
    """策略匹配结果。children 非空表示进子节点（树扩展）。"""

    matched: bool
    confidence: float
    payload: dict
    children: list[IntentNode] | None = None


class IntentStrategy(Protocol):
    """匹配策略。MVP 正则，未来可换 LLM（实现 LLMIntentStrategy 替换，接口不变）。"""

    def match(self, text: str) -> MatchResult: ...


class IntentAction(Protocol):
    """匹配后执行的动作。解耦：策略不管动作，动作不管匹配逻辑。

    返回 True = 已处理（不进 graph），False = 未处理（继续进 graph）。
    """

    def execute(self, intent: Intent, runtime: Any) -> bool: ...


@dataclass
class IntentNode:
    """意图树节点。叶子有 action，中间节点有 children。"""

    strategy: IntentStrategy
    action: IntentAction | None = None
    children: list[IntentNode] | None = None


class IntentTree:
    """意图树遍历器。

    detect_and_dispatch: root 匹配 → 进 children 或执行 action。
    返回 True = 已处理，False = 无匹配进 graph。
    """

    def __init__(self, root: IntentNode) -> None:
        self._root = root

    def detect_and_dispatch(self, text: str, runtime: Any) -> bool:
        return self._traverse(self._root, text, runtime)

    def _traverse(self, node: IntentNode, text: str, runtime: Any) -> bool:
        result = node.strategy.match(text)
        if not result.matched:
            return False
        if node.action is not None:
            intent = Intent(
                type=result.payload.get("type", IntentType.REPORT),
                confidence=result.confidence,
                payload=result.payload,
            )
            return bool(node.action.execute(intent, runtime))
        children = result.children or node.children
        if children:
            for child in children:
                if self._traverse(child, text, runtime):
                    return True
        return False


# --------------------------------------------------------------------------- #
# MVP 策略 + 动作
# --------------------------------------------------------------------------- #


class AnyMatchStrategy:
    """root 节点策略：匹配任意输入（进入 children）。"""

    def match(self, text: str) -> MatchResult:
        return MatchResult(matched=True, confidence=1.0, payload={}, children=None)


class ReportIntentStrategy:
    """报告意图策略：保守关键词整句匹配（^锚定），防误触发。

    匹配模式：生成报告 / 出报告 / 整理成报告 / 现在开始生成报告 /
    写一份报告 / 给我一份报告 / /report。
    不匹配："如何写报告"（不以关键词开头）。
    """

    PATTERNS = [
        r"^生成报告",
        r"^出报告",
        r"^整理成报告",
        r"^现在开始生成报告",
        r"^写一份报告",
        r"^给我一份报告",
        r"^/report\b",
    ]

    def match(self, text: str) -> MatchResult:
        stripped = text.strip()
        for pattern in self.PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                topic = _extract_topic(stripped)
                return MatchResult(
                    matched=True,
                    confidence=1.0,
                    payload={"type": IntentType.REPORT, "topic": topic},
                    children=None,
                )
        return MatchResult(matched=False, confidence=0.0, payload={}, children=None)


class ReportAction:
    """报告动作：调 handler 触发报告合成。

    handler 由调用方注入（CLI/API/IM），保持 intent 模块与报告合成流程解耦。
    handler 签名：(intent: Intent, runtime: Any) -> bool。
    """

    def __init__(self, handler: Callable[[Intent, Any], bool] | None = None) -> None:
        self._handler = handler

    def execute(self, intent: Intent, runtime: Any) -> bool:
        if self._handler is None:
            return False
        return bool(self._handler(intent, runtime))


def _extract_topic(text: str) -> str | None:
    """从 "/report 天气" 或 "生成报告 天气" 提取 topic。无则 None。"""
    m = re.match(r"^/report\s+(.+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(?:生成|出|整理成|现在开始生成|写一份|给我一份)报告\s+(.+)$", text)
    if m:
        return m.group(1).strip()
    return None


def default_intent_tree(report_handler: Callable[[Intent, Any], bool] | None = None) -> IntentTree:
    """MVP 单层树：root → ReportIntent leaf。

    report_handler 由调用方注入；None 时 ReportAction.execute 返回 False（不处理）。
    """
    root = IntentNode(
        strategy=AnyMatchStrategy(),
        children=[
            IntentNode(
                strategy=ReportIntentStrategy(),
                action=ReportAction(handler=report_handler),
            ),
        ],
    )
    return IntentTree(root)
