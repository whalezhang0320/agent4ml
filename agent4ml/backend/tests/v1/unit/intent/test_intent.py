"""意图识别树测试：MVP 单层树 ReportIntent + 整句匹配防误触发 + 可扩展性。"""

from agent4ml.backend.agents.intent import (
    AnyMatchStrategy,
    Intent,
    IntentType,
    IntentTree,
    ReportAction,
    ReportIntentStrategy,
    default_intent_tree,
)
from agent4ml.backend.agents.intent.engine import _extract_topic


# --------------------------------------------------------------------------- #
# ReportIntentStrategy 整句匹配
# --------------------------------------------------------------------------- #


def test_report_strategy_matches_generate_report() -> None:
    s = ReportIntentStrategy()
    r = s.match("生成报告")
    assert r.matched is True
    assert r.payload["type"] == IntentType.REPORT
    assert r.confidence == 1.0


def test_report_strategy_matches_slash_report() -> None:
    s = ReportIntentStrategy()
    assert s.match("/report").matched is True


def test_report_strategy_matches_with_topic() -> None:
    s = ReportIntentStrategy()
    r = s.match("/report 天气调研")
    assert r.matched is True
    assert r.payload["topic"] == "天气调研"


def test_report_strategy_matches_variants() -> None:
    s = ReportIntentStrategy()
    for text in ["出报告", "整理成报告", "现在开始生成报告", "写一份报告", "给我一份报告"]:
        assert s.match(text).matched is True, f"应匹配: {text}"


def test_report_strategy_no_match_how_to_write_report() -> None:
    """整句匹配防误触发：'如何写报告' 不以关键词开头。"""
    s = ReportIntentStrategy()
    assert s.match("如何写报告").matched is False


def test_report_strategy_no_match_normal_chat() -> None:
    s = ReportIntentStrategy()
    assert s.match("你好").matched is False
    assert s.match("查一下北京天气").matched is False
    assert s.match("报告说今天会下雨").matched is False


def test_report_strategy_case_insensitive() -> None:
    s = ReportIntentStrategy()
    assert s.match("/REPORT").matched is True
    assert s.match("/Report").matched is True


# --------------------------------------------------------------------------- #
# _extract_topic
# --------------------------------------------------------------------------- #


def test_extract_topic_from_slash_report() -> None:
    assert _extract_topic("/report 天气") == "天气"


def test_extract_topic_from_generate_report() -> None:
    assert _extract_topic("生成报告 知识图谱调研") == "知识图谱调研"


def test_extract_topic_none_when_no_topic() -> None:
    assert _extract_topic("/report") is None
    assert _extract_topic("生成报告") is None


# --------------------------------------------------------------------------- #
# IntentTree 遍历
# --------------------------------------------------------------------------- #


def test_tree_dispatches_report_with_handler() -> None:
    called: list[Intent] = []

    def handler(intent: Intent, runtime) -> bool:
        called.append(intent)
        return True

    tree = default_intent_tree(report_handler=handler)
    assert tree.detect_and_dispatch("生成报告", runtime=None) is True
    assert len(called) == 1
    assert called[0].type == IntentType.REPORT


def test_tree_no_match_returns_false() -> None:
    tree = default_intent_tree()
    assert tree.detect_and_dispatch("你好", runtime=None) is False


def test_tree_handler_none_returns_false() -> None:
    """ReportAction 无 handler → execute 返回 False → detect_and_dispatch 返回 False。"""
    tree = default_intent_tree(report_handler=None)
    # 匹配 ReportIntent 但无 handler → False（未处理）
    assert tree.detect_and_dispatch("生成报告", runtime=None) is False


def test_any_match_strategy_always_matches() -> None:
    s = AnyMatchStrategy()
    assert s.match("任意文本").matched is True
    assert s.match("").matched is True


# --------------------------------------------------------------------------- #
# 可扩展性（未来加意图不改 IntentTree / 主循环）
# --------------------------------------------------------------------------- #


def test_tree_add_new_intent_node() -> None:
    """加新 IntentNode 到 root.children 不改 IntentTree 遍历逻辑。"""
    from agent4ml.backend.agents.intent import IntentNode

    class _AlwaysMatchStrategy:
        def match(self, text: str):
            from agent4ml.backend.agents.intent import MatchResult
            return MatchResult(matched=True, confidence=1.0, payload={"type": IntentType.REPORT}, children=None)

    class _StubAction:
        def __init__(self):
            self.called = False

        def execute(self, intent, runtime) -> bool:
            self.called = True
            return True

    stub = _StubAction()
    tree = default_intent_tree()
    tree._root.children.append(IntentNode(strategy=_AlwaysMatchStrategy(), action=stub))
    tree.detect_and_dispatch("test", runtime=None)
    assert stub.called is True


def test_report_action_handler_return_value_propagates() -> None:
    def handler_true(intent, runtime):
        return True

    def handler_false(intent, runtime):
        return False

    tree_true = default_intent_tree(report_handler=handler_true)
    tree_false = default_intent_tree(report_handler=handler_false)
    assert tree_true.detect_and_dispatch("生成报告", runtime=None) is True
    assert tree_false.detect_and_dispatch("生成报告", runtime=None) is False
