"""DefaultStrategy 真实 deepseek 集成测：P4 summarize 实际压缩。"""

from __future__ import annotations

import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

langchain_deepseek = pytest.importorskip("langchain_deepseek")
ChatDeepSeek = langchain_deepseek.ChatDeepSeek

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="需 DEEPSEEK_API_KEY",
)

from langchain_core.messages import HumanMessage, RemoveMessage

from agent4ml.backend.agents.context_engineering.contract import GovernanceContext
from agent4ml.backend.agents.context_engineering.strategies.default.strategy import (
    DefaultStrategy,
)
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    ContextAssembler,
    AGENT4ML_SUMMARY,
)


def test_p4_summarize_real(tmp_path) -> None:
    """P4 触发真实 deepseek summarize，产出 summary。"""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    strategy = DefaultStrategy(
        params={
            "preserve_recent": 2,
            "snapshot_dir": str(tmp_path / "snapshots"),
            "externalize_dir": str(tmp_path / "externalized"),
        },
        model=model,
    )
    messages = [
        HumanMessage(content="研究 LangGraph 上下文工程"),
        HumanMessage(content="deer-flow 有 SummarizationMiddleware"),
        HumanMessage(content="继续设计治理层"),
        HumanMessage(content="设计接入契约 GovernanceStrategy 6 hook"),
        HumanMessage(content="实现标签化"),
    ]
    governance = {"default": {"pending": ["P4"]}}
    ctx = GovernanceContext(
        state={},
        governance=governance,
        config={},
        token_counter=lambda m: 1000,
        runtime=None,
        hook="before_model",
        messages=messages,
    )
    result = strategy.before_model(ctx)
    assert result is not None
    assert result.messages_patch is not None
    assert len(result.messages_patch) >= 3
    summary_msg = next(
        m for m in result.messages_patch
        if isinstance(m, HumanMessage) and m.additional_kwargs.get(AGENT4ML_SUMMARY)
    )
    assert len(summary_msg.content) > 0
    gov = result.state_patch["governance"]
    assert gov["default"]["summary"]
    assert gov["default"]["metrics"]["summarize_count"] == 1
    # regression：summary 写 governance.default.summary + ContextAssembler 读 default.summary 路径打通
    rendered = ContextAssembler().render_context_block({}, gov)
    assert "<summary>" in rendered
    assert gov["default"]["summary"] in rendered
