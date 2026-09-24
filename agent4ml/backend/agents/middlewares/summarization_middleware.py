"""SummarizationMiddleware — AgentMiddleware no-op 占位（V1）。

后续可接入 langchain.agents.middleware.summarization.SummarizationMiddleware
或自定义摘要逻辑。当前 before_model 返回 None，不干预。
"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime


class SummarizationMiddleware(AgentMiddleware):
    @override
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return None

    @override
    async def abefore_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return None
