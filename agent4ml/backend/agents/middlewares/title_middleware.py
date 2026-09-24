"""TitleMiddleware — after_agent 设置 metadata.title（取自 research_question/user_input，截断 60）。"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime


class TitleMiddleware(AgentMiddleware):
    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        source = state.get("research_question") or state.get("user_input") or "Untitled"
        return {"metadata": {"title": str(source)[:60]}}

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
