"""SystemContextMiddleware — before_model 注入 system_context 元数据。"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value


class SystemContextMiddleware(AgentMiddleware):
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        tz = _get_runtime_value(runtime, "timezone", "Asia/Shanghai")
        return {"metadata": {"system_context": {"agent": "Agent4ML deep research agent", "timezone": tz}}}

    async def abefore_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
