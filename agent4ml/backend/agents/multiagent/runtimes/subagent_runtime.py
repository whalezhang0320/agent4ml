"""SubagentRuntime — 进程内调用 lead factory 实现（Agent4ML self-copy subagent）。

设计（spec.md SubagentRuntime Requirement + design.md §2）:
- sync only MVP（INV#5）
- 复用 lead agent factory（create_agent4ml_agent），参数化 toolsets（leaf role）
- leaf role 递归控制：tool_groups 不含 multiagent，不能 spawn（INV#4）
- isolated context：全新 ThreadState，不继承父 message history
- shared thread sandbox：复用父 sandbox_id（INV#3）
- max_steps 限制：超限抛 SubagentMaxStepsError
- 实现 SpecialistRuntime Protocol（invoke(SpecialistRequest) → SpecialistRawResult）
"""
from __future__ import annotations

import time
from typing import Any, Callable

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
)
from agent4ml.backend.agents.multiagent.types import (
    SpecialistRawResult,
    SpecialistRequest,
)


class SubagentRuntime:
    """进程内调用 lead factory runtime（Agent4ML self-copy subagent）。

    sync only MVP（INV#5）。leaf role（INV#4）+ isolated context + shared sandbox。
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        """agent_factory: callable returning a runnable lead agent (leaf role, no multiagent tools).

        If None, _get_agent raises SpecialistStartupError (bootstrap injects factory).
        """
        self._agent_factory = agent_factory

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        start = time.time()
        agent = self._get_agent()
        state = self._create_isolated_state(request)

        try:
            result = agent.invoke(
                state,
                config={"recursion_limit": request.max_steps * 2},
            )
        except RecursionError:
            from agent4ml.backend.agents.multiagent.exceptions import (
                SubagentMaxStepsError,
            )
            raise SubagentMaxStepsError(max_steps=request.max_steps)
        except Exception as e:
            raise SpecialistCrashError(str(e))

        raw_output = self._extract_output(result)

        return SpecialistRawResult(
            raw_output=raw_output,
            duration_seconds=time.time() - start,
        )

    def _get_agent(self) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory()
        raise SpecialistStartupError(
            "agent_factory not configured (bootstrap must inject leaf-role factory)"
        )

    def _create_isolated_state(self, request: SpecialistRequest) -> dict:
        """Create isolated ThreadState (only goal + context_summary, no inherited messages).

        isolated context（INV#4）：全新 ThreadState，不继承父 messages/observations/sources。
        shared thread sandbox（INV#3）：复用父 sandbox_id。
        """
        from agent4ml.backend.agents.state.thread_state import (
            create_initial_thread_state,
        )

        state = create_initial_thread_state(request.goal)
        if request.context_summary:
            state["metadata"]["context_summary"] = request.context_summary
        if request.sandbox_id:
            state["sandbox"] = {"sandbox_id": request.sandbox_id}
        return state

    def _extract_output(self, result: Any) -> str:
        """Extract output text from agent result (last message content)."""
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if content is not None:
                    return str(content)
                return str(last)
            return ""
        return str(result)
