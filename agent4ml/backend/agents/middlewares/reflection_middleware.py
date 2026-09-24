"""ReflectionMiddleware — ReAct 退出闸门，简单反思（L1 充分性）。

after_model：模型想退出时，用 observations 对 todos step 覆盖度判断实质充分性。
不足 → reflection_items + jump_to model 补研究。外壳 + 可替换 Strategy 架构，
ReflectionAction 契约预留 revise_plan/backtrack/rollback_to 供未来 L2/L3 扩展。
仅 general/expert 启用。完整设计见 design_docs/15-reflection-middleware-design.md。
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, TypedDict, override

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.middlewares import _jump_budget
from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.middlewares.todo_middleware import _has_persistent_failures, _has_tool_call_intent
from agent4ml.backend.agents.prompts import get_prompt_manager
from agent4ml.backend.agents.state.types import ReflectionItem, ThreadState


class ReflectionAction(TypedDict):
    """策略返回的动作契约。L1 只用 pass/continue；revise_plan/backtrack/rollback_to 预留。"""

    verdict: str  # "pass" | "continue" | "revise_plan" | "backtrack"
    reflection_items: list[Any]
    plan: Any  # revise_plan/backtrack 时的新 plan，否则 None
    guidance: str  # 注回模型的提示
    rollback_to: str  # backtrack 回退到的决策节点 id（L3 未来）


class ReflectionStrategy(Protocol):
    def reflect(self, state: dict[str, Any], runtime: Runtime) -> ReflectionAction: ...


def _make_reflection_id() -> str:
    return f"refl-{uuid.uuid4().hex[:12]}"


def _step_id(obs: Any) -> str | None:
    if isinstance(obs, dict):
        return obs.get("step_id")
    return getattr(obs, "step_id", None)


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class LightReflectionStrategy:
    """default 模式轻量策略：不 jump，直接 pass。

    default 模式不强制补研究——模型给答案即结束。保留 ReflectionMiddleware 外壳
    （触发时机 + jump 预算管理），但判断逻辑恒 pass。
    """

    def reflect(self, state: dict[str, Any], runtime: Runtime) -> ReflectionAction:
        return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")


class SufficiencyStrategy:
    """L1 充分性守门：todos 全完成时，检查每步是否有 observation 覆盖。

    F7：覆盖度判断交 LLM 评估（若提供 model），否则保持"每步有无 observation"判断（MVP 过渡）。
    未完成 todo 交给 Todo Layer 2，Reflection 只在 todo 全完成时判实质充分。
    非研究类问题（personal）直接 pass + 提示，避免 reflection 死循环。
    """

    _RESEARCH_KEYWORDS = frozenset({
        "调研", "分析", "对比", "搜索", "报告", "研究", "深度", "了解",
        "收集", "整理", "调查", "评估", "综述", "总结", "查找", "查询",
        "research", "analyze", "compare", "search", "report", "study",
        "investigate", "survey", "review", "summary", "find", "deep dive",
    })

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def _classify_question(self, state: dict[str, Any]) -> str:
        """分类问题类型：research / personal / mixed。"""
        question = (state.get("research_question") or state.get("user_input") or "").lower()
        has_research_kw = any(kw in question for kw in self._RESEARCH_KEYWORDS)
        has_observations = bool(state.get("observations"))
        has_todos = bool(state.get("todos"))
        if has_todos or has_observations:
            return "research" if has_research_kw else "mixed"
        return "research" if has_research_kw else "personal"

    def reflect(self, state: dict[str, Any], runtime: Runtime) -> ReflectionAction:
        # 非研究类问题直接 pass + 提示，避免 reflection 死循环
        qtype = self._classify_question(state)
        if qtype == "personal":
            return ReflectionAction(
                verdict="pass", reflection_items=[], plan=None,
                guidance=(
                    "<system_reminder>\n"
                    "该问题似乎不需要深度研究。建议 /default 模式对话，"
                    "或提供更具体的研究方向。当前将直接回答。\n"
                    "</system_reminder>"
                ),
                rollback_to="",
            )

        todos = state.get("todos") or []
        if todos and not all(t.get("status") == "completed" for t in todos):
            return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")

        # F8.4：失败超阈放宽
        if _has_persistent_failures(state):
            return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")

        observations = state.get("observations") or []
        if not observations:
            return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")

        # F7：有 LLM 则交模型评估充分性；否则保持每步覆盖度判断（MVP 过渡）
        if self._llm is not None:
            return self._llm_evaluate(state, todos, observations)

        return self._rule_based_check(todos, observations)

    def _rule_based_check(self, todos: list, observations: list) -> ReflectionAction:
        """MVP 过渡：每步有无 observation 覆盖度判断。"""
        covered = {_step_id(o) for o in observations if _step_id(o)}
        missing = [f"todo-{i}" for i, _ in enumerate(todos) if f"todo-{i}" not in covered]
        if not missing:
            return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")
        item = ReflectionItem(
            item_id=_make_reflection_id(), scope="run", kind="gap",
            question=f"以下步骤标记完成但缺少证据覆盖：{', '.join(missing)}",
            related_refs=tuple(missing),
        )
        guidance = (
            "<system_reminder>\n"
            f"以下研究步骤已标记完成但缺少证据支撑：{', '.join(missing)}。\n"
            "请针对这些步骤补充搜索/调研，确保每步都有 observations 证据后再给出最终答案。\n"
            "</system_reminder>"
        )
        return ReflectionAction(verdict="continue", reflection_items=[item], plan=None, guidance=guidance, rollback_to="")

    def _llm_evaluate(self, state: dict[str, Any], todos: list, observations: list) -> ReflectionAction:
        """F7：LLM 评估充分性——简单问题不误判浅覆盖，复杂问题证据不足判 continue。"""
        question = state.get("research_question") or state.get("user_input") or ""
        todos_desc = "\n".join(f"- {t.get('content', '')}" for t in todos) if todos else "（无 todo）"
        obs_desc = "\n".join(
            f"- [{_step_id(o) or '-'}] {(_field(o, 'content') or '')[:200]}"
            for o in observations[:10]
        )
        prompt = get_prompt_manager().load(
            "reflection", "sufficiency",
            question=question, todos_desc=todos_desc, obs_desc=obs_desc,
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = self._llm.invoke([HumanMessage(content=prompt)], config={"tags": ["internal_llm"]})
            content = getattr(resp, "content", str(resp))
            import json
            # 容错解析 JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                if data.get("sufficient", True):
                    return ReflectionAction(verdict="pass", reflection_items=[], plan=None, guidance="", rollback_to="")
        except Exception:
            pass  # LLM 评估失败则回退规则判断
        # 判定为不充分或解析失败 → continue
        item = ReflectionItem(
            item_id=_make_reflection_id(), scope="run", kind="gap",
            question="LLM 评估证据不足以支撑研究结论",
            related_refs=(),
        )
        guidance = (
            "<system_reminder>\n"
            "LLM 评估认为当前证据不足以充分支撑研究结论，请补充搜索/调研后再给出最终答案。\n"
            "</system_reminder>"
        )
        return ReflectionAction(verdict="continue", reflection_items=[item], plan=None, guidance=guidance, rollback_to="")


class ReflectionMiddleware(AgentMiddleware):
    """外壳：管触发时机 + jump 预算；判断逻辑委托给 ReflectionStrategy。"""

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, strategy: ReflectionStrategy | None = None, llm: Any = None) -> None:
        self._strategy = strategy or SufficiencyStrategy(llm=llm)

    @staticmethod
    def _emit(runtime: Runtime, event_type: str, payload: dict[str, Any]) -> None:
        """F3: 发 reflection 事件到 journal（经 _get_runtime_value 取 journal）。"""
        journal = _get_runtime_value(runtime, "journal", None)
        if journal is not None:
            journal.append(event_type, payload)

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai or _has_tool_call_intent(last_ai):
            return None

        action = self._strategy.reflect(state if isinstance(state, dict) else dict(state), runtime)
        if action["verdict"] == "pass":
            return None

        # F3: 共享 jump 预算门（与 Todo 合计 ≤3）；预算耗尽放行时发事件。
        if not _jump_budget.try_consume(runtime):
            self._emit(runtime, "reflection.budget_exhausted", {"max": _jump_budget._MAX_TOTAL_JUMPS})
            return None

        # F3: 触发 jump 时发 reflection.fired 事件。
        gap_steps = [getattr(it, "related_refs", ()) for it in action["reflection_items"]]
        self._emit(runtime, "reflection.fired", {
            "verdict": action["verdict"],
            "gap_steps": gap_steps,
            "remaining_budget": _jump_budget.remaining(runtime),
        })

        update: dict[str, Any] = {"reflection_items": action["reflection_items"]}
        if action["guidance"]:
            update["messages"] = [HumanMessage(
                name="reflection",
                additional_kwargs={"hide_from_ui": True},
                content=action["guidance"],
            )]
        if action["plan"]:
            update["plan"] = action["plan"]
        return {"jump_to": "model", **update}

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
