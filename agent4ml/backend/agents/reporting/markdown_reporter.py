from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.reporting.result import ReportResult


class MarkdownReporter:
    def generate_report(self, thread_state: dict[str, Any], run_context: Any) -> ReportResult:
        question = (
            thread_state.get("research_question")
            or thread_state.get("user_input")
            or "Research report"
        )
        observations = thread_state.get("observations", [])
        sources = thread_state.get("sources", [])
        final_report_field = thread_state.get("final_report")

        # 三级 fallback 优先级：
        # ① final_report 字段（ReportMiddleware 已合成完整报告，最优）
        # ② 渲染 observations/sources（结构化兜底，ReportMiddleware 未跑但有证据）
        # ③ _last_ai_message（无证据：fast 模式 / 单轮对话，保留旧行为）
        if final_report_field:
            final_report = final_report_field
        elif observations:
            final_report = _render_structured(question, observations, sources)
        else:
            ai_answer = _last_ai_message(thread_state)
            body = ai_answer or "No answer collected."
            final_report = f"# {question}\n\n{body}"

        # F1: report.generated 事件仅由 agent.py（LeaderAgent.run）发出，reporter 只产文本。
        return ReportResult(final_report=final_report)


def _render_structured(question: str, observations: list[Any], sources: list[Any]) -> str:
    """Tier 2 兜底：把 observations + sources 渲染成结构化 Markdown。"""
    lines = [f"# {question}", "", "## Summary", _summary(observations, ""), "", "## Findings"]
    if observations:
        lines.extend(f"- {_field(obs, 'content')}" for obs in observations)
    else:
        lines.append("- No observations collected.")
    lines += ["", "## Sources"]
    if sources:
        lines.extend(
            f"- [{_field(src, 'title') or _field(src, 'url')}]({_field(src, 'url')})"
            for src in sources
        )
    else:
        lines.append("- No sources.")
    return "\n".join(lines)


def _last_ai_message(thread_state: dict[str, Any]) -> str:
    messages = thread_state.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        type_name = type(msg).__name__
        if content and "AI" in type_name:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    item["text"] if isinstance(item, dict) and "text" in item else str(item)
                    for item in content
                    if item
                ]
                return "".join(parts)
    return ""


def _summary(observations: list[Any], ai_answer: str) -> str:
    if observations:
        return _field(observations[0], "content")
    if ai_answer:
        return ai_answer
    return "No evidence collected yet."


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
