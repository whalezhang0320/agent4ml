"""stream_handler — StreamEvent → rich 渲染。

消费 Agent4MLStreamClient 产出的 StreamEvent，用 rich Console 实时渲染：
- thinking：暗灰色逐 token（style="dim"）
- answer：正常色逐 token，done 后 Markdown 整体渲染
- tool_start：Spinner + 工具名 + 参数摘要
- tool_end：停 Spinner，折叠摘要 ✓ tool → N results
- error：红色
- done：换行 + 分隔线
"""

from __future__ import annotations

from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from agent4ml.backend.app.services.stream_service import StreamEvent


def _truncate_args(args: dict | None) -> str:
    if not args:
        return ""
    items = [f"{k}={str(v)[:50]}" for k, v in args.items() if k != "type"]
    return ", ".join(items)[:120]


def _result_summary(result: str | None) -> str:
    if not result:
        return "ok"
    if len(result) <= 80:
        return result
    return result[:77] + "..."


def _tool_color(tool_name: str) -> str:
    """按工具名前缀分色（D8）：search=cyan / fetch=blue / write=yellow / 其他=dim。"""
    name = (tool_name or "").lower()
    if name.startswith(("web_search", "tavily", "search")):
        return "cyan"
    if name.startswith(("fetch", "browse", "scrape")):
        return "blue"
    if name.startswith(("write", "save", "store", "persist")):
        return "yellow"
    return "dim"


class StreamRenderer:
    """渲染 StreamEvent 到 rich Console。

    state dict 存：
    - full_answer：累积 answer 文本（done 后 Markdown 渲染）
    - tool_results：上一轮工具结果全文（供 /expand）
    - thinking_enabled：是否展示 thinking（/thinking off 关闭）
    - current_spinner：当前 Live spinner（tool_start 时开，tool_end 时停）
    """

    def __init__(self, console: Console | None = None, cli_state: dict[str, Any] | None = None) -> None:
        self.console = console or Console()
        self._cli_state = cli_state
        self.state: dict[str, Any] = {
            "full_answer": "",
            "tool_results": [],
            "thinking_enabled": True,
            "_live": None,
            # Thought 折叠+计时：_thinking_t0 为 None 表示当前无活跃 thinking 段
            "thinking_log": [],          # [{"ms": int, "text": str}] 供 /expand
            "_thinking_t0": None,        # monotonic 起始时间
            "_thinking_buffer": "",      # 累计的 thinking 原文
            # 回答耗时尾行：main.py 在 round 开始时注入
            "round_t0": None,            # monotonic round 起始时间
            "model": None,               # 当前轮模型名（供 _render_done 尾行）
        }

    def render(self, event: StreamEvent) -> None:
        etype = event["type"]

        # budget_update：仅更新 cli_state（供 bottom_toolbar 实时刷新），不参与轮次状态机、不输出到 console
        if etype == "budget_update":
            self._update_budget(event)
            return

        # 新轮首个事件时清上一轮 state（保留 tool_results 供 /expand）
        if not self.state.get("_round_active"):
            self.state["full_answer"] = ""
            self.state["tool_results"] = []
            self.state["thinking_log"] = []
            self.state["_round_active"] = True

        # 从 thinking 切到其他事件类型时，先 flush 累计的 thinking 段（输出 `+ Thought: Xms`）
        if etype != "thinking" and self.state.get("_thinking_t0") is not None:
            self._flush_thinking()

        if etype == "thinking":
            self._render_thinking(event)
        elif etype == "answer":
            self._render_answer(event)
        elif etype == "tool_start":
            self._render_tool_start(event)
        elif etype == "tool_end":
            self._render_tool_end(event)
        elif etype == "done":
            self._render_done()
        elif etype == "error":
            self._render_error(event)
        elif etype == "compaction_start":
            self._render_compaction_start(event)
        elif etype == "compaction_progress":
            self._render_compaction_progress(event)
        elif etype == "compaction_end":
            self._render_compaction_end(event)

    def _render_thinking(self, event: StreamEvent) -> None:
        # /thinking off：完全跳过（不计时、不累计 buffer、不输出折叠行）
        if not self.state["thinking_enabled"]:
            return
        content = event["content"]
        if not content:
            return
        # 首个 thinking token 启动计时
        if self.state["_thinking_t0"] is None:
            import time
            self.state["_thinking_t0"] = time.monotonic()
        self.state["_thinking_buffer"] += content
        # 不再 console.print 逐 token——done 段时由 _flush_thinking 统一输出折叠行

    def _flush_thinking(self) -> None:
        """输出累计的 thinking 段为 `+ Thought: {ms}ms`（橙色 #FF8C42），原文存入 thinking_log。

        由 ``render()`` 在切到非 thinking 事件时调用。``_thinking_t0`` 为 None 时静默跳过。
        """
        t0 = self.state.get("_thinking_t0")
        if t0 is None:
            return
        import time
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        text = self.state.get("_thinking_buffer", "")
        self.console.print(f"[#FF8C42]+ Thought: {elapsed_ms}ms[/#FF8C42]")
        self.state["thinking_log"].append({"ms": elapsed_ms, "text": text})
        self.state["_thinking_t0"] = None
        self.state["_thinking_buffer"] = ""

    def _render_answer(self, event: StreamEvent) -> None:
        content = event["content"]
        if content:
            # 累积 answer 文本，不立即打印（done 后统一 Markdown 渲染，避免重复输出）
            self.state["full_answer"] += content

    def _render_tool_start(self, event: StreamEvent) -> None:
        tool_name = event["tool_name"] or "unknown"
        args_str = _truncate_args(event["tool_args"])
        spinner_text = f"{tool_name}({args_str})..." if args_str else f"{tool_name}..."

        # 停前一个 spinner（如有）
        self._stop_spinner()

        color = _tool_color(tool_name)
        spinner = Spinner("dots", text=Text(spinner_text, style=color))
        self.state["_live"] = Live(spinner, console=self.console, transient=True)
        self.state["_live"].start()

    def _render_tool_end(self, event: StreamEvent) -> None:
        self._stop_spinner()

        tool_name = event["tool_name"] or "unknown"
        summary = _result_summary(event["tool_result"])
        color = _tool_color(tool_name)
        self.console.print(f"  [{color}]✓ {tool_name}[/{color}] [dim]→ {summary}[/dim]", highlight=False)

        # 存全文供 /expand
        if event["tool_result"]:
            self.state["tool_results"].append({
                "tool": tool_name,
                "result": event["tool_result"],
            })

    def _render_done(self) -> None:
        self._stop_spinner()
        # answer 累积完毕，统一 Markdown 渲染输出一次（不重复纯文本）
        full = self.state["full_answer"].strip()
        if full:
            from agent4ml.backend.app.services.stream_service import _strip_skills_leak
            full = _strip_skills_leak(full)
        if full:
            self.console.print()
            self.console.print(Markdown(full))
        # 回答耗时尾行：■ Build · {model} · {elapsed}s（main.py 在 round 开始时注入 round_t0/model）
        round_t0 = self.state.get("round_t0")
        model = self.state.get("model")
        if round_t0 is not None and model:
            import time
            elapsed = time.monotonic() - round_t0
            self.console.print(f"[dim]■ Build · {model} · {elapsed:.1f}s[/dim]")
        self.state["full_answer"] = ""
        self.state["_round_active"] = False
        self.state["round_t0"] = None
        self.console.print("\n[dim]" + "─" * 40 + "[/dim]")

    def render_user_input(self, text: str) -> None:
        """用户输入卡片化回显——蓝紫竖线 Panel 包裹，在用户提交问题后立即调用。"""
        body = Text(text)
        self.console.print(Panel(body, border_style="#6A5ACD", box=ROUNDED, padding=(0, 1)))
        self.console.print()

    def _render_error(self, event: StreamEvent) -> None:
        self._stop_spinner()
        self.console.print(f"\n[red]✗ {event['content']}[/red]")

    def _render_compaction_start(self, event: StreamEvent) -> None:
        stage = event.get("tool_name") or ""
        self.console.print(f"\n[dim][compaction] {stage} 触发...[/dim]")

    def _render_compaction_progress(self, event: StreamEvent) -> None:
        self.console.print(f"[dim]  {event['content']}[/dim]")

    def _render_compaction_end(self, event: StreamEvent) -> None:
        saved = event.get("tool_result") or ""
        self.console.print(f"[dim][compaction] 完成 {event['content']} (saved={saved})[/dim]")

    def _stop_spinner(self) -> None:
        live = self.state.get("_live")
        if live is not None:
            live.stop()
            self.state["_live"] = None

    def _update_budget(self, event: StreamEvent) -> None:
        """budget_update 事件 → 更新 cli_state 供 bottom_toolbar 实时刷新。

        renderer 不直接渲染 budget 信息——展示归 ``status_bar.build_bottom_toolbar``
        负责；renderer 仅做数据透传。``cli_state`` 为 None 时（未注入）静默跳过。
        """
        if self._cli_state is None:
            return
        budget = event.get("budget")
        if not budget:
            return
        self._cli_state["current_tokens"] = budget.get("total", 0)
        self._cli_state["current_fraction"] = budget.get("fraction", 0.0)
        self._cli_state["current_window"] = budget.get("window", 0)

    def expand_last_round(self) -> None:
        """展开上一轮工具结果 + Thought 段原文（/expand 命令调用）。"""
        from rich.panel import Panel
        from rich.text import Text

        thinking_log = self.state.get("thinking_log", [])
        results = self.state.get("tool_results", [])

        if not results and not thinking_log:
            self.console.print("[dim]无上一轮工具结果或 Thought[/dim]")
            return

        # Thought 段在前（按时间顺序）
        for entry in thinking_log:
            body = Text(entry["text"], style="dim")
            self.console.print(
                Panel(body, title=f"[#FF8C42]Thought ({entry['ms']}ms)[/#FF8C42]", border_style="dim")
            )

        # 工具结果在后
        for r in results:
            body = Text(r["result"], style="dim")
            self.console.print(Panel(body, title=f"[cyan]{r['tool']}[/cyan]", border_style="dim"))
