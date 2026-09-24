"""commands — CLI 命令系统（/help /clear /expert /default /report /exit /expand /thinking /tools /model /thread /prompt）。

命令以 / 开头，``handle_command`` 从 ``CommandRegistry`` 查 handler 分发。
返回 True 表示退出 CLI，False 继续循环。

``_cmd_help`` 文案与 ``/`` 补全菜单的 ``display_meta`` 共用同一份 ``CommandSpec.description``，
避免两处维护。``CommandRegistry.register_skill()`` 预留接口见 ``registry.py``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

from agent4ml.backend.app.cli.registry import CommandRegistry, CommandSpec
from agent4ml.backend.app.cli.stream_handler import StreamRenderer


@dataclass
class CommandContext:
    """单次命令调用的上下文——统一 ``_cmd_*`` 函数签名，让 ``handle_command`` 能通过 registry 泛化分发。"""

    console: Console
    renderer: StreamRenderer
    state: dict[str, Any]
    runtime: Any
    arg: str


# ---- handler 函数（签名统一为 ctx → bool | None） ----


def _cmd_help(ctx: CommandContext) -> None:
    ctx.console.print("[bold]Available commands:[/bold]")
    for spec in _registry.list_all():
        ctx.console.print(f"  [cyan]{spec.name}[/cyan]     {spec.description}")


def _cmd_clear(ctx: CommandContext) -> None:
    ctx.console.clear()


def _cmd_expert(ctx: CommandContext) -> None:
    ctx.state["pending_expert_mode"] = True
    ctx.console.print("[green]Mode will switch to expert next round[/green]")


def _cmd_default(ctx: CommandContext) -> None:
    ctx.state["pending_expert_mode"] = False
    ctx.console.print("[green]Mode will switch to default next round[/green]")


def _cmd_report(ctx: CommandContext) -> None:
    """标记 pending_report，主循环检测后调 _trigger_report（避免 commands.py 依赖 main.py）。

    空字符串 "" 表"pending 无 topic"，None 表"未设 pending"——区分避免 sentinel 冲突。
    """
    ctx.state["pending_report"] = ctx.arg.strip()


def _cmd_exit(ctx: CommandContext) -> bool:
    return True


def _cmd_expand(ctx: CommandContext) -> None:
    ctx.renderer.expand_last_round()


def _cmd_thinking(ctx: CommandContext) -> None:
    # 语义：toggle Thought 折叠行（非逐 token 流）——与 stream_handler._render_thinking 联动
    if ctx.arg == "off":
        ctx.renderer.state["thinking_enabled"] = False
        ctx.console.print("[green]Thinking display off[/green]")
    elif ctx.arg == "on":
        ctx.renderer.state["thinking_enabled"] = True
        ctx.console.print("[green]Thinking display on[/green]")
    else:
        current = "on" if ctx.renderer.state["thinking_enabled"] else "off"
        ctx.console.print(f"[yellow]Usage: /thinking on|off (current: {current})[/yellow]")


def _cmd_tools(ctx: CommandContext) -> None:
    try:
        from agent4ml.backend.agents.agent_tools.available import get_available_tools
        tools = get_available_tools(include_mcp=True)
        ctx.console.print("[bold]Available tools:[/bold]")
        for t in tools:
            ctx.console.print(f"  [cyan]{t.name}[/cyan]")
    except Exception as exc:
        ctx.console.print(f"[red]Failed to list tools: {exc}[/red]")


def _cmd_model(ctx: CommandContext) -> None:
    arg = ctx.arg.strip()
    # 无参：显示当前路由链
    if not arg:
        try:
            from agent4ml.backend.agents.leader.agent import _resolve_actual_model_name
            name = _resolve_actual_model_name(ctx.runtime.capability_registry)
            ctx.console.print(f"[bold]Model routing:[/bold] [cyan]{name}[/cyan]")
        except Exception as exc:
            ctx.console.print(f"[red]Failed to get model info: {exc}[/red]")
        return

    # 有参：<provider> [model] —— 设 pending_model_switch，主循环检测后调 runtime.switch_model
    parts = arg.split(maxsplit=1)
    provider = parts[0]
    model = parts[1].strip() if len(parts) > 1 else None
    try:
        from agent4ml.backend.agents.config.provider_config import get_provider_config
        cfg = get_provider_config(provider)
    except Exception as exc:
        ctx.console.print(f"[red]Unknown provider '{provider}': {exc}[/red]")
        return
    if cfg.provider not in ("fake",) and not cfg.api_key:
        ctx.console.print(f"[red]Provider '{provider}' has no API key in .env (set {provider.upper()}_API_KEY)[/red]")
        return
    model_label = model or cfg.model
    ctx.state["pending_model_switch"] = (provider, model)
    ctx.console.print(f"[green]Model will switch to {provider}/{model_label} next round[/green]")


def _cmd_thread(ctx: CommandContext) -> None:
    ctx.console.print(f"[bold]Thread ID:[/bold] [cyan]{ctx.runtime.thread_id}[/cyan]")
    ctx.console.print(f"[bold]Thread dir:[/bold] [dim]{ctx.runtime.thread_dir}[/dim]")
    try:
        import json
        events_path = ctx.runtime.thread_journal.events_path
        if events_path.exists():
            lines = events_path.read_text(encoding="utf-8").strip().split("\n\n")
            runs = []
            for block in lines[-10:]:
                if block.strip():
                    try:
                        evt = json.loads(block.split("\n")[0] if "\n" in block else block)
                        if evt.get("event_type") == "run.started":
                            runs.append(evt.get("run_id", "?"))
                    except Exception:
                        pass
            if runs:
                ctx.console.print("[bold]Recent runs:[/bold]")
                for rid in runs[-5:]:
                    ctx.console.print(f"  [dim]{rid}[/dim]")
    except Exception:
        pass


def _cmd_prompt(ctx: CommandContext) -> None:
    """Prompt 管理命令：/prompt list | /prompt show <cat/name> | /prompt reload"""
    from agent4ml.backend.agents.prompts import get_prompt_manager

    pm = get_prompt_manager()
    arg = ctx.arg
    if not arg or arg == "list":
        prompts = pm.list_prompts()
        if not prompts:
            ctx.console.print("[dim]No prompts found[/dim]")
            return
        ctx.console.print("[bold]Available prompts:[/bold]")
        for p in prompts:
            text, source = pm.load_raw(p.split("/")[0], p.split("/")[1])
            tag = "[green](user)[/green]" if source == "user" else "[dim](system)[/dim]"
            ctx.console.print(f"  [cyan]{p}[/cyan] {tag}")
        return

    parts = arg.split(maxsplit=1)
    if parts[0] == "show" and len(parts) > 1:
        ref = parts[1].strip()
        if "/" not in ref:
            ctx.console.print("[yellow]Usage: /prompt show <category/name>[/yellow]")
            return
        cat, name = ref.split("/", 1)
        try:
            text, source = pm.load_raw(cat, name)
            tag = f"[green][{source}][/green]" if source == "user" else f"[dim][{source}][/dim]"
            ctx.console.print(f"{tag} [cyan]{cat}/{name}[/cyan]:")
            ctx.console.print(text)
        except FileNotFoundError:
            ctx.console.print(f"[red]Prompt not found: {ref}[/red]")
        return

    if parts[0] == "reload":
        pm.clear_cache()
        ctx.console.print("[green]Prompt cache cleared — next load reads from .md files[/green]")
        return

    ctx.console.print("[yellow]Usage: /prompt list | /prompt show <cat/name> | /prompt reload[/yellow]")


def _cmd_skill(ctx: CommandContext) -> None:
    """Skill 命令：list | <name> | off | enable <name> | disable <name> | install <path> [name]。

    - <name>：持久 override（每轮生效，直到 off）
    - off：清 override（不禁用 skill 本身，agent 仍可自动 select）
    - enable/disable <name>：运行时持久 enable/disable（store.set_enabled，跨重启）
    - install <path> [name]：parser.install 拷到 skills/ + re-discover（S2）
    """
    arg = ctx.arg.strip()
    if arg == "list":
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        skills = mgr.list_skills()
        if not skills:
            ctx.console.print("[dim]No skills registered[/dim]")
            return
        ctx.console.print("[bold]Active skills:[/bold]")
        for s in skills:
            tools = ",".join(s["allowed_tools"]) if s["allowed_tools"] else "-"
            ctx.console.print(
                f"  [cyan]{s['name']}[/cyan] eff={s['effective_rate']:.0%} "
                f"sel={s['total_selections']} tools={tools}"
            )
        return
    if arg.startswith("search "):
        query = arg[7:].strip()
        if not query:
            ctx.console.print("[yellow]Usage: /skill search <query>[/yellow]")
            return
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        # H7 改造：优先用 hub unified_search（跨 source），降级为 builtin
        try:
            from agent4ml.backend.agents.skill.hub.search import unified_search_as_dicts

            results = unified_search_as_dicts(query)
        except Exception:
            # hub 不可用降级为只搜 builtin
            results = mgr.search_builtin_skills(query)
        if not results:
            ctx.console.print(f"[dim]No skills matching '{query}'[/dim]")
            return
        ctx.console.print(f"[bold]Skills matching '{query}':[/bold]")
        for r in results:
            source = r.get("source", "builtin")
            status = "[green]installed[/green]" if r.get("is_installed") or r.get("is_active") else "[dim]on-demand[/dim]"
            ctx.console.print(
                f"  [cyan]{r['name']}[/cyan] ({r.get('category', '-')}) [{source}] {status}"
                f"\n    {r.get('description', '')}"
                f"\n    install: /skill install {r.get('identifier', r['name'])}"
            )
        return
    if arg == "off":
        ctx.state["skill_override"] = []
        ctx.console.print("[green]Skill override cleared (skill still enabled, agent may auto-select)[/green]")
        return
    # enable/disable <name>：运行时持久
    parts = arg.split(maxsplit=1)
    if parts and parts[0] in ("enable", "disable"):
        if len(parts) < 2 or not parts[1].strip():
            ctx.console.print("[yellow]Usage: /skill enable|disable <name>[/yellow]")
            return
        name = parts[1].strip()
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        rec = mgr.store.get_active(name)
        if rec is None:
            ctx.console.print(f"[red]Skill not found: {name}[/red]")
            return
        enabled = parts[0] == "enable"
        if mgr.store.set_enabled(rec.skill_id, enabled):
            label = "enabled" if enabled else "disabled"
            ctx.console.print(f"[green]Skill '{name}' {label}[/green]")
        else:
            ctx.console.print(f"[red]Skill not found: {name}[/red]")
        return
    # evolve <name>：手动 FIX 进化
    if parts and parts[0] == "evolve":
        if len(parts) < 2 or not parts[1].strip():
            ctx.console.print("[yellow]Usage: /skill evolve <name>[/yellow]")
            return
        name = parts[1].strip()
        mgr = getattr(ctx.runtime, "skill_manager", None)
        evo = mgr.get_evolution_manager() if mgr else None
        if evo is None:
            ctx.console.print("[dim]Skill evolution not enabled (AGENT4ML_SKILL_EVOLVE_ENABLED=true)[/dim]")
            return
        try:
            rec = evo.evolve_skill(name)
            ctx.console.print(
                f"[green]Evolved '{name}'[/green] type={rec.evolution_type} "
                f"score={rec.eval_score:.2f} decision={rec.gate_decision}"
            )
        except ValueError as exc:
            ctx.console.print(f"[red]{exc}[/red]")
        except Exception as exc:
            ctx.console.print(f"[red]Evolve failed: {exc}[/red]")
        return
    # capture <pattern> <name>：手动 CAPTURED 沉淀新 skill
    if parts and parts[0] == "capture":
        rest = parts[1].strip() if len(parts) > 1 else ""
        cap_parts = rest.split(maxsplit=1)
        if len(cap_parts) < 2:
            ctx.console.print("[yellow]Usage: /skill capture <pattern> <name>[/yellow]")
            return
        pattern, cap_name = cap_parts[0], cap_parts[1].strip()
        mgr = getattr(ctx.runtime, "skill_manager", None)
        evo = mgr.get_evolution_manager() if mgr else None
        if evo is None:
            ctx.console.print("[dim]Skill evolution not enabled (AGENT4ML_SKILL_EVOLVE_ENABLED=true)[/dim]")
            return
        try:
            rec = evo.capture_skill(pattern, cap_name)
            ctx.console.print(
                f"[green]Captured '{cap_name}'[/green] score={rec.eval_score:.2f} "
                f"decision={rec.gate_decision}"
            )
        except Exception as exc:
            ctx.console.print(f"[red]Capture failed: {exc}[/red]")
        return
    # history <name>：查看 evolution 历史
    if parts and parts[0] == "history":
        if len(parts) < 2 or not parts[1].strip():
            ctx.console.print("[yellow]Usage: /skill history <name>[/yellow]")
            return
        name = parts[1].strip()
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        history = mgr.store.get_evolution_history(name)
        if not history:
            ctx.console.print(f"[dim]No evolution history for '{name}'[/dim]")
            return
        ctx.console.print(f"[bold]Evolution history for '{name}':[/bold]")
        for row in history:
            ctx.console.print(
                f"  [{row['gate_decision']}] type={row['evolution_type']} "
                f"trigger={row['trigger']} score={row['eval_score']:.2f} "
                f"ts={row['timestamp']}"
            )
        return
    # health [name]：RuntimeTracker 健康报告
    if parts and parts[0] == "health":
        name = parts[1].strip() if len(parts) > 1 else ""
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        from agent4ml.backend.agents.skill.eval.runtime_tracker import RuntimeTracker
        tracker = RuntimeTracker(mgr.store)
        if name:
            rec = mgr.store.get_active(name)
            if rec is None:
                ctx.console.print(f"[red]Skill not found: {name}[/red]")
                return
            report = tracker.health_report(rec.skill_id)
            ctx.console.print(
                f"[bold]{report.skill_name}[/bold] sel={report.window_selections} "
                f"eff={report.effective_rate:.0%} app={report.applied_rate:.0%} "
                f"comp={report.completion_rate:.0%} fb={report.fallback_rate:.0%} "
                f"trend={report.trend}"
            )
            if report.advice:
                ctx.console.print(f"  [dim]{report.advice}[/dim]")
        else:
            for rec in mgr.store.list_active():
                report = tracker.health_report(rec.skill_id)
                ctx.console.print(
                    f"  [cyan]{report.skill_name}[/cyan] eff={report.effective_rate:.0%} "
                    f"trend={report.trend}"
                )
        return
    # eval-history <name>：查看 eval 历史（judgments）
    if parts and parts[0] == "eval-history":
        if len(parts) < 2 or not parts[1].strip():
            ctx.console.print("[yellow]Usage: /skill eval-history <name>[/yellow]")
            return
        name = parts[1].strip()
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        rec = mgr.store.get_active(name)
        if rec is None:
            ctx.console.print(f"[red]Skill not found: {name}[/red]")
            return
        judgments = mgr.store.get_judgments(rec.skill_id)
        if not judgments:
            ctx.console.print(f"[dim]No eval history for '{name}'[/dim]")
            return
        ctx.console.print(f"[bold]Eval history for '{name}':[/bold]")
        for j in judgments:
            applied = "[green]applied[/green]" if j.skill_applied else "[red]not applied[/red]"
            ctx.console.print(
                f"  {applied} task={j.task_id} ts={j.timestamp}"
                + (f"\n    {j.deviation_note}" if j.deviation_note else "")
            )
        return
    # install <path> [name]：parser.install 拷到 skills/ + re-discover
    if parts and parts[0] == "install":
        rest = parts[1].strip() if len(parts) > 1 else ""
        if not rest:
            ctx.console.print("[yellow]Usage: /skill install <path|identifier> [name][/yellow]")
            return
        mgr = getattr(ctx.runtime, "skill_manager", None)
        if mgr is None:
            ctx.console.print("[dim]Skill module not enabled[/dim]")
            return
        from pathlib import Path
        path_parts = rest.split()
        src_or_identifier = path_parts[0]
        name = path_parts[1] if len(path_parts) > 1 else None

        # H7 改造：判断是 remote identifier 还是本地路径
        is_remote = any(
            src_or_identifier.startswith(prefix)
            for prefix in ("github:", "well-known:", "claude-marketplace:", "builtin:")
        )

        if is_remote:
            # H7 改造：用 hub Installer 安装 remote skill
            try:
                from agent4ml.backend.agents.skill.hub.installer import Installer
                from agent4ml.backend.agents.skill.hub.sources.builtin_source import (
                    BuiltinSource,
                )
                from agent4ml.backend.agents.skill.hub.sources.github_source import (
                    GitHubSource,
                )
                from agent4ml.backend.agents.skill.hub.sources.well_known_source import (
                    WellKnownSource,
                )
                from agent4ml.backend.agents.skill.hub.sources.claude_marketplace_source import (
                    ClaudeMarketplaceSource,
                )

                installer = Installer(
                    sources={
                        "builtin": BuiltinSource(skill_manager=mgr),
                        "github": GitHubSource(),
                        "well-known": WellKnownSource(),
                        "claude-marketplace": ClaudeMarketplaceSource(),
                    },
                    dest_root=Path("skills"),
                )
                skill_id = installer.install(src_or_identifier, name=name)
                mgr.load_startup()  # re-discover
                ctx.console.print(f"[green]Skill installed (id={skill_id})[/green]")
            except ValueError as exc:
                ctx.console.print(f"[red]Install failed: {exc}[/red]")
            except Exception as exc:
                ctx.console.print(f"[red]Install failed: {exc}[/red]")
        else:
            # 本地路径安装（既有逻辑）
            from agent4ml.backend.agents.skill.parser import install as install_skill
            src = Path(src_or_identifier)
            if name is None:
                name = src.name
            dest_root = Path("skills")
            try:
                skill_id = install_skill(src, name, dest_root)
                mgr.load_startup()  # re-discover（idempotent upsert）
                ctx.console.print(f"[green]Skill '{name}' installed (id={skill_id})[/green]")
            except FileNotFoundError:
                ctx.console.print(f"[red]Source path not found or no SKILL.md: {src}[/red]")
            except ValueError as exc:
                ctx.console.print(f"[red]Install failed: {exc}[/red]")
            except Exception as exc:
                ctx.console.print(f"[red]Install failed: {exc}[/red]")
        return
    if not arg:
        cur = ctx.state.get("skill_override") or []
        cur_label = ",".join(cur) if cur else "(none)"
        ctx.console.print(
            f"[yellow]Usage: /skill <name> | /skill search <query> | /skill off (clear override) | "
            f"/skill enable <name> | /skill disable <name> | /skill install <path|identifier> [name] | "
            f"/skill evolve <name> | /skill capture <pattern> <name> | /skill history <name> | /skill list"
            f"  (current override: {cur_label})[/yellow]"
        )
        return
    ctx.state["skill_override"] = [arg]
    ctx.console.print(f"[green]Skill '{arg}' activated[/green]")


def _cmd_mcp(ctx: CommandContext) -> None:
    """MCP 命令：list | reload。

    - list：展示 servers + transport + 工具数 + 健康状态
    - reload：设 pending_mcp_reload，主循环检测后 runtime.reload_mcp_tools() 重建 graph
    """
    arg = ctx.arg.strip()
    mgr = getattr(ctx.runtime, "mcp_manager", None)
    if arg == "list":
        if mgr is None:
            ctx.console.print("[dim]MCP module not enabled[/dim]")
            return
        servers = mgr.list_servers()
        if not servers:
            ctx.console.print("[dim]No MCP servers configured[/dim]")
            return
        ctx.console.print("[bold]MCP servers:[/bold]")
        for s in servers:
            health_color = "green" if s["health_state"] == "healthy" else "red"
            ctx.console.print(
                f"  [cyan]{s['name']}[/cyan] {s['transport']} "
                f"tools={s['tool_count']} "
                f"[{health_color}]{s['health_state']}[/{health_color}]"
            )
        return
    if arg == "reload":
        ctx.state["pending_mcp_reload"] = True
        ctx.console.print("[green]MCP tools will reload next round[/green]")
        return
    ctx.console.print("[yellow]Usage: /mcp list | /mcp reload[/yellow]")


# ---- 模块级 registry：注册全部 builtin 命令 ----

_registry = CommandRegistry()
_registry.register(CommandSpec("/help", "Show this help", _cmd_help))
_registry.register(CommandSpec("/clear", "Clear screen", _cmd_clear))
_registry.register(CommandSpec("/expert", "Switch to expert mode (deep research), applies next round", _cmd_expert))
_registry.register(CommandSpec("/default", "Switch to default mode (lightweight chat), applies next round", _cmd_default))
_registry.register(CommandSpec("/report", "Generate report from current thread; optional topic", _cmd_report))
_registry.register(CommandSpec("/exit", "Exit (also /quit)", _cmd_exit))
_registry.register(CommandSpec("/quit", "Exit (alias of /exit)", _cmd_exit))
_registry.register(CommandSpec("/expand", "Expand last round tool results and Thought", _cmd_expand))
_registry.register(CommandSpec("/thinking", "Toggle Thought fold row display (on|off)", _cmd_thinking))
_registry.register(CommandSpec("/tools", "List available tools", _cmd_tools))
_registry.register(CommandSpec("/model", "Show or switch model (<provider> [model]); applies next round", _cmd_model))
_registry.register(CommandSpec("/thread", "Show thread info", _cmd_thread))
_registry.register(CommandSpec("/prompt", "Prompt management (list|show <cat/name>|reload)", _cmd_prompt))
_registry.register(CommandSpec("/skill", "Skill control (list|search|<name>|off|enable|disable|install|evolve|capture|history)", _cmd_skill))
_registry.register(CommandSpec("/mcp", "MCP control (list|reload)", _cmd_mcp))


def get_registry() -> CommandRegistry:
    """供 ``main.py`` 构造 ``SlashCommandCompleter`` 时获取注册表实例。"""
    return _registry


def handle_command(
    cmd: str,
    console: Console,
    renderer: StreamRenderer,
    state: dict[str, Any],
    runtime: Any,
) -> bool:
    """处理 / 命令。返回 True=退出 CLI，False=继续。

    从 ``CommandRegistry.get(name)`` 查 handler，构造 ``CommandContext`` 后调用。
    未命中时打印 ``Unknown command`` 并返回 False。
    """
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    spec = _registry.get(name)
    if spec is None:
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        return False

    ctx = CommandContext(console=console, renderer=renderer, state=state, runtime=runtime, arg=arg)
    result = spec.handler(ctx)
    return result if isinstance(result, bool) else False
