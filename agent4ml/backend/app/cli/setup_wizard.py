"""首次启动配置向导 — 检测 .env 不存在时引导用户完成最小配置。

设计原则：
- 接口小：ensure_config(project_root) → bool，一个入口
- 不依赖 .env（向导运行前 .env 不存在，不能读 env）
- 用 rich 渲染 + input() 交互，不引入新依赖
- 生成的 .env 从 .env.example 模板填充，保证字段完整
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_console = Console()


def ensure_config(project_root: Path) -> bool:
    """检查 .env 是否存在，不存在则启动交互式配置向导。

    Docker 模式下 env_file 注入环境变量但容器内无 .env 文件——
    若已检测到任意 provider API key 在环境变量中，跳过向导。

    Returns:
        True = 配置已就绪（文件存在 / env 已注入 / 刚创建），False = 用户取消
    """
    env_path = project_root / ".env"
    if env_path.exists():
        return True
    # Docker env_file 模式：env 变量已注入但 .env 文件不在容器内
    if _has_any_provider_key():
        return True
    return _run_wizard(project_root, env_path)


def _has_any_provider_key() -> bool:
    """检查环境变量中是否已有任意 provider 的 API key。"""
    import os
    from agent4ml.backend.agents.config.provider_profile import PROVIDER_PROFILES

    for profile in PROVIDER_PROFILES:
        if profile.name == "fake":
            continue
        if os.environ.get(profile.env_key, "").strip():
            return True
    return False


def _run_wizard(project_root: Path, env_path: Path) -> bool:
    """交互式配置向导主流程。"""
    # 延迟导入，避免在已有 .env 时加载 provider_profile
    from agent4ml.backend.agents.config.provider_profile import PROVIDER_PROFILES

    _console.print(Panel(
        Text("Welcome to Agent4ML\n", justify="center", style="bold cyan")
        + Text("首次使用需要配置 LLM API Key。本向导将帮你生成 .env 配置文件。\n", justify="center", style="dim")
        + Text("按 Ctrl+C 可随时取消。", justify="center", style="dim"),
        border_style="cyan",
    ))

    # 可选 provider（排除 fake）
    selectable = [p for p in PROVIDER_PROFILES if p.name != "fake"]

    configs: dict[str, dict[str, str]] = {}  # provider_name → {api_key, base_url, model}

    try:
        while True:
            _console.print("\n[bold]可用 LLM Provider:[/bold]")
            for i, p in enumerate(selectable, 1):
                marker = "[green]✓[/green]" if p.name in configs else " "
                default_tag = " [dim](默认)[/dim]" if p.is_default else ""
                _console.print(f"  {marker} [{i}] [cyan]{p.name}[/cyan]{default_tag} — {p.default_model}")

            choice = input("\n选择 provider 编号（回车跳过）: ").strip()
            if not choice:
                if not configs:
                    _console.print("[yellow]至少配置一个 provider。[/yellow]")
                    continue
                break
            try:
                idx = int(choice) - 1
                profile = selectable[idx]
            except (ValueError, IndexError):
                _console.print("[red]无效选择。[/red]")
                continue

            # 输入 API key
            key = input(f"输入 {profile.name} API Key: ").strip()
            if not key and not profile.no_key_required:
                _console.print("[yellow]API Key 为空，跳过此 provider。[/yellow]")
                continue

            configs[profile.name] = {
                "api_key": key,
                "base_url": "",  # 空则用 .env.example 默认
                "model": "",     # 空则用 .env.example 默认
            }
            _console.print(f"[green]✓ {profile.name} 已配置[/green]")

            more = input("\n继续添加其他 provider？(y/N): ").strip().lower()
            if more != "y":
                break

        # 可选功能
        _console.print("\n[bold]可选功能（回车=跳过）:[/bold]")
        skill_enabled = _ask_yes_no("启用 Skill 系统？", default=False)
        mcp_enabled = _ask_yes_no("启用 MCP 工具？", default=False)
        sandbox = _ask_sandbox()

        # 生成 .env
        content = _build_env_content(project_root, configs, skill_enabled, mcp_enabled, sandbox)
        env_path.write_text(content, encoding="utf-8")

        _console.print(f"\n[green]✓ 配置文件已生成: {env_path}[/green]")
        _console.print("[dim]后续可通过编辑 .env 修改配置，或删除 .env 重新运行向导。[/dim]\n")
        return True

    except (KeyboardInterrupt, EOFError):
        _console.print("\n[yellow]配置已取消。[/yellow]")
        return False


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """是/否提问，回车取默认值。"""
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} ({hint}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _ask_sandbox() -> str:
    """沙箱配置选择。返回 provider 路径或空串。"""
    _console.print("  [1] Local — 本机进程（开发调试）")
    _console.print("  [2] Docker — 容器隔离（需 Docker）")
    _console.print("  [回车] 不启用")
    choice = input("选择沙箱模式: ").strip()
    if choice == "1":
        return "agent4ml.backend.agents.sandbox.local.local_sandbox_provider:LocalSandboxProvider"
    if choice == "2":
        return "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider:DockerSandboxProvider"
    return ""


def _build_env_content(
    project_root: Path,
    configs: dict[str, dict[str, str]],
    skill: bool,
    mcp: bool,
    sandbox: str,
) -> str:
    """从模板 .env.example 读取，填充用户输入的值，生成 .env 内容。"""
    template_path = project_root / ".env.example"
    if template_path.exists():
        lines = template_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = _MINIMAL_TEMPLATE.splitlines()

    # 构建 key→value 覆盖映射（大写 env 变量名）
    overrides: dict[str, str] = {}
    for provider_name, cfg in configs.items():
        prefix = provider_name.upper()
        if cfg["api_key"]:
            overrides[f"{prefix}_API_KEY"] = cfg["api_key"]
        # base_url 和 model 留空则保留模板默认值

    # 功能开关
    overrides["AGENT4ML_SKILL_ENABLED"] = "true" if skill else "false"
    overrides["AGENT4ML_MCP_ENABLED"] = "true" if mcp else "false"
    if sandbox:
        overrides["AGENT4ML_SANDBOX_USE"] = sandbox

    # 逐行替换：找到 KEY= 行，用 override 值替换
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in overrides:
                result.append(f"{key}={overrides[key]}")
                continue
        result.append(line)
    return "\n".join(result) + "\n"


# 最小模板（.env.example 不存在时的 fallback）
_MINIMAL_TEMPLATE = """\
# Agent4ML .env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENT4ML_SKILL_ENABLED=false
AGENT4ML_MCP_ENABLED=false
AGENT4ML_SANDBOX_USE=
"""


def update_env_file(env_path: Path, overrides: dict[str, str]) -> None:
    """更新 .env 文件中指定 key 的值，保留其他行原样。

    供 TUI 配置面板（Ctrl+B）调用：读取当前 .env → 替换 KEY= 行 → 写回。
    """
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in overrides:
                result.append(f"{key}={overrides[key]}")
                continue
        result.append(line)
    env_path.write_text("\n".join(result) + "\n", encoding="utf-8")

