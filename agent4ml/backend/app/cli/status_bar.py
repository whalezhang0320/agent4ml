"""status_bar — prompt_toolkit bottom_toolbar 渲染。

纯函数：读 cli_state 里的 mode/model/token/fraction，返回 HTML formatted text 供
``PromptSession(bottom_toolbar=...)`` 调用。每次 UI 刷新都会被调用，天然支持实时更新。

与 rich 输出不冲突——bottom_toolbar 占终端最后一行，rich Live spinner 在流式输出期间
接管，prompt_async 期间 bottom_toolbar 激活，两者不重叠。
"""

from __future__ import annotations

from typing import Any

from prompt_toolkit.formatted_text import HTML


def build_bottom_toolbar(cli_state: dict[str, Any]) -> HTML:
    """渲染常驻状态栏：``mode · model | {tokens}K ({fraction}%) | /help``。

    Args:
        cli_state: 主循环持有的状态 dict，需含 ``mode``/``model``/``current_tokens``/
            ``current_fraction`` 四个 key（缺省时回退到安全默认值）。

    Returns:
        prompt_toolkit HTML 对象，供 ``bottom_toolbar`` callable 返回。
    """
    mode = cli_state.get("mode", "default")
    model = cli_state.get("model", "?")
    tokens = cli_state.get("current_tokens", 0)
    fraction = cli_state.get("current_fraction", 0.0)
    running = cli_state.get("_running", False)
    activity = cli_state.get("_current_activity", "")

    tokens_k = tokens / 1000.0
    pct = fraction * 100.0

    activity_part = f" ● {activity} " if running and activity else ""
    return HTML(
        f'<style fg="#aaaaaa" bg="#2b2b2b">'
        f" {mode} · {model} "
        f"</style>"
        f'<style fg="#aaaaaa" bg="#2b2b2b">'
        f"| {tokens_k:.1f}K ({pct:.1f}%) |{activity_part}/help"
        f"</style>"
    )
