"""Agent4ML CLI banner — rich 渲染，冷色系渐变（cyan→blue→purple）。

不再用 colorama 手写 ANSI 码——改用 rich Text + Style，避免与 rich Console 冲突。
"""

from __future__ import annotations

from rich.text import Text

# 冷色系渐变：cyan → blue → indigo → purple，6 行逐行变色
_GRADIENT = [
    "#00CED1",  # dark turquoise
    "#00BFFF",  # deep sky blue
    "#1E90FF",  # dodger blue
    "#4169E1",  # royal blue
    "#6A5ACD",  # slate blue
    "#8A2BE2",  # blue violet
]
_VERSION_COLOR = "#48D1CC"  # medium turquoise

# AGENT4ML in box-drawing pixel font, 6 rows.
_PIXEL_AGENT4ML = [
    "██████╗  ██████╗ ██╗██████╗  ██████╗ ████████╗",
    "██╔══██╗██╔═══██╗██║██╔══██╗██╔═══██╗╚══██╔══╝",
    "██████╔╝██║   ██║██║██████╔╝██║   ██║   ██║   ",
    "██╔═══╝ ██║   ██║██║██╔══██╗██║   ██║   ██║   ",
    "██║     ╚██████╔╝██║██║  ██║╚██████╔╝   ██║   ",
    "╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   ",
]


def render_logo() -> Text:
    """仅返回渐变 ASCII logo（不含版本/tagline），供 TUI 居中展示。

    每行独立着色，行尾不留多余空格，便于 ``width: auto`` 容器精确居中。
    """
    result = Text(justify="center")
    for idx, row in enumerate(_PIXEL_AGENT4ML):
        color = _GRADIENT[idx % len(_GRADIENT)]
        result.append(row, style=color)
        if idx < len(_PIXEL_AGENT4ML) - 1:
            result.append("\n")
    return result


def render_banner(text: str = "AGENT4ML") -> Text:
    """返回 rich Text 对象，供 console.print() 渲染。

    冷色系渐变 ASCII art + 版本号 + tagline。不用手写 ANSI 码。
    """
    result = Text()

    upper = text.upper()
    if upper == "AGENT4ML":
        for idx, row in enumerate(_PIXEL_AGENT4ML):
            color = _GRADIENT[idx % len(_GRADIENT)]
            result.append(row, style=color)
            result.append("\n")
    else:
        result.append(upper, style=_GRADIENT[0])
        result.append("\n")

    result.append("\n")
    result.append("v1.0.0  |  ", style=_VERSION_COLOR)
    result.append("The little grey cells are working...", style="italic " + _VERSION_COLOR)

    return result
