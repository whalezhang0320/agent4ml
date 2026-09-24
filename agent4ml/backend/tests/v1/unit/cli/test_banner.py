from agent4ml.backend.app.cli.banner import render_banner


def test_banner_uses_cold_gradient_pixel_style() -> None:
    banner = render_banner("AGENT4ML")

    # render_banner 返回 rich Text 对象
    text_str = str(banner)

    # Box-drawing pixel font
    assert "█" in text_str
    # Version line
    assert "v1.0.0" in text_str
    # Tagline
    assert "The little grey cells are working..." in text_str
    # 冷色系渐变（不再用暖色 #FFB347）
    assert "\x1b[38;2;255;179;71m" not in text_str  # 无手写 ANSI 码
    # 装饰移除
    assert "♦" not in text_str
    assert "★" not in text_str
    assert ":-)" not in text_str
    assert "( * )" not in text_str
