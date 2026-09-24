"""PiInstaller — 后台安装 pi coding agent（决策 2：不阻塞主流程 + TUI 提示）。

设计（spec.md PiInstaller Requirement + design_docs/46 §10.4）:
- pi 装到与 Agent4ML 相同环境（沙箱镜像或 WSL2），保证 PATH 可见
- daemon thread 后台 npm install -g @earendil-works/pi-coding-agent --ignore-scripts
- 不阻塞主流程（subagent specialist 立即可用，pi 后台装）
- 装完写 flag file（~/.agent4ml/pi-installed.flag），下次启动 pi specialist 自动可用
- 首次启动无 coding specialist 时 TUI 显示提示（status 属性供 TUI 查询）
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class PiInstaller:
    """后台安装 Pi coding agent（不阻塞主流程，决策 2）。

    pi 装到与 Agent4ML 相同环境（沙箱镜像或 WSL2），保证 PATH 可见。
    """

    PACKAGE_NAME = "@earendil-works/pi-coding-agent"
    FLAG_FILE = Path.home() / ".agent4ml" / "pi-installed.flag"

    def __init__(self, auto_install: bool = True) -> None:
        self._auto_install = auto_install
        self._install_status = "not_started"  # not_started / installing / done / failed
        self._install_thread: threading.Thread | None = None

    def ensure_installed(self) -> bool:
        """检测 pi 是否已装；未装时启动后台安装（非阻塞）。

        返回：
        - True：pi 已装，立即可用
        - False：pi 不可用（未装 + auto_install=false，或 npm 不可用，或后台安装中）
        """
        if self._is_installed():
            return True

        if not self._auto_install:
            logger.warning(
                "[PiSpecialist] pi CLI not installed and auto_install=false. "
                "Install manually: npm install -g @earendil-works/pi-coding-agent"
            )
            return False

        if not self._is_npm_available():
            logger.warning(
                "[PiSpecialist] npm not available, cannot auto-install pi. "
                "Install Node.js first: https://nodejs.org/"
            )
            return False

        # 后台安装（不阻塞）
        self._install_status = "installing"
        self._install_thread = threading.Thread(
            target=self._install_in_background,
            daemon=True,
            name="pi-installer",
        )
        self._install_thread.start()
        logger.info(
            "[PiSpecialist] pi CLI not installed. "
            "Background install started (npm install -g %s). "
            "Pi specialist will be available after restart.",
            self.PACKAGE_NAME,
        )
        return False

    @property
    def status(self) -> str:
        """供 TUI 显示安装状态（决策 2）。

        返：
        - "not_started"：未启动安装
        - "installing"：后台安装中
        - "done"：安装完成（重启后可用）
        - "failed"：安装失败
        """
        return self._install_status

    def _is_installed(self) -> bool:
        """检测 pi 是否已装（PATH 查找 或 flag file 存在）。"""
        return shutil.which("pi") is not None or self.FLAG_FILE.exists()

    def _is_npm_available(self) -> bool:
        """检测 npm 是否可用（pi 是 npm 包）。"""
        return shutil.which("npm") is not None

    def _install_in_background(self) -> None:
        """后台 npm install（daemon thread，不阻塞）。

        决策 2：装到与 Agent4ML 相同环境（沙箱镜像或 WSL2）。
        装完写 flag file，下次启动 pi specialist 自动可用。
        Windows 兼容：npm 是 .cmd 文件，subprocess 需 shell=True 或全路径。
        """
        try:
            npm_path = shutil.which("npm")
            if not npm_path:
                self._install_status = "failed"
                logger.error("[PiSpecialist] npm not found in PATH")
                return
            result = subprocess.run(
                [
                    npm_path,
                    "install",
                    "-g",
                    "--ignore-scripts",  # Pi 官方推荐，不跑 lifecycle scripts
                    self.PACKAGE_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 分钟超时
            )
            if result.returncode == 0:
                self.FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
                self.FLAG_FILE.touch()
                self._install_status = "done"
                logger.info(
                    "[PiSpecialist] pi CLI installed successfully. "
                    "Restart Agent4ML to enable pi specialist."
                )
            else:
                self._install_status = "failed"
                logger.error(
                    "[PiSpecialist] pi install failed (exit %s): %s",
                    result.returncode,
                    result.stderr[:500],
                )
        except subprocess.TimeoutExpired:
            self._install_status = "failed"
            logger.error("[PiSpecialist] pi install timed out (5 min)")
        except Exception as e:
            self._install_status = "failed"
            logger.error("[PiSpecialist] pi install crashed: %s", e)
