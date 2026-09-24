"""PiInstaller 单测 — 后台 npm install + 状态查询 + TUI 提示接口（P3，决策 2）。

验证（tasks.md 3.2）：
- test_ensure_installed_already_installed: pi 已装，返 True，不启动后台安装
- test_ensure_installed_start_background: pi 未装 + auto_install=true，启动 daemon thread，返 False
- test_ensure_installed_no_npm: npm 不可用，返 False + warn
- test_ensure_installed_auto_install_false: auto_install=false，返 False + warn
- test_install_in_background_success: mock subprocess.run 返 0，验证 flag file 写入 + status="done"
- test_install_in_background_failure: mock subprocess.run 返非 0，验证 status="failed" + error log
- test_install_in_background_timeout: mock subprocess.run 抛 TimeoutExpired，验证 status="failed"
- test_status_property: status 属性供 TUI 查询
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.installer.pi_installer import PiInstaller


def _mock_pi_installed(installed: bool = True):
    """mock shutil.which 返 pi 路径或 None。"""
    return patch(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        return_value="/usr/local/bin/pi" if installed else None,
    )


def _mock_npm_available(available: bool = True):
    """mock shutil.which 返 npm 路径或 None（需区分 pi 和 npm 的 which 调用）。"""
    def _which_side_effect(cmd: str):
        if cmd == "pi":
            return None  # pi 未装
        if cmd == "npm":
            return "/usr/local/bin/npm" if available else None
        return None
    return patch(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        side_effect=_which_side_effect,
    )


# ---------------------------------------------------------------------------
# ensure_installed
# ---------------------------------------------------------------------------


def test_ensure_installed_already_installed(tmp_path):
    """pi 已装（PATH 查找）→ 返 True，不启动后台安装。"""
    installer = PiInstaller(auto_install=True)
    with _mock_pi_installed(True):
        with patch.object(installer, "_install_in_background") as mock_bg:
            result = installer.ensure_installed()
    assert result is True
    mock_bg.assert_not_called()  # 不启动后台安装


def test_ensure_installed_already_installed_via_flag_file(tmp_path, monkeypatch):
    """pi flag file 存在 → 返 True（即使 PATH 查不到）。"""
    # mock PATH 查不到 pi
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: None,
    )
    # mock flag file 存在
    mock_flag = tmp_path / "pi-installed.flag"
    mock_flag.touch()
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        mock_flag,
    )
    installer = PiInstaller(auto_install=True)
    result = installer.ensure_installed()
    assert result is True


def test_ensure_installed_start_background(tmp_path, monkeypatch):
    """pi 未装 + auto_install=true + npm 可用 → 启动 daemon thread，返 False。"""
    # mock pi + npm 都查得到（pi=None, npm=路径）
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: "/usr/local/bin/npm" if cmd == "npm" else None,
    )
    # mock flag file 不存在
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    installer = PiInstaller(auto_install=True)

    # mock _install_in_background 不实际跑（避免真 npm install）
    with patch.object(installer, "_install_in_background"):
        with patch("threading.Thread") as mock_thread_cls:
            result = installer.ensure_installed()

    assert result is False  # 本次不可用
    assert installer.status == "installing"
    mock_thread_cls.assert_called_once()  # 启动了 daemon thread


def test_ensure_installed_no_npm(tmp_path, monkeypatch):
    """pi 未装 + npm 不可用 → 返 False + warn。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: None,  # pi + npm 都查不到
    )
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    installer = PiInstaller(auto_install=True)

    with patch("threading.Thread") as mock_thread_cls:
        result = installer.ensure_installed()

    assert result is False
    mock_thread_cls.assert_not_called()  # npm 不可用不启动后台安装


def test_ensure_installed_auto_install_false(tmp_path, monkeypatch):
    """pi 未装 + auto_install=false → 返 False + warn，不启动后台安装。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: "/usr/local/bin/npm" if cmd == "npm" else None,
    )
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    installer = PiInstaller(auto_install=False)

    with patch("threading.Thread") as mock_thread_cls:
        result = installer.ensure_installed()

    assert result is False
    mock_thread_cls.assert_not_called()  # auto_install=false 不启动


# ---------------------------------------------------------------------------
# _install_in_background（实际 npm install 逻辑）
# ---------------------------------------------------------------------------


def test_install_in_background_success(tmp_path, monkeypatch):
    """mock subprocess.run 返 0 → flag file 写入 + status="done"。"""
    # mock flag file 路径到 tmp_path
    mock_flag = tmp_path / "pi-installed.flag"
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        mock_flag,
    )
    installer = PiInstaller(auto_install=True)

    # mock subprocess.run 返成功
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        installer._install_in_background()

    assert installer.status == "done"
    assert mock_flag.exists()  # flag file 写入


def test_install_in_background_failure(tmp_path, monkeypatch):
    """mock subprocess.run 返非 0 → status="failed" + error log。"""
    mock_flag = tmp_path / "pi-installed.flag"
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        mock_flag,
    )
    installer = PiInstaller(auto_install=True)

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "npm install failed"

    with patch("subprocess.run", return_value=mock_result):
        installer._install_in_background()

    assert installer.status == "failed"
    assert not mock_flag.exists()  # flag file 不写


def test_install_in_background_timeout(tmp_path, monkeypatch):
    """mock subprocess.run 抛 TimeoutExpired → status="failed"。"""
    mock_flag = tmp_path / "pi-installed.flag"
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        mock_flag,
    )
    installer = PiInstaller(auto_install=True)

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="npm", timeout=300),
    ):
        installer._install_in_background()

    assert installer.status == "failed"
    assert not mock_flag.exists()


def test_install_in_background_crash(tmp_path, monkeypatch):
    """mock subprocess.run 抛 Exception → status="failed"。"""
    mock_flag = tmp_path / "pi-installed.flag"
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        mock_flag,
    )
    installer = PiInstaller(auto_install=True)

    with patch("subprocess.run", side_effect=RuntimeError("crash")):
        installer._install_in_background()

    assert installer.status == "failed"


# ---------------------------------------------------------------------------
# status 属性（供 TUI 查询，决策 2）
# ---------------------------------------------------------------------------


def test_status_initial():
    """新构造的 installer status="not_started"。"""
    installer = PiInstaller()
    assert installer.status == "not_started"


def test_status_installing(tmp_path, monkeypatch):
    """ensure_installed 启动后台安装后 status="installing"。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: "/usr/local/bin/npm" if cmd == "npm" else None,
    )
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    installer = PiInstaller(auto_install=True)

    with patch.object(installer, "_install_in_background"):
        with patch("threading.Thread"):
            installer.ensure_installed()

    assert installer.status == "installing"


def test_status_done(tmp_path, monkeypatch):
    """_install_in_background 成功后 status="done"。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "pi-installed.flag",
    )
    installer = PiInstaller(auto_install=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        installer._install_in_background()

    assert installer.status == "done"


def test_status_failed(tmp_path, monkeypatch):
    """_install_in_background 失败后 status="failed"。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "pi-installed.flag",
    )
    installer = PiInstaller(auto_install=True)

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "failed"

    with patch("subprocess.run", return_value=mock_result):
        installer._install_in_background()

    assert installer.status == "failed"


# ---------------------------------------------------------------------------
# npm install 命令构造（验证 --ignore-scripts）
# ---------------------------------------------------------------------------


def test_install_uses_ignore_scripts(tmp_path, monkeypatch):
    """npm install 命令含 --ignore-scripts（Pi 官方推荐）。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "pi-installed.flag",
    )
    installer = PiInstaller(auto_install=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        installer._install_in_background()

    # 验证 npm install 命令含 --ignore-scripts
    call_args = mock_run.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    assert "--ignore-scripts" in cmd
    assert "@earendil-works/pi-coding-agent" in cmd


def test_install_uses_global_flag(tmp_path, monkeypatch):
    """npm install 命令含 -g（全局安装）。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "pi-installed.flag",
    )
    installer = PiInstaller(auto_install=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        installer._install_in_background()

    call_args = mock_run.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    assert "-g" in cmd or "--global" in cmd
