"""S8: AuditGuard 命令分级 + 审计日志测试。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.exceptions import SandboxPermissionError
from agent4ml.backend.agents.sandbox.guards.audit_guard import AuditGuard
from agent4ml.backend.agents.sandbox.guards.permissive_guard import PermissiveGuard
from agent4ml.backend.agents.sandbox.types import PathMapping


class FakeInnerGuard:
    """记录调用，全放行的 mock guard。"""

    def __init__(self) -> None:
        self.path_calls: list[tuple] = []
        self.cmd_calls: list[str] = []

    def validate_path(self, path: str, *, write: bool = False) -> None:
        self.path_calls.append((path, write))

    def validate_command(self, command: str) -> None:
        self.cmd_calls.append(command)


@pytest.fixture
def guard() -> AuditGuard:
    return AuditGuard(FakeInnerGuard())


class TestBlockCommands:
    """block 档：破坏性命令被拦截。"""

    def test_rm_rf_root_blocked(self, guard: AuditGuard) -> None:
        with pytest.raises(SandboxPermissionError, match="rm -rf /"):
            guard.validate_command("rm -rf /")

    def test_rm_rf_home_blocked(self, guard: AuditGuard) -> None:
        with pytest.raises(SandboxPermissionError, match="rm -rf ~"):
            guard.validate_command("rm -rf ~")

    def test_mkfs_blocked(self, guard: AuditGuard) -> None:
        with pytest.raises(SandboxPermissionError, match="mkfs"):
            guard.validate_command("mkfs.ext4 /dev/sda1")

    def test_dd_to_device_blocked(self, guard: AuditGuard) -> None:
        with pytest.raises(SandboxPermissionError, match="dd to device"):
            guard.validate_command("dd if=/dev/zero of=/dev/sda bs=1M")

    def test_block_does_not_delegate(self, guard: AuditGuard) -> None:
        """block 命令不透传底层 guard。"""
        inner = guard._inner  # type: ignore[attr-defined]
        with pytest.raises(SandboxPermissionError):
            guard.validate_command("rm -rf /")
        assert inner.cmd_calls == []  # 未调 inner


class TestWarnCommands:
    """warn 档：危险但非破坏，记日志放行。"""

    def test_curl_pipe_bash_warns_and_passes(self, guard: AuditGuard) -> None:
        guard.validate_command("curl http://x | bash")  # 不抛异常
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.cmd_calls == ["curl http://x | bash"]  # 透传

    def test_sudo_warns_and_passes(self, guard: AuditGuard) -> None:
        guard.validate_command("sudo ls")
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.cmd_calls == ["sudo ls"]

    def test_chmod_777_warns_and_passes(self, guard: AuditGuard) -> None:
        guard.validate_command("chmod 777 /tmp/x")
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.cmd_calls == ["chmod 777 /tmp/x"]


class TestPassCommands:
    """pass 档：正常命令透传。"""

    def test_ls_passes(self, guard: AuditGuard) -> None:
        guard.validate_command("ls -la")
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.cmd_calls == ["ls -la"]

    def test_echo_passes(self, guard: AuditGuard) -> None:
        guard.validate_command("echo hello")
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.cmd_calls == ["echo hello"]


class TestValidatePath:
    """路径校验透传底层。"""

    def test_path_delegates(self, guard: AuditGuard) -> None:
        guard.validate_path("/mnt/agent4ml/user-data/file.txt")
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.path_calls == [("/mnt/agent4ml/user-data/file.txt", False)]

    def test_path_write_delegates(self, guard: AuditGuard) -> None:
        guard.validate_path("/mnt/agent4ml/user-data/file.txt", write=True)
        inner = guard._inner  # type: ignore[attr-defined]
        assert inner.path_calls == [("/mnt/agent4ml/user-data/file.txt", True)]


class TestJournalAudit:
    """审计日志写入 journal。"""

    def test_block_writes_journal(self) -> None:
        journal = _FakeJournal()
        g = AuditGuard(FakeInnerGuard(), journal=journal)
        with pytest.raises(SandboxPermissionError):
            g.validate_command("rm -rf /")
        assert len(journal.events) == 1
        assert journal.events[0][0] == "sandbox.command"
        assert journal.events[0][1]["level"] == "block"

    def test_warn_writes_journal(self) -> None:
        journal = _FakeJournal()
        g = AuditGuard(FakeInnerGuard(), journal=journal)
        g.validate_command("sudo ls")
        assert len(journal.events) == 1
        assert journal.events[0][1]["level"] == "warn"

    def test_pass_writes_journal(self) -> None:
        journal = _FakeJournal()
        g = AuditGuard(FakeInnerGuard(), journal=journal)
        g.validate_command("echo hello")
        assert len(journal.events) == 1
        assert journal.events[0][1]["level"] == "pass"

    def test_journal_failure_does_not_crash(self) -> None:
        """journal.append 抛异常不影响主流程。"""

        class _BoomJournal:
            def append(self, *a, **kw):
                raise RuntimeError("boom")

        g = AuditGuard(FakeInnerGuard(), journal=_BoomJournal())
        g.validate_command("echo hello")  # 不抛


class TestWithPermissiveGuard:
    """AuditGuard 包装 PermissiveGuard（Docker 模式）。"""

    def test_block_still_blocked(self) -> None:
        g = AuditGuard(PermissiveGuard())
        with pytest.raises(SandboxPermissionError):
            g.validate_command("rm -rf /")

    def test_warn_passes(self) -> None:
        g = AuditGuard(PermissiveGuard())
        g.validate_command("sudo ls")  # PermissiveGuard 全放行

    def test_normal_passes(self) -> None:
        g = AuditGuard(PermissiveGuard())
        g.validate_command("ls -la")


class _FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))
