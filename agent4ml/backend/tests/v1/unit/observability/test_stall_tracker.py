"""Tests for StallTracker stall detection signals.

阈值同步 stall_tracker.py 改动（2026-07-27）：
- capability_failure_threshold: 3 → 5
- error_pattern_threshold: 3 → 5
- todo_stagnation_rounds: 5 → 15
- 加 success decay：成功工具调用后 120s 内不报 stuck（_success_decay_window）
"""

from __future__ import annotations

import time

from agent4ml.backend.agents.observability.stall_tracker import (
    StallTracker,
    classify_capability,
    classify_error_class,
)


class TestClassify:
    def test_postgres_capability(self) -> None:
        assert classify_capability("bash", {"command": "apt install postgresql"}, "denied") == "postgres"

    def test_root_capability(self) -> None:
        assert classify_capability("bash", {"command": "apt-get update"}, "permission denied") == "root"

    def test_docker_capability(self) -> None:
        assert classify_capability("bash", {"command": "docker ps"}, "not found") == "docker"

    def test_network_error_class(self) -> None:
        assert classify_error_class("504 Gateway Time-out") == "network"

    def test_sandbox_error_class(self) -> None:
        assert classify_error_class("write failed: [WinError 10061] connection refused") == "sandbox"

    def test_sandbox_capability_for_winerror(self) -> None:
        assert classify_capability("write_file", {"path": "/x"}, "write failed: [WinError 10061]") == "sandbox"

    def test_permission_error_class(self) -> None:
        assert classify_error_class("Permission denied") == "permission"

    def test_not_found_error_class(self) -> None:
        assert classify_error_class("command not found") == "not_found"


class TestCapabilityExhaustion:
    def test_four_different_commands_same_capability_does_not_trigger_yet(self) -> None:
        """阈值 5：4 次不同命令同 capability 不触发。"""
        tracker = StallTracker()
        tracker.record_tool_failure("bash", {"command": "apt install postgresql"}, "permission denied")
        tracker.record_tool_failure("bash", {"command": "find / -name postgres"}, "not found")
        tracker.record_tool_failure("bash", {"command": "dpkg -l | grep postgres"}, "not installed")
        tracker.record_tool_failure("bash", {"command": "which postgres"}, "not found")
        assert not tracker.stuck

    def test_five_different_commands_same_capability_triggers_stuck(self) -> None:
        """阈值 5：5 次不同命令同 capability 触发 stuck。"""
        tracker = StallTracker()
        tracker.record_tool_failure("bash", {"command": "apt install postgresql"}, "permission denied")
        tracker.record_tool_failure("bash", {"command": "find / -name postgres"}, "not found")
        tracker.record_tool_failure("bash", {"command": "dpkg -l | grep postgres"}, "not installed")
        tracker.record_tool_failure("bash", {"command": "which postgres"}, "not found")
        assert not tracker.stuck
        tracker.record_tool_failure("bash", {"command": "service postgres status"}, "no such service")
        assert tracker.stuck
        assert "capability exhausted" in tracker.get_stuck_reason()

    def test_same_command_twice_does_not_trigger(self) -> None:
        tracker = StallTracker()
        tracker.record_tool_failure("bash", {"command": "apt install postgresql"}, "denied")
        tracker.record_tool_failure("bash", {"command": "apt install postgresql"}, "denied")
        assert not tracker.stuck

    def test_success_decays_capability_failures(self) -> None:
        """成功工具调用后 120s 内不报 capability stuck（success decay）。"""
        tracker = StallTracker()
        # 累积 5 次失败（超过阈值）
        for cmd in ("cmd1", "cmd2", "cmd3", "cmd4", "cmd5"):
            tracker.record_tool_failure("bash", {"command": cmd}, "permission denied")
        # 此时本应 stuck，但插入一次成功调用后 decay
        tracker.record_tool_success()
        assert not tracker.stuck


class TestErrorPatternRepetition:
    def test_five_same_error_class_triggers_stuck(self) -> None:
        """阈值 5：5 次同类错误触发 stuck。

        用 permission class（permission denied / 403 forbidden / EACCES 都匹配 permission 模式）。
        """
        tracker = StallTracker()
        tracker.record_tool_failure("bash", {"command": "cmd1"}, "permission denied")
        tracker.record_tool_failure("bash", {"command": "cmd2"}, "403 forbidden")
        tracker.record_tool_failure("bash", {"command": "cmd3"}, "EACCES")
        tracker.record_tool_failure("bash", {"command": "cmd4"}, "forbidden access")
        assert not tracker.stuck
        tracker.record_tool_failure("bash", {"command": "cmd5"}, "permission denied again")
        assert tracker.stuck
        assert "error pattern" in tracker.get_stuck_reason()

    def test_success_decays_error_pattern(self) -> None:
        """成功调用后 120s 内不报 error pattern stuck。"""
        tracker = StallTracker()
        for i in range(5):
            tracker.record_tool_failure("bash", {"command": f"cmd{i}"}, "permission denied")
        tracker.record_tool_success()
        assert not tracker.stuck


class TestTodoStagnation:
    def test_fifteen_rounds_same_todo_triggers_stuck(self) -> None:
        """阈值 15：15 轮同 todo 触发 stuck。"""
        tracker = StallTracker()
        todos = [{"content": "install postgres", "status": "in_progress"}]
        for _ in range(14):
            tracker.record_todo_state(todos)
            assert not tracker.stuck
        tracker.record_todo_state(todos)
        assert tracker.stuck
        assert "todo stagnated" in tracker.get_stuck_reason()

    def test_changing_todo_resets_counter(self) -> None:
        tracker = StallTracker()
        tracker.record_todo_state([{"content": "task A", "status": "in_progress"}])
        tracker.record_todo_state([{"content": "task A", "status": "in_progress"}])
        tracker.record_todo_state([{"content": "task B", "status": "in_progress"}])
        assert not tracker.stuck
        assert tracker._todo_stagnation_count == 1

    def test_success_decays_todo_stagnation(self) -> None:
        """成功调用后 120s 内不报 todo stagnation stuck。"""
        tracker = StallTracker()
        todos = [{"content": "task", "status": "in_progress"}]
        for _ in range(15):
            tracker.record_todo_state(todos)
        # 本应 stuck，但插入一次成功调用后 decay
        tracker.record_tool_success()
        assert not tracker.stuck


class TestReset:
    def test_reset_clears_all_signals(self) -> None:
        """reset 清空所有信号（阈值 5）。"""
        tracker = StallTracker()
        for cmd in ("apt install postgresql", "which postgres", "pg_isready", "find / -name pg", "service pg status"):
            tracker.record_tool_failure("bash", {"command": cmd}, "permission denied")
        assert tracker.stuck
        tracker.reset()
        assert not tracker.stuck
        assert len(tracker.get_failures()) == 0

    def test_reset_clears_last_success_ts(self) -> None:
        """reset 后 _last_success_ts=0，_recent_success() 返 False（不抑制失败信号）。"""
        tracker = StallTracker()
        tracker.record_tool_success()
        assert tracker._recent_success()  # 成功后 decay 生效
        tracker.reset()
        # reset 后 _last_success_ts=0，_recent_success() 返 False
        assert not tracker._recent_success()

