from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.sandbox.docker.local_container_backend import (
    _extract_host_port,
    _format_mount,
    _is_no_such_container_error,
    _parse_docker_timestamp,
    LocalContainerBackend,
)
from agent4ml.backend.agents.sandbox.types import PathMapping, SandboxInfo


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _make_backend(**kwargs) -> LocalContainerBackend:
    defaults = {"image": "test-image:latest", "base_port": 9000, "container_prefix": "test-sb"}
    defaults.update(kwargs)
    with patch.object(LocalContainerBackend, "_detect_runtime", return_value="docker"):
        return LocalContainerBackend(**defaults)


class TestDetectRuntime:
    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_macos_apple_container_available(self, mock_run, _mock_sys) -> None:
        mock_run.return_value = _completed(stdout="container 0.1", returncode=0)
        p = LocalContainerBackend()
        assert p._runtime == "container"

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_macos_fallback_docker(self, mock_run, _mock_sys) -> None:
        mock_run.side_effect = FileNotFoundError()
        p = LocalContainerBackend()
        assert p._runtime == "docker"

    @patch("platform.system", return_value="Windows")
    def test_non_macos_docker(self, _mock_sys) -> None:
        p = LocalContainerBackend()
        assert p._runtime == "docker"


class TestContainerName:
    def test_format(self) -> None:
        p = _make_backend()
        assert p._container_name("abc12345") == "test-sb-abc12345"


class TestFormatMount:
    def test_docker_readwrite(self) -> None:
        result = _format_mount("docker", "/host/path", "/container/path", False)
        assert result == ["--mount", "type=bind,src=/host/path,dst=/container/path"]

    def test_docker_readonly(self) -> None:
        result = _format_mount("docker", "/host/path", "/container/path", True)
        assert result == ["--mount", "type=bind,src=/host/path,dst=/container/path,readonly"]

    def test_container_runtime(self) -> None:
        result = _format_mount("container", "/host/path", "/container/path", False)
        assert result == ["-v", "/host/path:/container/path"]

    def test_container_readonly(self) -> None:
        result = _format_mount("container", "/host/path", "/container/path", True)
        assert result == ["-v", "/host/path:/container/path:ro"]


class TestSandboxIdValidation:
    """S6: create/discover 入口校验 sandbox_id 格式。"""

    def test_create_rejects_invalid_id(self) -> None:
        backend = _make_backend()
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            backend.create("t1", "../../etc")

    def test_discover_rejects_invalid_id(self) -> None:
        backend = _make_backend()
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            backend.discover("../../../")

    def test_create_accepts_valid_id(self) -> None:
        """合法 id 不在入口抛 ValueError（后续 docker 不可用会抛其他异常）。"""
        backend = _make_backend()
        with patch.object(backend, "_start_container", side_effect=RuntimeError("docker not running")):
            with pytest.raises(RuntimeError, match="docker not running"):
                backend.create("t1", "a1b2c3d4")


class TestParseDockerTimestamp:
    def test_valid_iso(self) -> None:
        ts = _parse_docker_timestamp("2026-07-14T10:30:00Z")
        assert ts > 0

    def test_nanosecond_precision(self) -> None:
        ts = _parse_docker_timestamp("2026-07-14T10:30:00.123456789Z")
        assert ts > 0

    def test_empty(self) -> None:
        assert _parse_docker_timestamp("") == 0.0

    def test_invalid(self) -> None:
        assert _parse_docker_timestamp("not-a-date") == 0.0


class TestExtractHostPort:
    def test_found(self) -> None:
        entry = {"NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9090"}]}}}
        assert _extract_host_port(entry, 8080) == 9090

    def test_not_found(self) -> None:
        assert _extract_host_port({}, 8080) is None

    def test_malformed(self) -> None:
        assert _extract_host_port({"NetworkSettings": "bad"}, 8080) is None


class TestIsNoSuchContainer:
    def test_no_such_object(self) -> None:
        assert _is_no_such_container_error("Error: no such object: test-sb-abc", "test-sb-abc") is True

    def test_no_such_container(self) -> None:
        assert _is_no_such_container_error("No such container: test-sb-abc", "test-sb-abc") is True

    def test_not_found_with_name(self) -> None:
        assert _is_no_such_container_error("not found: test-sb-abc", "test-sb-abc") is True

    def test_transient_error(self) -> None:
        assert _is_no_such_container_error("connection refused", "test-sb-abc") is False

    def test_command_not_found(self) -> None:
        assert _is_no_such_container_error("docker: command not found", "test-sb-abc") is False


class TestCreate:
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_success(self, mock_run, _mock_port) -> None:
        # discover returns None (container not found), _start_container returns container_id
        mock_run.side_effect = [
            _completed(stdout="", stderr="no such container", returncode=1),  # discover inspect
            _completed(stdout="container123\n", returncode=0),  # docker run
        ]
        p = _make_backend()
        info = p.create("thread-1", "abc12345")
        assert info.sandbox_id == "abc12345"
        assert "9090" in info.sandbox_url
        assert info.container_name == "test-sb-abc12345"
        assert info.container_id == "container123"

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_idempotent_discover_hits(self, mock_run) -> None:
        # discover finds running container with port
        mock_run.side_effect = [
            _completed(stdout="true\n", returncode=0),  # inspect running
            _completed(stdout="0.0.0.0:9090\n", returncode=0),  # docker port
        ]
        p = _make_backend()
        info = p.create("thread-1", "abc12345")
        assert info.sandbox_url == "http://localhost:9090"
        # _start_container NOT called (only 2 subprocess calls: inspect + port)
        assert mock_run.call_count == 2

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port")
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_port_conflict_retry(self, mock_run, mock_port) -> None:
        mock_port.side_effect = [9090, 9091]
        mock_run.side_effect = [
            _completed(stdout="", stderr="no such container", returncode=1),  # discover inspect
            subprocess.CalledProcessError(1, [], stderr="port is already allocated"),  # _start_container fail
            _completed(stdout="", stderr="no such container", returncode=1),  # discover inspect (2nd port)
            _completed(stdout="container456\n", returncode=0),  # _start_container success
        ]
        p = _make_backend()
        info = p.create("thread-1", "abc12345")
        assert "9091" in info.sandbox_url

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_name_conflict_discover(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stdout="", stderr="no such container", returncode=1),  # discover inspect (initial)
            subprocess.CalledProcessError(1, [], stderr="is already in use by container"),  # _start_container fail
            _completed(stdout="true\n", returncode=0),  # discover inspect (after conflict)
            _completed(stdout="0.0.0.0:9090\n", returncode=0),  # docker port
        ]
        p = _make_backend()
        info = p.create("thread-1", "abc12345")
        assert info.sandbox_url == "http://localhost:9090"

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port")
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_retry_exhausted(self, mock_run, mock_port) -> None:
        mock_port.side_effect = list(range(9000, 9000 + _MAX_PORT_RETRIES_TEST))
        # discover inspect (1 call) + 10 _start_container failures
        side_effects = [_completed(stderr="no such container", returncode=1)]  # discover
        for _ in range(_MAX_PORT_RETRIES_TEST):
            side_effects.append(subprocess.CalledProcessError(1, [], stderr="port is already allocated"))
        mock_run.side_effect = side_effects
        p = _make_backend()
        with pytest.raises(RuntimeError, match="all candidate ports"):
            p.create("thread-1", "abc12345")

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_start_container_cmd_has_rm_and_env(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stderr="no such container", returncode=1),  # discover
            _completed(stdout="cid\n", returncode=0),  # docker run
        ]
        p = _make_backend()
        p.create("thread-1", "abc12345")
        run_cmd = mock_run.call_args_list[1].args[0]
        assert "--rm" in run_cmd
        assert "-d" in run_cmd
        assert "--name" in run_cmd
        assert "test-sb-abc12345" in run_cmd
        assert "-e" in run_cmd
        assert "SANDBOX_ID=abc12345" in run_cmd
        assert "THREAD_ID=thread-1" in run_cmd
        assert "test-image:latest" in run_cmd

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_start_container_bind_mount(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stderr="no such container", returncode=1),  # discover
            _completed(stdout="cid\n", returncode=0),  # docker run
        ]
        p = _make_backend(sandbox_root="/tmp/agent4ml-sandbox")
        p.create("thread-1", "abc12345")
        run_cmd = mock_run.call_args_list[1].args[0]
        mount_args = [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--mount"]
        assert any("type=bind" in m for m in mount_args)
        assert any("/mnt/agent4ml/user-data" in m for m in mount_args)
        assert any("abc12345" in m for m in mount_args)

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_start_container_extra_mounts(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stderr="no such container", returncode=1),  # discover
            _completed(stdout="cid\n", returncode=0),  # docker run
        ]
        p = _make_backend()
        extra = [PathMapping("/mnt/agent4ml/skills", "/host/skills", read_only=True)]
        p.create("thread-1", "abc12345", extra_mounts=extra)
        run_cmd = mock_run.call_args_list[1].args[0]
        mount_args = [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--mount"]
        assert any("/host/skills" in m and "readonly" in m for m in mount_args)

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_start_container_config_env(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stderr="no such container", returncode=1),  # discover
            _completed(stdout="cid\n", returncode=0),  # docker run
        ]
        p = _make_backend(environment={"NODE_ENV": "production"})
        p.create("thread-1", "abc12345")
        run_cmd = mock_run.call_args_list[1].args[0]
        assert "NODE_ENV=production" in run_cmd

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend._get_free_port", return_value=9090)
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_docker_seccomp_unconfined(self, mock_run, _mock_port) -> None:
        mock_run.side_effect = [
            _completed(stderr="no such container", returncode=1),  # discover
            _completed(stdout="cid\n", returncode=0),  # docker run
        ]
        p = _make_backend()
        p.create("thread-1", "abc12345")
        run_cmd = mock_run.call_args_list[1].args[0]
        assert "seccomp=unconfined" in run_cmd


class TestDiscover:
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_found(self, mock_run) -> None:
        mock_run.side_effect = [
            _completed(stdout="true\n", returncode=0),  # inspect running
            _completed(stdout="0.0.0.0:9090\n", returncode=0),  # docker port
        ]
        p = _make_backend()
        info = p.discover("abc12345")
        assert info is not None
        assert info.sandbox_id == "abc12345"
        assert "9090" in info.sandbox_url
        assert info.container_name == "test-sb-abc12345"

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_not_running(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="false\n", returncode=0)
        p = _make_backend()
        assert p.discover("abc12345") is None

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_not_found(self, mock_run) -> None:
        mock_run.return_value = _completed(stderr="no such container", returncode=1)
        p = _make_backend()
        assert p.discover("abc12345") is None

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_port_missing(self, mock_run) -> None:
        mock_run.side_effect = [
            _completed(stdout="true\n", returncode=0),  # inspect running
            _completed(stdout="", returncode=1),  # docker port fails
        ]
        p = _make_backend()
        assert p.discover("abc12345") is None

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_timeout_returns_none(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=5)
        p = _make_backend()
        assert p.discover("abc12345") is None


class TestDestroy:
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_stop_with_container_id(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="", returncode=0)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="http://localhost:9090", container_name="test-sb-abc", container_id="cid123")
        p.destroy(info)
        mock_run.assert_called_once()
        assert "cid123" in mock_run.call_args.args[0]

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_stop_fallback_to_name(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="", returncode=0)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="http://localhost:9090", container_name="test-sb-abc")
        p.destroy(info)
        assert "test-sb-abc" in mock_run.call_args.args[0]

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_idempotent_silent(self, mock_run) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, [])
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        p.destroy(info)  # should not raise

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_timeout_silent(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=15)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        p.destroy(info)  # should not raise

    def test_no_target_noop(self) -> None:
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="")
        p.destroy(info)  # no container_id or container_name, no-op


class TestIsAlive:
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_running_true(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="true\n", returncode=0)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        assert p.is_alive(info) is True

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_running_false(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="false\n", returncode=0)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        assert p.is_alive(info) is False

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_not_found_false(self, mock_run) -> None:
        mock_run.return_value = _completed(stderr="no such container: test-sb-abc", returncode=1)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        assert p.is_alive(info) is False

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_transient_error_none(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=5)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        assert p.is_alive(info) is None

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_unknown_stderr_none(self, mock_run) -> None:
        mock_run.return_value = _completed(stderr="daemon error", returncode=1)
        p = _make_backend()
        info = SandboxInfo(sandbox_id="abc", sandbox_url="", container_name="test-sb-abc")
        assert p.is_alive(info) is None

    def test_no_name_none(self) -> None:
        p = _make_backend()
        # container_name=None, sandbox_id="" 鈫?_container_name("") returns "test-sb-"
        # but name would be "test-sb-" which is truthy. Test the real edge: sandbox_id empty
        info = SandboxInfo(sandbox_id="", sandbox_url="")
        # _container_name("") = "test-sb-" which is truthy, so is_alive proceeds to docker
        # On Windows without docker, FileNotFoundError 鈫?None
        result = p.is_alive(info)
        assert result is None


class TestListRunning:
    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_success(self, mock_run) -> None:
        inspect_payload = json.dumps([
            {"Name": "/test-sb-abc", "Created": "2026-07-14T10:30:00Z",
             "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9090"}]}}},
            {"Name": "/test-sb-def", "Created": "2026-07-14T11:00:00Z",
             "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9091"}]}}},
        ])
        mock_run.side_effect = [
            _completed(stdout="test-sb-abc\ntest-sb-def\n", returncode=0),  # docker ps
            _completed(stdout=inspect_payload, returncode=0),  # docker inspect
        ]
        p = _make_backend()
        infos = p.list_running()
        assert len(infos) == 2
        assert infos[0].sandbox_id == "abc"
        assert "9090" in infos[0].sandbox_url
        assert infos[1].sandbox_id == "def"
        assert "9091" in infos[1].sandbox_url

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_empty(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="", returncode=0)
        p = _make_backend()
        assert p.list_running() == []

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_ps_error(self, mock_run) -> None:
        mock_run.return_value = _completed(stderr="daemon error", returncode=1)
        p = _make_backend()
        assert p.list_running() == []

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_prefix_filter(self, mock_run) -> None:
        # docker filter is substring: "test-sb-extra-xyz" matches "test-sb-"
        # but should be filtered out by startswith check... actually it DOES start with "test-sb-"
        # So the real concern is filtering names like "other-test-sb-abc"
        mock_run.side_effect = [
            _completed(stdout="test-sb-abc\nother-test-sb-xyz\n", returncode=0),  # ps
            _completed(stdout=json.dumps([
                {"Name": "/test-sb-abc", "Created": "2026-07-14T10:30:00Z",
                 "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9090"}]}}},
            ]), returncode=0),  # inspect only finds test-sb-abc (other-test-sb-xyz filtered by startswith)
        ]
        p = _make_backend()
        infos = p.list_running()
        assert len(infos) == 1
        assert infos[0].sandbox_id == "abc"

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_container_disappears(self, mock_run) -> None:
        # ps returns 2 names, but inspect only finds 1 (other disappeared)
        mock_run.side_effect = [
            _completed(stdout="test-sb-abc\ntest-sb-gone\n", returncode=0),  # ps
            _completed(stdout=json.dumps([
                {"Name": "/test-sb-abc", "Created": "2026-07-14T10:30:00Z",
                 "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9090"}]}}},
            ]), returncode=0),  # inspect only has abc
        ]
        p = _make_backend()
        infos = p.list_running()
        assert len(infos) == 1
        assert infos[0].sandbox_id == "abc"

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_created_at_iso(self, mock_run) -> None:
        mock_run.side_effect = [
            _completed(stdout="test-sb-abc\n", returncode=0),  # ps
            _completed(stdout=json.dumps([
                {"Name": "/test-sb-abc", "Created": "2026-07-14T10:30:00Z",
                 "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "9090"}]}}},
            ]), returncode=0),
        ]
        p = _make_backend()
        infos = p.list_running()
        assert "2026-07-14T10:30:00" in infos[0].created_at or "2026-07-14" in infos[0].created_at

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_no_port_mapping(self, mock_run) -> None:
        # Container without port mapping 鈫?empty sandbox_url
        mock_run.side_effect = [
            _completed(stdout="test-sb-abc\n", returncode=0),  # ps
            _completed(stdout=json.dumps([
                {"Name": "/test-sb-abc", "Created": "2026-07-14T10:30:00Z",
                 "NetworkSettings": {"Ports": {}}},
            ]), returncode=0),
        ]
        p = _make_backend()
        infos = p.list_running()
        assert len(infos) == 1
        assert infos[0].sandbox_url == ""


class TestBatchInspect:
    def test_empty_names(self) -> None:
        p = _make_backend()
        assert p._batch_inspect([]) == {}

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_json_decode_error(self, mock_run) -> None:
        mock_run.return_value = _completed(stdout="not json", returncode=0)
        p = _make_backend()
        assert p._batch_inspect(["test-sb-abc"]) == {}

    @patch("agent4ml.backend.agents.sandbox.docker.local_container_backend.subprocess.run")
    def test_inspect_error(self, mock_run) -> None:
        mock_run.return_value = _completed(stderr="daemon error", returncode=1)
        p = _make_backend()
        assert p._batch_inspect(["test-sb-abc"]) == {}


_MAX_PORT_RETRIES_TEST = 10
