from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import anyio  # noqa: F401
    HAS_ANYIO = True
except ImportError:
    HAS_ANYIO = False

from agent4ml.backend.agents.sandbox.docker import readiness  # noqa: E402
from agent4ml.backend.agents.sandbox.docker.readiness import (  # noqa: E402
    wait_for_sandbox_ready,
    wait_for_sandbox_ready_async,
)


class TestWaitForSandboxReady:
    @patch.object(readiness, "httpx")
    def test_ready(self, mock_httpx) -> None:
        mock_httpx.get.return_value = MagicMock(status_code=200)
        assert wait_for_sandbox_ready("http://localhost:8080", timeout=5) is True

    @patch.object(readiness, "httpx")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.sleep")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.time")
    def test_timeout(self, mock_time, mock_sleep, mock_httpx) -> None:
        mock_httpx.get.return_value = MagicMock(status_code=503)
        mock_time.side_effect = [0, 0, 6]
        assert wait_for_sandbox_ready("http://localhost:8080", timeout=5) is False

    @patch.object(readiness, "httpx")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.sleep")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.time")
    def test_connection_error_continues_then_succeeds(self, mock_time, mock_sleep, mock_httpx) -> None:
        mock_httpx.get.side_effect = [
            ConnectionError("refused"),
            MagicMock(status_code=200),
        ]
        mock_time.side_effect = [0, 1, 1]
        assert wait_for_sandbox_ready("http://localhost:8080", timeout=5) is True

    @patch.object(readiness, "httpx")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.sleep")
    @patch("agent4ml.backend.agents.sandbox.docker.readiness.time.time")
    def test_request_error_does_not_raise(self, mock_time, mock_sleep, mock_httpx) -> None:
        mock_httpx.get.side_effect = RuntimeError("timeout")
        mock_time.side_effect = [0, 0, 6]
        assert wait_for_sandbox_ready("http://localhost:8080", timeout=5) is False


class TestWaitForSandboxReadyAsync:
    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_ready(self) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch("agent4ml.backend.agents.sandbox.docker.readiness.httpx.AsyncClient", return_value=mock_client):
            result = await wait_for_sandbox_ready_async("http://localhost:8080", timeout=5)
        assert result is True

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_timeout(self) -> None:
        mock_resp = MagicMock(status_code=503)
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch("agent4ml.backend.agents.sandbox.docker.readiness.httpx.AsyncClient", return_value=mock_client):
            with patch("agent4ml.backend.agents.sandbox.docker.readiness.asyncio.sleep", new=AsyncMock(return_value=None)):
                with patch("agent4ml.backend.agents.sandbox.docker.readiness.asyncio.get_running_loop") as mock_loop:
                    mock_loop_obj = MagicMock()
                    mock_loop_obj.time.side_effect = [0, 0, 0, 6, 6]
                    mock_loop.return_value = mock_loop_obj
                    result = await wait_for_sandbox_ready_async("http://localhost:8080", timeout=5, poll_interval=0.01)
        assert result is False

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_connection_error_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch("agent4ml.backend.agents.sandbox.docker.readiness.httpx.AsyncClient", return_value=mock_client):
            with patch("agent4ml.backend.agents.sandbox.docker.readiness.asyncio.sleep", new=AsyncMock(return_value=None)):
                with patch("agent4ml.backend.agents.sandbox.docker.readiness.asyncio.get_running_loop") as mock_loop:
                    mock_loop_obj = MagicMock()
                    mock_loop_obj.time.side_effect = [0, 0, 6]
                    mock_loop.return_value = mock_loop_obj
                    result = await wait_for_sandbox_ready_async("http://localhost:8080", timeout=5)
        assert result is False
