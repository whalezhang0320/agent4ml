"""Sandbox readiness polling — sync + async."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def wait_for_sandbox_ready(sandbox_url: str, timeout: int = 60) -> bool:
    """轮询 /v1/sandbox 直到 ready 或超时。sync 版本。"""
    if httpx is None:
        raise RuntimeError("httpx required for readiness polling")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{sandbox_url}/v1/sandbox", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


async def wait_for_sandbox_ready_async(
    sandbox_url: str, timeout: int = 60, poll_interval: float = 1.0,
) -> bool:
    """轮询 /v1/sandbox 直到 ready 或超时。async 版本，不阻塞事件循环。"""
    if httpx is None:
        raise RuntimeError("httpx required for readiness polling")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                resp = await client.get(
                    f"{sandbox_url}/v1/sandbox", timeout=min(5.0, remaining),
                )
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))
    return False
