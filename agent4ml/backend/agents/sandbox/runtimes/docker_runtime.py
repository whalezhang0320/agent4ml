from __future__ import annotations

import base64
import fnmatch
import logging
import re as _re
import shlex
import threading
import uuid

from agent_sandbox import Sandbox as AioSandboxClient

from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxError,
    SandboxFileError,
    SandboxFileNotFoundError,
    SandboxPermissionError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.types import GrepMatch
from agent4ml.backend.agents.sandbox.utils.search import (
    should_ignore_path,
    truncate_line,
)

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
_DEFAULT_NO_CHANGE_TIMEOUT = 1800
_ERROR_OBSERVATION_SIGNATURE = "'ErrorObservation' object has no attribute 'exit_code'"
VIRTUAL_PATH_PREFIX = "/mnt/agent4ml/user-data"


class DockerRuntime:
    """DockerRuntime — agent_sandbox SDK 调容器内 AIO runtime。

    方案 C 三组件之一。通过 agent_sandbox SDK（Fern 生成，sync）调容器内
    all-in-one-sandbox 镜像。配合 IdentityTranslator 使用（容器内虚拟路径直传）。

    INVARIANT:
    - sync 方法（匹配 SandboxRuntime Protocol + LocalRuntime + Sandbox 编排）
    - exec_command 用 threading.Lock 串行化（防并发破坏持久 shell session，#1433）
    - exec_command 检测 ErrorObservation 签名 → fresh session 重试（仿 deer-flow）
    - close() 遍历 _client_wrapper → httpx_client → httpx_client 属性链找真实
      httpx.Client.close()（SDK 不暴露 close，#2872）
    - list_dir 走 shell find（无专用端点）
    - grep 走 list_path + 逐文件 search_in_file 客户端编排（无服务端 grep）
    - download_file 路径遍历检查（.. + VIRTUAL_PATH_PREFIX）+ 100MB 上限
    - 全抛 SandboxError 子类（Grill #5）
    - 容器内 AIO runtime 接收虚拟路径（/mnt/agent4ml/user-data/...），DockerRuntime 不翻译
    """

    def __init__(self, sandbox_url: str) -> None:
        self._url = sandbox_url
        self._client = AioSandboxClient(base_url=sandbox_url, timeout=_DEFAULT_NO_CHANGE_TIMEOUT)
        self._lock = threading.Lock()
        self._closed = False

    def exec_command(self, command: str) -> str:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                result = self._client.shell.exec_command(
                    command=command, no_change_timeout=_DEFAULT_NO_CHANGE_TIMEOUT,
                )
                output = result.data.output if result.data else ""
                if output and _ERROR_OBSERVATION_SIGNATURE in output:
                    logger.warning("ErrorObservation detected, retrying on fresh session")
                    fresh_id = str(uuid.uuid4())
                    self._client.shell.create_session(id=fresh_id)
                    try:
                        result = self._client.shell.exec_command(
                            command=command, id=fresh_id,
                            no_change_timeout=_DEFAULT_NO_CHANGE_TIMEOUT,
                        )
                        output = result.data.output if result.data else ""
                    finally:
                        try:
                            self._client.shell.cleanup_session(fresh_id)
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Failed to release recovery session {fresh_id}: {cleanup_error}"
                            )
                if output and _ERROR_OBSERVATION_SIGNATURE in output:
                    raise SandboxCommandError(
                        "sandbox returned ErrorObservation after fresh-session retry",
                        command=command,
                        exit_code=None,
                    )
                return output if output else "(no output)"
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxCommandError(
                    f"exec failed: {exc}", command=command, exit_code=-1,
                ) from exc

    def read_file(self, path: str) -> str:
        try:
            result = self._client.file.read_file(file=path)
            return result.data.content if result.data else ""
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "no such file" in msg:
                raise SandboxFileNotFoundError(
                    f"file not found: {path}", path=path, operation="read",
                ) from exc
            if "permission denied" in msg or "403" in msg:
                raise SandboxPermissionError(
                    f"permission denied: {path}", path=path, operation="read",
                ) from exc
            raise SandboxFileError(
                f"read failed: {exc}", path=path, operation="read",
            ) from exc

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                self._client.file.write_file(file=path, content=content, append=append)
            except SandboxError:
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if "permission denied" in msg or "403" in msg:
                    raise SandboxPermissionError(
                        f"permission denied: {path}", path=path, operation="write",
                    ) from exc
                raise SandboxFileError(
                    f"write failed: {exc}", path=path, operation="write",
                ) from exc

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                cmd = (
                    f"find {shlex.quote(path)} -maxdepth {max_depth} "
                    f"-type f -o -type d 2>/dev/null | head -500"
                )
                result = self._client.shell.exec_command(
                    command=cmd, no_change_timeout=_DEFAULT_NO_CHANGE_TIMEOUT,
                )
                output = result.data.output if result.data else ""
                if output:
                    return [line.strip() for line in output.strip().split("\n") if line.strip()]
                return []
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxFileError(
                    f"list_dir failed: {exc}", path=path, operation="list",
                ) from exc

    def glob(
        self, path: str, pattern: str, *,
        include_dirs: bool = False, max_results: int = 200,
    ) -> tuple[list[str], bool]:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                if not include_dirs:
                    result = self._client.file.find_files(path=path, glob=pattern)
                    files = result.data.files if result.data and result.data.files else []
                    filtered = [f for f in files if not should_ignore_path(f)]
                    truncated = len(filtered) > max_results
                    return filtered[:max_results], truncated
                result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
                entries = result.data.files if result.data and result.data.files else []
                matches: list[str] = []
                root = path.rstrip("/") or "/"
                root_prefix = root if root == "/" else f"{root}/"
                for entry in entries:
                    if entry.path != root and not entry.path.startswith(root_prefix):
                        continue
                    if should_ignore_path(entry.path):
                        continue
                    rel = entry.path[len(root):].lstrip("/")
                    if fnmatch.fnmatch(rel, pattern):
                        matches.append(entry.path)
                        if len(matches) >= max_results:
                            return matches, True
                return matches, False
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxFileError(
                    f"glob failed: {exc}", path=path, operation="glob",
                ) from exc

    def grep(
        self, path: str, pattern: str, *,
        glob: str | None = None, literal: bool = False,
        case_sensitive: bool = False, max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                regex_source = _re.escape(pattern) if literal else pattern
                _re.compile(regex_source, 0 if case_sensitive else _re.IGNORECASE)
                regex = regex_source if case_sensitive else f"(?i){regex_source}"

                if glob is not None:
                    find_result = self._client.file.find_files(path=path, glob=glob)
                    candidate_paths = (
                        find_result.data.files if find_result.data and find_result.data.files else []
                    )
                else:
                    list_result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
                    entries = list_result.data.files if list_result.data and list_result.data.files else []
                    candidate_paths = [e.path for e in entries if not e.is_directory]

                matches: list[GrepMatch] = []
                for file_path in candidate_paths:
                    if should_ignore_path(file_path):
                        continue
                    search_result = self._client.file.search_in_file(file=file_path, regex=regex)
                    data = search_result.data
                    if data is None:
                        continue
                    line_numbers = data.line_numbers or []
                    matched_lines = data.matches or []
                    for line_number, line in zip(line_numbers, matched_lines):
                        matches.append(GrepMatch(
                            path=file_path,
                            line_number=line_number if isinstance(line_number, int) else 0,
                            line=truncate_line(line),
                        ))
                        if len(matches) >= max_results:
                            return matches, True
                return matches, False
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxFileError(
                    f"grep failed: {exc}", path=path, operation="grep",
                ) from exc

    def download_file(self, path: str) -> bytes:
        normalised = path.replace("\\", "/")
        for segment in normalised.split("/"):
            if segment == "..":
                raise SandboxPermissionError(
                    f"path traversal detected: {path}", path=path, operation="download",
                )
        stripped = normalised.lstrip("/")
        allowed = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped != allowed and not stripped.startswith(f"{allowed}/"):
            raise SandboxPermissionError(
                f"path must be under {VIRTUAL_PATH_PREFIX}: {path}",
                path=path, operation="download",
            )
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                chunks: list[bytes] = []
                total = 0
                for chunk in self._client.file.download_file(path=path):
                    total += len(chunk)
                    if total > _MAX_DOWNLOAD_SIZE:
                        raise SandboxFileError(
                            f"file exceeds {_MAX_DOWNLOAD_SIZE} bytes: {path}",
                            path=path, operation="download",
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxFileError(
                    f"download failed: {exc}", path=path, operation="download",
                ) from exc

    def update_file(self, path: str, content: bytes) -> None:
        with self._lock:
            if self._closed:
                raise SandboxRuntimeError("runtime already closed")
            try:
                b64 = base64.b64encode(content).decode("utf-8")
                self._client.file.write_file(file=path, content=b64, encoding="base64")
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxFileError(
                    f"update failed: {exc}", path=path, operation="update",
                ) from exc

    def close(self) -> None:
        """Best-effort close。遍历属性链找真实 httpx.Client.close()。

        SDK 不暴露 close()，需遍历：
            _client._client_wrapper  -> SyncClientWrapper
                .httpx_client         -> Fern HttpClient（wrapper，非 httpx.Client）
                    .httpx_client      -> httpx.Client  <- 真实 socket owner

        幂等 + 线程安全 + 失败不阻断（仿 deer-flow #2872）。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None

        if client is None:
            return
        wrapper = getattr(client, "_client_wrapper", None)
        fern_http = getattr(wrapper, "httpx_client", None)
        real_httpx = getattr(fern_http, "httpx_client", None)
        target = next(
            (c for c in (real_httpx, fern_http, client)
             if c is not None and hasattr(c, "close")),
            None,
        )
        if target is None:
            logger.debug("DockerRuntime: no closable client found")
            return
        try:
            target.close()
        except Exception as exc:
            logger.warning(f"Error closing DockerRuntime client: {exc}")
