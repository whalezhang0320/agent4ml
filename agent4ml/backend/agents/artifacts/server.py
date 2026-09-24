"""Lightweight HTTP server for artifact download/preview.

Binds 127.0.0.1, serves files registered via register() at /artifacts/{sandbox_id}/{filename}.
Started by bootstrap, stopped on exit. No third-party deps (stdlib http.server).
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 18080
_MAX_PORT_RETRIES = 10


class ArtifactServer:
    """Local HTTP server serving registered artifact files."""

    def __init__(self, host: str = "127.0.0.1", port: int = _DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._registry: dict[str, str] = {}  # f"{sandbox_id}/{filename}" -> host_path
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start HTTP server in background thread. Auto-increment port if occupied."""
        for attempt in range(_MAX_PORT_RETRIES):
            port = self._port + attempt
            try:
                server = ThreadingHTTPServer((self._host, port), self._make_handler())
                self._port = port
                self._server = server
                self._thread = threading.Thread(target=server.serve_forever, daemon=True)
                self._thread.start()
                logger.info(f"ArtifactServer listening on {self.base_url}")
                return
            except OSError:
                logger.warning(f"Port {port} occupied, trying {port + 1}")
                continue
        raise RuntimeError(f"Could not bind ArtifactServer on ports {self._port}-{self._port + _MAX_PORT_RETRIES}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def register(self, sandbox_id: str, filename: str, host_path: str) -> str:
        """Register a host path, return downloadable URL. filename may contain subdirs."""
        key = f"{sandbox_id}/{filename}"
        with self._lock:
            self._registry[key] = host_path
        return f"{self.base_url}/artifacts/{sandbox_id}/{quote(filename, safe='')}"

    def _resolve(self, sandbox_id: str, filename: str) -> str | None:
        key = f"{sandbox_id}/{filename}"
        with self._lock:
            return self._registry.get(key)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        registry_resolve = self._resolve

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                logger.debug("ArtifactServer: " + fmt, *args)

            def do_GET(self) -> None:
                path = unquote(self.path)
                if not path.startswith("/artifacts/"):
                    self.send_error(404)
                    return
                rest = path[len("/artifacts/"):]
                parts = rest.split("/", 1)
                if len(parts) != 2 or ".." in rest:
                    self.send_error(403, "path traversal blocked")
                    return
                sandbox_id = parts[0]
                filename = parts[1]  # may contain subdirs (e.g. workspace/x.pptx)
                host_path = registry_resolve(sandbox_id, filename)
                if host_path is None:
                    self.send_error(404, "artifact not registered")
                    return
                p = Path(host_path)
                if not p.exists() or not p.is_file():
                    self.send_error(404, "file not found")
                    return
                data = p.read_bytes()
                display_name = Path(filename).name
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'inline; filename="{display_name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return _Handler

