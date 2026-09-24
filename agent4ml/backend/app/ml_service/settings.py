from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLServiceSettings:
    redis_url: str
    redis_namespace: str
    work_root: Path
    allowed_root: Path
    host: str
    port: int
    sse_heartbeat_seconds: float

    @classmethod
    def from_env(cls) -> MLServiceSettings:
        project_root = Path.cwd().resolve()
        return cls(
            redis_url=os.environ.get("AGENT4ML_ML_REDIS_URL", "redis://127.0.0.1:6379/0"),
            redis_namespace=os.environ.get("AGENT4ML_ML_REDIS_NAMESPACE", "agent4ml:ml"),
            work_root=Path(
                os.environ.get("AGENT4ML_ML_WORK_ROOT", str(project_root / ".agent4ml/ml-tasks"))
            ).expanduser().resolve(),
            allowed_root=Path(
                os.environ.get("AGENT4ML_ML_ALLOWED_ROOT", str(project_root))
            ).expanduser().resolve(),
            host=os.environ.get("AGENT4ML_ML_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("AGENT4ML_ML_API_PORT", "8000")),
            sse_heartbeat_seconds=float(
                os.environ.get("AGENT4ML_ML_SSE_HEARTBEAT_SECONDS", "15")
            ),
        )
