from __future__ import annotations

import signal

from agent4ml.backend.agents.ml_research.task_store import RedisTaskStore
from agent4ml.backend.agents.ml_research.task_worker import MLTaskWorker
from agent4ml.backend.app.ml_service.settings import MLServiceSettings


def api_main() -> None:
    import uvicorn

    settings = MLServiceSettings.from_env()
    uvicorn.run(
        "agent4ml.backend.app.ml_service.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )


def worker_main() -> None:
    settings = MLServiceSettings.from_env()
    store = RedisTaskStore(settings.redis_url, namespace=settings.redis_namespace)
    worker = MLTaskWorker(store)

    def stop_worker(signum: int, frame: object) -> None:
        del signum, frame
        worker.stop()

    signal.signal(signal.SIGINT, stop_worker)
    signal.signal(signal.SIGTERM, stop_worker)
    try:
        worker.run_forever()
    finally:
        store.close()
