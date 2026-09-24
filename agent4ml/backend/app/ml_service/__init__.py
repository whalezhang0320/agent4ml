"""FastAPI control plane for local Agent4ML ML tasks."""

from agent4ml.backend.app.ml_service.api import create_app

__all__ = ["create_app"]
