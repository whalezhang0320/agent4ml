"""Tests for recursion_limit derivation from RuntimeConfig."""

from __future__ import annotations

from types import SimpleNamespace

from agent4ml.backend.agents.config.schema import RuntimeConfig
from agent4ml.backend.app.cli.main import _build_stream_config


def _make_runtime_context(max_loop_steps: int = 50, multiplier: int = 10) -> SimpleNamespace:
    rc = RuntimeConfig(max_loop_steps=max_loop_steps, graph_node_multiplier=multiplier)
    return SimpleNamespace(
        config=SimpleNamespace(runtime=rc),
        run_id="run-test",
        thread_id="thread-test",
        journal=None,
        output_dir="/tmp/test",
    )


def _make_app_runtime() -> SimpleNamespace:
    return SimpleNamespace(capability_registry=SimpleNamespace())


class TestRecursionLimitDerivation:
    def test_default_multiplier_produces_500(self) -> None:
        ctx = _make_runtime_context(max_loop_steps=50, multiplier=10)
        rt = _make_app_runtime()
        config = _build_stream_config(rt, ctx)
        assert config["recursion_limit"] == 500

    def test_expert_mode_100_steps_produces_1000(self) -> None:
        ctx = _make_runtime_context(max_loop_steps=100, multiplier=10)
        rt = _make_app_runtime()
        config = _build_stream_config(rt, ctx)
        assert config["recursion_limit"] == 1000

    def test_custom_multiplier(self) -> None:
        ctx = _make_runtime_context(max_loop_steps=30, multiplier=20)
        rt = _make_app_runtime()
        config = _build_stream_config(rt, ctx)
        assert config["recursion_limit"] == 600

    def test_not_hardcoded_300(self) -> None:
        ctx = _make_runtime_context(max_loop_steps=50, multiplier=10)
        rt = _make_app_runtime()
        config = _build_stream_config(rt, ctx)
        assert config["recursion_limit"] != 300
