from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Mock langchain_core with MagicMock if not installed (bootstrap.py imports it at top level)
try:
    import langchain_core  # noqa: F401
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    for _mod_name in ("langchain_core", "langchain_core.language_models",
                       "langchain_core.tools", "langchain"):
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = MagicMock()

from agent4ml.backend.agents.sandbox.local.local_sandbox_provider import (  # noqa: E402
    LocalSandboxProvider,
)
from agent4ml.backend.agents.sandbox.types import PathMapping  # noqa: E402


class TestLocalSandboxProviderCompat:
    def test_accepts_sandbox_config(self) -> None:
        config = MagicMock()
        provider = LocalSandboxProvider(path_mappings=[], sandbox_config=config)
        assert provider is not None

    def test_sandbox_config_defaults_none(self) -> None:
        provider = LocalSandboxProvider(path_mappings=[])
        assert provider is not None

    def test_sandbox_config_ignored(self) -> None:
        config = MagicMock()
        config.image = "should-be-ignored"
        provider = LocalSandboxProvider(path_mappings=[], sandbox_config=config)
        # LocalSandboxProvider doesn't use sandbox_config 鈥?no image attribute
        assert not hasattr(provider, "_image")


class TestBuildPathMappings:
    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain_core not installed")
    def test_contains_local_prefix(self) -> None:
        from agent4ml.backend.app.bootstrap import _build_path_mappings

        config = MagicMock()
        config.mounts = []
        mappings = _build_path_mappings(config)
        for m in mappings:
            if "workspace" in m.local_path:
                assert "local" in m.local_path
                break

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain_core not installed")
    def test_custom_mounts_appended(self) -> None:
        from agent4ml.backend.app.bootstrap import _build_path_mappings

        config = MagicMock()
        mount = MagicMock()
        mount.host_path = "/skills"
        mount.container_path = "/mnt/agent4ml/skills"
        mount.read_only = True
        config.mounts = [mount]
        mappings = _build_path_mappings(config)
        custom = [m for m in mappings if m.container_path == "/mnt/agent4ml/skills"]
        assert len(custom) == 1
        assert custom[0].local_path == "/skills"
        assert custom[0].read_only is True

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain_core not installed")
    def test_default_three_mappings(self) -> None:
        from agent4ml.backend.app.bootstrap import _build_path_mappings

        config = MagicMock()
        config.mounts = []
        mappings = _build_path_mappings(config)
        paths = [m.container_path for m in mappings]
        assert "/mnt/agent4ml/user-data/workspace" in paths
        assert "/mnt/agent4ml/user-data/uploads" in paths
        assert "/mnt/agent4ml/user-data/outputs" in paths


class TestLoadSandboxProvider:
    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain_core not installed")
    def test_use_empty_returns_none(self) -> None:
        from agent4ml.backend.app.bootstrap import _load_sandbox_provider

        config = MagicMock()
        config.sandbox = MagicMock()
        config.sandbox.use = ""
        assert _load_sandbox_provider(config) is None

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain_core not installed")
    def test_passes_sandbox_config(self) -> None:
        from agent4ml.backend.app.bootstrap import _load_sandbox_provider

        config = MagicMock()
        config.sandbox = MagicMock()
        config.sandbox.use = "test_module:TestProvider"
        mock_provider_cls = MagicMock(return_value=MagicMock())
        mock_module = MagicMock(TestProvider=mock_provider_cls)
        with patch("importlib.import_module", return_value=mock_module):
            _load_sandbox_provider(config)
        mock_provider_cls.assert_called_once()
        call_kwargs = mock_provider_cls.call_args.kwargs
        assert "sandbox_config" in call_kwargs
        assert "path_mappings" in call_kwargs
