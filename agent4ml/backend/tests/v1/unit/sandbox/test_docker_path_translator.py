from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.contracts import PathTranslator
from agent4ml.backend.agents.sandbox.translators.docker_path_translator import (
    DockerPathTranslator,
)


class TestDockerPathTranslator:
    def test_translate_path_passthrough(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        assert translator.translate_path("/mnt/agent4ml/user-data/x") == "/mnt/agent4ml/user-data/x"

    def test_translate_command_passthrough(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        cmd = "ls /mnt/agent4ml/user-data"
        assert translator.translate_command(cmd) == cmd

    def test_mask_output_passthrough(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        assert translator.mask_output("output") == "output"

    def test_is_path_translator(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        assert isinstance(translator, PathTranslator)

    def test_empty_path_passthrough(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        assert translator.translate_path("") == ""

    def test_reverse_translate_nested_path(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data/outputs/foo.txt")
        assert result == "/data/aio_docker/abc123/outputs/foo.txt"

    def test_reverse_translate_prefix_only(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data")
        assert result == "/data/aio_docker/abc123"

    def test_reverse_translate_prefix_with_trailing_slash(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data/")
        assert result == "/data/aio_docker/abc123"

    def test_reverse_translate_single_segment(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data/foo")
        assert result == "/data/aio_docker/abc123/foo"

    def test_reverse_translate_non_prefix_raises(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        with pytest.raises(ValueError, match="path not under"):
            translator.reverse_translate("/tmp/foo")

    def test_reverse_translate_partial_prefix_raises(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        with pytest.raises(ValueError, match="path not under"):
            translator.reverse_translate("/mnt/agent4ml/user-data-extra/foo")

    def test_reverse_translate_windows_root(self) -> None:
        translator = DockerPathTranslator("D:/data/aio_docker", "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data/foo")
        assert "abc123" in result
        assert "foo" in result

    def test_reverse_translate_pathlib_root(self) -> None:
        from pathlib import Path

        translator = DockerPathTranslator(Path("/data/aio_docker"), "abc123")
        result = translator.reverse_translate("/mnt/agent4ml/user-data/foo")
        assert result == "/data/aio_docker/abc123/foo"

    def test_reverse_translate_deep_nested(self) -> None:
        translator = DockerPathTranslator("/data/aio_docker", "abc123")
        result = translator.reverse_translate(
            "/mnt/agent4ml/user-data/workspace/project/src/main.py"
        )
        assert result == "/data/aio_docker/abc123/workspace/project/src/main.py"
