from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.sandbox.contracts import PathTranslator
from agent4ml.backend.agents.sandbox.translators.local_path_translator import (
    LocalPathTranslator,
)
from agent4ml.backend.agents.sandbox.types import PathMapping


@pytest.fixture
def translator(tmp_path) -> LocalPathTranslator:
    ws = tmp_path / "ws"
    ws.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    mappings = [
        PathMapping("/mnt/agent4ml/user-data/workspace", str(ws)),
        PathMapping("/mnt/agent4ml/skills", str(skills), read_only=True),
    ]
    return LocalPathTranslator(mappings)


class TestTranslatePath:
    def test_translate(self, translator: LocalPathTranslator) -> None:
        result = translator.translate_path("/mnt/agent4ml/user-data/workspace/file.txt")
        assert Path(result).name == "file.txt"
        assert "ws" in Path(result).parts

    def test_translate_root(self, translator: LocalPathTranslator, tmp_path) -> None:
        result = translator.translate_path("/mnt/agent4ml/user-data/workspace")
        assert result == str((tmp_path / "ws").resolve())

    def test_no_match_returns_original(self, translator: LocalPathTranslator) -> None:
        assert translator.translate_path("/unknown/path") == "/unknown/path"

    def test_traversal_rejected(self, translator: LocalPathTranslator) -> None:
        with pytest.raises(PermissionError, match="path traversal"):
            translator.translate_path("/mnt/agent4ml/user-data/workspace/../../../etc/passwd")

    def test_idempotent(self, translator: LocalPathTranslator) -> None:
        path = "/mnt/agent4ml/user-data/workspace/file.txt"
        assert translator.translate_path(path) == translator.translate_path(path)

    def test_longest_prefix_priority(self, tmp_path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        ws_extra = tmp_path / "ws-extra"
        ws_extra.mkdir()
        mappings = [
            PathMapping("/mnt/agent4ml/skills", str(ws)),
            PathMapping("/mnt/agent4ml/skills-extra", str(ws_extra)),
        ]
        translator = LocalPathTranslator(mappings)
        result = translator.translate_path("/mnt/agent4ml/skills-extra/x")
        assert str(ws_extra.resolve()) in result


class TestTranslateCommand:
    def test_replace_path_in_command(self, translator: LocalPathTranslator, tmp_path) -> None:
        cmd = "ls /mnt/agent4ml/user-data/workspace"
        result = translator.translate_command(cmd)
        assert "/mnt/agent4ml/user-data/workspace" not in result
        assert str((tmp_path / "ws").resolve()) in result

    def test_shell_boundary_semicolon(self, translator: LocalPathTranslator) -> None:
        cmd = "ls /mnt/agent4ml/user-data/workspace;echo done"
        result = translator.translate_command(cmd)
        assert "echo done" in result

    def test_no_match_passthrough(self, translator: LocalPathTranslator) -> None:
        cmd = "echo hello"
        assert translator.translate_command(cmd) == "echo hello"


class TestMaskOutput:
    def test_reverse_translate(self, translator: LocalPathTranslator, tmp_path) -> None:
        physical = str((tmp_path / "ws" / "file.txt").resolve())
        result = translator.mask_output(physical)
        assert "/mnt/agent4ml/user-data/workspace/file.txt" in result

    def test_inverse_of_translate_path(self, translator: LocalPathTranslator, tmp_path) -> None:
        virtual = "/mnt/agent4ml/user-data/workspace/file.txt"
        physical = translator.translate_path(virtual)
        masked = translator.mask_output(physical)
        assert masked == virtual
    def test_no_match_passthrough(self, translator: LocalPathTranslator) -> None:
        assert translator.mask_output("no paths here") == "no paths here"


class TestCachedProperty:
    def test_command_pattern_cached(self, translator: LocalPathTranslator) -> None:
        p1 = translator._command_pattern
        p2 = translator._command_pattern
        assert p1 is p2

    def test_resolved_local_paths_cached(self, translator: LocalPathTranslator) -> None:
        d1 = translator._resolved_local_paths
        d2 = translator._resolved_local_paths
        assert d1 is d2

    def test_empty_mappings_command_pattern_none(self) -> None:
        translator = LocalPathTranslator([])
        assert translator._command_pattern is None


class TestProtocolConformance:
    def test_is_path_translator(self, translator: LocalPathTranslator) -> None:
        assert isinstance(translator, PathTranslator)
