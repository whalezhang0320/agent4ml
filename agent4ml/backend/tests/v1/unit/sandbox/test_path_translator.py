from __future__ import annotations

from agent4ml.backend.agents.sandbox.contracts import PathTranslator


class _CompleteTranslator:
    """Mock 实现 PathTranslator 全部 3 方法。"""

    def translate_path(self, virtual_path: str) -> str:
        return virtual_path

    def translate_command(self, command: str) -> str:
        return command

    def mask_output(self, output: str) -> str:
        return output


class _IncompleteTranslator:
    """缺 mask_output 方法。"""
    def translate_path(self, virtual_path: str) -> str:
        return virtual_path


class TestPathTranslatorProtocol:
    def test_complete_impl_is_instance(self) -> None:
        translator = _CompleteTranslator()
        assert isinstance(translator, PathTranslator)

    def test_incomplete_impl_not_instance(self) -> None:
        translator = _IncompleteTranslator()
        assert not isinstance(translator, PathTranslator)

    def test_identity_semantics(self) -> None:
        """IdentityTranslator 直传，translate_path == mask_output 逆操作。"""
        translator = _CompleteTranslator()
        assert translator.translate_path("/mnt/x") == "/mnt/x"
        assert translator.mask_output("/mnt/x") == "/mnt/x"
