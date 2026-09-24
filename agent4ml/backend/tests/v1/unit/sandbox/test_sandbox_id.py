"""S6: sandbox_id 格式校验测试。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.utils.sandbox_id import validate_sandbox_id


class TestValidateSandboxId:
    def test_valid_hex8_passes(self) -> None:
        validate_sandbox_id("a1b2c3d4")

    def test_all_zeros_passes(self) -> None:
        validate_sandbox_id("00000000")

    def test_all_f_passes(self) -> None:
        validate_sandbox_id("ffffffff")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("../../etc")

    def test_too_short_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("a1b2c3")

    def test_too_long_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("a1b2c3d4e5")

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("A1B2C3D4")

    def test_non_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("g1b2c3d4")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("")

    def test_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id(None)  # type: ignore[arg-type]

    def test_dot_dot_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("../../../")

    def test_injection_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sandbox_id"):
            validate_sandbox_id("a1b2c3d4;rm -rf /")
