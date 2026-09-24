"""策略 bundle 实现。每策略自成子包，实现 GovernanceStrategy 6 hook。

import 此包即触发各族 bundle 的 @register_strategy 注册。
"""

from __future__ import annotations

from agent4ml.backend.agents.context_engineering.strategies.default import DefaultStrategy

__all__ = ["DefaultStrategy"]
