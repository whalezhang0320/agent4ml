"""Skill Hub 模块 — 多 source skill 发现 + 安装 + 安全 + provenance。

设计（design_docs/46 §2）:
- SkillSource Protocol + 多 source adapter（Builtin/GitHub/WellKnown/ClaudeMarketplace）
- HubLockFile 跟踪 provenance + SkillsGuard 安全扫描 + AuditLog 留痕
- CLI + slash command 双入口
- 零外部依赖：复用既有 agents/skill/ 模块（parser/store/selector）
"""
from __future__ import annotations
