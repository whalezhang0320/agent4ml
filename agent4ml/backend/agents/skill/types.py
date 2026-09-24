"""Skill 数据模型 — frozen dataclass + 4 rate property。

INVARIANT:
- 全部 frozen（不可变值对象）
- SkillRecord 4 计数器 + 4 rate property（零除保护，selections/applied=0 返 0.0）
- SkillLineage.parent_skill_ids: () IMPORTED/CAPTURED（root）| (prev,) FIXED | (multi,) DERIVED
- origin: IMPORTED / CAPTURED / FIXED / DERIVED
- 内容在文件（path），SkillRecord 只存引用 + metrics，不存全文
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillLineage:
    """skill 版本血缘。

    parent_skill_ids: () IMPORTED/CAPTURED（root） | (prev,) FIXED | (multi,) DERIVED
    generation: 距 root 深度
    origin: IMPORTED / CAPTURED / FIXED / DERIVED
    version_hash: SKILL.md content sha256
    created_by: "human" | model name | None
    """

    parent_skill_ids: tuple[str, ...] = ()
    generation: int = 0
    origin: str = "IMPORTED"
    version_hash: str = ""
    created_by: str | None = None


@dataclass(frozen=True)
class SkillRecord:
    """skill 注册条目。内容在文件（path），SQLite 只存引用 + metrics。

    is_active: 单指针，每 name 仅 1 个 active 版本
    4 计数器: selections/applied/completions/fallbacks（基础层打点，B2/B3 实现）
    4 rate: applied_rate/completion_rate/effective_rate/fallback_rate（零除保护）
    """

    skill_id: str
    name: str
    path: str
    content_hash: str
    is_active: bool = True
    lineage: SkillLineage = field(default_factory=SkillLineage)
    description: str = ""
    allowed_tools: tuple[str, ...] = ()
    enabled: bool = True
    total_selections: int = 0
    total_applied: int = 0
    total_completions: int = 0
    total_fallbacks: int = 0
    created_at: str = ""
    last_updated: str = ""

    @property
    def applied_rate(self) -> float:
        return self.total_applied / self.total_selections if self.total_selections else 0.0

    @property
    def completion_rate(self) -> float:
        return self.total_completions / self.total_applied if self.total_applied else 0.0

    @property
    def effective_rate(self) -> float:
        return self.total_completions / self.total_selections if self.total_selections else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.total_fallbacks / self.total_selections if self.total_selections else 0.0


@dataclass(frozen=True)
class SkillMetrics:
    """skill quality metrics 快照（get_metrics 返回）。"""

    skill_id: str
    selections: int
    applied: int
    completions: int
    fallbacks: int
    applied_rate: float
    completion_rate: float
    effective_rate: float
    fallback_rate: float


@dataclass(frozen=True)
class SkillHealth:
    """skill 健康状态（health_check 返回）。

    degraded: effective_rate < threshold AND total_selections >= min_selections
    """

    skill_id: str
    name: str
    effective_rate: float
    fallback_rate: float
    total_selections: int
    degraded: bool
