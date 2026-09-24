"""Memory 辅助类型：MemoryQuery / MemoryFilter / RetrievalResult。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.5
+ `48-memory-l1-base-layer.md` §4 Step 3。

INVARIANT: RetrievalResult.score 复合分数公式 `score = similarity * 0.7 +
strength * 0.3`（00 §5.5），语义相关性占主导（70%），记忆强度参与排序（30%）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType


@dataclass(frozen=True)
class MemoryQuery:
    """检索查询。"""

    text: str
    top_k: int = 5
    type_filter: MemoryType | None = None
    min_strength: float = 0.0
    metadata_filter: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryFilter:
    """记忆过滤（遗忘策略用）。"""

    type_filter: MemoryType | None = None
    min_strength: float = 0.0
    max_age_hours: float | None = None       # 最大年龄（小时），None=不限
    metadata_filter: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果（00 §7.5）。"""

    trace: MemoryTrace
    similarity: float                        # 语义相似度 0.0~1.0
    strength: float                          # 当前强度（retrieve 时按需计算）
    score: float                             # = similarity * 0.7 + strength * 0.3（00 §5.5）

    @classmethod
    def compute_score(
        cls, trace: MemoryTrace, similarity: float, strength: float
    ) -> "RetrievalResult":
        """复合检索分数：score = similarity * 0.7 + strength * 0.3。"""
        score = similarity * 0.7 + strength * 0.3
        return cls(trace=trace, similarity=similarity, strength=strength, score=score)
