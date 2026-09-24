"""MemoryTrace — 记忆原子单元（frozen dataclass）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §5.2 + §5.6
+ `48-memory-l1-base-layer.md` §4 Step 2。

INVARIANT:
- MemoryTrace 不可变：strength 等可变字段通过 `with_strength()` / `with_operation()`
  创建新实例替换（类似 skill version DAG 的 is_active 指针）
- operation_log traceability：上限 20 条 FIFO；retrieve 不记（高频，强化在
  strength/access_count 已体现）；actor 字段预留 turn_id（Layer 4 Middleware 注入）
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """记忆类型（认知科学映射，00 §5.2）。"""

    EPISODIC = "episodic"        # 事件记忆：衰减快，需反复检索强化
    SEMANTIC = "semantic"        # 语义记忆：衰减慢，提炼后的稳定知识
    PROCEDURAL = "procedural"    # 过程记忆：几乎不衰减，习得的技能（skill 系统管理）


@dataclass(frozen=True)
class Association:
    """记忆关联（扩散激活用）。"""

    target_id: str
    strength: float = 0.5
    type: str = "related"        # related / causal / temporal / contrast 等


@dataclass(frozen=True)
class OperationLog:
    """操作日志条目（traceability，debug 用）。

    每次 manager 操作（encode/associate/consolidate/reconsolidate/forget）append 一条。
    retrieve 不记（高频，强化在 strength/access_count 已体现）。
    actor 字段预留 turn_id（Layer 4 Middleware 注入），Layer 2 留 None。
    """

    timestamp: float                      # 操作发生时间（unix timestamp）
    operation: str                        # encode/associate/consolidate/reconsolidate/forget
    actor: str | None = None              # thread_id / turn_id（Layer 4 注入）
    diff: dict[str, Any] | None = None    # 变更摘要，如 {"content": ("old...", "new..."), "strength": (0.5, 0.7)}


@dataclass(frozen=True)
class MemoryTrace:
    """记忆痕迹 —：strength 等可变字段通过 `with_strength()` / `with_operation()`
    创建新实例替换（类似 skill version DAG 的 is_active 指针）。

    traceability：operation_log 记录操作历史（上限 20 条，超出丢最老），
    支持 debug 回溯"谁何时对这条 trace 做了什么"。
    """

    id: str
    content: str
    type: MemoryType
    # 强度与衰减（00 §5.5）
    strength: float = 0.0                # 当前强度 0.0~1.0（lazy decay，retrieve 时计算）
    base_strength: float = 0.7           # 初始强度（由 type 决定，defaults 填充）
    decay_rate: float = 0.1              # 衰减速率（由 type 决定）
    access_count: int = 0                # 累计访问次数
    last_accessed: float = 0.0           # 最后访问时间（unix timestamp）
    importance: float = 0.5              # 语义重要性 0.0~1.0
    # 关联（扩散激活）
    associations: tuple[Association, ...] = field(default_factory=tuple)
    # embedding 预留（00 §5.6，默认 None，向量库阶段填充）
    embedding: tuple[float, ...] | None = None
    # Agent4ML 扩展字段（00 §5.6）
    source: str | None = None            # 来源（thread_id / run_id / user_input）
    created_at: float = 0.0              # 创建时间
    metadata: dict[str, Any] = field(default_factory=dict)  # tags / project / specialist_id
    # traceability（操作日志，上限 20 条，append-only）
    operation_log: tuple[OperationLog, ...] = field(default_factory=tuple)

    def with_strength(self, new_strength: float, accessed_at: float) -> "MemoryTrace":
        """创建新 trace 替换旧 trace（frozen 语义，检索强化用）。

        00 §5.3 Retrieve 操作：检索时自动增强 strength + access_count + last_accessed。
        retrieve 不记 operation_log（高频，强化在 strength/access_count 已体现）。
        """
        return replace(
            self,
            strength=new_strength,
            access_count=self.access_count + 1,
            last_accessed=accessed_at,
        )

    def with_operation(self, log: OperationLog, *, max_log: int = 20) -> "MemoryTrace":
        """append 一条操作日志（frozen 语义，创建新实例）。

        上限 max_log 条（默认 20），超出丢最老（FIFO）。
        manager 各操作（encode/associate/consolidate/reconsolidate/forget）调用。
        """
        new_log = self.operation_log + (log,)
        if len(new_log) > max_log:
            new_log = new_log[-max_log:]  # 保留最近 max_log 条
        return replace(self, operation_log=new_log)
