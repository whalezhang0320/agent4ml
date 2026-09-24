"""默认策略常量（衰减参数 / 遗忘阈值 / 检索权重 / 巩固参数 / 关联默认）。

Layer 1：仅衰减参数 + 检索权重（供 schema 默认值参考）。
Layer 2：补全遗忘阈值 / 巩固参数 / 关联默认。
Layer 3：BM25 参数。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §5.2 + §5.5
+ `49-memory-l2-default-strategies.md` §4 Step 1。
"""

from __future__ import annotations

# 00 §5.2 三类记忆衰减参数（示例值，可调）
DECAY_PARAMS = {
    "episodic": {"base_strength": 0.7, "decay_rate": 0.1},      # 衰减快
    "semantic": {"base_strength": 0.8, "decay_rate": 0.02},     # 衰减慢
    "procedural": {"base_strength": 0.9, "decay_rate": 0.005},  # 几乎不衰减
}

# 00 §5.5 复合检索分数权重
RETRIEVAL_WEIGHTS = {
    "similarity": 0.7,   # 语义相关性占主导
    "strength": 0.3,     # 记忆强度参与排序
}

# 00 §5.5 衰减公式系数
DECAY_COEFFICIENTS = {
    "access_boost": 0.1,        # log(1 + access_count) × 0.1
    "importance_boost": 0.05,   # importance × 0.05
}

# 遗忘阈值（00 §7.4 CompositeForgetPolicy）
FORGET_THRESHOLDS = {
    "strength_threshold": 0.1,   # strength 低于此值 → 遗忘
    "ttl_hours": 720,            # 30 天未访问 → 遗忘（TTL）
    "conflict_window_hours": 24, # 矛盾检测窗口（新记忆 24h 内覆盖旧记忆，预留 Layer 5）
}

# 巩固参数（00 §5.3 Consolidate）
CONSOLIDATE_PARAMS = {
    "min_traces_to_consolidate": 2,  # 最少 2 条才能合并
    "max_traces_to_consolidate": 10, # 最多 10 条一次合并（避免 LLM 上下文爆炸，E1）
    "default_consolidated_type": "semantic",  # 合并后默认转 semantic
    "default_importance_boost": 0.1, # 合并后 importance 提升（稳定知识更重要）
}

# associate 默认参数（00 §5.3 Associate）
ASSOCIATE_DEFAULTS = {
    "default_strength": 0.5,     # 默认关联强度
    "default_type": "related",   # related / causal / temporal / contrast
    "max_associations_per_trace": 20,  # 单 trace 最大关联数（防膨胀，D3 LRU 淘汰）
}

# BM25 检索参数（50 §4 Step 2 HybridRetriever）
BM25_PARAMS = {
    "k1": 1.5,       # TF 饱和参数（词频归一，1.2~2.0 典型）
    "b": 0.75,       # 文档长度归一参数（0=不考虑长度，1=完全归一）
    "epsilon": 0.25, # IDF 下限平滑（防负 IDF）
}
