"""Skill 配置层 — .env 读取 + frozen dataclass。

INVARIANT:
- AGENT4ML_SKILL_ENABLED 缺省 false（opt-in，与 MCP 一致）→ build_skill_manager 返 None，既有行为不影响
- AGENT4ML_SKILL_DB_PATH 缺省 .agent4ml/skills.db，相对项目根
- AGENT4ML_SKILL_DIRS 逗号分隔多个扫描目录，缺省 ("skills/",)
- AGENT4ML_SKILL_MAX_INJECT 缺省 3，单轮最多注入 skill 数
- AGENT4ML_SKILL_QUALITY_THRESHOLD 缺省 0.3，quality filter 淘汰阈值
- AGENT4ML_SKILL_MIN_SELECTIONS 缺省 5，淘汰判定最少 selections（anti-loop）
- int/float 转换失败 → 用默认值，不抛
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根——config.py 位于 agent4ml/backend/agents/skill/，parents[4] 即项目根。
# db_path / skill_dirs 默认相对路径，锚定到项目根后 CWD 无关，避免从非项目根
# 启动时 skill 模块因找不到 skills/ 目录或 .env 被误跳过（与 logs_root 同款处理）。
_PROJECT_ROOT = Path(__file__).parents[4]


def _anchor(path: str) -> str:
    """相对路径锚到项目根；绝对路径原样返回。"""
    p = Path(path)
    if p.is_absolute():
        return path
    return str((_PROJECT_ROOT / p).resolve())


@dataclass(frozen=True)
class SkillConfig:
    """Skill 模块顶层配置。

    enabled: 是否启用 skill 模块（缺省 false，false 时 build_skill_manager 返 None）
    db_path: SQLite 路径（相对项目根）
    skill_dirs: skill 扫描目录元组
    max_inject: 单轮最多注入 skill 数
    quality_threshold: quality filter 淘汰阈值（effective_rate < threshold 且 selections >= min）
    min_selections: 淘汰判定最少 selections（anti-loop，给新 skill 数据积累机会）
    """
    enabled: bool = False
    db_path: str = ".agent4ml/skills.db"
    skill_dirs: tuple[str, ...] = ("skills/",)
    include_builtin: bool = True
    max_inject: int = 3
    quality_threshold: float = 0.3
    min_selections: int = 5
    # 自进化（Layer 2a，默认 false opt-in）
    evolve_enabled: bool = False
    evolve_threshold: float = 0.3
    evolve_min_selections: int = 5
    evolve_cooldown_turns: int = 10
    evolve_mutate_budget: int = 20
    evolve_max_steps: int = 5
    eval_config: "SkillEvalConfig" = field(default_factory=lambda: SkillEvalConfig())
    # Skill Hub 配置（H8，默认 true opt-in）
    hub_enabled: bool = True
    hub_quarantine_enabled: bool = True
    hub_audit_log: bool = True


@dataclass(frozen=True)
class SkillEvalConfig:
    """Layer 3 eval 配置（D-L3-8 默认 opt-in false）。"""
    enabled: bool = False
    judgment_enabled: bool = True
    task_judge_enabled: bool = True
    contract_check: bool = True
    async_eval: bool = True
    skip_no_skill: bool = True
    runtime_window: int = 20
    degradation_delta: float = 0.15
    captured_min_score: float = 0.5
    max_messages_chars: int = 80000
    task_weights: tuple[float, ...] = (0.50, 0.35, 0.05, 0.10)


def load_skill_config() -> SkillConfig:
    """从 os.environ 读 AGENT4ML_SKILL_* + 默认值。

    int/float 转换失败时用默认值，不抛异常。
    """
    enabled = os.environ.get("AGENT4ML_SKILL_ENABLED", "false").lower() == "true"
    db_path = _anchor(os.environ.get("AGENT4ML_SKILL_DB_PATH", ".agent4ml/skills.db"))
    include_builtin = os.environ.get("AGENT4ML_SKILL_INCLUDE_BUILTIN", "true").lower() == "true"
    evolve_enabled = os.environ.get("AGENT4ML_SKILL_EVOLVE_ENABLED", "false").lower() == "true"
    # H8: hub 配置（默认 true opt-in）
    hub_enabled = os.environ.get("AGENT4ML_SKILL_HUB_ENABLED", "true").lower() == "true"
    hub_quarantine = os.environ.get("AGENT4ML_SKILL_HUB_QUARANTINE", "true").lower() == "true"
    hub_audit = os.environ.get("AGENT4ML_SKILL_HUB_AUDIT", "true").lower() == "true"

    dirs_raw = os.environ.get("AGENT4ML_SKILL_DIRS", "")
    if dirs_raw:
        skill_dirs = tuple(_anchor(d.strip()) for d in dirs_raw.split(",") if d.strip())
    else:
        skill_dirs = (_anchor("skills"),)

    try:
        max_inject = int(os.environ.get("AGENT4ML_SKILL_MAX_INJECT", "3"))
    except (ValueError, TypeError):
        max_inject = 3

    try:
        quality_threshold = float(os.environ.get("AGENT4ML_SKILL_QUALITY_THRESHOLD", "0.3"))
    except (ValueError, TypeError):
        quality_threshold = 0.3

    try:
        min_selections = int(os.environ.get("AGENT4ML_SKILL_MIN_SELECTIONS", "5"))
    except (ValueError, TypeError):
        min_selections = 5

    try:
        evolve_threshold = float(os.environ.get("AGENT4ML_SKILL_EVOLVE_THRESHOLD", "0.3"))
    except (ValueError, TypeError):
        evolve_threshold = 0.3

    try:
        evolve_min_selections = int(os.environ.get("AGENT4ML_SKILL_EVOLVE_MIN_SELECTIONS", "5"))
    except (ValueError, TypeError):
        evolve_min_selections = 5

    try:
        evolve_cooldown_turns = int(os.environ.get("AGENT4ML_SKILL_EVOLVE_COOLDOWN_TURNS", "10"))
    except (ValueError, TypeError):
        evolve_cooldown_turns = 10

    try:
        evolve_mutate_budget = int(os.environ.get("AGENT4ML_SKILL_EVOLVE_MUTATE_BUDGET", "20"))
    except (ValueError, TypeError):
        evolve_mutate_budget = 20

    try:
        evolve_max_steps = int(os.environ.get("AGENT4ML_SKILL_EVOLVE_MAX_STEPS", "5"))
    except (ValueError, TypeError):
        evolve_max_steps = 5

    # Layer 3 eval 配置
    eval_enabled = os.environ.get("AGENT4ML_SKILL_EVAL_ENABLED", "false").lower() == "true"
    eval_judgment = os.environ.get("AGENT4ML_SKILL_EVAL_JUDGMENT_ENABLED", "true").lower() == "true"
    eval_task_judge = os.environ.get("AGENT4ML_SKILL_EVAL_TASK_JUDGE_ENABLED", "true").lower() == "true"
    eval_contract = os.environ.get("AGENT4ML_SKILL_EVAL_CONTRACT_CHECK", "true").lower() == "true"
    eval_async = os.environ.get("AGENT4ML_SKILL_EVAL_ASYNC", "true").lower() == "true"
    eval_skip_no_skill = os.environ.get("AGENT4ML_SKILL_EVAL_SKIP_NO_SKILL", "true").lower() == "true"

    try:
        eval_window = int(os.environ.get("AGENT4ML_SKILL_EVAL_RUNTIME_WINDOW", "20"))
    except (ValueError, TypeError):
        eval_window = 20

    try:
        eval_degradation = float(os.environ.get("AGENT4ML_SKILL_EVAL_DEGRADATION_DELTA", "0.15"))
    except (ValueError, TypeError):
        eval_degradation = 0.15

    try:
        eval_captured_min = float(os.environ.get("AGENT4ML_SKILL_EVAL_CAPTURED_MIN_SCORE", "0.5"))
    except (ValueError, TypeError):
        eval_captured_min = 0.5

    try:
        eval_max_chars = int(os.environ.get("AGENT4ML_SKILL_EVAL_MAX_MESSAGES_CHARS", "80000"))
    except (ValueError, TypeError):
        eval_max_chars = 80000

    weights_raw = os.environ.get("AGENT4ML_SKILL_EVAL_TASK_WEIGHTS", "0.50,0.35,0.05,0.10")
    try:
        eval_weights = tuple(float(w.strip()) for w in weights_raw.split(","))
    except (ValueError, TypeError):
        eval_weights = (0.50, 0.35, 0.05, 0.10)

    eval_config = SkillEvalConfig(
        enabled=eval_enabled,
        judgment_enabled=eval_judgment,
        task_judge_enabled=eval_task_judge,
        contract_check=eval_contract,
        async_eval=eval_async,
        skip_no_skill=eval_skip_no_skill,
        runtime_window=eval_window,
        degradation_delta=eval_degradation,
        captured_min_score=eval_captured_min,
        max_messages_chars=eval_max_chars,
        task_weights=eval_weights,
    )

    return SkillConfig(
        enabled=enabled,
        db_path=db_path,
        skill_dirs=skill_dirs,
        include_builtin=include_builtin,
        max_inject=max_inject,
        quality_threshold=quality_threshold,
        min_selections=min_selections,
        evolve_enabled=evolve_enabled,
        evolve_threshold=evolve_threshold,
        evolve_min_selections=evolve_min_selections,
        evolve_cooldown_turns=evolve_cooldown_turns,
        evolve_mutate_budget=evolve_mutate_budget,
        evolve_max_steps=evolve_max_steps,
        eval_config=eval_config,
        hub_enabled=hub_enabled,
        hub_quarantine_enabled=hub_quarantine,
        hub_audit_log=hub_audit,
    )
