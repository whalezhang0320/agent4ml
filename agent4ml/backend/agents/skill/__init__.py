"""Skill 模块 — 基础层（SkillStore + version DAG + 打点 + 注入）。

公共 API：SkillManager / build_skill_manager。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.config import SkillConfig, load_skill_config
from agent4ml.backend.agents.skill.selector import SkillSelector
from agent4ml.backend.agents.skill.store import SQLiteSkillStore, SkillStore

__all__ = [
    "SkillConfig",
    "SkillStore",
    "SQLiteSkillStore",
    "SkillSelector",
    "SkillManager",
    "build_skill_manager",
]

# 核心系统 skill 目录（随包提交，已验证）。与用户 skill（skills/，gitignore）同机制，
# 均可经 frontmatter enabled:false 或 store 禁用。包相对路径，不依赖 cwd。
# 只有 core/ 子目录在启动时 discover（元 skill 自动加载）；
# 其余子目录（research/software-development/creative/productivity）的 skill 不自动加载，
# 需通过 /skill search 命令或 find-skills skill 搜索后按需使用。
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin_skills"
_BUILTIN_CORE_SKILLS_DIR = _BUILTIN_SKILLS_DIR / "core"


class SkillManager:
    """skill 管理门面。聚合 store + selector + injection/metrics middleware。

    INVARIANT:
    - bootstrap 构造一次，随 AppRuntime 生命周期
    - load_startup(llm) discover skill_dirs + sync_from_files + 建 selector/middleware
    - get_injection_middleware / get_metrics_middleware 供 middleware 链注入
    - switch_expert_mode 不重建（store 持久，INV#持久）
    - middleware 类 lazy import（避免 skill→middlewares→skill 循环）
    """

    def __init__(self, config: SkillConfig) -> None:
        self._config = config
        self._store = SQLiteSkillStore(config.db_path)
        self._selector: SkillSelector | None = None
        self._injection: Any = None
        self._metrics: Any = None
        self._evolution: Any = None  # EvolutionManager，由 bootstrap 装配（避免循环依赖）
        self._eval_layer: Any = None  # L3 EvalLayer，由 bootstrap 装配

    def get_evolution_manager(self) -> Any:
        """返 EvolutionManager（未装配时 None）。"""
        return self._evolution

    def set_evolution_manager(self, manager: Any) -> None:
        """bootstrap 装配 EvolutionManager 注入。"""
        self._evolution = manager

    def get_eval_layer(self) -> Any:
        """返 L3 EvalLayer（未装配时 None）。"""
        return self._eval_layer

    def set_eval_layer(self, eval_layer: Any) -> None:
        """bootstrap 装配 L3 EvalLayer 注入。"""
        self._eval_layer = eval_layer

    def load_startup(self, llm: Any | None = None) -> None:
        """discover skill_dirs（用户 IMPORTED）+ builtin_skills（核心 BUILTIN）+ sync + 建 middleware。"""
        user_dirs = [Path(d) for d in self._config.skill_dirs]
        for rec in self._store.discover(user_dirs, origin="IMPORTED"):
            try:
                self._store.register(rec)
            except Exception:
                pass
        if self._config.include_builtin and _BUILTIN_CORE_SKILLS_DIR.exists():
            for rec in self._store.discover([_BUILTIN_CORE_SKILLS_DIR], origin="BUILTIN"):
                try:
                    self._store.register(rec)
                except Exception:
                    pass

        self._selector = SkillSelector(
            self._store, llm,
            max_skills=self._config.max_inject,
            quality_threshold=self._config.quality_threshold,
            min_selections=self._config.min_selections,
        )

        # lazy import 避免循环（middlewares 反向 import skill.injector/_ctx）
        from agent4ml.backend.agents.middlewares.skill_injection_middleware import (
            SkillInjectionMiddleware,
        )
        from agent4ml.backend.agents.middlewares.skill_metrics_middleware import (
            SkillMetricsMiddleware,
        )
        self._injection = SkillInjectionMiddleware(self._store, self._selector)
        self._metrics = SkillMetricsMiddleware(self._store)

    def get_injection_middleware(self) -> Any:
        return self._injection

    def get_metrics_middleware(self) -> Any:
        return self._metrics

    @property
    def store(self) -> SkillStore:
        return self._store

    @property
    def config(self) -> SkillConfig:
        """暴露 config 供 bootstrap 访问 evolve 参数。"""
        return self._config

    def list_skills(self) -> list[dict]:
        """返回 active skills 概览（供 TUI/CLI 展示）。"""
        result: list[dict] = []
        for rec in self._store.list_active():
            result.append({
                "skill_id": rec.skill_id,
                "name": rec.name,
                "description": rec.description,
                "effective_rate": round(rec.effective_rate, 3),
                "total_selections": rec.total_selections,
                "allowed_tools": list(rec.allowed_tools),
            })
        return result

    def search_builtin_skills(self, query: str) -> list[dict]:
        """搜索全部 builtin_skills/（含非 core 的未加载 skill）按关键词匹配。

        core/ 下的 skill 启动时已 discover 为 active；其余子目录的 skill 仅存在于
        文件系统，不进 DB。此方法扫全树找匹配 name/description 的 SKILL.md，返回
        元数据供 /skill search 命令或 find-skills skill 使用。
        """
        from agent4ml.backend.agents.skill.parser import parse_skill_file

        if not _BUILTIN_SKILLS_DIR.exists():
            return []
        query_lower = query.lower()
        results: list[dict] = []
        for skill_md in sorted(_BUILTIN_SKILLS_DIR.rglob("SKILL.md")):
            try:
                rec = parse_skill_file(skill_md, origin="BUILTIN")
            except Exception:
                continue
            # 匹配 name 或 description
            if query_lower in rec.name.lower() or query_lower in rec.description.lower():
                category = skill_md.parent.parent.name  # core/research/...
                results.append({
                    "name": rec.name,
                    "description": rec.description,
                    "category": category,
                    "path": str(skill_md),
                    "is_active": rec.name in {r["name"] for r in self.list_skills()},
                })
        return results


def build_skill_manager() -> SkillManager | None:
    """从 .env 读配置；用户目录或 builtin core 任一可用即可启动。"""
    config = load_skill_config()
    if not config.enabled:
        return None
    has_user_skills = any(Path(d).exists() for d in config.skill_dirs)
    has_builtin_skills = config.include_builtin and _BUILTIN_CORE_SKILLS_DIR.exists()
    if not has_user_skills and not has_builtin_skills:
        return None
    return SkillManager(config)
