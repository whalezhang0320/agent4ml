"""skill_search tool — agent graph ReAct 可调的 skill 搜索工具。

设计（设计文档 46 §3.2 改造点 A）：
- 让 agent 在接到任务时主动调，而不是只能用 /skill search 命令
- 调 SkillManager.search_builtin_skills + hub unified_search（hub 不可用降级为只搜 builtin）
- 返 JSON 列表（name/description/category/source/is_active/path）
- 归 core group（default + expert 都加载）

find-skills 主动触发的核心：agent 接到 specialized task 时先调此工具 check 是否有相关 skill。
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool("skill_search")
def skill_search(query: str, limit: int = 5) -> str:
    """Search builtin + hub skills by keyword. Returns JSON list of matches.

    Call this PROACTIVELY when:
    - User asks for a specialized task (frontend design, chart, github PR, debug, tdd, etc.)
    - You're about to start a complex task that might benefit from existing skill
    - Before writing code for a common pattern (check if skill exists first)

    Returns: [{"name", "description", "category", "source", "is_active", "path"}]

    Common trigger keywords:
    - frontend / UI / React / Vue → creative/frontend-design
    - chart / graph / visualization → creative/chart-visualization
    - diagram / architecture → creative/architecture-diagram
    - github / PR / code review → software-development/github-*
    - debug / bug → core/systematic-debugging
    - test / TDD → core/test-driven-development
    - plan / spike → core/plan / core/spike
    - simplify / refactor → core/simplify-code
    """
    from agent4ml.backend.agents.skill import build_skill_manager

    mgr = build_skill_manager()
    if mgr is None:
        return "[]"

    # builtin 搜索（零网络，扫 builtin_skills/ 树）
    builtin_results = mgr.search_builtin_skills(query)

    # hub 搜索（如果 hub 启用；不可用时降级为只搜 builtin）
    hub_results: list[dict] = []
    try:
        from agent4ml.backend.agents.skill.hub.search import unified_search

        hub_results = unified_search(query, limit=limit)
    except ImportError:
        # hub 模块未安装（add-skill-hub change 未实现），降级为只搜 builtin
        pass
    except Exception as e:
        logger.warning("skill_search hub search failed, degrading to builtin only: %s", e)

    # 合并 + 去重（按 name）+ 截断
    seen_names: set[str] = set()
    all_results: list[dict] = []
    for r in builtin_results + hub_results:
        # 统一转 dict（builtin 返 dict，hub 返 SkillMeta frozen dataclass）
        if hasattr(r, "name"):
            # SkillMeta → dict
            r_dict = {
                "name": r.name,
                "description": r.description,
                "category": r.category,
                "source": r.source,
                "identifier": r.identifier,
                "is_installed": r.is_installed,
                "install_path": r.install_path,
                "preview_url": r.preview_url,
            }
        else:
            # 已是 dict
            r_dict = r
        name = r_dict.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            # 统一字段格式（builtin 已有 source 缺失，补 "builtin"）
            if "source" not in r_dict:
                r_dict["source"] = "builtin"
            all_results.append(r_dict)

    return json.dumps(all_results[:limit], ensure_ascii=False, indent=2)
