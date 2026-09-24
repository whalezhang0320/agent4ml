"""LLMMutator — LLM 推理变异（FIX 编辑 + CAPTURED 生成）。

约束（防"凭感觉自由改"）：
- budget：单次最多改 max_changed_lines 行（超则截断，partial apply 前 N 改动）
- 单维度：LLM prompt 指示一次只改一个 section
- 保留 frontmatter：不改 name/allowed-tools（FIX 重新附加原 frontmatter）
- diff 输出：difflib.unified_diff，返 (candidate, diff)
- max_steps：迭代上限（调用方 EvolutionManager 控制，本类单次 mutate）

进化逻辑唯一 LLM 推理（RL/计算式不适用，37.md D-L2-3）。LLM 可自主调既有 web_search 工具。
candidate is_active=False（champion 隔离），写 staging 路径（.agent4ml/skills_staging/）。
"""
from __future__ import annotations

import difflib
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.evolution.types import EvolutionContext
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord

_STAGING_ROOT = Path(".agent4ml/skills_staging")


class LLMMutator:
    """LLM 编辑 SKILL.md body（FIX）或生成新 SKILL.md（CAPTURED）。"""

    def __init__(
        self,
        max_changed_lines: int = 20,
        max_steps: int = 5,
        llm: Any | None = None,
    ) -> None:
        self._max_changed_lines = max_changed_lines
        self._max_steps = max_steps
        self._llm = llm

    def mutate(self, ctx: EvolutionContext, llm: Any | None = None) -> tuple[SkillRecord, str]:
        """产 (candidate SkillRecord, diff str)。candidate is_active=False。"""
        llm = llm or self._llm
        if ctx.evolution_type == "FIX":
            return self._mutate_fix(ctx, llm)
        if ctx.evolution_type == "CAPTURED":
            return self._mutate_capture(ctx, llm)
        raise ValueError(f"unsupported evolution_type: {ctx.evolution_type}")

    # ── FIX ──────────────────────────────────────────────

    def _mutate_fix(self, ctx: EvolutionContext, llm: Any | None) -> tuple[SkillRecord, str]:
        baseline = ctx.target_skill
        assert baseline is not None
        orig_content = Path(baseline.path).read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(orig_content)

        new_body = self._llm_edit_body(body, ctx, llm)
        new_body = self._enforce_budget(body, new_body, self._max_changed_lines)
        diff = self._compute_diff(body, new_body)

        # 保留 frontmatter（不改 name/allowed-tools）
        new_content = frontmatter + new_body if frontmatter else new_body
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]
        staging_path = self._staging_path(baseline.name)
        staging_path.write_text(new_content, encoding="utf-8")

        candidate = SkillRecord(
            skill_id=f"{baseline.name}__cand_{uuid.uuid4().hex[:8]}",
            name=baseline.name,
            path=str(staging_path),
            content_hash=new_hash,
            is_active=False,
            lineage=SkillLineage(
                parent_skill_ids=(baseline.skill_id,),
                generation=baseline.lineage.generation + 1,
                origin="FIXED",
                version_hash=new_hash,
                created_by="llm_mutator",
            ),
            description=baseline.description,
            allowed_tools=baseline.allowed_tools,
            enabled=True,
        )
        return candidate, diff

    # ── CAPTURED ─────────────────────────────────────────

    def _mutate_capture(self, ctx: EvolutionContext, llm: Any | None) -> tuple[SkillRecord, str]:
        new_content = self._llm_generate_skill(ctx, llm)
        # 校验 frontmatter + 提取 name
        frontmatter, body = self._split_frontmatter(new_content)
        name = self._extract_frontmatter_name(frontmatter) or ctx.suggested_name
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise ValueError(f"invalid skill name from LLM: {name!r}")

        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]
        staging_path = self._staging_path(name)
        staging_path.write_text(new_content, encoding="utf-8")

        candidate = SkillRecord(
            skill_id=f"{name}__cand_{uuid.uuid4().hex[:8]}",
            name=name,
            path=str(staging_path),
            content_hash=new_hash,
            is_active=False,
            lineage=SkillLineage(
                parent_skill_ids=(),
                generation=0,
                origin="CAPTURED",
                version_hash=new_hash,
                created_by="llm_mutator",
            ),
            description=self._extract_frontmatter_desc(frontmatter),
            allowed_tools=(),
            enabled=True,
        )
        diff = "+ full new SKILL.md (CAPTURED)"
        return candidate, diff

    # ── LLM 调用 ─────────────────────────────────────────

    def _llm_edit_body(self, body: str, ctx: EvolutionContext, llm: Any | None) -> str:
        """LLM 编辑 body（单维度 + budget 指令）。失败返原 body（不变）。"""
        if llm is None:
            return body  # 无 LLM 不变异
        try:
            from langchain_core.messages import HumanMessage
            prompt = (
                f"修复方向: {ctx.fix_direction}\n\n"
                f"当前 SKILL.md body:\n{body}\n\n"
                f"约束：只改一个 section（何时使用/步骤/失败模式之一），"
                f"最多改 {self._max_changed_lines} 行，保留 frontmatter 不动。"
                f"只返编辑后的 body 全文。"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return body  # 失败不变异

    def _llm_generate_skill(self, ctx: EvolutionContext, llm: Any | None) -> str:
        """LLM 生成全新 SKILL.md（CAPTURED）。失败抛（CAPTURED 必须有 LLM）。"""
        if llm is None:
            raise ValueError("CAPTURED 需要 LLM 生成 SKILL.md")
        from langchain_core.messages import HumanMessage
        prompt = (
            f"可复用模式: {ctx.capture_pattern}\n"
            f"建议 skill name: {ctx.suggested_name}\n\n"
            f"生成完整 SKILL.md（含 frontmatter: name + description），"
            f"描述这研究过程知识。只返 SKILL.md 全文。"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)

    # ── budget 截断 ──────────────────────────────────────

    @staticmethod
    def _enforce_budget(orig_body: str, new_body: str, budget: int) -> str:
        """超 budget 截断：partial apply 前 budget 个改动，超出部分回退原 body。

        改动数按 difflib SequenceMatcher opcodes 的 max(i2-i1, j2-j1) 计。
        """
        orig_lines = orig_body.splitlines()
        new_lines = new_body.splitlines()
        sm = difflib.SequenceMatcher(a=orig_lines, b=new_lines)
        opcodes = sm.get_opcodes()

        # 统计改动数
        changed = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                changed += max(i2 - i1, j2 - j1)
        if changed <= budget:
            return new_body

        # 截断：partial apply 前 budget 改动
        result: list[str] = []
        applied = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                result.extend(orig_lines[i1:i2])
                continue
            remaining = budget - applied
            if remaining <= 0:
                result.extend(orig_lines[i1:i2])  # 回退原
                continue
            n_apply = min(remaining, j2 - j1)
            result.extend(new_lines[j1:j1 + n_apply])
            applied += n_apply
            if n_apply < (j2 - j1):
                # replace 的原部分补回
                if tag == "replace":
                    result.extend(orig_lines[i1 + n_apply:i2])
        trailing = "\n" if new_body.endswith("\n") else ""
        return "\n".join(result) + trailing

    # ── helpers ──────────────────────────────────────────

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[str, str]:
        """返 (frontmatter 含 --- 定界, body)。无 frontmatter 返 ("", content)。"""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm = "---" + parts[1] + "---"
                body = parts[2].lstrip("\r\n")
                return fm, body
        return "", content

    @staticmethod
    def _extract_frontmatter_name(frontmatter: str) -> str | None:
        m = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_frontmatter_desc(frontmatter: str) -> str:
        m = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _compute_diff(orig: str, new: str) -> str:
        diff = difflib.unified_diff(
            orig.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="body", tofile="body_edited", n=2,
        )
        return "".join(diff)

    @staticmethod
    def _staging_path(name: str) -> Path:
        d = _STAGING_ROOT / f"{name}__cand_{uuid.uuid4().hex[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "SKILL.md"
