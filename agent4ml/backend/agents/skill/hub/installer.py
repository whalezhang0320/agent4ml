"""Installer — install/uninstall/update 编排。

设计（design_docs/46 §2.6）:
- source.fetch → SkillsGuard.scan → parser.install → HubLockFile.add → AuditLog.append
- uninstall 反向流程
- update 对比 upstream hash 与本地 content_hash
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.hub.hub_store import (
    AuditLog,
    HubLockEntry,
    HubLockFile,
    ScanResult,
    SkillsGuard,
)
from agent4ml.backend.agents.skill.hub.source import SkillSource

logger = logging.getLogger(__name__)


class Installer:
    """install/uninstall/update 编排。

    流程：source.fetch → SkillsGuard.scan → parser.install → HubLockFile.add → AuditLog.append
    """

    def __init__(
        self,
        sources: dict[str, SkillSource],
        lock_file: HubLockFile | None = None,
        guard: SkillsGuard | None = None,
        audit_log: AuditLog | None = None,
        dest_root: Path | None = None,
    ) -> None:
        self._sources = sources
        self._lock_file = lock_file or HubLockFile()
        self._guard = guard or SkillsGuard()
        self._audit_log = audit_log or AuditLog()
        self._dest_root = dest_root or (Path.home() / ".agent4ml" / "skills")

    def install(
        self,
        identifier: str,
        name: str | None = None,
    ) -> str:
        """安装 skill：source.fetch → SkillsGuard.scan → parser.install → HubLockFile.add → AuditLog.append。

        identifier: source 特定的唯一标识（如 github:owner/repo@skill-name）
        name: 安装后的 skill 名（None 时从 identifier 推导）
        返 skill_id（parser.install 返）。
        """
        source = self._resolve_source(identifier)
        if source is None:
            raise ValueError(f"Cannot resolve source for identifier: {identifier}")

        # 推导 name（@ 后部分，或 repo 名）
        if name is None:
            name = self._derive_name(identifier)
        if not name:
            raise ValueError(f"Cannot derive name from identifier: {identifier}")

        # 1. source.fetch → 临时目录
        dest_dir = self._dest_root / name
        skill_dir = source.fetch(identifier, dest_dir)

        # 2. SkillsGuard.scan
        scan_result = self._guard.scan(skill_dir, source.name, identifier)
        if not scan_result.allowed:
            logger.warning(
                "Skill %s rejected by SkillsGuard: %s (quarantined to %s)",
                name, scan_result.reasons, scan_result.quarantine_path,
            )
            raise ValueError(
                f"Skill {name} rejected: {scan_result.reasons}"
            )

        # 3. parser.install → 写入 SkillStore（复用既有 install 函数）
        from agent4ml.backend.agents.skill.parser import install as parser_install

        skill_id = parser_install(skill_dir, name, self._dest_root)

        # 4. 计算 content_hash
        content_hash = self._compute_hash(skill_dir)

        # 5. HubLockFile.add → 记录 provenance
        entry = HubLockEntry(
            name=name,
            source=source.name,
            identifier=identifier,
            install_path=str(self._dest_root / name),
            installed_at=self._now_iso(),
            content_hash=content_hash,
            upstream_url=self._derive_upstream_url(source, identifier),
        )
        self._lock_file.add(entry)

        # 6. AuditLog.append → 留痕
        self._audit_log.append(
            "install", name, source=source.name,
            identifier=identifier, skill_id=skill_id,
        )

        logger.info("Skill %s installed (id=%s, source=%s)", name, skill_id, source.name)
        return skill_id

    def uninstall(self, name: str) -> None:
        """卸载 skill：HubLockFile.get → 删目录 → HubLockFile.remove → AuditLog.append。"""
        entry = self._lock_file.get(name)
        if entry is None:
            raise ValueError(f"Skill {name} not found in hub lock file")

        # 删目录
        skill_path = Path(entry.install_path)
        if skill_path.exists():
            import shutil

            shutil.rmtree(skill_path)

        # HubLockFile.remove
        self._lock_file.remove(name)

        # AuditLog.append
        self._audit_log.append("uninstall", name, source=entry.source)

        logger.info("Skill %s uninstalled", name)

    def update(self, name: str | None = None) -> list[str]:
        """检查更新（MVP：返空，需 upstream hash 对比实现）。

        name=None 时检查所有 hub skill。
        返有更新的 skill name 列表。
        """
        # MVP：不实现 update（需重新 fetch + hash 对比）
        # 进阶：对比 upstream_url hash 与本地 content_hash
        return []

    def _resolve_source(self, identifier: str) -> SkillSource | None:
        """从 identifier 前缀解析 source（github: / well-known: / claude-marketplace: / builtin:）。"""
        for prefix, source in self._sources.items():
            if identifier.startswith(prefix + ":") or identifier.startswith(prefix):
                return source
        return None

    def _derive_name(self, identifier: str) -> str:
        """从 identifier 推导 skill name（@ 后部分，或 repo 名）。"""
        # github:owner/repo@skill-name → skill-name
        if "@" in identifier:
            return identifier.rsplit("@", 1)[1]
        # github:owner/repo → repo
        if "/" in identifier:
            return identifier.split("/")[-1]
        # builtin:name → name
        if ":" in identifier:
            return identifier.split(":", 1)[1]
        return identifier

    def _compute_hash(self, skill_dir: Path) -> str:
        """计算 SKILL.md sha256。"""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return ""
        return hashlib.sha256(skill_md.read_bytes()).hexdigest()

    def _now_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _derive_upstream_url(self, source: SkillSource, identifier: str) -> str | None:
        """推导 upstream URL（用于 update 检查）。"""
        # MVP：返 None（进阶实现）
        return None
