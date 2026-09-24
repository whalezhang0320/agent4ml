"""HubLockFile + SkillsGuard + AuditLog — provenance + 安全 + 留痕。

设计（design_docs/46 §2.5）:
- HubLockFile：~/.agent4ml/skills/.hub/lock.json，跟踪 hub 安装的 skill provenance
- SkillsGuard：install 前安全扫描（prompt injection + 敏感路径 + 可疑 URL）
- AuditLog：~/.agent4ml/skills/.hub/audit.log，记录每次 install/uninstall/update
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HubLockEntry:
    """hub 安装的 skill provenance 记录。"""

    name: str
    source: str                              # "github" / "well-known" / "claude-marketplace"
    identifier: str                          # 安装时用的 identifier
    install_path: str                        # ~/.agent4ml/skills/<name>/
    installed_at: str                        # ISO timestamp
    content_hash: str                        # SKILL.md sha256
    upstream_url: str | None = None          # 用于 update 检查


@dataclass(frozen=True)
class ScanResult:
    """SkillsGuard 扫描结果。

    allowed=True 时可安装；False 时进 quarantine。
    reasons 含检测到的风险描述。
    quarantine_path 是可疑 skill 的隔离路径（allowed=False 时填）。
    """

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    quarantine_path: str | None = None


class HubLockFile:
    """~/.agent4ml/skills/.hub/lock.json — 跟踪 hub 安装的 skill provenance。

    记录每个 hub skill 的 source/identifier/install_path/installed_at/content_hash/upstream_url。
    """

    def __init__(self, lock_path: Path | None = None) -> None:
        self._lock_path = lock_path or (
            Path.home() / ".agent4ml" / "skills" / ".hub" / "lock.json"
        )

    def add(self, entry: HubLockEntry) -> None:
        """添加 entry（同名覆盖）。"""
        entries = self._load()
        entries = [e for e in entries if e.get("name") != entry.name]
        entries.append(self._entry_to_dict(entry))
        self._save(entries)

    def remove(self, name: str) -> None:
        """删除 entry（按 name）。"""
        entries = self._load()
        entries = [e for e in entries if e.get("name") != name]
        self._save(entries)

    def list_installed(self) -> list[HubLockEntry]:
        """列出所有 hub 安装的 skill。"""
        return [self._dict_to_entry(d) for d in self._load()]

    def get(self, name: str) -> HubLockEntry | None:
        """按 name 查 entry。"""
        for d in self._load():
            if d.get("name") == name:
                return self._dict_to_entry(d)
        return None

    def _load(self) -> list[dict[str, Any]]:
        """从 lock.json 加载 entries。"""
        if not self._lock_path.exists():
            return []
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        """保存 entries 到 lock.json。"""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _entry_to_dict(self, entry: HubLockEntry) -> dict[str, Any]:
        return {
            "name": entry.name,
            "source": entry.source,
            "identifier": entry.identifier,
            "install_path": entry.install_path,
            "installed_at": entry.installed_at,
            "content_hash": entry.content_hash,
            "upstream_url": entry.upstream_url,
        }

    def _dict_to_entry(self, d: dict[str, Any]) -> HubLockEntry:
        return HubLockEntry(
            name=d.get("name", ""),
            source=d.get("source", ""),
            identifier=d.get("identifier", ""),
            install_path=d.get("install_path", ""),
            installed_at=d.get("installed_at", ""),
            content_hash=d.get("content_hash", ""),
            upstream_url=d.get("upstream_url"),
        )


# SkillsGuard 安全扫描模式
_SENSITIVE_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"mkfs\.", re.IGNORECASE),
    re.compile(r"dd\s+of=/dev/", re.IGNORECASE),
    re.compile(r">\s*/etc/passwd", re.IGNORECASE),
    re.compile(r"chmod\s+777\s+/", re.IGNORECASE),
]
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"curl\s+[^|]+\s*\|\s*(?:bash|sh)", re.IGNORECASE),
    re.compile(r"wget\s+[^|]+\s*\|\s*(?:bash|sh)", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"exec\s*\(", re.IGNORECASE),
]
_SUSPICIOUS_URL_PATTERNS = [
    re.compile(r"https?://[^\s]+\.(?:exe|bat|cmd|ps1|sh)(?:\s|$)", re.IGNORECASE),
]

# 信任的 repo（hermes 模式，TRUSTED_REPOS 自动放行）
TRUSTED_REPOS = {
    "earendil-works/pi-mono",
    "bytedance/deer-flow",
    "vercel-labs/agent-skills",
    "openai/codex",
    "anthropic/claude-code",
}


class SkillsGuard:
    """install 前安全扫描（借鉴 hermes skills_guard）。

    检查 SKILL.md 内容：
    - prompt injection 模式（curl | bash / eval / exec 等）
    - 敏感路径访问（rm -rf / / mkfs / dd of=/dev/ 等）
    - 可疑外部 URL（.exe / .bat / .sh 下载）
    可疑内容进 quarantine 隔离，拒 install。
    """

    def __init__(self, quarantine_dir: Path | None = None) -> None:
        self._quarantine_dir = quarantine_dir or (
            Path.home() / ".agent4ml" / "skills" / ".hub" / "quarantine"
        )

    def scan(
        self,
        skill_dir: Path,
        source: str = "",
        identifier: str = "",
    ) -> ScanResult:
        """扫描 skill 目录的 SKILL.md 内容。

        返 ScanResult：allowed=True 可安装；False 进 quarantine。
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return ScanResult(allowed=True, reasons=["no SKILL.md found, skip scan"])

        content = skill_md.read_text(encoding="utf-8")
        reasons: list[str] = []

        # 检查信任 repo（自动放行）
        if self._is_trusted(source, identifier):
            return ScanResult(allowed=True, reasons=["trusted repo"])

        # 检查敏感模式
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(content):
                reasons.append(f"sensitive pattern: {pattern.pattern}")

        # 检查 prompt injection
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(content):
                reasons.append(f"prompt injection: {pattern.pattern}")

        # 检查可疑 URL
        for pattern in _SUSPICIOUS_URL_PATTERNS:
            if pattern.search(content):
                reasons.append(f"suspicious URL: {pattern.pattern}")

        if reasons:
            # 进 quarantine
            quarantine_path = self._quarantine(skill_dir)
            return ScanResult(
                allowed=False,
                reasons=reasons,
                quarantine_path=str(quarantine_path),
            )

        return ScanResult(allowed=True)

    def _is_trusted(self, source: str, identifier: str) -> bool:
        """检查是否信任 repo（TRUSTED_REPOS）。"""
        # 从 identifier 提取 owner/repo（github:owner/repo@... → owner/repo）
        if "github:" in identifier:
            repo_part = identifier.replace("github:", "").split("@")[0]
            if repo_part in TRUSTED_REPOS:
                return True
        return source == "builtin"  # builtin 总是信任

    def _quarantine(self, skill_dir: Path) -> Path:
        """移动可疑 skill 到 quarantine 目录。"""
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        dest = self._quarantine_dir / skill_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(skill_dir), str(dest))
        return dest


class AuditLog:
    """~/.agent4ml/skills/.hub/audit.log — 记录每次 install/uninstall/update。

    每行一条 JSON 记录：action/skill_name/source/timestamp/extra。
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or (
            Path.home() / ".agent4ml" / "skills" / ".hub" / "audit.log"
        )

    def append(
        self,
        action: str,
        skill_name: str,
        source: str = "",
        **extra: Any,
    ) -> None:
        """追加一条审计记录。"""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "action": action,
            "skill_name": skill_name,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        """读取所有审计记录。"""
        if not self._log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
