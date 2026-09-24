"""GitHubSource — git clone owner/repo 安装 skill。

设计（design_docs/46 §2.4）:
- git clone --depth 1 owner/repo 到 ~/.agent4ml/skills/<name>/
- 支持 GITHUB_TOKEN env var 提升限速
- cache index 1 小时（hermes 模式）
- identifier 格式：github:owner/repo@skill-name 或 github:owner/repo
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from agent4ml.backend.agents.skill.hub.source import SkillMeta


# cache：repo URL → (timestamp, SkillMeta list)
_search_cache: dict[str, tuple[float, list[SkillMeta]]] = {}
_CACHE_TTL = 3600  # 1 hour


class GitHubSource:
    """从 GitHub repo 安装 skill 的 source（git clone）。

    支持 GITHUB_TOKEN env var 提升限速。
    cache index 1 小时（hermes 模式）。
    """

    name = "github"

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """搜索 GitHub skill（MVP：返空，需 GitHub API 集成）。

        MVP 不实现 GitHub API 搜索（需 GitHub Search API + token）。
        用户通过 identifier 直接 install：github:owner/repo@skill-name。
        进阶：实现 GitHub Search API 搜 SKILL.md 文件。
        """
        # MVP：不实现搜索，返空（用户用 identifier 直接 install）
        return []

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        """git clone owner/repo 到 dest_dir，返 skill 目录路径。

        identifier 格式：github:owner/repo@skill-name 或 github:owner/repo
        """
        owner, repo, skill_name = self._parse_identifier(identifier)

        clone_url = self._build_clone_url(owner, repo)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # git clone --depth 1
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(dest_dir)],
            capture_output=True,
            text=True,
            check=True,
        )

        if skill_name:
            skill_path = dest_dir / skill_name
            if skill_path.exists():
                return skill_path
        return dest_dir

    def preview(self, identifier: str) -> str | None:
        """预览 GitHub SKILL.md（MVP：返 None，需 GitHub Contents API）。

        MVP 不实现预览（需 GitHub Contents API + token）。
        进阶：HTTP GET GitHub raw content。
        """
        return None

    def _parse_identifier(self, identifier: str) -> tuple[str, str, str | None]:
        """解析 identifier：github:owner/repo@skill-name → (owner, repo, skill_name)。

        支持格式：
        - github:owner/repo
        - github:owner/repo@skill-name
        - owner/repo（无 github: 前缀）
        - owner/repo@skill-name
        """
        # 去 github: 前缀
        if identifier.startswith("github:"):
            identifier = identifier[7:]

        # 分离 @skill-name
        skill_name: str | None = None
        if "@" in identifier:
            identifier, skill_name = identifier.rsplit("@", 1)

        # 分离 owner/repo
        parts = identifier.split("/")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid github identifier: {identifier}. "
                "Expected: github:owner/repo@skill-name or owner/repo"
            )
        return parts[0], parts[1], skill_name

    def _build_clone_url(self, owner: str, repo: str) -> str:
        """构造 git clone URL（支持 GITHUB_TOKEN 提升限速）。"""
        token = os.getenv("GITHUB_TOKEN")
        if token:
            return f"https://{token}@github.com/{owner}/{repo}.git"
        return f"https://github.com/{owner}/{repo}.git"

    def _get_cached(self, repo_url: str) -> list[SkillMeta] | None:
        """从 cache 读（TTL 1h）。"""
        if repo_url in _search_cache:
            ts, results = _search_cache[repo_url]
            if time.time() - ts < _CACHE_TTL:
                return results
        return None

    def _set_cached(self, repo_url: str, results: list[SkillMeta]) -> None:
        """写 cache。"""
        _search_cache[repo_url] = (time.time(), results)
