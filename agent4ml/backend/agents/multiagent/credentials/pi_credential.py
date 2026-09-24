"""PiCredentialProvider — 双轨凭证解析（决策 3：config 优先 + env 兜底）。

设计（spec.md PiCredentialProvider Requirement + design_docs/46 §10.3.4）:
- 双轨凭证解析：config 优先 + env 兜底
- 国内 provider 优先（DeepSeek/Kimi/MiniMax/Xiaomi 靠前，便宜优先）
- pi CLI 可执行 + 任意 API key 即可用（不强制 auth.json 文件存在）
- 凭证缺失返 None（specialist 标 disabled）
- 凭证不写 ThreadState（INV#8）

凭证解析优先级（决策 3）：
1. config 显式 api_key（config_provider + config_api_key）
2. config 显式 provider → 找对应 env var
3. 遍历所有 provider env var（国内优先顺序）
4. auth.json 文件（~/.pi/agent/auth.json）
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent4ml.backend.agents.multiagent.credential_provider import Credential


@dataclass(frozen=True)
class PiCredential(Credential):
    """Pi agent 凭证（kind="pi"）。

    provider: 用户偏好的 provider（anthropic/deepseek/kimi/...）
    api_key: 直接 API key（可选，config 显式时填）
    auth_file: ~/.pi/agent/auth.json 路径（env 兜底时填）
    """

    kind: str = "pi"
    provider: str | None = None
    api_key: str | None = None
    auth_file: str | None = None


class PiCredentialProvider:
    """双轨凭证解析：config 优先 + env 兜底（决策 3）。

    与 CodexCredentialProvider（读 ~/.codex/auth.json）不同：
    - Pi 不强制要求凭证文件存在
    - 任何 API key env var 都能用
    - 只需 pi CLI 可执行（PATH 查找）+ 任意一个 API key 即可

    凭证解析优先级（决策 3）：
    1. config 显式 api_key（最高优先级）
    2. config 显式 provider → 找对应 env var
    3. 遍历所有 provider env var（国内 provider 优先：DeepSeek/Kimi/MiniMax/Xiaomi 靠前）
    4. auth.json 文件（~/.pi/agent/auth.json，支持 PI_CODING_AGENT_DIR 覆盖）
    """

    def __init__(
        self,
        config_provider: str | None = None,
        config_api_key: str | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._config_api_key = config_api_key

    def get_credential(self) -> PiCredential | None:
        # 1. 检测 pi CLI 是否可执行
        if not self._is_pi_installed():
            return None

        # 2. config 显式 api_key（最高优先级）
        if self._config_api_key:
            provider = self._config_provider or "anthropic"  # 默认 anthropic
            return PiCredential(provider=provider, api_key=self._config_api_key)

        # 3. config 显式 provider → 找对应 env var
        if self._config_provider:
            env_var = self._provider_env_map().get(self._config_provider)
            if env_var:
                key = os.getenv(env_var)
                if key:
                    return PiCredential(
                        provider=self._config_provider, api_key=key
                    )

        # 4. 遍历所有 provider env var（国内 provider 优先顺序）
        for provider, env_var in self._provider_env_map().items():
            key = os.getenv(env_var)
            if key:
                return PiCredential(provider=provider, api_key=key)

        # 5. auth.json 文件（~/.pi/agent/auth.json）
        auth_path = self._resolve_auth_path()
        if auth_path and auth_path.exists():
            return PiCredential(auth_file=str(auth_path))

        return None

    def _is_pi_installed(self) -> bool:
        """检测 pi CLI 是否在 PATH。"""
        return shutil.which("pi") is not None

    def _resolve_auth_path(self) -> Path | None:
        """~/.pi/agent/auth.json 路径（支持 PI_CODING_AGENT_DIR 覆盖）。"""
        configured = os.getenv("PI_CODING_AGENT_DIR")
        if configured:
            return Path(configured) / "auth.json"
        return Path.home() / ".pi" / "agent" / "auth.json"

    def _provider_env_map(self) -> dict[str, str]:
        """Pi 支持的 provider → env var 映射（国内优先顺序，决策 3）。

        顺序即优先级：国内常用 provider 靠前（便宜优先），国外靠后。
        """
        return {
            # 国内 provider（便宜优先）
            "deepseek": "DEEPSEEK_API_KEY",
            "kimi-coding": "KIMI_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "xiaomi": "XIAOMI_API_KEY",
            "zai": "ZAI_API_KEY",
            # 国外大厂
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GEMINI_API_KEY",
            # 聚合/其他
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "xai": "XAI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
