"""token utility：token_counter + resolve_window_size。

tiktoken 懒加载 + 失败 cooldown 缓存 + CJK-aware char fallback。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_TIKTOKEN_RETRY_COOLDOWN_S = 600
_DEFAULT_WINDOW = 128_000

# model_name → window 映射表。长前缀优先（避免 "gpt-4" 误匹配 "gpt-4o"）。
# langchain ChatModel 不一定暴露 max_input_tokens，靠 model_name 前缀匹配兜底。
_MODEL_WINDOW_MAP: dict[str, int] = {
    # OpenAI（长前缀优先，避免 gpt-4 误匹配 gpt-4o）
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-110": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o1-mini": 128_000,
    "o1": 200_000,
    # Anthropic
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # DeepSeek
    "deepseek-v4-flash": 200_000,
    "deepseek-reasoner": 64_000,
    "deepseek-chat": 64_000,
    # Qwen
    "qwen-max": 32_768,
    "qwen-plus": 131_072,
    "qwen-turbo": 131_072,
    "qwen2.5": 131_072,
    "qwen": 32_768,
    # GLM
    "glm-4-plus": 131_072,
    "glm-4": 131_072,
    # Yi
    "yi-large": 32_768,
    "yi-34b": 4_096,
    # Moonshot
    "moonshot-v1-128k": 131_072,
    "moonshot-v1-32k": 32_768,
    "moonshot-v1-8k": 8_192,
    "moonshot-v1": 8_192,
    # Stepfun
    "step-2": 8_192,
    "step-1": 8_192,
    # MiniMax
    "abab6.5": 245_760,
    # xAI Grok
    "grok-3": 131_072,
    "grok-2": 131_072,
    # Google Gemini
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini": 1_000_000,
    # Anthropic Claude 4+（3.x 已在上方）
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude": 200_000,
    # vLLM / openai-compatible 默认
    "vllm": 128_000,
}

_ENCODING_CACHE: dict[str, Any] = {}
_last_failure_ts: float = 0.0
_loading_sentinel = object()

# CJK Unified Ideographs + Extension A + CJK Symbols
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F))


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _extract_text(message: Any) -> str:
    """从 message（str / BaseMessage / dict）提取纯文本。"""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _char_estimate(messages: list) -> int:
    """CJK-aware char 估算：CJK 字符 ~1 token，非 CJK ~4 字符/token。"""
    cjk = 0
    other = 0
    for msg in messages:
        for ch in _extract_text(msg):
            if _is_cjk(ch):
                cjk += 1
            else:
                other += 1
    return cjk + other // 4


def _get_encoding(model_name: str | None) -> Any | None:
    """tiktoken encoding 懒加载 + 失败 cooldown + 并发 LOADING sentinel。"""
    global _last_failure_ts
    key = model_name or "default"
    if key in _ENCODING_CACHE:
        cached = _ENCODING_CACHE[key]
        return None if cached is _loading_sentinel else cached
    if time.time() - _last_failure_ts < _TIKTOKEN_RETRY_COOLDOWN_S:
        return None
    try:
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.encoding_for_model(key) if model_name else tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning("tiktoken unavailable, fallback to char estimate: %s", exc)
        _last_failure_ts = time.time()
        _ENCODING_CACHE[key] = _loading_sentinel
        return None
    _ENCODING_CACHE[key] = enc
    return enc


def token_counter(messages: list, model_name: str | None = None) -> int:
    """算 messages 总 token。tiktoken 优先，失败 fallback char 估算。"""
    enc = _get_encoding(model_name)
    if enc is None:
        return _char_estimate(messages)
    total = 0
    for msg in messages:
        try:
            total += len(enc.encode(_extract_text(msg)))
        except Exception:
            total += _char_estimate([msg])
    return total


def resolve_window_size(model: Any) -> int:
    """取模型上下文窗口大小。

    解析顺序：
    1. FallbackChatModel 穿透——读 ``.models[active]`` 内层真实模型（ChatDeepSeek
       等），递归解析其 model_name → 命中映射表（deepseek-v4-flash→200k 等）。
       FallbackChatModel 本身的 _identifying_params 只含 providers/active，不含
       窗口信息，必须穿透到内层才能拿到真实容量。
    2. model 属性（max_input_tokens / model_max_tokens / max_tokens）
    3. _identifying_params dict
    4. model_name / model 前缀匹配 _MODEL_WINDOW_MAP
    5. default 128000
    """
    # FallbackChatModel 穿透：取活跃内层模型递归解析
    inner_models = getattr(model, "models", None)
    if isinstance(inner_models, list) and inner_models:
        active = getattr(model, "_active", 0) or 0
        inner = inner_models[active] if active < len(inner_models) else inner_models[0]
        # bind_tools 后的 RunnableBinding 需剥 .bound 拿到原始 BaseChatModel
        bound = getattr(inner, "bound", None)
        if bound is not None:
            inner = bound
        w = resolve_window_size(inner)
        if w != _DEFAULT_WINDOW:
            return w
    for attr in ("max_input_tokens", "model_max_tokens", "max_tokens"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    params = getattr(model, "_identifying_params", None)
    if callable(params):
        try:
            params = params()
        except Exception:
            params = None
    if isinstance(params, dict):
        for k in ("max_input_tokens", "model_max_tokens", "max_tokens"):
            v = params.get(k)
            if isinstance(v, int) and v > 0:
                return v
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    if isinstance(model_name, str):
        for prefix, window in _MODEL_WINDOW_MAP.items():
            if model_name.startswith(prefix):
                return window
    return _DEFAULT_WINDOW


def resolve_model_name(model: Any) -> str | None:
    """取当前活跃 provider 的真实模型名（如 ``deepseek-v4-flash``）。

    与 ``resolve_window_size`` 同一穿透逻辑：FallbackChatModel 本身没有模型名，
    只有 ``.models[active]`` 内层真实 ChatModel 才有 ``model_name``/``model``
    属性——TUI 的 Build 信息行要展示的是这个真实名字，不是 provider 链
    （如 ``openai,qwen,deepseek``）。
    """
    inner_models = getattr(model, "models", None)
    if isinstance(inner_models, list) and inner_models:
        active = getattr(model, "_active", 0) or 0
        inner = inner_models[active] if active < len(inner_models) else inner_models[0]
        bound = getattr(inner, "bound", None)
        if bound is not None:
            inner = bound
        name = resolve_model_name(inner)
        if name:
            return name
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    if isinstance(name, str) and name:
        return name
    params = getattr(model, "_identifying_params", None)
    if callable(params):
        try:
            params = params()
        except Exception:
            params = None
    if isinstance(params, dict):
        v = params.get("model_name") or params.get("model")
        if isinstance(v, str) and v:
            return v
    return None
