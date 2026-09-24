"""DuckDuckGo web search tool（ddgs 库直调，无需 API key，无 Node 中间层）。

借鉴 deer-flow 的 ddg_search 模块。优先于 freeweb-mcp 的 web_search 使用，降低失败率。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
DEFAULT_REGION = "wt-wt"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_BACKEND = "duckduckgo"


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the web for information. Use this tool to find current information, news, articles, and facts from the internet.

    Args:
        query: Search keywords describing what you want to find. Be specific for better results.
        max_results: Maximum number of results to return. Default is 5.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return json.dumps({"error": "ddgs library not installed. Run: pip install ddgs", "query": query}, ensure_ascii=False)

    ddgs = DDGS(timeout=30)
    try:
        results = ddgs.text(
            query,
            region=DEFAULT_REGION,
            safesearch=DEFAULT_SAFESEARCH,
            max_results=max_results,
            backend=DEFAULT_BACKEND,
        )
        results = list(results) if results else []
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return json.dumps({"error": f"Search failed: {e}", "query": query}, ensure_ascii=False)

    if not results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    normalized = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "content": r.get("body", r.get("snippet", "")),
        }
        for r in results
    ]
    return json.dumps({"query": query, "total_results": len(normalized), "results": normalized}, indent=2, ensure_ascii=False)
