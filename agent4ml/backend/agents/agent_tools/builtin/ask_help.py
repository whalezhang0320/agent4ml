"""ask_help tool — LLM-initiated help request.

The actual interception and graph-pause logic is in HelpRequestMiddleware.
This tool definition exists so the LLM can call it; the middleware
intercepts before execution and returns Command(goto=END).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool("ask_help", return_direct=True)
def ask_help_tool(
    question: str,
    help_type: Literal["missing_info", "approach_choice", "risk_confirmation", "stuck_report"],
    context: str | None = None,
    options: list[str] | None = None,
) -> str:
    """Ask the user for help when you are stuck or need direction.

    Use this tool when you cannot proceed without user input:
    - **missing_info**: A required detail was not provided (e.g., file path, URL, credential).
    - **approach_choice**: Multiple valid approaches exist and you need user preference.
    - **risk_confirmation**: You are about to perform a potentially risky operation.
    - **stuck_report**: You have tried multiple approaches and all failed; you need guidance.

    After calling this tool, execution will be paused automatically and the
    question will be presented to the user. Wait for the user's response
    before continuing.

    Args:
        question: The help question to ask the user. Be specific and clear.
        help_type: The type of help needed.
        context: Optional background explaining why help is needed.
        options: Optional list of choices for the user to pick from.
    """
    return "Help request processed by HelpRequestMiddleware"
