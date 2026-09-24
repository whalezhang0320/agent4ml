"""测试用 fake chat model —— 支持 bind_tools（FakeListChatModel 未实现，导致集成测试失败）。"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel


class FakeChatModelWithTools(FakeListChatModel):
    """FakeListChatModel + bind_tools 支持。

    bind_tools 返回 self（工具 schema 被接受但不影响响应），让 create_agent 编译通过。
    用于集成测试，避免 NotImplementedError。
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self
