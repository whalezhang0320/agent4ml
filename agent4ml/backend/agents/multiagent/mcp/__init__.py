"""Multi-Agent MCP — SpecialistMcpServer 暴露 Agent4ML 沙箱接口给 specialist。

模式 B MCP（design.md §5）：specialist 通过 SpecialistMcpServer 调用 Agent4ML 8 个沙箱接口，
经过安全层（PathTranslator + SecurityGuard），数据共享（shared thread sandbox，INV#3/INV#9）。
"""
