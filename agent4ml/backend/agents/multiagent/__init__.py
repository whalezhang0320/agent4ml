"""Multi-Agent Orchestration 系统 — L1 基础编排层 + L2 智能编排层。

specialist 作为 tool + soft routing 接入 lead agent graph ReAct。
lead agent 编排型：能委派就委派，自己做是 fallback。

详见 design_docs/40-multi-agent-orchestration-overview.md +
41-multi-agent-orchestration-three-layer-foundation.md +
Hezao-MultiAgentDesign-Docs/agent4ml/42-multi-agent-l2-evolution-layer.md。

L1 INVARIANT（10 条）:
1. specialist 黑盒——Agent4ML 不管理 specialist 内部 context，只传 goal + context_summary + sandbox_id
2. specialist 自带 model——Agent4ML 不为 specialist 配置 model，只发现凭证
3. shared thread sandbox——lead + self-copy + specialist 同一 sandbox_id
4. leaf role 递归控制——子 agent tool_groups 不含 multiagent，不能 spawn
5. sync only MVP——不做 async ainvoke
6. pairing 完整性——specialist 失败抛 SpecialistError，由 OrchestrationMiddleware 转 error ToolMessage
7. 消息角色交替——specialist 调用作为 tool result 回流，不插入 synthetic user message
8. 凭证不进 LLM 主态——CredentialProvider 返回 token 只传给 specialist runtime，不写 ThreadState
9. 8 接口抽象——specialist 通过 SpecialistMcpServer 调用 Agent4ML 8 个沙箱接口，经过安全层
10. programmatic eval floor——success_criteria 强制，ResultSummarizer 内校验

L2 INVARIANT（40 条，详见 evolution/__init__.py）:
- 不演化 Router（Router = LLM，L1 D3）
- 演化产物仅 W2 + W4（ContextSummaryTemplate + SkillInjectionTemplate）
- hot swap 不破 cache（per-call 产物，不进 cache prefix）
- L2TriggerMiddleware 不调 LLM 不改 ThreadState
- per-profile 串行（daemon thread 单 worker）
- Wilson 95% CI + hash 防环 5 版
- BudgetGuard 超限 fallback lead
- VersionDAG SQLite + is_active 单指针
- 演化失败保持旧 is_active
- IntentEngine 不进 middleware（ContextSummarizer 输入源）
- L2 CLI 暂不实现
"""
