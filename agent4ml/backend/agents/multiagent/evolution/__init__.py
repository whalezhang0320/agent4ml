"""Multi-Agent L2 Evolution Layer - evolve ContextSummaryTemplate / SkillInjectionTemplate.

承接 design_docs/41-multi-agent-orchestration-three-layer-foundation.md S13 +
Hezao-MultiAgentDesign-Docs/agent4ml/42-multi-agent-l2-evolution-layer.md.

核心立场（解读 B）：L2 不演化 Router（L1 D3 已定 Router = LLM），
L2 只演化 LLM 能看到但不进 system prompt cache prefix 的 per-call 产物
（ContextSummaryTemplate + SkillInjectionTemplate），hot swap 不破 cache。

L2 INVARIANT（40 条，42 文档 S8）:

核心架构（INV-1 ~ INV-10）:
1. L2 不直接读 OrchestrationStore，只通过 MetricsView Protocol
2. L2 不演化 Router（Router 不存在，解读 B）
3. L2 演化产物仅限 ContextSummaryTemplate + SkillInjectionTemplate（W2 + W4）
4. L2TriggerMiddleware 不调 LLM，不修改 ThreadState
5. L2EvolutionWorker per-profile 串行，不并发演化
6. 演化产物 hot swap 不破坏 prompt caching（因不进 cache prefix）
7. PromotionGate 95% CI 决策 + hash 防环，拒绝单次偶然
8. L2 不引入自动 retry（D-5=a，retry 仍由 LLM 决策）
9. L2 演化产物形态为结构化 dataclass（D-F=b），非字符串模板
10. BudgetGuard 超限 fallback 到 lead 自己做，不 fallback 另一 specialist

持久化与读取（INV-11 ~ INV-13）:
11. VersionDAG 持久化用 SQLite（multiagent.db 加表，Z3 模式）
12. L1 每次 specialist 调用查 DB 取 is_active，不缓存（保 hot swap）
13. 演化失败时 is_active 不变（演化失败 = 不演化）

EvolutionMutator（INV-14 ~ INV-18）:
14. EvolutionMutator 单次 LLM 调用，最多重试 2 次（含首次共 3 次）
15. 演化失败保持旧 is_active，不阻塞后续 L2 任务
16. 连续 3 次演化失败 -> 标记 failure pattern "evolution_blocked"，需人工 inspect
17. 演化输入样本数 <= 5，按 failure_category 聚类取 top
18. EvolutionMutator 默认用 lead 同 model，可配置覆盖

PromotionGate eval（INV-19 ~ INV-24）:
19. PromotionGate eval 样本数 10-15，混合 80% 失败 task + 20% 成功 task
20. CI 用 Wilson score interval（z=1.96），不用正态近似
21. 单个 task 累计被 eval 用 <= 3 次，超过从 pool 淘汰
22. eval 整体超时 30 min -> 中断 + 保持旧 is_active
23. 连续 3 次 eval 失败 -> 标 "eval_blocked"，需人工 inspect
24. candidate 95% CI 下界 > baseline 95% CI 上界 -> accept；否则 reject

触发与节流（INV-25 ~ INV-29）:
25. L2 cron 周期默认 6h，冷却默认 1h，均可配置
26. 失败聚焦触发窗口 24h + 阈值 5 次；specialist 降级阈值 invoked>=5 + rate<0.4
27. anti-loop hash 窗口默认 5 版
28. L2EvolutionWorker 走 daemon thread 单 worker 串行，不加额外锁
29. evolution_blocked / eval_blocked 24h 自动解除 + CLI 手动解除

BudgetGuard（INV-30 ~ INV-33）:
30. BudgetGuard 三维度记账（token / cost_usd / 调用次数），cost_usd 为主触发维度
31. budget per-day UTC 0 点重置，跨 session 持久化于 multiagent.db
32. budget 超限 fallback 到 lead，通过 tool 返回 JSON 通知 LLM（不污染 system prompt）
33. budget 80% 预警写 metrics，不主动注入 system prompt

IntentEngine（INV-34 ~ INV-37）:
34. IntentEngineStrengthened 不作为 before_model middleware，不注入 system prompt
35. candidate metadata 通过 ContextSummarizer 渲染进 context_summary（per-call 产物）
36. IntentTree 始终启用 + 零 LLM 成本；LLM 兜底仅数据触发后启用
37. LLM 兜底用 lead 同 model，失败 fallback 到 IntentTree

可观测性（INV-38 ~ INV-40）:
38. L2 metrics 写 multiagent.db l2_metrics 表，不进 ThreadState
39. L2 告警不主动 push，用户通过 CLI 主动 inspect
40. L2 CLI 命令树 agent4ml multiagent l2 <verb>，设计保留暂不实现
"""
