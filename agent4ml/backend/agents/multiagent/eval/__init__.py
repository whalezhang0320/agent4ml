"""L3 eval 编排评估层 — 可插拔评估方法库 + 健康监控 + 跨 run 学习.

设计（43 文档 §1 + spec.md multiagent-l3-eval-layer）:
- L3 不演化（EvalAdapter 是可插拔库，新增/替换由人工 + L2 触发，避免无限递归到 L4）
- L2 调 L3 唯一入口：OrchestrationBridge.evaluate(ctx) → EvalResult（fail-closed）
- sync only（与 L1 D10 一致）
- L3 不进 L1 graph（与 L2 一致）
- 复用 L2 daemon thread pattern（独立 thread，不嵌套）
- 复用 L2 MetricsView Protocol（扩展加 get_specialist_l2_events）
- 复用 L2 自建 EvalTask / EvalResult（import from evolution/promotion_gate.py）
- 复用 L2 _wilson_ci（import from evolution/promotion_gate.py）
- L3 metrics 复用 l2_metrics 表（event_type 加 l3_ 前缀）
- L3 默认 enabled=false（数据驱动触发后才实现）

L3 INVARIANT 21 条（43 文档 §12，不可违反）:

OrchestrationBridge INVARIANT（L3-INV-1 到 L3-INV-7，Tier 2）:
1. OrchestrationBridge 实现 L3 自建 EvalBridge Protocol（不共享 skill EvalBridge——skill 用 SkillRecord 不能跨模块共享）
2. L2 调 L3 唯一入口是 OrchestrationBridge.evaluate(ctx) → EvalResult
3. L3 evaluate 是同步阻塞调用（与 L1 D10 sync only 一致）
4. L3 evaluate 失败返 EvalResult(success=False)，不抛异常（fail-closed）
5. L2 收 success=False → reject candidate + 保持旧 is_active
6. L3 自建 EvalBridge Protocol（不共享 skill EvalBridge——skill 用 SkillRecord 专属类型不能跨模块共享）
7. task_sample 由 L2 抽样传入，L3 不重复实现 task 池管理

EvalAdapter INVARIANT（L3-INV-8 到 L3-INV-9，Tier 3）:
8. L3 不演化（EvalAdapter 是可插拔库，新增/替换由人工 + L2 触发，避免无限递归到 L4）
9. L3 3 adapter（programmatic / llm_judge / longitudinal_pairs），选择由 Bridge 自动

HealthMonitor + RuntimeTracker INVARIANT（L3-INV-10 到 L3-INV-11，Tier 4+5）:
10. SpecialistRuntimeTracker 自建（pattern 复用 skill RuntimeTracker 趋势判定算法，不实现 skill Protocol——返回类型 SkillHealthReport 字段不兼容）
11. degraded_specialists 命中 → enqueue L2 daemon thread queue（不直接调 L2 TriggerManager，解耦）

架构 INVARIANT（L3-INV-12 到 L3-INV-15，Tier 8）:
12. L3 不进 L1 graph
13. L3 复用 L2 daemon thread pattern（独立 thread，不嵌套）
14. L3 读 L1 metrics 复用 L2 MetricsView Protocol（扩展加 get_specialist_l2_events 方法）
15. L3 启用时 L2 改调 OrchestrationBridge，L1 ResultSummarizer 不动（向后兼容）

DecisionLog INVARIANT（L3-INV-16 到 L3-INV-18，Tier 7）:
16. decision log 异步写（fire-and-forget，不阻塞 L1 turn）
17. decision log 不直接注入 prompt，作为 EvolutionMutator 输入样本（类似 L2 R2.3 failure cases）
18. decision log 保留 90 天 + 归档（不删除，移到 archive 表）

可观测性 + 实现 INVARIANT（L3-INV-19 到 L3-INV-21，Tier 9）:
19. L3 metrics 复用 l2_metrics 表（event_type 加 l3_ 前缀，不新建 metrics 表）
20. L3 CLI 设计保留暂不实现（同 L2 R7.2，命令方式交互复杂不便观测，等待更好可观测形态）
21. L3 默认 enabled=false，数据触发后才实现（L2 演化产物 > 5 版 + floor eval 不足 / 多 specialist 对比需求 / specialist completion_rate < 0.4 持续 / 跨 session lessons 累积需求 / 用户主动需求）
"""
