"""Memory 模块 — 长期记忆系统。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` 奠基（D1-D16 ADR
+ 三层解耦架构）+ `01-agent4ml-integration-points.md` 介入清单 + `48-memory-l1-base-layer.md`
Layer 1 完整设计 + `49-memory-l2-default-strategies.md` Layer 2 完整设计
+ `50-memory-l3-store-retriever.md` Layer 3 完整设计。

三层解耦架构（north-star）：
- Layer 1（已落地）：基础可用性层（Schema + 9 Protocol + 状态接入 + Registry slot）
- Layer 2（已落地）：记忆管理层（默认策略实现：EbbinghausDecayPolicy / CompositeForgetPolicy / DefaultMemoryManager）
- Layer 3（已落地）：存储检索层（MarkdownFileStore + HybridRetriever）
- Layer 4（本模块当前状态）：中间件接入与 bootstrap 装配（MemoryMiddleware + bootstrap lifecycle）
- Layer 4：集成层（MemoryMiddleware + bootstrap + 5 处 make_lead_agent 透传）
- Layer 5：演化层（Phase 2 LLM 决策 cron）
- Layer 6：扩展层（adapter 具体实现 + Persona）

依赖方向：`app → agents/memory → (agents/capabilities, agents/state, agents/config)`。
memory 包不反向依赖 `app`；跨层用 Protocol 破循环（boundary §3.1）。

INVARIANT（L1+L2+L3+L4 合并，分 L1/L2/L3/L4 段）：

## L1 不变量（16 条）

1. **MemoryTrace 不可变**：frozen dataclass，strength 等可变字段通过 `with_strength()`
   / `with_operation()` 创建新实例替换（类似 skill version DAG 的 is_active 指针）
2. **四操作无 LLM**：Encode/Associate/Consolidate/Reconsolidate 是纯存储操作，LLM 生成内容
   外部传入（Retrieve 移至 Retriever）
3. **检索是一切基础**：Phase 1 对话与 Phase 2 记忆管理都从 retrieve 开始（00 D4），
   retrieve 统一走 Retriever
4. **lazy decay**：strength 在 retrieve 时按需计算，不跑后台衰减任务（00 §5.5）
5. **Protocol 纯契约**：9 Protocol（7 核心 + 2 adapter）零实现，可 mock 可替换（00 D8）
6. **strategies/default/ 同构**：与 context_engineering/strategies/default/ 同构，策略可插拔
7. **CapabilityRegistry 注册**：memory_provider 作为第 9 个 capability，禁全局单例（00 D11）
8. **CORE_FIELDS 保护**：recalled_memories / memory_updates 加入 CORE_FIELDS，防 metadata 冲突
9. **向后兼容**：memory_provider=None / use="" 时 Agent4ML 行为不变
10. **import 防火墙**：`agents/memory/` 不反向依赖 `app`（00 D12）
11. **runtime 可切**：enable_recall/extract/token_budget/decay/forget/phase2 通过
    `set_memory_config()` 整替，Provider/Middleware/Retriever 不缓存 config，每次从
    `get_memory_config()` 取最新（参照 deer-flow memory_config.py 模式）
12. **STARTUP_ONLY 仅 4 字段**：use/storage_path/vector_store/graph_store 换需重启
    （重建 Provider/adapter + 重建 derived index）；其余 runtime 可切
13. **叠加非互斥**：Markdown truth（总在）+ optional VectorStore + optional GraphStore
    三层叠加，可同时启用（00 §8.2 derived shadow index），vector_store 与 graph_store 非三选一
14. **adapter 是 Retriever 子组件**：VectorStore/GraphStore 非独立 MemoryProvider，由
    HybridRetriever 组合；adapter 加载失败 → no-op 跳过 + log warning，系统不崩
15. **lifecycle duck-type**：MemoryProvider/adapter 构造即就绪，无强制 initialize()；
    shutdown 走 `hasattr(provider, "shutdown")` 检查，MarkdownFileStore 不实现，
    SQLiteShadowStore/VectorStore 实现（参照 deer-flow SandboxProvider 模式）
16. **operation_log traceability**：MemoryTrace.operation_log 记录操作历史
    （encode/associate/consolidate/reconsolidate/forget），上限 20 条 FIFO；retrieve 不记
    （高频，强化在 strength/access_count 已体现）；actor 字段预留 turn_id（Layer 4 注入）；
    支持 debug 回溯"谁何时对哪条 trace 做了什么"

## L2 不变量（20 条，默认策略实现层，承接 49 §8）

1. **四操作无 LLM**：Encode/Associate/Consolidate/Reconsolidate 是纯存储操作，
   merged_content/new_content 外部传入（00 D3）
2. **Ebbinghaus 公式完整**：strength = base_strength×(1-decay_rate)^time_hours
   + log(1+access_count)×0.1 + importance×0.05（00 §5.5）
3. **lazy decay**：compute_strength 不修改 trace，retrieve 时由 Retriever 调用
   + with_strength 更新（00 §5.5）
4. **strength 钳制**：compute_strength 返回值永远在 [0.0, 1.0]
5. **两规则遗忘**（B3）：should_forget 检查 TTL + strength（规则 1+2）；
   resolve_conflict 已删，矛盾解决走 reconsolidate/consolidate（00 §7.4）
6. **矛盾不删除**：resolve_conflict 返回 new，old 标记 forgotten 不删除（保留回滚）（00 §5.3）
7. **frozen 语义**：manager 所有操作返回依赖注入**：DefaultMemoryManager 接收 store + decay_policy + forget_policy + journal
   （策略组合可替换）
9. **runtime 可切**：策略类不缓存 config，每次从 get_memory_config() 取（48 INVARIANT 11）
10. **关联膨胀防护**（D3）：associate 检查 max_associations_per_trace，超限 LRU 淘汰最弱
11. **consolidate 数量校验**（E1）：trace_ids 数量在 [min=2, max=10] 范围，否则抛 ValueError
12. **reconsolidate 保留强度**（B3）：content 替换 + strength 保留原值 + last_accessed=now
    （视为一次访问，不重置 base_strength）+ operation_log 记 content diff（旧前 200 字符不丢）
13. **shutdown duck-type**：DefaultMemoryProvider.shutdown() 委托 store/retriever（hasattr 检查）
14. **encode id = content hash**（F2）：SHA256(content+type.value) 前 16 位，同内容同 type 自动去重
15. **encode 幂等**（点4）：同 id 已存在 → 返回旧 trace，不重复 add，emit `memory.encode.duplicate`
16. **associate LRU 淘汰**（D3）：超 max_associations_per_trace 时淘汰 strength 最低的关联
    （非静默跳过），保持关联质量
17. **forgotten 标记不删除**（C1）：consolidate 后旧 trace 标记 `metadata.forgotten=True`，
    store 不删除（保留回滚），Retriever 过滤留 L3
18. **traceability A**（operation_log）：manager 每操作 append OperationLog
    （encode/associate/consolidate/reconsolidate/forget），上限 20 条 FIFO；retrieve 不记
19. **traceability B**（journal）：manager 每操作 emit `memory.*` 事件
    （journal callback 注入，Layer 4 接 RunJournal）
20. **traceability C**（actor 预留）：OperationLog.actor + journal payload.actor 从 ContextVar
    `_turn_id_var` 取（Layer 4 MemoryMiddleware 注入 turn_id），Layer 2 测试时为 None

## L3 不变量（12 条，存储检索实现层，承接 50 §9）

1. **Markdown-as-Truth**：traces.md 是 truth source，内存索引是 derived（可重建）（00 §8.2）
2. **单文件 + 分隔符**：所有 trace 在 traces.md，用 `<!-- trace: {id} -->` 分隔（方案 B）
3. **frontmatter + content**：每条 trace = YAML frontmatter（所有字段除 content）+ content 正文
4. **无事务**（2A）：add/update/remove 逐个操作，失败 log，接受最终一致（Phase 2 cron 修复）
5. **文件锁**（6B）：`threading.Lock` 保护 update/remove（单进程）；跨进程锁留后续
6. **list_by_filter 粗筛**（7A）：store 只按 max_age_hours/type/metadata 过滤，strength 精算由调用方
7. **retrieve 强化写回**（1A）：HybridRetriever.retrieve 内部调 store.update，caller 不负责
8. **forgotten 过滤在 Retriever**（3B）：store 不感知 forgotten，Retriever 过滤 metadata.forgotten
9. **BM25 增量索引**（5B）：构造全量建 + on_trace_* 增量维护
10. **无 vector/graph**：HybridRetriever 纯 BM25，不依赖 adapters（空壳保留 Layer 6）
11. **storage_path 锚定**（01 D12）：相对路径锚定（L3 cwd fallback，Layer 4 bootstrap 传绝对路径）
12. **解析容错**（2A）：frontmatter 损坏 log + 跳过，不崩（最终一致）

## L4 不变量（11 条，中间件接入与 bootstrap 装配层，承接 53 §8）

1. **记忆是 middleware**：MemoryMiddleware 挂载，不进 leader agent 主体（00 D9）
2. **不进 system prompt cache**：per-call HumanMessage(hide_from_ui=True)，
   recalled_memories 只存索引（00 D10）
3. **挂载顺序**：Sandbox 后，HelpRequest/ToolCall 前（记忆引用 sandbox 结果，不进 tool pairing）
4. **set_turn_id 注入/清除**：before_model 注入，after_model 清除（traceability C）
5. **懒加载双检锁**：get_memory_provider 线程安全
6. **shutdown duck-type**：hasattr(provider, "shutdown") 委托 store/retriever
7. **store 包装 5B**：add/update/remove 后调 retriever.on_trace_*（装饰器，不改 store 类）
8. **5 处透传**：make_lead_agent 5 调用点都透传 memory_provider（漏传该路径记忆丢失）
9. **storage_path 锚定**：_resolve_relative_paths 锚定 _PROJECT_ROOT（01 D12）
10. **import 防火墙**：bootstrap.py 不 import app（00 D12）；MemoryMiddleware 迁到 agents/middlewares/memory_recall_middleware.py（块 B）
11. **向后兼容**：use="" 默认禁用，行为不变；memory_provider=None 不挂载
"""

from __future__ import annotations
