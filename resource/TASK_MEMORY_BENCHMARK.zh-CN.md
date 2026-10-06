# Task Memory A/B 基准测试

这个基准使用完全固定的工具输出，对以下三组进行可复现比较：

- `raw`：不做上下文治理。
- `legacy`：使用原有工具结果外化和摘要，不启用任务图。
- `task_memory`：启用 `refs + offload.jsonl + task_graph.json`。

## 快速运行

```bash
.venv/bin/python -m agent4ml.backend.benchmarks.task_memory_ab --repeats 3
```

也可以在重新安装项目后使用：

```bash
agent4ml-task-memory-bench --repeats 3
```

只跑一个案例：

```bash
.venv/bin/python -m agent4ml.backend.benchmarks.task_memory_ab \
  --scenario needle_recovery \
  --repeats 5
```

默认使用 `accelerated` profile，将 P1/P4 阈值调低，以较低成本验证行为。保持生产阈值运行：

```bash
.venv/bin/python -m agent4ml.backend.benchmarks.task_memory_ab \
  --profile production \
  --window 128000 \
  --repeats 5
```

## 输出

默认输出到 `.agent4ml/benchmarks/task-memory-ab-<timestamp>/`：

- `runs.jsonl`：每次运行的原始指标。
- `summary.json`：按案例和变体聚合后的中位数及差值。
- `summary.md`：适合直接阅读的对比表。
- `runs/`：每次 replay 产生的外化文件、快照和任务记忆文件。

## 指标解释

- `cumulative_input_tokens`：每次模拟模型调用时上下文 token 的累计值，最接近输入 token 成本。
- `peak_context_tokens`：单次模型调用的最大上下文。
- `governance_ms`：本地外化、fsync、WAL、Graph 和投影所花时间。
- `recoverable_fact_rate`：预埋在工具输出尾部的精确事实，能否从当前上下文或证据文件恢复。
- `constraint_retention_rate`：早期用户约束在最终上下文中的保留比例。
- `traceability_rate`：工具结果是否仍有可用的消息路径或 Graph → Event → Ref 链路。

`replay_wall_ms` 不包含真实 LLM 或网络调用，因此不能直接解释为线上响应速度。真实端到端提速需要在下一层 live benchmark 中记录每次模型调用的 usage 和 latency。
