# Agent4ML 本地机器学习科研助理方案

## 可行性结论

本地单机版本可实现，且现有 Agent4ML 已具备适合复用的 Agent 循环、sandbox、skill、
run journal、artifact 和报告能力。第一阶段应聚焦“论文到可审计实验”的闭环，不引入
Redis、Prometheus、远程训练平台或跨服务器调度。

能力边界：

| 能力 | 本地阶段结论 | 说明 |
|---|---|---|
| 有代码论文复现 | 可实现 | 审计仓库、隔离环境、smoke test、兼容性补丁、正式运行 |
| 无代码论文复现 | 可实现但不保证结果一致 | 需显式记录论文未说明的实现假设 |
| 训练日志持续记录 | 可实现 | 后台 worker 持续写本地日志和 JSONL 指标 |
| Agent 观察训练进度 | 可实现（轮询快照） | 当前 TUI 不是逐行实时推流 |
| 结果对比分析 | 可实现 | 需要相同数据、指标、评测和 checkpoint 选择协议 |
| 任意论文全自动成功 | 不可承诺 | 受数据许可、缺失细节、算力和不可复现性影响 |
| 跨机器/分布式训练 | 暂不纳入 | 后续再增加调度、心跳、资源租约和远程 artifact 存储 |

## 本地执行闭环

```text
论文/PDF/代码
    ↓
复现计划与资源盘点
    ↓
隔离环境 + 最小实现或兼容性补丁
    ↓
smoke test → 小规模代理实验 → 忠实实验
    ↓
后台训练 + stdout/events/metrics/checkpoints
    ↓
协议对齐 + 多 seed 统计 + 复现结论
```

三个 research skill 分工如下：

- `ml-paper-reproduction`：负责论文证据抽取、代码审计、环境构建、缺失实现和执行策略；
- `ml-experiment-tracking`：负责非阻塞启动、原始日志、结构化指标和异常证据；
- `ml-result-analysis`：负责协议对齐、统计比较、曲线、失败归因和结论校准。

通用的 skill eval 只评价“skill 是否帮助 Agent 完成任务”，不能代替科研结果分析。
科研结果必须由 `ml-result-analysis` 基于实验产物完成。

## 数据与目录协议

每篇论文使用独立目录，任何 headline 数字都必须能追溯到 run id：

```text
experiments/<paper-slug>/
  reproduction-plan.md
  environment/
  source/
  configs/
  runs/<run-id>/
    manifest.json
    events.jsonl
    metrics.jsonl
    stdout.log
    checkpoints/
    artifacts/
  analysis/result-analysis.md
  reproduction-report.md
```

训练代码可输出以下单行记录，tracker 会在保留原始日志的同时写入 `metrics.jsonl`：

```text
AGENT4ML_METRIC {"step": 100, "split": "train", "metrics": {"loss": 0.42}}
```

## 与旧方案的取舍

保留：分阶段 Agent 工作流、环境构建、训练执行、异常诊断、结果验证、失败恢复和人工审批。

调整：

- MCP 适合接外部工具，不应成为本地训练的必经层；本地进程直接由 sandbox 执行更简单；
- Docker 是可选隔离层，不应阻塞 macOS/CPU 或已有 Conda/venv 环境的首版复现；
- Redis + SSE 适合多用户服务端任务，单机 TUI 首版用后台 worker + 文件状态即可；
- Prometheus 适合系统监控，科研指标应先用可携带的 JSONL/TensorBoard/CSV；
- NCCL 超时属于分布式阶段；本地首版重点处理 OOM、NaN、data loader、磁盘和 checkpoint。

## 后续优先级

1. 在 TUI 增加实验状态面板和自动刷新，而不是把整份训练日志塞进对话；
2. 增加 TensorBoard、CSV、Hugging Face Trainer 等常见日志适配器；
3. 增加 GPU/磁盘资源预算、断点续训和进程取消工具；
4. 建立小型基准集，覆盖“有代码、无代码、OOM、指标不一致、训练失败”场景；
5. 单机闭环稳定后，再设计跨服务器调度和远程 artifact 存储。
