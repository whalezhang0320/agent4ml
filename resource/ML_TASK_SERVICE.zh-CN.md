# Agent4ML ML 长任务服务

该服务把本地 `LocalExperimentTracker` 包装成 FastAPI 控制面，使用 Redis
Streams 保存队列和可重放事件。API 和 Worker 必须分别运行；API 不直接执行训练。

## 启动

```bash
docker compose up -d redis
uv sync --extra dev
uv run agent4ml-ml-api
```

在另一个终端启动 Worker：

```bash
uv run agent4ml-ml-worker
```

服务默认监听 `127.0.0.1:8000`。当前接口面向可信的本机调用方，没有内置用户鉴权；
不得直接暴露到公网。`AGENT4ML_ML_ALLOWED_ROOT` 限制任务工作目录，但提交的命令仍会以
Worker 用户权限执行。

## 提交和查看任务

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "smoke-test",
    "cwd": "/absolute/path/under/allowed-root",
    "command": ["python", "train.py", "--config", "config.yaml"]
  }'
```

响应为 HTTP 202，并包含 `task_id`。查询状态：

```bash
curl http://127.0.0.1:8000/v1/tasks/<task_id>
```

订阅 SSE：

```bash
curl -N http://127.0.0.1:8000/v1/tasks/<task_id>/events
```

断线重放可发送 `Last-Event-ID`，或使用 `?after=<event-id>`。事件包括：

- `task.created`、`task.queued`、`task.started`；
- `task.log`、`metric.reported`；
- `failure.detected`；
- `task.cancellation_requested`、`task.termination_started`、`task.cancelled`；
- `task.succeeded`、`task.failed`。

取消任务：

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks/<task_id>/cancel
```

Worker 会先终止整个任务进程组，等待宽限期后再强制终止，避免只杀父进程而遗留
GPU 子进程。排队中的任务会立即进入 `cancelled`。

## 论文复现工作流

`POST /v1/reproductions` 创建固定版本的四节点工作流：

```text
analyze_code → build_environment → [人工审批] → run_training → validate_result
```

示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/reproductions \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "reproduce-paper",
    "repository_path": "/absolute/path/under/allowed-root/repository",
    "paper_path": "/absolute/path/under/allowed-root/paper.pdf",
    "target_metric": "accuracy",
    "resource_limits": {"gpu_count": 1},
    "training_command": ["python", "train.py", "--config", "config.yaml"]
  }'
```

训练节点执行前，任务进入 `waiting_approval`。从查询响应读取当前
`approval.approval_id` 和 `version` 后批准：

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks/<task_id>/approve \
  -H 'Content-Type: application/json' \
  -d '{"approval_id":"<approval-id>","expected_version":5}'
```

也可调用 `/reject` 拒绝，或对仍有尝试次数的失败节点调用 `/retry`：

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks/<task_id>/retry \
  -H 'Content-Type: application/json' \
  -d '{"expected_version":8}'
```

审批和重试均使用乐观锁版本；配置已变化时返回 HTTP 409。工作流队列消息包含
`node_id`、`attempt` 和 `expected_version`，所以重复或过期消息只会被 ACK，不会重复
执行副作用。节点日志和产物按 `nodes/<node-id>/attempt-<n>/` 保存。

## 故障 Eval

```bash
uv run agent4ml-ml-eval
uv run agent4ml-ml-eval --json
```

当前确定性 Eval 覆盖 PyTorch/CUDA OOM、cuDNN allocation failure、pip resolver
依赖冲突、网络超时、配置缺失、语法错误和普通失败误报控制。网络类暂时故障可只重试
当前节点；OOM 和依赖冲突进入人工审批，不会静默修改 batch size 或依赖版本。

## 持久化边界

- Redis：任务状态、队列、取消标记和有限长度事件流；
- `.agent4ml/ml-tasks/<task_id>/`：manifest、原始 stdout、指标和完整实验事件；
- Checkpoint 和大型 artifact 不应写入 Redis。

Redis Consumer Group 的 pending claim 会由存活 Worker 持续刷新。Worker 丢失后，
只读或幂等节点重新入队；已有 `external_job_id` 的训练继续对账；提交状态不确定的训练
进入人工审批，不会未经确认地重复提交。这里保证的是工作流节点级恢复；训练 step/epoch
级恢复仍要求训练脚本自行保存并加载 checkpoint。
