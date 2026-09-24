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

## 故障 Eval

```bash
uv run agent4ml-ml-eval
uv run agent4ml-ml-eval --json
```

当前确定性 Eval 覆盖 PyTorch/CUDA OOM、cuDNN allocation failure、pip resolver
依赖冲突、`pip check` 依赖冲突和普通失败误报控制。检测结果写入任务状态和
`failure.detected` 事件，但不会自动修改 batch size 或依赖版本。

## 持久化边界

- Redis：任务状态、队列、取消标记和有限长度事件流；
- `.agent4ml/ml-tasks/<task_id>/`：manifest、原始 stdout、指标和完整实验事件；
- Checkpoint 和大型 artifact 不应写入 Redis。

Redis Consumer Group 的 pending claim 会由存活 Worker 持续刷新。Worker 丢失后，
过期 claim 会被新 Worker 回收，并将原任务标记为 `worker_lost`，不会未经确认地重复
执行训练任务。
