"""MarkdownFileStore — Markdown 持久化 truth source（00 §8.2）。

承接 `Hezao-MemDesign-Docs/agent4ml/50-memory-l3-store-retriever.md` §4 Step 1。

实现 MemoryStore Protocol。Markdown 单文件 traces.md + 内存索引。
无事务（2A）：逐个操作，失败 log，接受最终一致。
文件锁（6B）：threading.Lock 保护 update（单进程）。
list_by_filter（7A）：按 max_age_hours 粗筛内存索引。
解析容错（2A）：frontmatter 损坏 log + 跳过，不崩。

INVARIANT：
- Markdown-as-Truth：traces.md 是 truth source，内存索引是 derived（可重建）
- 单文件 + 分隔符：所有 trace 在 traces.md，用 `<!-- trace: {id} -->` 分隔（方案 B）
- storage_path 锚定：相对路径用 cwd fallback（Layer 4 bootstrap 传绝对路径锚定 _PROJECT_ROOT）
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from agent4ml.backend.agents.memory.exceptions import (
    MemoryConflictError,
    MemoryNotFoundError,
)
from agent4ml.backend.agents.memory.schema import (
    Association,
    MemoryTrace,
    MemoryType,
    OperationLog,
)
from agent4ml.backend.agents.memory.types import MemoryFilter

logger = logging.getLogger(__name__)

# <!-- trace: {id} --> 分隔符，捕获 id + body（到下一个分隔符或文件尾）
_TRACE_SEPARATOR = re.compile(
    r"<!-- trace: ([a-f0-9]+) -->\n(.*?)(?=<!-- trace:|$)", re.DOTALL
)


class MarkdownFileStore:
    """Markdown 文件持久化（00 §8.2 truth source）。

    单文件 traces.md + 内存索引 dict[str, MemoryTrace]。
    构造即就绪：读 storage_path/traces.md，解析所有 trace 建内存索引。
    文件锁：threading.Lock 保护写操作（6B 单进程）。
    """

    def __init__(self, storage_path: str | Path) -> None:
        """初始化：解析 storage_path + 加载 traces.md 建内存索引。

        Args:
            storage_path: Markdown 持久化根目录（相对路径锚定 cwd fallback，
                          Layer 4 bootstrap 传绝对路径锚定 _PROJECT_ROOT）
        """
        self._root = self._resolve_storage_path(storage_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._traces_file = self._root / "traces.md"
        self._lock = threading.Lock()  # 6B 文件锁（单进程）
        # 内存索引：trace_id → MemoryTrace（启动加载 + 增量维护）
        self._traces: dict[str, MemoryTrace] = {}
        self._load()

    def _resolve_storage_path(self, storage_path: str | Path) -> Path:
        """解析 storage_path（相对路径锚定 cwd fallback，01 D12）。

        Layer 4 bootstrap 负责传绝对路径（_resolve_relative_paths 锚定 _PROJECT_ROOT）。
        L3 fallback：相对路径用 Path.resolve() 锚定 cwd。
        """
        p = Path(storage_path)
        if p.is_absolute():
            return p
        return p.resolve()  # 相对路径锚定 cwd

    def _load(self) -> None:
        """启动加载：读 traces.md 解析所有 trace + 建内存索引。

        traces.md 不存在时创建空文件。
        解析容错（2A）：单条 frontmatter 损坏 log + 跳过，不崩。
        """
        if not self._traces_file.exists():
            self._traces_file.write_text("# Memory Traces\n\n", encoding="utf-8")
            return
        content = self._traces_file.read_text(encoding="utf-8")
        for match in _TRACE_SEPARATOR.finditer(content):
            trace_id = match.group(1)
            trace_body = match.group(2).strip()
            trace = self._parse_trace(trace_id, trace_body)
            if trace is not None:
                self._traces[trace.id] = trace
        logger.info(
            "MarkdownFileStore loaded %d traces from %s",
            len(self._traces), self._traces_file,
        )

    def _parse_trace(self, trace_id: str, body: str) -> MemoryTrace | None:
        """解析单条 trace（frontmatter + content）→ MemoryTrace。

        格式：
        ---
        {yaml frontmatter：所有字段除 content}
        ---
        {content 正文}

        容错（2A）：yaml 解析失败 / 字段缺失 / 类型错时 log warning + 返 None（跳过）。
        associations/operation_log list → tuple（MemoryTrace frozen 要求）。
        embedding list → tuple 或 None。
        type string → MemoryType 枚举。
        """
        try:
            # 分离 frontmatter + content
            if not body.startswith("---\n"):
                logger.warning("trace %s missing frontmarker, skipped", trace_id)
                return None
            parts = body[4:].split("\n---\n", 1)
            if len(parts) != 2:
                logger.warning("trace %s malformed frontmatter, skipped", trace_id)
                return None
            frontmatter_text, content = parts
            data = yaml.safe_load(frontmatter_text)
            if not isinstance(data, dict):
                logger.warning("trace %s frontmatter not dict, skipped", trace_id)
                return None

            # 必填字段校验
            if "id" not in data or "type" not in data:
                logger.warning("trace %s missing id/type, skipped", trace_id)
                return None

            # type string → MemoryType 枚举
            type_val = data.pop("type")
            mem_type = MemoryType(type_val) if not isinstance(type_val, MemoryType) else type_val

            # associations list[dict] → tuple[Association]
            assocs_data = data.pop("associations", [])
            associations = tuple(
                Association(**a) for a in assocs_data
            ) if assocs_data else ()

            # operation_log list[dict] → tuple[OperationLog]
            log_data = data.pop("operation_log", [])
            operation_log = tuple(
                OperationLog(**log) for log in log_data
            ) if log_data else ()

            # embedding list → tuple 或 None
            embedding = data.pop("embedding", None)
            if embedding is not None:
                embedding = tuple(embedding)

            return MemoryTrace(
                content=content,
                type=mem_type,
                associations=associations,
                operation_log=operation_log,
                embedding=embedding,
                **data,
            )
        except Exception as exc:
            logger.warning("Failed to parse trace %s: %s", trace_id, exc)
            return None

    def _serialize_trace(self, trace: MemoryTrace) -> str:
        """序列化 MemoryTrace → frontmatter + content 字符串。

        frontmatter（YAML）：所有字段除 content（content 放正文）。
        手动构建 dict 避免 yaml python/tuple tag（safe_load 不认）：
        MemoryType → value string；associations/operation_log tuple → list；
        embedding tuple → list 或 None；diff tuple → list。
        """
        data = {
            "id": trace.id,
            "type": trace.type.value,
            "strength": trace.strength,
            "base_strength": trace.base_strength,
            "decay_rate": trace.decay_rate,
            "access_count": trace.access_count,
            "last_accessed": trace.last_accessed,
            "importance": trace.importance,
            "associations": [
                {"target_id": a.target_id, "strength": a.strength, "type": a.type}
                for a in trace.associations
            ],
            "embedding": list(trace.embedding) if trace.embedding is not None else None,
            "source": trace.source,
            "created_at": trace.created_at,
            "metadata": trace.metadata,
            "operation_log": [
                {
                    "timestamp": log.timestamp,
                    "operation": log.operation,
                    "actor": log.actor,
                    "diff": self._diff_to_serializable(log.diff),
                }
                for log in trace.operation_log
            ],
        }
        frontmatter = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        ).strip()
        return f"---\n{frontmatter}\n---\n{trace.content}"

    @staticmethod
    def _diff_to_serializable(diff: dict[str, Any] | None) -> dict[str, Any] | None:
        """OperationLog.diff tuple → list（yaml safe_load 兼容）。"""
        if diff is None:
            return None
        result: dict[str, Any] = {}
        for k, v in diff.items():
            if isinstance(v, tuple):
                result[k] = list(v)
            else:
                result[k] = v
        return result

    def add(self, trace: MemoryTrace) -> None:
        """新增记忆。trace.id 已存在抛 MemoryConflictError。

        6B 文件锁保护：并发 add 序列化。
        """
        with self._lock:
            if trace.id in self._traces:
                raise MemoryConflictError(
                    f"trace already exists: {trace.id}",
                    old_id=trace.id, new_id=trace.id,
                )
            self._traces[trace.id] = trace
            self._append_to_file(trace)

    def get(self, trace_id: str) -> MemoryTrace | None:
        """按 id 取记忆，不存在返 None。"""
        return self._traces.get(trace_id)

    def _append_to_file(self, trace: MemoryTrace) -> None:
        """追加单条 trace 到 traces.md（增量写）。"""
        block = f"<!-- trace: {trace.id} -->\n{self._serialize_trace(trace)}\n\n"
        with open(self._traces_file, "a", encoding="utf-8") as f:
            f.write(block)

    def _rewrite_file(self) -> None:
        """全量重写 traces.md（update/remove/batch_update 后）。

        Layer 3 先用全量重写（简化），增量改留后续优化（记忆量大时）。
        """
        with open(self._traces_file, "w", encoding="utf-8") as f:
            f.write("# Memory Traces\n\n")
            for trace in self._traces.values():
                f.write(f"<!-- trace: {trace.id} -->\n{self._serialize_trace(trace)}\n\n")

    def update(self, trace: MemoryTrace) -> None:
        """更新记忆（frozen 语义：替换）。trace.id 不存在抛 MemoryNotFoundError。

        6B 文件锁保护：并发 update 序列化，防丢更新。
        """
        with self._lock:
            if trace.id not in self._traces:
                raise MemoryNotFoundError(trace.id)
            self._traces[trace.id] = trace
            self._rewrite_file()

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        """批量更新（F2 决策，consolidate 标记 N 条旧 trace forgotten 用）。

        原子性：任一 trace.id 不存在抛 MemoryNotFoundError（全成功或全失败）。
        一次 _rewrite_file 全量重写（非 N 次 O(N²)）。
        6B 文件锁保护（与 update 同锁）。
        """
        with self._lock:
            for trace in traces:
                if trace.id not in self._traces:
                    raise MemoryNotFoundError(trace.id)
            for trace in traces:
                self._traces[trace.id] = trace
            self._rewrite_file()

    def remove(self, trace_id: str) -> None:
        """删除记忆。不存在静默（幂等）。"""
        with self._lock:
            if trace_id in self._traces:
                del self._traces[trace_id]
                self._rewrite_file()

    def list_by_type(self, type: MemoryType) -> list[MemoryTrace]:
        """按类型列出。"""
        type_key = type.value if isinstance(type, MemoryType) else str(type)
        return [t for t in self._traces.values() if t.type.value == type_key]

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]:
        """按过滤器列出（7A 粗筛 + 调用方精算 strength）。

        7A：store 只按 max_age_hours / type / metadata 粗筛（内存索引），
        strength 精算由调用方（forget_policy）逐条 compute_strength。
        """
        result = list(self._traces.values())
        # type 过滤
        if filter.type_filter is not None:
            type_key = (
                filter.type_filter.value
                if isinstance(filter.type_filter, MemoryType)
                else str(filter.type_filter)
            )
            result = [t for t in result if t.type.value == type_key]
        # max_age_hours 粗筛（按 last_accessed，<=0 用 created_at，7A）
        if filter.max_age_hours is not None:
            now = time.time()
            max_age_seconds = filter.max_age_hours * 3600.0
            result = [
                t for t in result
                if (now - (t.last_accessed if t.last_accessed > 0 else t.created_at)) <= max_age_seconds
            ]
        # metadata 过滤（全匹配）
        if filter.metadata_filter:
            result = [
                t for t in result
                if all(t.metadata.get(k) == v for k, v in filter.metadata_filter.items())
            ]
        return result

    def list_all(self) -> list[MemoryTrace]:
        """列出全部。"""
        return list(self._traces.values())
