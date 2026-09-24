from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from agent4ml.backend.agents.memory.schema import (
    Association,
    MemoryTrace,
    MemoryType,
    OperationLog,
)
from agent4ml.backend.agents.memory.strategies.default.store import MarkdownFileStore


def _make_trace(
    *,
    id: str = "a1b2c3d4e5f67890",
    content: str = "用户下周去东京出差",
    type: MemoryType = MemoryType.EPISODIC,
    associations: tuple = (),
    operation_log: tuple = (),
    metadata: dict | None = None,
    embedding: tuple | None = None,
    importance: float = 0.5,
    source: str | None = "thread:abc",
    last_accessed: float = 0.0,
    created_at: float = 0.0,
) -> MemoryTrace:
    return MemoryTrace(
        id=id,
        content=content,
        type=type,
        importance=importance,
        associations=associations,
        operation_log=operation_log,
        metadata=metadata or {},
        embedding=embedding,
        source=source,
        last_accessed=last_accessed,
        created_at=created_at,
    )


class TestAdd:
    def test_add_writes_to_file_and_index(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        trace = _make_trace()
        store.add(trace)
        # 内存索引
        assert store.get(trace.id) is trace or store.get(trace.id).id == trace.id
        # traces.md 写入
        content = (tmp_path / "traces.md").read_text(encoding="utf-8")
        assert f"<!-- trace: {trace.id} -->" in content

    def test_add_duplicate_raises_conflict(self, tmp_path: Path) -> None:
        from agent4ml.backend.agents.memory.exceptions import MemoryConflictError

        store = MarkdownFileStore(tmp_path)
        trace = _make_trace()
        store.add(trace)
        with pytest.raises(MemoryConflictError):
            store.add(trace)


class TestGet:
    def test_get_existing_returns_trace(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        trace = _make_trace()
        store.add(trace)
        result = store.get(trace.id)
        assert result is not None
        assert result.id == trace.id

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        assert store.get("nonexistent") is None


class TestStartupLoad:
    def test_load_existing_traces(self, tmp_path: Path) -> None:
        """启动加载：traces.md 存在时解析所有 trace 建索引。"""
        # 第一次 store 写入 2 条
        store1 = MarkdownFileStore(tmp_path)
        store1.add(_make_trace(id="aaaa1111aaaa1111", content="content one"))
        store1.add(_make_trace(id="bbbb2222bbbb2222", content="content two", type=MemoryType.SEMANTIC))
        # 第二次 store 从同一目录加载
        store2 = MarkdownFileStore(tmp_path)
        assert len(store2._traces) == 2

    def test_load_creates_empty_file_when_missing(self, tmp_path: Path) -> None:
        """启动加载：traces.md 不存在时创建空文件。"""
        assert not (tmp_path / "traces.md").exists()
        MarkdownFileStore(tmp_path)
        assert (tmp_path / "traces.md").exists()
        content = (tmp_path / "traces.md").read_text(encoding="utf-8")
        assert "# Memory Traces" in content


class TestSerializeParseRoundtrip:
    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        """序列化 + 解析往返一致（基础字段）。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(content="hello world", importance=0.8)
        store.add(original)
        # 重新加载解析
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert parsed.id == original.id
        assert parsed.content == original.content
        assert parsed.type == original.type
        assert parsed.importance == original.importance
        assert parsed.strength == original.strength

    def test_associations_roundtrip(self, tmp_path: Path) -> None:
        """associations tuple 序列化往返。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(
            associations=(
                Association(target_id="t2", strength=0.9, type="causal"),
                Association(target_id="t3", strength=0.3),
            )
        )
        store.add(original)
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert len(parsed.associations) == 2
        assert parsed.associations[0].target_id == "t2"
        assert parsed.associations[0].strength == 0.9
        assert parsed.associations[0].type == "causal"
        assert parsed.associations[1].target_id == "t3"

    def test_operation_log_roundtrip(self, tmp_path: Path) -> None:
        """operation_log tuple 序列化往返。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(
            operation_log=(
                OperationLog(
                    timestamp=1000.0, operation="encode", actor="turn-1",
                    diff={"content": [None, "hello"]},
                ),
            )
        )
        store.add(original)
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert len(parsed.operation_log) == 1
        assert parsed.operation_log[0].operation == "encode"
        assert parsed.operation_log[0].actor == "turn-1"

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        """metadata dict 序列化往返。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(metadata={"tag": "travel", "project": "agent4ml"})
        store.add(original)
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert parsed.metadata == {"tag": "travel", "project": "agent4ml"}

    def test_embedding_roundtrip(self, tmp_path: Path) -> None:
        """embedding tuple 序列化往返。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(embedding=(0.1, 0.2, 0.3))
        store.add(original)
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert parsed.embedding == (0.1, 0.2, 0.3)

    def test_embedding_none_roundtrip(self, tmp_path: Path) -> None:
        """embedding None 序列化往返。"""
        store = MarkdownFileStore(tmp_path)
        original = _make_trace(embedding=None)
        store.add(original)
        store2 = MarkdownFileStore(tmp_path)
        parsed = store2.get(original.id)
        assert parsed is not None
        assert parsed.embedding is None


class TestStoragePath:
    def test_relative_path_anchors_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """相对路径锚定 cwd fallback。"""
        monkeypatch.chdir(tmp_path)
        store = MarkdownFileStore("memdir")
        assert store._root == (tmp_path / "memdir").resolve()
        assert (tmp_path / "memdir" / "traces.md").exists()


class TestUpdate:
    def test_update_existing_replaces(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        trace = _make_trace(content="original")
        store.add(trace)
        updated = _make_trace(id=trace.id, content="updated")
        store.update(updated)
        assert store.get(trace.id).content == "updated"

    def test_update_missing_raises_not_found(self, tmp_path: Path) -> None:
        from agent4ml.backend.agents.memory.exceptions import MemoryNotFoundError

        store = MarkdownFileStore(tmp_path)
        with pytest.raises(MemoryNotFoundError):
            store.update(_make_trace(id="nonexistent12345"))

    def test_concurrent_update_no_lost_update(self, tmp_path: Path) -> None:
        """6B 文件锁：并发 update 序列化，不丢更新。"""
        store = MarkdownFileStore(tmp_path)
        trace = _make_trace(id="cccc1111cccc1111", content="base")
        store.add(trace)

        errors: list[Exception] = []

        def update_content(i: int) -> None:
            try:
                updated = _make_trace(id="cccc1111cccc1111", content=f"v{i}")
                store.update(updated)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update_content, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # 最终值是某次 update 的结果（不丢，不是 base）
        assert store.get("cccc1111cccc1111").content.startswith("v")


class TestBatchUpdate:
    def test_batch_update_all_exist(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        t1 = _make_trace(id="dddd1111dddd1111", content="c1")
        t2 = _make_trace(id="eeee2222eeee2222", content="c2")
        store.add(t1)
        store.add(t2)
        store.batch_update([
            _make_trace(id="dddd1111dddd1111", content="updated1"),
            _make_trace(id="eeee2222eeee2222", content="updated2"),
        ])
        assert store.get("dddd1111dddd1111").content == "updated1"
        assert store.get("eeee2222eeee2222").content == "updated2"

    def test_batch_update_atomic_all_fail(self, tmp_path: Path) -> None:
        """任一不存在全失败（原子性），已存在的内存索引不变。"""
        from agent4ml.backend.agents.memory.exceptions import MemoryNotFoundError

        store = MarkdownFileStore(tmp_path)
        t1 = _make_trace(id="ffff1111ffff1111", content="c1")
        store.add(t1)
        with pytest.raises(MemoryNotFoundError):
            store.batch_update([
                _make_trace(id="ffff1111ffff1111", content="updated1"),
                _make_trace(id="nonexistent12345", content="updated2"),
            ])
        # t1 内存索引不变（原子性，未替换）
        assert store.get("ffff1111ffff1111").content == "c1"


class TestRemove:
    def test_remove_existing(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        trace = _make_trace(id="1111aaaa1111aaaa")
        store.add(trace)
        store.remove(trace.id)
        assert store.get(trace.id) is None

    def test_remove_missing_silent(self, tmp_path: Path) -> None:
        """不存在静默（幂等），不抛错。"""
        store = MarkdownFileStore(tmp_path)
        store.remove("nonexistent12345")  # 不抛错


class TestListByType:
    def test_filter_by_type(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", type=MemoryType.EPISODIC))
        store.add(_make_trace(id="bbbb2222bbbb2222", type=MemoryType.SEMANTIC))
        episodic = store.list_by_type(MemoryType.EPISODIC)
        assert len(episodic) == 1
        assert episodic[0].type is MemoryType.EPISODIC


class TestListAll:
    def test_returns_all(self, tmp_path: Path) -> None:
        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="cccc1111cccc1111"))
        store.add(_make_trace(id="dddd2222dddd2222"))
        assert len(store.list_all()) == 2


class TestListByFilter:
    def test_max_age_hours_filter(self, tmp_path: Path) -> None:
        """7A max_age_hours 粗筛：超龄 trace 排除。"""
        from agent4ml.backend.agents.memory.types import MemoryFilter

        store = MarkdownFileStore(tmp_path)
        old_trace = _make_trace(id="oldd1111oldd1111", last_accessed=1.0)
        new_trace = _make_trace(id="neww2222neww2222", last_accessed=time.time())
        store.add(old_trace)
        store.add(new_trace)
        # max_age_hours=1（1h 内），old_trace 远超
        result = store.list_by_filter(MemoryFilter(max_age_hours=1))
        ids = [t.id for t in result]
        assert "neww2222neww2222" in ids
        assert "oldd1111oldd1111" not in ids

    def test_last_accessed_zero_uses_created_at(self, tmp_path: Path) -> None:
        """last_accessed<=0 时用 created_at 算年龄。"""
        from agent4ml.backend.agents.memory.types import MemoryFilter

        store = MarkdownFileStore(tmp_path)
        now = time.time()
        trace = _make_trace(id="aaaa1111aaaa1111", last_accessed=0.0, created_at=now)
        store.add(trace)
        result = store.list_by_filter(MemoryFilter(max_age_hours=1))
        assert any(t.id == "aaaa1111aaaa1111" for t in result)

    def test_metadata_filter(self, tmp_path: Path) -> None:
        from agent4ml.backend.agents.memory.types import MemoryFilter

        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", metadata={"tag": "travel"}))
        store.add(_make_trace(id="bbbb2222bbbb2222", metadata={"tag": "work"}))
        result = store.list_by_filter(MemoryFilter(metadata_filter={"tag": "travel"}))
        assert len(result) == 1
        assert result[0].id == "aaaa1111aaaa1111"

    def test_type_filter(self, tmp_path: Path) -> None:
        from agent4ml.backend.agents.memory.types import MemoryFilter

        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", type=MemoryType.EPISODIC))
        store.add(_make_trace(id="bbbb2222bbbb2222", type=MemoryType.SEMANTIC))
        result = store.list_by_filter(MemoryFilter(type_filter=MemoryType.SEMANTIC))
        assert len(result) == 1
        assert result[0].type is MemoryType.SEMANTIC


class TestParseFaultTolerance:
    def test_malformed_frontmatter_skipped(self, tmp_path: Path, caplog) -> None:
        """2A 容错：frontmatter 损坏 log + 跳过，不崩。"""
        traces_file = tmp_path / "traces.md"
        traces_file.write_text(
            "# Memory Traces\n\n"
            "<!-- trace: aaaa1111aaaa1111 -->\n"
            "this is not valid yaml frontmatter\n\n"
            "<!-- trace: bbbb2222bbbb2222 -->\n"
            "---\n"
            "id: bbbb2222bbbb2222\n"
            "type: episodic\n"
            "strength: 0.0\n"
            "---\n"
            "valid content\n\n",
            encoding="utf-8",
        )
        store = MarkdownFileStore(tmp_path)
        # 损坏的跳过，正常的加载
        assert store.get("aaaa1111aaaa1111") is None
        assert store.get("bbbb2222bbbb2222") is not None
