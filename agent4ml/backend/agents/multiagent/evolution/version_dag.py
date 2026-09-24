"""VersionDAG — 演化产物 version DAG + is_active 单指针 + 回滚 + hash 防环（R1）。

设计（42 文档 §7.8 + spec.md VersionDAG Requirement + R1）:
- commit 写 evolution_artifacts + evolution_experiments 表 + is_active 单指针更新
- get_active L1 每次调用查 DB（不缓存，保 hot swap，INV-12）
- rollback is_active 指针回退
- hash_exists_in_recent 检查近 5 版防环（INV-7/INV-27）
- reject candidate 也存（防重复尝试）
- 持久化 SQLite（multiagent.db 加表，Z3 模式，与 L1 metrics 同 db 不同表）
- 演化失败保持旧 is_active（INV-13）
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    EvolutionArtifact,
    SkillInjectionTemplate,
)

# L2 表 schema（加到 multiagent.db，与 L1 specialist_records 同 db 不同表）
_VERSION_DAG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evolution_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    artifact_type   TEXT NOT NULL,
    version         TEXT NOT NULL,
    template_id     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    artifact_hash   TEXT NOT NULL,
    rationale       TEXT,
    created_at      TEXT NOT NULL,
    is_active       INTEGER DEFAULT 0,
    UNIQUE(artifact_type, template_id, version)
);

CREATE TABLE IF NOT EXISTS evolution_experiments (
    experiment_id   TEXT PRIMARY KEY,
    artifact_id     TEXT NOT NULL REFERENCES evolution_artifacts(artifact_id),
    from_artifact_id TEXT,
    trigger_source  TEXT NOT NULL,
    trigger_detail  TEXT,
    eval_method     TEXT,
    eval_result_json TEXT,
    decision        TEXT NOT NULL,
    timestamp       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evo_exp_artifact ON evolution_experiments(artifact_id);
"""


@dataclass(frozen=True)
class ArtifactRow:
    """evolution_artifacts 表行（get_active / get_history 返回）。"""

    artifact_id: str
    artifact_type: str
    version: str
    template_id: str
    payload_json: str
    artifact_hash: str
    rationale: str
    created_at: str
    is_active: bool


def _artifact_type_name(artifact: EvolutionArtifact) -> str:
    """artifact_type 字段值（'context_summary' | 'skill_injection'）。"""
    if isinstance(artifact, ContextSummaryTemplate):
        return "context_summary"
    if isinstance(artifact, SkillInjectionTemplate):
        return "skill_injection"
    return type(artifact).__name__.lower()


def _serialize_payload(artifact: EvolutionArtifact) -> str:
    """序列化 artifact payload 为 JSON（结构化 dataclass，可 diff / 回滚）."""
    if isinstance(artifact, ContextSummaryTemplate):
        return json.dumps({
            "version": artifact.version,
            "template_id": artifact.template_id,
            "extractors": [type(e).__name__ for e in artifact.extractors],
            "filters": [type(f).__name__ for f in artifact.filters],
            "max_tokens": artifact.max_tokens,
            "prompt_skeleton": artifact.prompt_skeleton,
        }, sort_keys=True)
    if isinstance(artifact, SkillInjectionTemplate):
        return json.dumps({
            "version": artifact.version,
            "template_id": artifact.template_id,
            "skill_selector": type(artifact.skill_selector).__name__,
            "injection_format": artifact.injection_format,
            "max_skills": artifact.max_skills,
        }, sort_keys=True)
    return json.dumps({"version": artifact.version, "template_id": artifact.template_id})


def _deserialize_payload(row: ArtifactRow) -> EvolutionArtifact:
    """从 payload_json 反序列化 artifact."""
    payload = json.loads(row.payload_json)
    if row.artifact_type == "context_summary":
        # extractors/filters 仅存类名，反序列化为空 tuple（hot swap 时 L1 重新构造）
        return ContextSummaryTemplate(
            version=payload["version"],
            template_id=payload["template_id"],
            extractors=(),
            filters=(),
            max_tokens=payload.get("max_tokens", 2000),
            prompt_skeleton=payload.get("prompt_skeleton", ""),
        )
    if row.artifact_type == "skill_injection":
        return SkillInjectionTemplate(
            version=payload["version"],
            template_id=payload["template_id"],
            skill_selector=_DummySelector(),  # placeholder，L1 重新构造
            injection_format=payload.get("injection_format", ""),
            max_skills=payload.get("max_skills", 3),
        )
    raise ValueError(f"unknown artifact_type: {row.artifact_type}")


class _DummySelector:
    """反序列化 placeholder（L1 hot swap 时重新构造真实 selector）."""

    def select(self, goal: str, available_skills: list) -> tuple:
        return ()


class VersionDAG:
    """演化产物 version DAG + is_active 单指针 + 回滚 + hash 防环（R1）。

    INVARIANT:
    - 持久化 SQLite（multiagent.db 加表，Z3 模式，INV-11）
    - is_active 单指针（同 template_id 仅 1 行 is_active=1，INV-7）
    - get_active 每次查 DB（不缓存，保 hot swap，INV-12）
    - reject candidate 也存（防重复尝试）
    - hash_exists_in_recent 近 5 版防环（INV-27）
    - 演化失败保持旧 is_active（INV-13）
    """

    def __init__(self, db_path: str = ".agent4ml/multiagent.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_VERSION_DAG_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def commit(
        self,
        artifact: EvolutionArtifact,
        eval_result: Any,
        *,
        from_artifact_id: str | None = None,
        trigger_source: str = "manual",
        trigger_detail: str = "",
        rationale: str = "",
        decision: str = "accept",
    ) -> str:
        """commit artifact + eval_result 到 DB + is_active 单指针更新（accept 时）.

        reject candidate 也存（防重复尝试），但 is_active 不更新.
        返 artifact_id.
        """
        artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        artifact_type = _artifact_type_name(artifact)
        payload_json = _serialize_payload(artifact)
        artifact_hash = artifact.artifact_hash
        now = utc_now_iso()

        with self._lock:
            conn = self._connect()
            try:
                # 插入 artifact
                conn.execute(
                    """INSERT INTO evolution_artifacts
                        (artifact_id, artifact_type, version, template_id,
                         payload_json, artifact_hash, rationale, created_at, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (artifact_id, artifact_type, artifact.version,
                     artifact.template_id, payload_json, artifact_hash,
                     rationale, now, 0),
                )
                # accept 时 is_active 单指针更新（同 artifact_type + template_id 仅 1 行 active）
                if decision == "accept":
                    conn.execute(
                        "UPDATE evolution_artifacts SET is_active=0 WHERE artifact_type=? AND template_id=?",
                        (artifact_type, artifact.template_id),
                    )
                    conn.execute(
                        "UPDATE evolution_artifacts SET is_active=1 WHERE artifact_id=?",
                        (artifact_id,),
                    )
                # 插入 experiment 记录
                exp_id = f"exp_{uuid.uuid4().hex[:12]}"
                eval_json = json.dumps({
                    "candidate_score": getattr(eval_result, "candidate_score", 0.0),
                    "baseline_score": getattr(eval_result, "baseline_score", 0.0),
                    "ci_low": getattr(eval_result, "ci_low", 0.0),
                    "ci_high": getattr(eval_result, "ci_high", 0.0),
                    "sample_size": getattr(eval_result, "sample_size", 0),
                    "success": getattr(eval_result, "success", True),
                }) if eval_result else "{}"
                conn.execute(
                    """INSERT INTO evolution_experiments
                        (experiment_id, artifact_id, from_artifact_id,
                         trigger_source, trigger_detail, eval_method,
                         eval_result_json, decision, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (exp_id, artifact_id, from_artifact_id,
                     trigger_source, trigger_detail, "longitudinal_pairs",
                     eval_json, decision, now),
                )
                conn.commit()
            finally:
                conn.close()
        return artifact_id

    def get_active(self, artifact_type: type) -> EvolutionArtifact | None:
        """L1 每次调用查 DB 取 is_active（不缓存，保 hot swap，INV-12）."""
        type_name = "context_summary" if artifact_type is ContextSummaryTemplate else "skill_injection"
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT artifact_id, artifact_type, version, template_id,
                              payload_json, artifact_hash, rationale, created_at, is_active
                        FROM evolution_artifacts
                        WHERE artifact_type=? AND is_active=1
                        ORDER BY created_at DESC LIMIT 1""",
                    (type_name,),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        art_row = ArtifactRow(
            artifact_id=row[0], artifact_type=row[1], version=row[2],
            template_id=row[3], payload_json=row[4], artifact_hash=row[5],
            rationale=row[6] or "", created_at=row[7], is_active=bool(row[8]),
        )
        return _deserialize_payload(art_row)

    def get_history(
        self, artifact_type: type, template_id: str
    ) -> list[ArtifactRow]:
        """查询同 template_id 的所有版本（按 created_at 降序）."""
        type_name = "context_summary" if artifact_type is ContextSummaryTemplate else "skill_injection"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT artifact_id, artifact_type, version, template_id,
                              payload_json, artifact_hash, rationale, created_at, is_active
                        FROM evolution_artifacts
                        WHERE artifact_type=? AND template_id=?
                        ORDER BY created_at DESC""",
                    (type_name, template_id),
                ).fetchall()
            finally:
                conn.close()
        return [
            ArtifactRow(
                artifact_id=r[0], artifact_type=r[1], version=r[2],
                template_id=r[3], payload_json=r[4], artifact_hash=r[5],
                rationale=r[6] or "", created_at=r[7], is_active=bool(r[8]),
            )
            for r in rows
        ]

    def rollback(self, artifact_id: str) -> None:
        """is_active 指针回退到指定 artifact_id."""
        with self._lock:
            conn = self._connect()
            try:
                # 查 artifact 的 artifact_type + template_id
                row = conn.execute(
                    "SELECT artifact_type, template_id FROM evolution_artifacts WHERE artifact_id=?",
                    (artifact_id,),
                ).fetchone()
                if row is None:
                    return
                artifact_type, template_id = row[0], row[1]
                # 同 artifact_type + template_id 全部 is_active=0
                conn.execute(
                    "UPDATE evolution_artifacts SET is_active=0 WHERE artifact_type=? AND template_id=?",
                    (artifact_type, template_id),
                )
                # 指定 artifact is_active=1
                conn.execute(
                    "UPDATE evolution_artifacts SET is_active=1 WHERE artifact_id=?",
                    (artifact_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def hash_exists_in_recent(
        self, artifact_hash: str, window: int = 5
    ) -> bool:
        """检查近 N 版是否含此 hash（防环，INV-27）."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT artifact_hash FROM evolution_artifacts
                        ORDER BY created_at DESC LIMIT ?""",
                    (window,),
                ).fetchall()
            finally:
                conn.close()
        return any(r[0] == artifact_hash for r in rows)
