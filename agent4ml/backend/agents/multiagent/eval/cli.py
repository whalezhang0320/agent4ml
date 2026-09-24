"""L3 CLI 命令树设计 skeleton（保留设计，暂不实现）.

设计（43 文档 §9 + §11.8 L3-9.2 + spec.md L3 CLI Requirement）:
- agent4ml multiagent l3 <verb> 命令树：status / health / decision-log / eval-history / degraded
- 仅 docstring + 接口签名，不实现（L3-9.2 决策 b：命令方式交互复杂不便观测，等待更好可观测形态）
- 未来实现触发条件：用户主动需求 / 更好可观测形态出现 / L3 实际落地后需要 inspect 工具

verb 说明：
- status: L3 状态总览（enabled / 已注册 adapter / 当前 default method / 健康监控窗口）
- health: specialist 健康报告（per-specialist SpecialistHealthReport 列表）
- decision-log: decision log 查询（按 specialist / failure_category / 时间过滤）
- eval-history: 评估历史（最近 N 次 EvalRun）
- degraded: 当前 degraded specialist（degraded_specialists() 输出）
"""
from __future__ import annotations

from typing import Any

L3_VERBS = ("status", "health", "decision-log", "eval-history", "degraded")


def l3_status(*args: Any, **kwargs: Any) -> dict:
    """L3 状态总览（enabled / 已注册 adapter / 当前 default method / 健康监控窗口）.

    暂不实现（L3-9.2 决策 b：命令方式交互复杂不便观测）.
    """
    raise NotImplementedError("L3 CLI status not implemented (L3-9.2: await better observability)")


def l3_health(*args: Any, **kwargs: Any) -> dict:
    """specialist 健康报告（per-specialist SpecialistHealthReport 列表）.

    暂不实现（L3-9.2 决策 b）.
    """
    raise NotImplementedError("L3 CLI health not implemented (L3-9.2: await better observability)")


def l3_decision_log(*args: Any, **kwargs: Any) -> dict:
    """decision log 查询（按 specialist / failure_category / 时间过滤）.

    暂不实现（L3-9.2 决策 b）.
    """
    raise NotImplementedError("L3 CLI decision-log not implemented (L3-9.2: await better observability)")


def l3_eval_history(*args: Any, **kwargs: Any) -> dict:
    """评估历史（最近 N 次 EvalRun）.

    暂不实现（L3-9.2 决策 b）.
    """
    raise NotImplementedError("L3 CLI eval-history not implemented (L3-9.2: await better observability)")


def l3_degraded(*args: Any, **kwargs: Any) -> dict:
    """当前 degraded specialist（degraded_specialists() 输出）.

    暂不实现（L3-9.2 决策 b）.
    """
    raise NotImplementedError("L3 CLI degraded not implemented (L3-9.2: await better observability)")
