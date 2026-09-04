"""Data structures exchanged by the OSAF-SORS routing planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from service import ServiceTag


@dataclass
class NodeResource:
    """Resource-state record consumed by the control-plane algorithm."""

    node_id: Any
    compute_capacity: float
    available_compute: float
    bandwidth_capacity: dict[Any, float]
    available_bandwidth: dict[Any, float]
    capabilities: Mapping[str, int] | set[str] = field(default_factory=lambda: {"cpu": 1})
    arrival_queue: list[dict[str, Any]] = field(default_factory=list)
    computing_tasks: list[dict[str, Any]] = field(default_factory=list)
    processing_delay: float = 0.0
    max_arrival_queue: int = 100
    max_computing_queue: int = 100
    active: bool = True


@dataclass(frozen=True)
class CandidatePath:
    instance: Any
    path: tuple[Any, ...]
    path_kind: str
    satisfaction: tuple[float, float, float, float, float]
    model_score: float
    learned_score: float
    final_score: float


@dataclass(frozen=True)
class RoutingDecision:
    source: Any
    stag: ServiceTag
    instance: Any | None
    path: tuple[Any, ...] | None
    score: float | None
    candidates: tuple[CandidatePath, ...] = ()
    failure_reason: str | None = None

    @property
    def successful(self) -> bool:
        return self.instance is not None and self.path is not None
