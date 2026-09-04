"""Service-tag and routing-policy definitions for OSAF-SORS."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Callable, Mapping


@dataclass(frozen=True)
class ServiceTag:
    """The 128-bit STag Option Data."""

    value: int
    sid_type: int = 0
    control_flags: int = 0
    traffic_hint: int = 0
    indicator: int = 1
    reserved_1: int = 0
    reserved_2: int = 0

    def __post_init__(self) -> None:
        limits = {
            "indicator": (self.indicator, 8),
            "sid_type": (self.sid_type, 2),
            "control_flags": (self.control_flags, 6),
            "reserved_1": (self.reserved_1, 16),
            "value": (self.value, 32),
            "traffic_hint": (self.traffic_hint, 32),
            "reserved_2": (self.reserved_2, 32),
        }
        for name, (number, bits) in limits.items():
            if not isinstance(number, int) or not 0 <= number < (1 << bits):
                raise ValueError(f"{name} must fit in {bits} bits")

    def to_option_data(self) -> bytes:
        sid_and_flags = (self.sid_type << 6) | self.control_flags
        return struct.pack(
            "!BBHIII",
            self.indicator,
            sid_and_flags,
            self.reserved_1,
            self.value,
            self.traffic_hint,
            self.reserved_2,
        )

    @classmethod
    def from_option_data(cls, data: bytes) -> "ServiceTag":
        if len(data) != 16:
            raise ValueError("STag option data must be exactly 128 bits (16 bytes)")
        indicator, sid_and_flags, reserved_1, value, hint, reserved_2 = struct.unpack(
            "!BBHIII", data
        )
        return cls(
            value=value,
            sid_type=sid_and_flags >> 6,
            control_flags=sid_and_flags & 0b0011_1111,
            traffic_hint=hint,
            indicator=indicator,
            reserved_1=reserved_1,
            reserved_2=reserved_2,
        )


@dataclass(frozen=True)
class ServiceProfile:
    """SLA profile dereferenced from an STag."""

    required_capabilities: Mapping[str, int]
    min_available_compute: float
    min_available_bandwidth: float
    max_end_to_end_delay: float
    max_arrival_queue: float = 1.0
    max_computing_queue: float = 1.0

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.required_capabilities.values()):
            raise ValueError("required_capabilities must be a binary capability vector")
        if self.min_available_compute < 0 or self.min_available_bandwidth < 0:
            raise ValueError("minimum compute and bandwidth must be non-negative")
        if self.max_end_to_end_delay <= 0:
            raise ValueError("maximum end-to-end delay must be positive")
        if not 0 <= self.max_arrival_queue <= 1 or not 0 <= self.max_computing_queue <= 1:
            raise ValueError("queue bounds must be normalized values in [0, 1]")


@dataclass(frozen=True)
class RoutingPolicy:
    """Inference-time coefficients produced by ``g(Phi(st))``."""

    cost_alpha: float
    queue_cost_beta: float
    score_weights: tuple[float, float, float, float, float]
    model_score_weight: float
    q_score_weight: float

    def __post_init__(self) -> None:
        if not 0 <= self.cost_alpha <= 1 or self.queue_cost_beta < 0:
            raise ValueError("invalid network-cost coefficients")
        if len(self.score_weights) != 5 or not math.isclose(sum(self.score_weights), 1.0, abs_tol=1e-8):
            raise ValueError("the five final-score weights must sum to one")
        if any(weight < 0 for weight in self.score_weights):
            raise ValueError("final-score weights must be non-negative")
        if not math.isclose(self.model_score_weight + self.q_score_weight, 1.0, abs_tol=1e-8):
            raise ValueError("model_score_weight and q_score_weight must sum to one")


PolicyMapper = Callable[[ServiceTag, ServiceProfile], RoutingPolicy]


class ServiceTagRegistry:
    """Control-plane STag-to-SLA mapping; packet data never contains the SLA."""

    def __init__(self) -> None:
        self._profiles: dict[int, ServiceProfile] = {}

    def register(self, tag: ServiceTag, profile: ServiceProfile) -> None:
        self._profiles[tag.value] = profile

    def resolve(self, tag: ServiceTag) -> ServiceProfile:
        try:
            return self._profiles[tag.value]
        except KeyError as error:
            raise KeyError(f"unknown, expired, or unauthorized STag {tag.value}") from error

    def __contains__(self, tag: ServiceTag) -> bool:
        return tag.value in self._profiles
