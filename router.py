"""OSAF-SORS routing and scheduling control-plane implementation."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from models import D3QN, MMGAT
from routing_types import CandidatePath, NodeResource, RoutingDecision
from service import (
    PolicyMapper,
    RoutingPolicy,
    ServiceProfile,
    ServiceTag,
    ServiceTagRegistry,
)


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _attribute(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _queue_length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return len(value)


StateBuilder = Callable[["ServiceAwareRouter"], Tensor]


class ServiceAwareRouter:
    """OSAF-SORS Routing Planner (RPL)."""

    def __init__(
        self,
        network: nx.Graph,
        nodes: Mapping[Any, Any],
        device: str | torch.device | None = None,
        *,
        policy_mapper: PolicyMapper,
        state_builder: StateBuilder,
        reference_bandwidth: float,
        registry: ServiceTagRegistry | None = None,
        embedding_dim: int = 28,
    ) -> None:
        if reference_bandwidth <= 0:
            raise ValueError("reference_bandwidth must be positive")
        if not callable(state_builder):
            raise TypeError("state_builder must be callable")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.registry = registry or ServiceTagRegistry()
        self.policy_mapper = policy_mapper
        self.state_builder = state_builder
        self.reference_bandwidth = float(reference_bandwidth)
        self.embedding_dim = embedding_dim
        self.network: nx.Graph
        self.nodes: Mapping[Any, Any]
        self.node_ids: list[Any] = []
        self.node_index: dict[Any, int] = {}
        self.capability_names: tuple[str, ...] = ()
        self.directed_edges: list[tuple[Any, Any]] = []
        self.edge_to_action: dict[tuple[Any, Any], int] = {}
        self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        self.gat: MMGAT
        self.dqn: D3QN
        self.update_network(network, nodes, rebuild=True)

    def update_network(
        self, network: nx.Graph, nodes: Mapping[Any, Any], *, rebuild: bool = False
    ) -> None:
        if set(network.nodes) != set(nodes):
            raise ValueError("nodes must provide a resource record for every topology node")
        self.network = network
        self.nodes = nodes
        node_ids = list(network.nodes)
        capability_names = self._discover_capabilities(nodes.values())
        directed_edges = self._directed_edges(network)
        if not directed_edges:
            raise ValueError("SORS requires at least one directed forwarding edge")

        topology_changed = (
            rebuild
            or node_ids != self.node_ids
            or capability_names != self.capability_names
            or directed_edges != self.directed_edges
        )
        self.node_ids = node_ids
        self.node_index = {node: index for index, node in enumerate(node_ids)}
        self.capability_names = capability_names
        self.directed_edges = directed_edges
        self.edge_to_action = {edge: index for index, edge in enumerate(directed_edges)}
        self.edge_index = torch.tensor(
            [[self.node_index[u] for u, _ in directed_edges], [self.node_index[v] for _, v in directed_edges]],
            dtype=torch.long,
            device=self.device,
        )
        if topology_changed:
            self._initialise_models()

    @staticmethod
    def _directed_edges(network: nx.Graph) -> list[tuple[Any, Any]]:
        if network.is_directed():
            return list(network.edges)
        return [(u, v) for u, v in network.edges for u, v in ((u, v), (v, u))]

    @staticmethod
    def _capability_vector(node: Any) -> dict[str, int]:
        capabilities = _attribute(node, "capabilities", None)
        if capabilities is None:
            return {"cpu": 1}
        if isinstance(capabilities, Mapping):
            return {str(name): int(bool(enabled)) for name, enabled in capabilities.items()}
        return {str(name): 1 for name in capabilities}

    @classmethod
    def _discover_capabilities(cls, nodes: Iterable[Any]) -> tuple[str, ...]:
        names = set().union(*(cls._capability_vector(node) for node in nodes))
        return tuple(sorted(names or {"cpu"}))

    def _initialise_models(self) -> None:
        feature_size = 6 + len(self.capability_names)
        state_size = len(self.node_ids) * self.embedding_dim
        action_size = len(self.directed_edges)
        self.gat = MMGAT(feature_size, output_dim=self.embedding_dim).to(self.device)
        self.dqn = D3QN(state_size, action_size).to(self.device)

    @property
    def feature_size(self) -> int:
        return 6 + len(self.capability_names)

    def _node_active(self, node_id: Any) -> bool:
        return bool(_attribute(self.nodes[node_id], "active", True))

    def _available_compute(self, node_id: Any) -> float:
        return max(0.0, float(_attribute(self.nodes[node_id], "available_compute", 0.0)))

    def _queue_ratio(self, node_id: Any, queue_name: str, max_name: str) -> float:
        node = self.nodes[node_id]
        queue_length = _queue_length(_attribute(node, queue_name, None))
        maximum = float(_attribute(node, max_name, 100) or 100)
        return _clamp_unit(queue_length / max(maximum, 1.0))

    def _link_value(self, node_id: Any, neighbor: Any, name: str, fallback: float = 0.0) -> float:
        node = self.nodes[node_id]
        values = _attribute(node, name, None)
        if isinstance(values, Mapping) and neighbor in values:
            return float(values[neighbor])
        edge = self.network.get_edge_data(node_id, neighbor, default={}) or {}
        graph_name = {"bandwidth_capacity": "bandwidth", "available_bandwidth": "available_bandwidth"}.get(
            name, name
        )
        return float(edge.get(graph_name, edge.get(name, fallback)))

    def _bandwidth_capacity(self, u: Any, v: Any) -> float:
        return max(self._link_value(u, v, "bandwidth_capacity", 0.0), 0.0)

    def _available_bandwidth(self, u: Any, v: Any) -> float:
        capacity = self._bandwidth_capacity(u, v)
        return min(capacity, max(self._link_value(u, v, "available_bandwidth", capacity), 0.0))

    def _link_delay(self, u: Any, v: Any) -> float:
        edge = self.network.get_edge_data(u, v, default={}) or {}
        return max(0.0, float(edge.get("delay", edge.get("latency", 1.0))))

    def resource_features(self) -> Tensor:
        """Obtain model input features from the injected state builder."""

        features = self.state_builder(self)
        if not isinstance(features, Tensor):
            raise TypeError("state_builder must return a torch.Tensor")
        expected_shape = (len(self.node_ids), self.feature_size)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"state_builder returned shape {tuple(features.shape)}; expected {expected_shape}"
            )
        return features.to(device=self.device, dtype=torch.float32)

    def _online_q_values(self, features: Tensor) -> Tensor:
        self.gat.eval()
        self.dqn.eval()
        with torch.no_grad():
            return self.dqn(self.gat(features, self.edge_index).reshape(1, -1))[0]

    def _capability_matches(self, node_id: Any, profile: ServiceProfile) -> bool:
        capabilities = self._capability_vector(self.nodes[node_id])
        return all(
            capabilities.get(name, 0) >= required
            for name, required in profile.required_capabilities.items()
        )

    def _feasible_instances(self, source: Any, profile: ServiceProfile) -> list[Any]:
        instances: list[Any] = []
        for node_id in self.node_ids:
            if not self._node_active(node_id):
                continue
            if not self._capability_matches(node_id, profile):
                continue
            if self._available_compute(node_id) < profile.min_available_compute:
                continue
            if self._queue_ratio(node_id, "arrival_queue", "max_arrival_queue") > profile.max_arrival_queue:
                continue
            if self._queue_ratio(node_id, "computing_tasks", "max_computing_queue") > profile.max_computing_queue:
                continue
            if nx.has_path(self.network, source, node_id):
                instances.append(node_id)
        return instances

    def _valid_next_actions(self, current: Any, visited: set[Any]) -> list[int]:
        return [
            self.edge_to_action[(current, neighbor)]
            for neighbor in self.network.neighbors(current)
            if neighbor not in visited
            and self._node_active(neighbor)
            and self._available_bandwidth(current, neighbor) > 0
            and (current, neighbor) in self.edge_to_action
        ]

    @staticmethod
    def _choose_action(q_values: Tensor, valid_actions: Sequence[int]) -> int:
        if not valid_actions:
            raise ValueError("no valid forwarding action")
        return max(valid_actions, key=lambda action: float(q_values[action]))

    def computing_aware_path(
        self, source: Any, instance: Any, features: Tensor | None = None
    ) -> tuple[Any, ...] | None:
        """Generate a greedy D3QN next-hop path."""

        if source == instance:
            return (source,)
        features = self.resource_features() if features is None else features
        q_values = self._online_q_values(features)
        current, visited, path = source, {source}, [source]
        while current != instance and len(path) <= len(self.node_ids):
            valid_actions = self._valid_next_actions(current, visited)
            if not valid_actions:
                return None
            action = self._choose_action(q_values, valid_actions)
            _, next_hop = self.directed_edges[action]
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
        return tuple(path) if current == instance else None

    def _network_cost(self, policy: RoutingPolicy):
        def cost(u: Any, v: Any, _: Mapping[str, Any]) -> float:
            available = self._available_bandwidth(u, v)
            if available <= 0 or not self._node_active(v):
                return float("inf")
            return (
                policy.cost_alpha * self._link_delay(u, v)
                + (1.0 - policy.cost_alpha) * self.reference_bandwidth / available
                + policy.queue_cost_beta * self._queue_ratio(v, "arrival_queue", "max_arrival_queue")
            )

        return cost

    def network_aware_path(
        self, source: Any, instance: Any, policy: RoutingPolicy
    ) -> tuple[Any, ...] | None:
        """Compute the delay-bandwidth-queue shortest path."""

        try:
            path = nx.shortest_path(self.network, source, instance, weight=self._network_cost(policy))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return tuple(path)

    def _path_delay(self, path: Sequence[Any]) -> float:
        if not path:
            return float("inf")
        return sum(self._link_delay(u, v) for u, v in zip(path, path[1:])) + max(
            0.0, float(_attribute(self.nodes[path[-1]], "processing_delay", 0.0))
        )

    def _path_feasible(self, instance: Any, path: Sequence[Any], profile: ServiceProfile) -> bool:
        if not path or path[0] not in self.nodes or path[-1] != instance:
            return False
        if not self._capability_matches(instance, profile):
            return False
        if self._available_compute(instance) < profile.min_available_compute:
            return False
        if any(not self._node_active(node) for node in path):
            return False
        if any(
            self._available_bandwidth(u, v) < profile.min_available_bandwidth
            for u, v in zip(path, path[1:])
        ):
            return False
        if self._path_delay(path) > profile.max_end_to_end_delay:
            return False
        return (
            self._queue_ratio(instance, "arrival_queue", "max_arrival_queue") <= profile.max_arrival_queue
            and self._queue_ratio(instance, "computing_tasks", "max_computing_queue")
            <= profile.max_computing_queue
        )

    @staticmethod
    def _normalised_q_values(
        q_values: Tensor, feasible_actions: Iterable[int]
    ) -> dict[int, float]:
        feasible_actions = sorted(set(feasible_actions))
        if not feasible_actions:
            return {}
        values = q_values[feasible_actions]
        minimum, maximum = float(values.min()), float(values.max())
        if math.isclose(minimum, maximum, abs_tol=1e-12):
            return {action: 0.5 for action in feasible_actions}
        return {action: _clamp_unit((float(q_values[action]) - minimum) / (maximum - minimum)) for action in feasible_actions}

    def _candidate_score(
        self,
        instance: Any,
        path: Sequence[Any],
        path_kind: str,
        profile: ServiceProfile,
        policy: RoutingPolicy,
        features: Tensor,
        normalised_q: Mapping[int, float],
    ) -> CandidatePath:
        instance_index = self.node_index[instance]
        edge_utilisation = [
            _clamp_unit(
                1.0 - self._available_bandwidth(u, v) / max(self._bandwidth_capacity(u, v), 1e-12)
            )
            for u, v in zip(path, path[1:])
        ]
        network_util = float(np.mean(edge_utilisation)) if edge_utilisation else 0.0
        compute_util = float(features[instance_index, 0])
        delay = _clamp_unit(self._path_delay(path) / profile.max_end_to_end_delay)
        arrival_queue = float(features[instance_index, 4])
        computing_queue = float(features[instance_index, 5])
        satisfaction = (
            1.0 - float(network_util),
            1.0 - compute_util,
            1.0 - delay,
            1.0 - arrival_queue,
            1.0 - computing_queue,
        )
        model_score = float(np.dot(np.asarray(policy.score_weights), np.asarray(satisfaction)))
        edge_q = [normalised_q.get(self.edge_to_action[(u, v)], 0.0) for u, v in zip(path, path[1:])]
        learned_score = float(np.mean(edge_q)) if edge_q else 0.0
        final_score = policy.model_score_weight * model_score + policy.q_score_weight * learned_score
        return CandidatePath(
            instance=instance,
            path=tuple(path),
            path_kind=path_kind,
            satisfaction=tuple(float(value) for value in satisfaction),
            model_score=model_score,
            learned_score=learned_score,
            final_score=final_score,
        )

    def plan(self, source: Any, stag: ServiceTag) -> RoutingDecision:
        """Select the highest-scoring feasible ``(instance, path)`` pair."""

        if source not in self.nodes:
            raise ValueError(f"unknown source node {source!r}")
        profile = self.registry.resolve(stag)
        policy = self.policy_mapper(stag, profile)
        if not isinstance(policy, RoutingPolicy):
            raise TypeError("policy_mapper must return a RoutingPolicy")
        if not self._node_active(source):
            return RoutingDecision(source, stag, None, None, None, failure_reason="source is unavailable")
        features = self.resource_features()
        instances = self._feasible_instances(source, profile)
        if not instances:
            return RoutingDecision(
                source, stag, None, None, None, failure_reason="no capability- and SLA-matched instance"
            )
        feasible_paths: list[tuple[Any, tuple[Any, ...], str]] = []
        for instance in instances:
            computed = self.computing_aware_path(source, instance, features)
            network = self.network_aware_path(source, instance, policy)
            seen: set[tuple[Any, ...]] = set()
            for path_kind, path in (("computing-aware", computed), ("network-aware", network)):
                if path is None or path in seen:
                    continue
                seen.add(path)
                if self._path_feasible(instance, path, profile):
                    feasible_paths.append((instance, path, path_kind))
        if not feasible_paths:
            return RoutingDecision(
                source, stag, None, None, None, failure_reason="all instance-path pairs violate the SLA"
            )

        # Normalize Q only over actions belonging to the feasible instance-path
        # set F(st, s, R_t), rather than over unrelated edges.
        feasible_actions = {
            self.edge_to_action[(u, v)]
            for _, path, _ in feasible_paths
            for u, v in zip(path, path[1:])
        }
        normalised_q = self._normalised_q_values(
            self._online_q_values(features), feasible_actions
        )
        candidates = [
            self._candidate_score(
                instance, path, path_kind, profile, policy, features, normalised_q
            )
            for instance, path, path_kind in feasible_paths
        ]
        best = max(candidates, key=lambda candidate: candidate.final_score)
        return RoutingDecision(
            source=source,
            stag=stag,
            instance=best.instance,
            path=best.path,
            score=best.final_score,
            candidates=tuple(candidates),
        )
