from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING

import numpy as np

from perception import EPSILON, pairwise_visible_mask, unit_rows, visible_exit_mask
from tom0 import ToM0Params, distance_penalty, occupancy_penalty, uncertainty

if TYPE_CHECKING:
    from obstacle import Obstacle


@dataclass(frozen=True)
class ToM1Params:
    model_range: float = 12.0
    memory_agents: int = 16
    memory_horizon: float = 30.0
    confidence_decay: float = 10.0
    knowledge_prior: float = 0.2
    uncertainty_prior: float = 0.5
    choice_temperature: float = 1.0
    demand_half_count: float = 2.0
    demand_weight: float = -1.0

    def __post_init__(self) -> None:
        if self.model_range <= 0:
            raise ValueError("ToM1 model_range must be positive")
        if self.memory_agents < 0:
            raise ValueError("ToM1 memory_agents cannot be negative")
        if self.memory_horizon <= 0 or self.confidence_decay <= 0:
            raise ValueError("ToM1 memory times must be positive")
        if not 0 <= self.knowledge_prior < 1:
            raise ValueError("ToM1 knowledge_prior must be in [0, 1)")
        if not 0 <= self.uncertainty_prior <= 1:
            raise ValueError("ToM1 uncertainty_prior must be in [0, 1]")
        if self.choice_temperature <= 0:
            raise ValueError("ToM1 choice_temperature must be positive")
        if not np.isfinite(self.demand_half_count) or self.demand_half_count <= 0:
            raise ValueError("ToM1 demand_half_count must be positive and finite")
        if self.demand_weight > 0:
            raise ValueError("ToM1 demand_weight must be nonpositive")


def demand_penalty(demand: np.ndarray, half_count: float) -> np.ndarray:
    demand = np.maximum(demand, 0.0)
    return demand / (demand + half_count)


class ToM1Reasoner:
    """Predict visible agents' ToM0 choices from each observer's limited records."""

    def __init__(self, params: ToM1Params) -> None:
        self.params = params
        # observer ID -> observed ID -> exit index -> latest witnessed time
        self.memories: dict[int, dict[int, dict[int, float]]] = {}

    def _prune(self, active_ids: set[int], now: float) -> None:
        horizon = self.params.memory_horizon
        for observer_id in list(self.memories):
            if observer_id not in active_ids:
                del self.memories[observer_id]
                continue
            records = self.memories[observer_id]
            for observed_id in list(records):
                if observed_id not in active_ids:
                    del records[observed_id]
                    continue
                for exit_index, witnessed_at in list(records[observed_id].items()):
                    if now - witnessed_at > horizon:
                        del records[observed_id][exit_index]
                if not records[observed_id]:
                    del records[observed_id]
            if not records:
                del self.memories[observer_id]

    def _witness(
        self,
        observer_id: int,
        observed: np.ndarray,
        ids: np.ndarray,
        distances: np.ndarray,
        known: np.ndarray,
        witnessed: np.ndarray,
        now: float,
    ) -> None:
        if self.params.memory_agents == 0 or len(observed) == 0:
            return

        # Process farther agents first so a full, equal-age memory retains nearer ones.
        records = self.memories.setdefault(observer_id, {})
        # Heap keys preserve the dict insertion-order tie break of the old min scan.
        # An updated record leaves a stale heap entry, skipped when evicting.
        latest = {agent_id: max(sightings.values()) for agent_id, sightings in records.items()}
        insertion_order = {agent_id: order for order, agent_id in enumerate(records)}
        oldest_first = [
            (timestamp, insertion_order[agent_id], agent_id)
            for agent_id, timestamp in latest.items()
        ]
        heapify(oldest_first)
        next_order = len(insertion_order)
        for row in np.argsort(distances[observed])[::-1]:
            exits_seen = known[witnessed[row]]
            if len(exits_seen) == 0:
                continue
            observed_id = int(ids[observed[row]])
            if observed_id not in records:
                if len(records) >= self.params.memory_agents:
                    while True:
                        timestamp, order, oldest = heappop(oldest_first)
                        if latest.get(oldest) == timestamp and insertion_order.get(oldest) == order:
                            del records[oldest]
                            del latest[oldest]
                            del insertion_order[oldest]
                            break
                records[observed_id] = {}
                insertion_order[observed_id] = next_order
                next_order += 1
            records[observed_id].update({int(k): now for k in exits_seen})
            previous_latest = latest.get(observed_id)
            if previous_latest is None or now > previous_latest:
                latest[observed_id] = now
                heappush(oldest_first, (now, insertion_order[observed_id], observed_id))
        if not records:
            del self.memories[observer_id]

    def _estimated_occupancy(
        self,
        observer: int,
        observed: np.ndarray,
        pos: np.ndarray,
        targets: np.ndarray,
        pair_directions: np.ndarray,
        occupancy_eligible: np.ndarray,
        tom0: ToM0Params,
    ) -> np.ndarray:
        other_indices = np.concatenate(([observer], observed))
        directions = pair_directions[np.ix_(observed, other_indices)]

        axes = targets[None, :, :] - pos[observed][:, None, :]
        axes /= np.maximum(np.linalg.norm(axes, axis=2, keepdims=True), EPSILON)
        in_cone = np.matmul(directions, np.swapaxes(axes, 1, 2)) >= np.cos(
            0.5 * tom0.occupancy_fov_rad
        )
        eligible = occupancy_eligible[np.ix_(observed, other_indices)]
        count = np.sum(eligible[:, :, None] & in_cone, axis=1)
        area = max(
            0.5 * tom0.occupancy_range**2 * tom0.occupancy_fov_rad,
            EPSILON,
        )
        return count / area

    def choose(
        self,
        *,
        ids: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        mu: np.ndarray,
        has_belief: np.ndarray,
        base_scores: np.ndarray,
        visible_agents: np.ndarray,
        obstacles: list[Obstacle],
        view_range: float,
        fov_rad: float,
        tom0: ToM0Params,
        now: float,
        observer_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return desired headings, chosen exits, and predicted demand by exit."""
        n, exit_count = has_belief.shape
        if observer_mask is None:
            observer_mask = np.ones(n, dtype=bool)
        chosen = np.full(n, -1, dtype=int)
        demand = np.zeros((n, exit_count), dtype=float)
        heading = np.zeros((n, 2), dtype=float)
        self._prune(set(map(int, ids)), now)
        if n == 0 or exit_count == 0:
            return heading, chosen, demand

        # Each observer reuses these pair measurements for its predicted crowding.
        pair_offset = pos[None, :, :] - pos[:, None, :]
        pair_distance = np.linalg.norm(pair_offset, axis=2)
        pair_directions = pair_offset / np.maximum(pair_distance[:, :, None], EPSILON)
        modeled = visible_agents & (pair_distance <= self.params.model_range) & observer_mask[:, None]
        witnessed = np.zeros((n, n, exit_count), dtype=bool)
        observer_index, observed_index = np.nonzero(modeled)
        for k in range(exit_count):
            candidates = has_belief[observer_index, k]
            ii = observer_index[candidates]
            jj = observed_index[candidates]
            if len(ii):
                witnessed[ii, jj, k] = visible_exit_mask(
                    pos[jj],
                    vel[jj],  # Observable motion approximates a hidden heading.
                    mu[ii, k],
                    obstacles,
                    view_range,
                    fov_rad,
                )
        # Actual desired headings are private. Use visible velocity as a heading proxy.
        estimated_visibility = (
            visible_agents
            if fov_rad >= 2.0 * np.pi - 1e-9
            else pairwise_visible_mask(pos, vel, obstacles, view_range, fov_rad)
        )
        occupancy_eligible = (
            estimated_visibility
            & (pair_distance > EPSILON)
            & (pair_distance <= tom0.occupancy_range)
        )

        for i in range(n):
            if not observer_mask[i]:
                continue
            known = np.flatnonzero(has_belief[i])
            if len(known) == 0:
                continue
            observer_id = int(ids[i])
            observed = np.flatnonzero(modeled[i])
            targets = mu[i, known]
            self._witness(
                observer_id, observed, ids, pair_distance[i], known,
                witnessed[i][np.ix_(observed, known)], now,
            )
            if len(known) == 1:
                # With one candidate, its predicted choice probability is q_any.
                # Witness records still update for later multi-exit decisions.
                exit_index = int(known[0])
                chosen[i] = exit_index
                if len(observed):
                    records = self.memories.get(observer_id, {})
                    confidence = np.full(len(observed), self.params.knowledge_prior)
                    for row, j in enumerate(observed):
                        witnessed_at = records.get(int(ids[j]), {}).get(exit_index)
                        if witnessed_at is not None:
                            age = max(now - witnessed_at, 0.0)
                            confidence[row] += (
                                (1.0 - self.params.knowledge_prior)
                                * np.exp(-age / self.params.confidence_decay)
                            )
                    demand[i, exit_index] = np.sum(1.0 - (1.0 - confidence))
                continue
            own_scores = base_scores[i, known].copy()
            if len(observed):
                records = self.memories.get(observer_id, {})
                confidence = np.full(
                    (len(observed), len(known)),
                    self.params.knowledge_prior,
                    dtype=float,
                )
                uncertain = np.full_like(confidence, self.params.uncertainty_prior)
                for row, j in enumerate(observed):
                    sightings = records.get(int(ids[j]), {})
                    for column, k in enumerate(known):
                        witnessed_at = sightings.get(int(k))
                        if witnessed_at is None:
                            continue
                        age = max(now - witnessed_at, 0.0)
                        confidence[row, column] += (
                            (1.0 - self.params.knowledge_prior)
                            * np.exp(-age / self.params.confidence_decay)
                        )
                        uncertain[row, column] = uncertainty(age, tom0.uncertainty_time)

                distance = np.linalg.norm(
                    targets[None, :, :] - pos[observed][:, None, :],
                    axis=2,
                )
                occupancy = self._estimated_occupancy(
                    i, observed, pos, targets, pair_directions,
                    occupancy_eligible, tom0,
                )
                estimated_scores = (
                    tom0.w_distance * distance_penalty(distance, tom0.distance_scale)
                    + tom0.w_occupancy * occupancy_penalty(
                        occupancy, tom0.occupancy_half_density
                    )
                    + tom0.w_uncertainty * uncertain
                )
                q_any = 1.0 - np.prod(1.0 - confidence, axis=1)
                scaled = estimated_scores / self.params.choice_temperature
                scaled -= np.max(scaled, axis=1, keepdims=True)
                weighted = confidence * np.exp(scaled)
                total = weighted.sum(axis=1, keepdims=True)
                probabilities = q_any[:, None] * np.divide(
                    weighted,
                    total,
                    out=np.zeros_like(weighted),
                    where=total > 0,
                )
                demand[i, known] = probabilities.sum(axis=0)
                own_scores += self.params.demand_weight * demand_penalty(
                    demand[i, known], self.params.demand_half_count,
                )

            chosen[i] = int(known[np.argmax(own_scores)])

        selected = chosen >= 0
        heading[selected] = unit_rows(
            mu[np.flatnonzero(selected), chosen[selected]] - pos[selected]
        )
        return heading, chosen, demand
