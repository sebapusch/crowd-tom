from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception import EPSILON, unit_rows


def profile_greedy() -> ToM0Params:
    return ToM0Params(
        uncertainty_time=1.0,
    )


@dataclass(frozen=True)
class ToM0Params:
    uncertainty_time: float = 30.0
    distance_scale: float = 10.0
    w_distance: float = -1.0
    w_occupancy: float = -1.0
    w_uncertainty: float = -1.0
    occupancy_range: float = 12.0
    occupancy_fov_rad: float = np.deg2rad(60.0)
    occupancy_half_density: float = 0.04

    def __post_init__(self) -> None:
        if not np.isfinite(self.uncertainty_time) or self.uncertainty_time <= 0:
            raise ValueError("ToM0 uncertainty_time must be positive and finite")
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("ToM0 distance_scale must be positive and finite")
        if not np.isfinite(self.occupancy_half_density) or self.occupancy_half_density <= 0:
            raise ValueError("ToM0 occupancy_half_density must be positive and finite")


def minmax_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] among masked entries of each row. Unmasked -> 0."""
    lo = np.min(np.where(mask, values, np.inf), axis=1, keepdims=True)
    hi = np.max(np.where(mask, values, -np.inf), axis=1, keepdims=True)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 0.0)
    span = hi - lo
    equal = span < EPSILON
    scaled = (values - lo) / np.maximum(span, EPSILON)
    scaled = np.where(equal, 0.0, scaled)
    return np.where(mask, scaled, 0.0)


def uncertainty(age: np.ndarray, uncertainty_time: float) -> np.ndarray:
    age = np.maximum(age, 0.0)
    return age / (age + uncertainty_time)


def distance_penalty(distance: np.ndarray, distance_scale: float) -> np.ndarray:
    return np.sqrt(1.0 + np.maximum(distance, 0.0) / distance_scale) - 1.0


def occupancy_penalty(density: np.ndarray, half_density: float) -> np.ndarray:
    density = np.maximum(density, 0.0)
    return density / (density + half_density)


def cone_density(
        pos: np.ndarray,
        targets: np.ndarray,
        occupancy_range: float,
        occupancy_fov_rad: float,
        visible: np.ndarray | None = None,
) -> np.ndarray:
    """Density of other agents in a cone from each agent toward each target. Shape (N, E)."""
    n = len(pos)
    e = targets.shape[1]
    if n == 0 or e == 0:
        return np.zeros((n, e))

    delta = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    in_range = (dist > EPSILON) & (dist <= occupancy_range)
    if visible is not None:
        in_range &= visible

    to_agent = np.zeros_like(delta)
    safe_dist = np.maximum(dist, EPSILON)
    to_agent[in_range] = delta[in_range] / safe_dist[in_range, None]

    to_exit = targets - pos[:, None, :]
    exit_norm = np.linalg.norm(to_exit, axis=2, keepdims=True)
    to_exit = to_exit / np.maximum(exit_norm, EPSILON)

    cosine = np.einsum('nkd,njd->nkj', to_agent, to_exit)
    in_cone = cosine >= np.cos(0.5 * occupancy_fov_rad)
    counts = np.sum(in_range[:, :, None] & in_cone, axis=1).astype(float)
    area = 0.5 * occupancy_range ** 2 * occupancy_fov_rad
    return counts / max(area, EPSILON)


def utilities(
        distance: np.ndarray,
        occupancy: np.ndarray,
        age: np.ndarray,
        has_belief: np.ndarray,
        params: ToM0Params,
) -> np.ndarray:
    u = uncertainty(age, params.uncertainty_time)
    d_penalty = distance_penalty(distance, params.distance_scale)
    o_penalty = occupancy_penalty(occupancy, params.occupancy_half_density)
    score = (
        params.w_distance * d_penalty
        + params.w_occupancy * o_penalty
        + params.w_uncertainty * u
    )
    return np.where(has_belief, score, -np.inf)


def headings_from_beliefs(
        pos: np.ndarray,
        mu: np.ndarray,
        has_belief: np.ndarray,
        age: np.ndarray,
        params: ToM0Params,
        visible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heading, chosen_exit, occupancy). chosen_exit is -1 if no belief."""
    n = len(pos)
    e = has_belief.shape[1] if has_belief.ndim == 2 else 0
    if n == 0 or e == 0:
        return np.zeros((n, 2)), np.full(n, -1, dtype=int), np.zeros((n, e))

    occupancy = cone_density(
        pos, mu, params.occupancy_range, params.occupancy_fov_rad, visible=visible,
    )
    distance = np.linalg.norm(mu - pos[:, None, :], axis=2)
    score = utilities(distance, occupancy, age, has_belief, params)
    chosen = np.argmax(score, axis=1)
    none = ~np.any(has_belief, axis=1)
    chosen = np.where(none, -1, chosen)

    gathered_mu = mu[np.arange(n), np.clip(chosen, 0, e - 1)]
    heading = unit_rows(gathered_mu - pos)
    heading[none] = 0.0
    return heading, chosen, occupancy
