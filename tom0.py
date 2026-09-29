from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception import EPSILON, unit_rows


@dataclass(frozen=True)
class ToM0Params:
    sigma0: float = 0.1
    sigma_alpha: float = 0.5
    sigma_ref: float = 8.0
    w_distance: float = -1.0
    w_occupancy: float = -1.0
    w_uncertainty: float = -1.0
    occupancy_range: float = 12.0
    occupancy_fov_rad: float = np.deg2rad(60.0)


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


def uncertainty(sigma: np.ndarray, sigma_ref: float) -> np.ndarray:
    return 1.0 - np.exp(-sigma / max(sigma_ref, EPSILON))


def cone_density(
        pos: np.ndarray,
        targets: np.ndarray,
        occupancy_range: float,
        occupancy_fov_rad: float,
) -> np.ndarray:
    """Density of other agents in a cone from each agent toward each target. Shape (N, E)."""
    n = len(pos)
    e = targets.shape[1]
    if n == 0 or e == 0:
        return np.zeros((n, e))

    delta = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    in_range = (dist > EPSILON) & (dist <= occupancy_range)

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
        sigma: np.ndarray,
        has_belief: np.ndarray,
        params: ToM0Params,
) -> np.ndarray:
    u = uncertainty(sigma, params.sigma_ref)
    d_hat = minmax_masked(distance, has_belief)
    o_hat = minmax_masked(occupancy, has_belief)
    u_hat = minmax_masked(u, has_belief)
    score = (
        params.w_distance * d_hat
        + params.w_occupancy * o_hat
        + params.w_uncertainty * u_hat
    )
    return np.where(has_belief, score, -np.inf)


def headings_from_beliefs(
        pos: np.ndarray,
        mu: np.ndarray,
        has_belief: np.ndarray,
        sigma: np.ndarray,
        params: ToM0Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heading, chosen_exit, occupancy). chosen_exit is -1 if no belief."""
    n = len(pos)
    e = has_belief.shape[1] if has_belief.ndim == 2 else 0
    if n == 0 or e == 0:
        return np.zeros((n, 2)), np.full(n, -1, dtype=int), np.zeros((n, e))

    occupancy = cone_density(pos, mu, params.occupancy_range, params.occupancy_fov_rad)
    distance = np.linalg.norm(mu - pos[:, None, :], axis=2)
    score = utilities(distance, occupancy, sigma, has_belief, params)
    chosen = np.argmax(score, axis=1)
    none = ~np.any(has_belief, axis=1)
    chosen = np.where(none, -1, chosen)

    gathered_mu = mu[np.arange(n), np.clip(chosen, 0, e - 1)]
    heading = unit_rows(gathered_mu - pos)
    heading[none] = 0.0
    return heading, chosen, occupancy
