from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from obstacle import Circle, Wall, EPSILON

if TYPE_CHECKING:
    from obstacle import Obstacle


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.maximum(norms, EPSILON)
    out = vectors / safe
    out[norms[:, 0] < EPSILON] = 0.0
    return out


def rays_hit_segment(
        origins: np.ndarray,
        targets: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
) -> np.ndarray:
    movement = targets - origins
    segment = end - start
    denominator = movement[:, 0] * segment[1] - movement[:, 1] * segment[0]
    parallel = np.abs(denominator) < EPSILON
    safe = np.where(parallel, 1.0, denominator)

    to_start = start - origins
    ray_fraction = (to_start[:, 0] * segment[1] - to_start[:, 1] * segment[0]) / safe
    seg_fraction = (to_start[:, 0] * movement[:, 1] - to_start[:, 1] * movement[:, 0]) / safe

    return (
        ~parallel
        & (ray_fraction > EPSILON)
        & (ray_fraction < 1.0 - EPSILON)
        & (seg_fraction >= 0.0)
        & (seg_fraction <= 1.0)
    )


def rays_hit_circle(
        origins: np.ndarray,
        targets: np.ndarray,
        center: np.ndarray,
        radius: float,
) -> np.ndarray:
    direction = targets - origins
    offset = origins - center
    a = np.sum(direction * direction, axis=1)
    b = 2.0 * np.sum(offset * direction, axis=1)
    c = np.sum(offset * offset, axis=1) - radius * radius
    discriminant = b * b - 4.0 * a * c
    no_hit = (a < EPSILON) | (discriminant < 0.0)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    safe_a = np.where(a < EPSILON, 1.0, a)
    t1 = (-b - sqrt_disc) / (2.0 * safe_a)
    t2 = (-b + sqrt_disc) / (2.0 * safe_a)
    hit = ((t1 > EPSILON) & (t1 < 1.0 - EPSILON)) | ((t2 > EPSILON) & (t2 < 1.0 - EPSILON))
    return hit & ~no_hit


def line_of_sight_blocked(
        origins: np.ndarray,
        targets: np.ndarray,
        obstacles: list[Obstacle],
) -> np.ndarray:
    blocked = np.zeros(len(origins), dtype=bool)
    for obstacle in obstacles:
        if isinstance(obstacle, Wall):
            blocked |= rays_hit_segment(origins, targets, obstacle.start, obstacle.end)
        elif isinstance(obstacle, Circle):
            blocked |= rays_hit_circle(origins, targets, obstacle.center, obstacle.radius)
    return blocked


def visible_exit_mask(
        positions: np.ndarray,
        headings: np.ndarray,
        targets: np.ndarray,
        obstacles: list[Obstacle],
        view_range: float,
        fov_rad: float,
) -> np.ndarray:
    """Return (N,) booleans: whether each agent can see the corresponding target."""
    displacement = targets - positions
    distance = np.linalg.norm(displacement, axis=1)
    direction = unit_rows(displacement)
    heading = unit_rows(headings)
    in_range = (distance > EPSILON) & (distance <= view_range)
    in_fov = True if fov_rad >= 2.0 * np.pi - 1e-9 else np.sum(heading * direction, axis=1) >= np.cos(0.5 * fov_rad)
    blocked = line_of_sight_blocked(positions, targets, obstacles)
    return in_range & in_fov & ~blocked
