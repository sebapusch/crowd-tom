from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from agent import Agent


EPSILON = 1e-12
Point = tuple[float, float] | np.ndarray


@dataclass
class Distance:
    distance: float
    normal: np.ndarray
    closest_point: np.ndarray


@dataclass
class Distances:
    distance: np.ndarray
    normal: np.ndarray
    closest_point: np.ndarray


def _as_positions(positions: np.ndarray) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, 2)
    return array


class Obstacle(ABC):
    @abstractmethod
    def distance_to(self, agent: Agent) -> Distance:
        """Return the surface distance, outward normal, and closest point."""
        ...

    @abstractmethod
    def distances(self, positions: np.ndarray) -> Distances:
        """Vectorized distance query for an (N, 2) array of positions."""
        ...


def _distance_from_batch(batch: Distances) -> Distance:
    return Distance(
        distance=float(batch.distance[0]),
        normal=batch.normal[0],
        closest_point=batch.closest_point[0],
    )


class Wall(Obstacle):
    def __init__(self, start: Point, end: Point) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def distance_to(self, agent: Agent) -> Distance:
        return _distance_from_batch(self.distances(agent.position))

    def distances(self, positions: np.ndarray) -> Distances:
        positions = _as_positions(positions)
        segment = self.end - self.start
        length_squared = float(np.dot(segment, segment))

        if length_squared < EPSILON:
            difference = positions - self.start
            distance = np.linalg.norm(difference, axis=1)
            normal = np.zeros_like(positions)
            degenerate = distance < EPSILON
            ok = ~degenerate
            if np.any(ok):
                normal[ok] = difference[ok] / distance[ok, None]
            normal[degenerate] = np.array([1.0, 0.0])
            closest_point = np.repeat(self.start.reshape(1, 2), len(positions), axis=0)
            distance = np.where(degenerate, 0.0, distance)
            return Distances(distance=distance, normal=normal, closest_point=closest_point)

        projection = np.dot(positions - self.start, segment) / length_squared
        projection = np.clip(projection, 0.0, 1.0)
        closest_point = self.start + projection[:, None] * segment
        difference = positions - closest_point
        distance = np.linalg.norm(difference, axis=1)

        normal = np.empty_like(positions)
        on_surface = distance < EPSILON
        ok = ~on_surface
        if np.any(ok):
            normal[ok] = difference[ok] / distance[ok, None]
        if np.any(on_surface):
            fallback = np.array([-segment[1], segment[0]])
            fallback /= np.linalg.norm(fallback)
            normal[on_surface] = fallback
            closest_point[on_surface] = positions[on_surface]
            distance[on_surface] = 0.0

        return Distances(distance=distance, normal=normal, closest_point=closest_point)


class Circle(Obstacle):
    def __init__(
        self,
        center: Point,
        radius: float,
    ) -> None:
        if radius < 0:
            raise ValueError("Circle radius cannot be negative")

        self.center = np.asarray(center, dtype=float)
        self.radius = radius

    def distance_to(self, agent: Agent) -> Distance:
        return _distance_from_batch(self.distances(agent.position))

    def distances(self, positions: np.ndarray) -> Distances:
        positions = _as_positions(positions)
        difference = positions - self.center
        center_distance = np.linalg.norm(difference, axis=1)

        normal = np.empty_like(positions)
        at_center = center_distance < EPSILON
        ok = ~at_center
        if np.any(ok):
            normal[ok] = difference[ok] / center_distance[ok, None]
        normal[at_center] = np.array([1.0, 0.0])

        distance = center_distance - self.radius
        closest_point = self.center + normal * self.radius

        return Distances(distance=distance, normal=normal, closest_point=closest_point)
