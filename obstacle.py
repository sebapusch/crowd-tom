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



class Obstacle(ABC):
    @abstractmethod
    def distance_to(self, agent: Agent) -> Distance:
        """Return the surface distance, outward normal, and closest point."""
        ...


class Wall(Obstacle):
    def __init__(self, start: Point, end: Point) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def distance_to(self, agent: Agent) -> Distance:
        segment = self.end - self.start
        length_squared = np.dot(segment, segment)

        if length_squared < EPSILON:
            difference = agent.position - self.start
            distance = np.linalg.norm(difference)

            if distance < EPSILON:
                return Distance(
                    distance=0.0,
                    normal=np.array([1.0, 0.0]),
                    closest_point=self.start.copy(),
                )

            return Distance(
                distance=distance,
                normal=difference / distance,
                closest_point=self.start.copy(),
            )

        # Project the agent onto the infinite line.
        projection = (
            np.dot(agent.position - self.start, segment)
            / length_squared
        )

        # Restrict the projection to the finite segment.
        projection = np.clip(projection, 0.0, 1.0)

        closest_point = self.start + projection * segment
        difference = agent.position - closest_point
        distance = np.linalg.norm(difference)

        if distance < EPSILON:
            normal = np.array([-segment[1], segment[0]])
            normal /= np.linalg.norm(normal)

            return Distance(
                distance=0.0,
                normal=normal,
                closest_point=agent.position,
            )

        normal = difference / distance

        return Distance(
            distance=distance,
            normal=normal,
            closest_point=closest_point,
        )


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
        difference = agent.position - self.center
        center_distance = np.linalg.norm(difference)

        if center_distance < EPSILON:
            normal = np.array([1.0, 0.0])
        else:
            normal = difference / center_distance

        distance = center_distance - self.radius
        closest_point = self.center + normal * self.radius

        return Distance(
            distance=distance,
            normal=normal,
            closest_point=closest_point,
        )
