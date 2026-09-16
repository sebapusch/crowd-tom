from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from environment import Environment
from exit import Exit

if TYPE_CHECKING:
    from obstacle import Obstacle, Distance

COMPRESSION_COEFF = 1.2e5
SLIDING_COEFF = 2.4e5
MIN_DIST = 1e-12



class Agent:
    def __init__(self,
                 idx: int,
                 mass: float,
                 radius: float,
                 social_repulsion: tuple[float, float],
                 position: np.ndarray,
                 velocity: np.ndarray,
                 desired_speed: np.ndarray,
                 desired_direction: np.ndarray,
                 tau: float):
        self.idx = idx
        self.mass = mass
        self.radius = radius
        self.social_repulsion = social_repulsion
        self.tau = tau
        self.position = position
        self.velocity = velocity
        self.desired_speed = desired_speed
        self.desired_direction = desired_direction
        self.acceleration = 0

    def tick_acceleration(self, env: Environment) -> None:
        self.update_desired(env)

        f_drive = self.mass * (self.desired_speed * self.desired_direction - self.velocity) / self.tau
        f_other = self.compute_social_force(env.agents)
        f_obst = self.compute_obstacle_force(env.obstacles)

        f = f_drive + f_other + f_obst

        self.acceleration = f / self.mass

    def tick_position(self, dt: float) -> None:
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def update_desired(self, env: Environment) -> None:
        min_distance = None

        for i, ex in enumerate(env.exits):
            distance = ex.distance_to(self)
            if min_distance is None or distance.distance > min_distance:
                min_distance = distance

        if min_distance is None: return

        displacement = min_distance.closest_point - self.position
        self.desired_direction = displacement / np.linalg.norm(displacement)

    def compute_social_force(self, others: list[Agent]) -> np.ndarray:
        f_others = np.zeros(2)

        for other in others:
            if other.idx == self.idx: continue

            displacement = self.position - other.position
            distance = np.linalg.norm(displacement)

            if distance < MIN_DIST: continue

            normal_ij = displacement / distance
            tangent_ij = np.array([-normal_ij[1], normal_ij[0]])

            combined_radius = self.radius + other.radius
            overlap = max(0.0, combined_radius - distance)

            A, B = self.social_repulsion

            force_ij = A * np.exp((combined_radius - distance) / B) * normal_ij

            if overlap > 0.0:
                force_ij += COMPRESSION_COEFF * overlap * normal_ij

                delta_v = np.dot(other.velocity - self.velocity, tangent_ij)

                force_ij += SLIDING_COEFF * overlap * delta_v * tangent_ij

            f_others += force_ij

        return f_others

    def compute_obstacle_force(self, objects: list[Obstacle]) -> np.ndarray:
        f_objects = np.zeros(2)

        for obj in objects:
            obstacle_distance = obj.distance_to(self)
            distance = obstacle_distance.distance
            obj_normal = obstacle_distance.normal
            obj_tangent = np.array([-obj_normal[1], obj_normal[0]])
            overlap = max(0.0, self.radius - distance)

            A, B = self.social_repulsion

            f = A * np.exp((self.radius - distance) / B) * obj_normal

            if overlap > 0.0:
                f += COMPRESSION_COEFF * overlap * obj_normal
                speed_along_wall = np.dot(self.velocity, obj_tangent)
                f -= SLIDING_COEFF * overlap * speed_along_wall * obj_tangent

            f_objects += f

        return f_objects
