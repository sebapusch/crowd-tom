from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent import Agent
    from obstacle import Obstacle

from exit import Exit
from perception import unit_rows, visible_exit_mask


# Newtons
F_CUT = 1
COMPRESSION_COEFF = 1.2e5
SLIDING_COEFF = 2.4e5
MIN_DIST = 1e-12


class Environment:
    """
    Agents must be passed in constructor as visibility size is computed once in __init__
    """
    def __init__(
            self,
            width: int,
            height: int,
            agents: list[Agent],
            view_range: float = 40.0,
            fov_rad: float = 2.0 * np.pi,
            wander_turn: float = 1.5,
            speed_limit_factor: float = 1.5,
    ) -> None:
        self.width = width
        self.height = height
        self.agents: list[Agent] = agents
        self.exits: list[Exit] = []
        self.obstacles: list[Obstacle] = []
        self.view_range = view_range
        self.fov_rad = fov_rad
        self.wander_turn = wander_turn
        self.speed_limit_factor = speed_limit_factor
        self._rng = np.random.default_rng()
        self._grid = {}
        self._cell_size = self._compute_visibility()
        self._seen_exits = np.zeros((len(agents), 0), dtype=bool)
        self._pack_agents()

    def add_exit(self, exit_: Exit) -> None:
        self.exits.append(exit_)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def tick(self, dt: float) -> None:
        n = len(self.agents)
        if n == 0:
            return

        pos = self._pos
        vel = self._vel
        radii = self._radii
        masses = self._masses
        taus = self._taus
        desired_speeds = self._desired_speeds
        A = self._A
        B = self._B

        desired_direction = self._desired_directions(pos, dt)
        np.copyto(self._heading, desired_direction)
        f_drive = masses[:, None] * (desired_speeds[:, None] * desired_direction - vel) / taus[:, None]
        f_other = self._social_forces(pos, vel, radii, A, B)
        f_obst = self._obstacle_forces(pos, vel, radii, A, B)

        acceleration = (f_drive + f_other + f_obst) / masses[:, None]
        previous_position = pos.copy()
        vel += acceleration * dt
        self._clip_speeds(vel, desired_speeds)
        pos += vel * dt

        crossed = np.zeros(n, dtype=bool)
        for exit_position in self.exits:
            crossed |= exit_position.is_crossed_many(previous_position, pos)

        if not np.any(crossed):
            return

        keep = ~crossed
        self.agents = [agent for agent, stayed in zip(self.agents, keep) if stayed]
        self._pack_agents()

    def _clip_speeds(self, vel: np.ndarray, desired_speeds: np.ndarray) -> None:
        speed = np.linalg.norm(vel, axis=1)
        vmax = self.speed_limit_factor * desired_speeds
        too_fast = speed > vmax
        if np.any(too_fast):
            vel[too_fast] *= (vmax[too_fast] / np.maximum(speed[too_fast], MIN_DIST))[:, None]

    def get_visible_agents(self, agent: Agent) -> list[Agent]:
        """
        Includes self
        """
        x, y = self._cell_key(agent)
        agents = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                k = x + dx, y + dy
                if k in self._grid:
                    agents.extend(self._grid[k])

        return agents

    def _pack_agents(self) -> None:
        n = len(self.agents)
        if n == 0:
            self._pos = np.empty((0, 2), dtype=float)
            self._vel = np.empty((0, 2), dtype=float)
            self._radii = np.empty(0, dtype=float)
            self._masses = np.empty(0, dtype=float)
            self._taus = np.empty(0, dtype=float)
            self._desired_speeds = np.empty(0, dtype=float)
            self._A = np.empty(0, dtype=float)
            self._B = np.empty(0, dtype=float)
            self._heading = np.empty((0, 2), dtype=float)
            self._seen_exits = np.zeros((0, len(self.exits)), dtype=bool)
            return
        self._pos = np.array([agent.position for agent in self.agents], dtype=float)
        self._vel = np.array([agent.velocity for agent in self.agents], dtype=float)
        self._radii = np.array([agent.radius for agent in self.agents], dtype=float)
        self._masses = np.array([agent.mass for agent in self.agents], dtype=float)
        self._taus = np.array([agent.tau for agent in self.agents], dtype=float)
        self._desired_speeds = np.array([agent.desired_speed for agent in self.agents], dtype=float)
        repulsion = np.array([agent.social_repulsion for agent in self.agents], dtype=float)
        self._A = repulsion[:, 0]
        self._B = repulsion[:, 1]
        headings = np.array([agent.desired_direction for agent in self.agents], dtype=float)
        heading_norm = np.linalg.norm(headings, axis=1, keepdims=True)
        bad = heading_norm[:, 0] < MIN_DIST
        headings = headings / np.maximum(heading_norm, MIN_DIST)
        headings[bad] = np.array([1.0, 0.0])
        self._heading = headings
        for i, agent in enumerate(self.agents):
            agent.position = self._pos[i]
            agent.velocity = self._vel[i]
            agent.desired_direction = self._heading[i]
        self._seen_exits = np.zeros((n, len(self.exits)), dtype=bool)

    def _desired_directions(self, pos: np.ndarray, dt: float) -> np.ndarray:
        n = len(pos)
        if not self.exits:
            return self._heading.copy()

        look = self._heading
        best_distance = np.full(n, np.inf)
        best_closest = np.zeros((n, 2))
        seen_any = np.zeros(n, dtype=bool)
        seen_exits = np.zeros((n, len(self.exits)), dtype=bool)

        for j, exit_position in enumerate(self.exits):
            query = exit_position.distances(pos)
            seen = visible_exit_mask(
                pos,
                look,
                query.closest_point,
                self.obstacles,
                self.view_range,
                self.fov_rad,
            )
            seen_exits[:, j] = seen
            closer = seen & (query.distance < best_distance)
            best_distance = np.where(closer, query.distance, best_distance)
            best_closest = np.where(closer[:, None], query.closest_point, best_closest)
            seen_any |= seen

        self._seen_exits = seen_exits

        displacement = best_closest - pos
        toward_exit = unit_rows(displacement)
        fallback = self._follow_or_wander(pos, seen_any, dt)
        return np.where(seen_any[:, None], toward_exit, fallback)

    def _follow_or_wander(self, pos: np.ndarray, informed: np.ndarray, dt: float) -> np.ndarray:
        n = len(pos)
        turn = self._rng.normal(0.0, self.wander_turn * dt, size=n)
        wander = self._rotate_headings(self._heading, turn)
        if n < 2 or not np.any(informed):
            return wander

        offset = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(offset, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist[:, ~informed] = np.inf
        nearest = np.argmin(dist, axis=1)
        can_follow = np.isfinite(dist[np.arange(n), nearest])
        follow = unit_rows(pos[nearest] - pos)
        return np.where(can_follow[:, None], follow, wander)

    @staticmethod
    def _rotate_headings(headings: np.ndarray, angles: np.ndarray) -> np.ndarray:
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        hx, hy = headings[:, 0], headings[:, 1]
        return np.stack((cos_a * hx - sin_a * hy, sin_a * hx + cos_a * hy), axis=1)

    def _social_forces(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            radii: np.ndarray,
            A: np.ndarray,
            B: np.ndarray,
    ) -> np.ndarray:
        n = len(pos)
        if n < 2:
            return np.zeros((n, 2))

        displacement = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        combined_radius = radii[:, None] + radii[None, :]
        cutoff = combined_radius - B[:, None] * np.log(F_CUT / A[:, None])
        interacting = (distance >= MIN_DIST) & (distance < cutoff)
        np.fill_diagonal(interacting, False)

        distance_safe = np.maximum(distance, MIN_DIST)
        normal = displacement / distance_safe[:, :, None]
        tangent = np.empty_like(normal)
        tangent[:, :, 0] = -normal[:, :, 1]
        tangent[:, :, 1] = normal[:, :, 0]
        overlap = np.maximum(0.0, combined_radius - distance)

        magnitude = A[:, None] * np.exp((combined_radius - distance) / B[:, None])
        delta_v = np.sum((vel[None, :, :] - vel[:, None, :]) * tangent, axis=2)
        force = (
            (magnitude + COMPRESSION_COEFF * overlap)[:, :, None] * normal
            + (SLIDING_COEFF * overlap * delta_v)[:, :, None] * tangent
        )
        force *= interacting[:, :, None]
        return force.sum(axis=1)

    def _obstacle_forces(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            radii: np.ndarray,
            A: np.ndarray,
            B: np.ndarray,
    ) -> np.ndarray:
        f_objects = np.zeros_like(pos)
        for obj in self.obstacles:
            query = obj.distances(pos)
            obj_tangent = np.stack((-query.normal[:, 1], query.normal[:, 0]), axis=1)
            overlap = np.maximum(0.0, radii - query.distance)
            force = (A * np.exp((radii - query.distance) / B))[:, None] * query.normal
            force += (COMPRESSION_COEFF * overlap)[:, None] * query.normal
            speed_along_wall = np.sum(vel * obj_tangent, axis=1)
            force -= (SLIDING_COEFF * overlap * speed_along_wall)[:, None] * obj_tangent
            f_objects += force
        return f_objects

    def _update_grid(self) -> None:
        self._grid = {}
        for agent in self.agents:
            key = self._cell_key(agent)

            if not key in self._grid:
                self._grid[key] = [agent]
            else:
                self._grid[key].append(agent)

    def _compute_visibility(self) -> float:
        if len(self.agents) < 2:
            raise ValueError('Invalid cell size found')

        max_radius = max(agent.radius for agent in self.agents)
        largest = float('-inf')
        for agent in self.agents:
            A, B = agent.social_repulsion
            dist = agent.radius + max_radius - B * log(F_CUT / A)
            if dist > largest:
                largest = dist

        return largest

    def _cell_key(self, agent: Agent) -> tuple:
        cell = agent.position // self._cell_size
        return cell[0], cell[1]
