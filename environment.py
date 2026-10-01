from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent import Agent
    from obstacle import Obstacle

from exit import Exit
from perception import pairwise_visible_mask, unit_rows, visible_exit_mask
from tom0 import ToM0Params, headings_from_beliefs


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
            tom0: ToM0Params | None = None,
            tom_order: int = 0,
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
        self.tom0 = tom0 or ToM0Params()
        self.tom_order = tom_order
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._grid = {}
        self._cell_size = self._compute_visibility()
        self._seen_exits = np.zeros((len(agents), 0), dtype=bool)
        self._has_belief = np.zeros((len(agents), 0), dtype=bool)
        self._belief_t = np.zeros((len(agents), 0), dtype=float)
        self._belief_mu = np.zeros((len(agents), 0, 2), dtype=float)
        self._chosen_exit = np.full(len(agents), -1, dtype=int)
        self.initial_agent_count = len(agents)
        self.escaped_by_exit: list[int] = []
        self.evacuation_time: float | None = None
        self.peak_memory_guided = 0
        self.blind_committed_s = 0.0
        self.peak_commit_distance = 0.0
        self._pack_agents()

    def add_exit(self, exit_: Exit) -> None:
        self.exits.append(exit_)
        self.escaped_by_exit.append(0)
        self._ensure_beliefs()

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

        claimed = np.zeros(n, dtype=bool)
        for j, exit_position in enumerate(self.exits):
            hit = exit_position.is_crossed_many(previous_position, pos) & ~claimed
            if j < len(self.escaped_by_exit):
                self.escaped_by_exit[j] += int(hit.sum())
            claimed |= hit

        if not np.any(claimed):
            self._t += dt
            self._update_tom0_peaks(dt)
            return

        keep = ~claimed
        self.agents = [agent for agent, stayed in zip(self.agents, keep) if stayed]
        self._has_belief = self._has_belief[keep]
        self._belief_t = self._belief_t[keep]
        self._belief_mu = self._belief_mu[keep]
        self._chosen_exit = self._chosen_exit[keep]
        if self._seen_exits.shape[0] == keep.shape[0]:
            self._seen_exits = self._seen_exits[keep]
        self._pack_agents()
        self._t += dt
        if len(self.agents) == 0 and self.evacuation_time is None:
            self.evacuation_time = self._t
        self._update_tom0_peaks(dt)

    def _clip_speeds(self, vel: np.ndarray, desired_speeds: np.ndarray) -> None:
        speed = np.linalg.norm(vel, axis=1)
        vmax = self.speed_limit_factor * desired_speeds
        too_fast = speed > vmax
        if np.any(too_fast):
            vel[too_fast] *= (vmax[too_fast] / np.maximum(speed[too_fast], MIN_DIST))[:, None]

    def _update_tom0_peaks(self, dt: float) -> None:
        stats = self.tom0_metrics()
        self.peak_memory_guided = max(self.peak_memory_guided, stats['memory_guided'])
        self.peak_commit_distance = max(self.peak_commit_distance, stats['max_commit_distance'])
        self.blind_committed_s += stats['blind_committed'] * dt

    def tom0_metrics(self) -> dict:
        n = len(self.agents)
        e = len(self.exits)
        seeing = 0
        with_belief = 0
        memory_guided = 0
        blind_committed = 0
        no_belief = n
        mean_sigma = 0.0
        mean_age = 0.0
        max_commit_distance = 0.0
        if n > 0 and e > 0 and self._has_belief.shape == (n, e):
            seeing_any = self._seen_exits.any(axis=1) if self._seen_exits.shape == (n, e) else np.zeros(n, dtype=bool)
            seeing = int(seeing_any.sum())
            has_any = self._has_belief.any(axis=1)
            with_belief = int(has_any.sum())
            no_belief = n - with_belief
            blind_committed = int((~seeing_any & has_any).sum())
            valid = self._chosen_exit >= 0
            if np.any(valid) and self._seen_exits.shape == (n, e):
                rows = np.flatnonzero(valid)
                chosen = self._chosen_exit[rows]
                memory_guided = int((~self._seen_exits[rows, chosen]).sum())
                age = np.maximum(self._t - self._belief_t[rows, chosen], 0.0)
                sigma = self.tom0.sigma0 + self.tom0.sigma_alpha * np.sqrt(age)
                mean_age = float(age.mean())
                mean_sigma = float(sigma.mean())
                commit_dist = np.linalg.norm(self._belief_mu[rows, chosen] - self._pos[rows], axis=1)
                max_commit_distance = float(commit_dist.max())
        escaped = int(sum(self.escaped_by_exit))
        return {
            'n': n,
            'initial': self.initial_agent_count,
            'escaped': escaped,
            'escaped_by_exit': list(self.escaped_by_exit),
            'seeing': seeing,
            'with_belief': with_belief,
            'memory_guided': memory_guided,
            'blind_committed': blind_committed,
            'no_belief': no_belief,
            'mean_sigma': mean_sigma,
            'mean_age': mean_age,
            'max_commit_distance': max_commit_distance,
            'peak_memory_guided': self.peak_memory_guided,
            'peak_commit_distance': self.peak_commit_distance,
            'blind_committed_s': self.blind_committed_s,
            't': self._t,
            'evacuation_time': self.evacuation_time,
        }

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
            self._has_belief = np.zeros((0, len(self.exits)), dtype=bool)
            self._belief_t = np.zeros((0, len(self.exits)), dtype=float)
            self._belief_mu = np.zeros((0, len(self.exits), 2), dtype=float)
            self._chosen_exit = np.empty(0, dtype=int)
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
        if self._seen_exits.shape != (n, len(self.exits)):
            self._seen_exits = np.zeros((n, len(self.exits)), dtype=bool)
        self._ensure_beliefs()

    def _ensure_beliefs(self) -> None:
        n = len(self.agents)
        e = len(self.exits)
        old_has = getattr(self, '_has_belief', None)
        if old_has is not None and old_has.shape == (n, e):
            return

        has_belief = np.zeros((n, e), dtype=bool)
        belief_t = np.zeros((n, e), dtype=float)
        belief_mu = np.zeros((n, e, 2), dtype=float)
        if old_has is not None and old_has.size:
            n_copy = min(n, old_has.shape[0])
            e_copy = min(e, old_has.shape[1])
            has_belief[:n_copy, :e_copy] = old_has[:n_copy, :e_copy]
            belief_t[:n_copy, :e_copy] = self._belief_t[:n_copy, :e_copy]
            belief_mu[:n_copy, :e_copy] = self._belief_mu[:n_copy, :e_copy]
        self._has_belief = has_belief
        self._belief_t = belief_t
        self._belief_mu = belief_mu
        if getattr(self, '_chosen_exit', np.array([])).shape != (n,):
            self._chosen_exit = np.full(n, -1, dtype=int)

    def _desired_directions(self, pos: np.ndarray, dt: float) -> np.ndarray:
        n = len(pos)
        if not self.exits:
            return self._heading.copy()

        self._ensure_beliefs()
        params = self.tom0
        seen_exits = np.zeros((n, len(self.exits)), dtype=bool)

        for j, exit_position in enumerate(self.exits):
            query = exit_position.distances(pos)
            seen = visible_exit_mask(
                pos,
                self._heading,
                query.closest_point,
                self.obstacles,
                self.view_range,
                self.fov_rad,
            )
            seen_exits[:, j] = seen
            if np.any(seen) and self.tom_order >= 0:
                self._has_belief[seen, j] = True
                self._belief_t[seen, j] = self._t
                self._belief_mu[seen, j] = query.closest_point[seen]

        self._seen_exits = seen_exits
        visible_agents = pairwise_visible_mask(
            pos, self._heading, self.obstacles, self.view_range, self.fov_rad,
        )
        if self.tom_order < 0:
            best_distance = np.full(n, np.inf)
            best_closest = np.zeros((n, 2))
            seen_any = seen_exits.any(axis=1)
            for j, exit_position in enumerate(self.exits):
                query = exit_position.distances(pos)
                closer = seen_exits[:, j] & (query.distance < best_distance)
                best_distance = np.where(closer, query.distance, best_distance)
                best_closest = np.where(closer[:, None], query.closest_point, best_closest)
            toward_exit = unit_rows(best_closest - pos)
            fallback = self._follow_or_wander(pos, seen_any, dt, visible_agents)
            chosen = np.full(n, -1, dtype=int)
            for j, exit_position in enumerate(self.exits):
                query = exit_position.distances(pos)
                pick = seen_exits[:, j] & (query.distance <= best_distance + 1e-9)
                chosen = np.where(pick & (chosen < 0), j, chosen)
            self._chosen_exit = chosen
            return np.where(seen_any[:, None], toward_exit, fallback)

        age = np.maximum(self._t - self._belief_t, 0.0)
        sigma = params.sigma0 + params.sigma_alpha * np.sqrt(age)
        sigma = np.where(self._has_belief, sigma, np.inf)

        tom_heading, chosen, _occupancy = headings_from_beliefs(
            pos,
            self._belief_mu,
            self._has_belief,
            sigma,
            params,
            visible=visible_agents,
        )
        self._chosen_exit = chosen

        has_any_belief = chosen >= 0
        fallback = self._follow_or_wander(pos, has_any_belief, dt, visible_agents)
        return np.where(has_any_belief[:, None], tom_heading, fallback)

    def _follow_or_wander(
            self,
            pos: np.ndarray,
            informed: np.ndarray,
            dt: float,
            visible: np.ndarray,
    ) -> np.ndarray:
        n = len(pos)
        turn = self._rng.normal(0.0, self.wander_turn * dt, size=n)
        wander = self._rotate_headings(self._heading, turn)
        if n < 2 or not np.any(informed):
            return wander

        offset = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(offset, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist[~visible] = np.inf
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
