from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent import Agent
    from obstacle import Obstacle

from exit import Exit
from perception import pairwise_visible_mask, unit_rows, visible_exit_mask
from tom0 import (
    ToM0Params, distance_penalty, headings_from_beliefs, occupancy_penalty,
    uncertainty, utilities,
)
from tom1 import ToM1Params, ToM1Reasoner, demand_penalty


# Newtons
F_CUT = 1
COMPRESSION_COEFF = 1.2e5
SLIDING_COEFF = 2.4e5
MIN_DIST = 1e-12
DEFAULT_TEST_SEED = 1000
AGENT_TYPES = ('reactive', 'tom0', 'tom1')


def _type_masks(orders: np.ndarray) -> dict[str, np.ndarray]:
    return {
        'reactive': orders < 0,
        'tom0': (orders >= 0) & (orders != 1),
        'tom1': orders == 1,
    }


@dataclass(frozen=True)
class ExitSign:
    position: tuple[float, float]
    exit_index: int


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
            tom1: ToM1Params | None = None,
            tom_order: int | None = 0,
            rng: np.random.Generator | None = None,
    ) -> None:
        if not np.isfinite(view_range) or view_range <= 0:
            raise ValueError('view_range must be positive and finite')
        self.width = width
        self.height = height
        self.agents: list[Agent] = agents
        self.exits: list[Exit] = []
        self.signs: list[ExitSign] = []
        self.obstacles: list[Obstacle] = []
        self.view_range = view_range
        self.fov_rad = fov_rad
        self.wander_turn = wander_turn
        self.speed_limit_factor = speed_limit_factor
        self.tom0 = replace(tom0 or ToM0Params(), occupancy_range=view_range)
        self.tom1 = ToM1Reasoner(replace(tom1 or ToM1Params(), model_range=view_range))
        self.tom_order = tom_order
        self.has_assigned_tom_orders = any(agent.tom_order is not None for agent in agents)
        self._rng = rng if rng is not None else np.random.default_rng(DEFAULT_TEST_SEED)
        self._t = 0.0
        self._grid = {}
        self._cell_size = self._compute_visibility()
        self._seen_exits = np.zeros((len(agents), 0), dtype=bool)
        self._signed_exits = np.zeros((len(agents), 0), dtype=bool)
        self._has_belief = np.zeros((len(agents), 0), dtype=bool)
        self._belief_t = np.zeros((len(agents), 0), dtype=float)
        self._belief_mu = np.zeros((len(agents), 0, 2), dtype=float)
        self._chosen_exit = np.full(len(agents), -1, dtype=int)
        self._tom1_changed = np.zeros(len(agents), dtype=bool)
        self._following = np.zeros(len(agents), dtype=bool)
        self._wandering = np.zeros(len(agents), dtype=bool)
        self._switches_step_by_type = {agent_type: 0 for agent_type in AGENT_TYPES}
        self._switches_total_by_type = {agent_type: 0 for agent_type in AGENT_TYPES}
        self._agent_seconds_by_type = {agent_type: 0.0 for agent_type in AGENT_TYPES}
        self.score_trace_enabled = False
        self.score_trace_agent_id: int | None = None
        self.score_trace_rows: list[dict] = []
        self.initial_agent_count = len(agents)
        self.escaped_by_exit: list[int] = []
        self.evacuation_time: float | None = None
        self.evacuation_times: dict[int, float] = {}
        self.escaped_by_type = {agent_type: 0 for agent_type in AGENT_TYPES}
        self.escaped_by_exit_by_type = {agent_type: [] for agent_type in AGENT_TYPES}
        self._departed_exit_step_by_type = {agent_type: [] for agent_type in AGENT_TYPES}
        self.peak_memory_guided = 0
        self.blind_committed_s = 0.0
        self.peak_commit_distance = 0.0
        self._pack_agents()
        initial_masks = _type_masks(self._agent_tom_orders if self.has_assigned_tom_orders else self._effective_orders())
        self._initial_by_type = {agent_type: int(mask.sum()) for agent_type, mask in initial_masks.items()}
        self._peak_memory_guided_by_type = {agent_type: 0 for agent_type in AGENT_TYPES}
        self._peak_commit_distance_by_type = {agent_type: 0.0 for agent_type in AGENT_TYPES}
        self._blind_committed_s_by_type = {agent_type: 0.0 for agent_type in AGENT_TYPES}
        self._evacuation_time_by_type = {agent_type: None for agent_type in AGENT_TYPES}

    def add_exit(self, exit_: Exit) -> None:
        self.exits.append(exit_)
        self.escaped_by_exit.append(0)
        for counts in self.escaped_by_exit_by_type.values():
            counts.append(0)
        for counts in self._departed_exit_step_by_type.values():
            counts.append(0)
        self._ensure_beliefs()

    def add_sign(self, sign: ExitSign) -> None:
        if not 0 <= sign.exit_index < len(self.exits):
            raise ValueError(f'Sign refers to missing exit {sign.exit_index}')
        self.signs.append(sign)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def tick(self, dt: float) -> None:
        n = len(self.agents)
        if n == 0:
            return
        self._departed_exit_step_by_type = {
            agent_type: [0] * len(self.exits) for agent_type in AGENT_TYPES
        }

        pos = self._pos
        vel = self._vel
        radii = self._radii
        masses = self._masses
        taus = self._taus
        desired_speeds = self._desired_speeds
        A = self._A
        B = self._B

        desired_direction = self._desired_directions(pos, dt)
        metric_orders = self._agent_tom_orders if self.has_assigned_tom_orders else self._effective_orders()
        for agent_type, mask in _type_masks(metric_orders).items():
            self._switches_total_by_type[agent_type] += self._switches_step_by_type[agent_type]
            self._agent_seconds_by_type[agent_type] += int(mask.sum()) * dt
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
        metric_orders = self._agent_tom_orders if self.has_assigned_tom_orders else self._effective_orders()
        type_masks = _type_masks(metric_orders)
        for j, exit_position in enumerate(self.exits):
            hit = exit_position.is_crossed_many(previous_position, pos) & ~claimed
            crossing_time = self._t + dt
            for idx in np.flatnonzero(hit):
                agent = self.agents[idx]
                if agent.idx not in self.evacuation_times:
                    self.evacuation_times[agent.idx] = crossing_time
            if j < len(self.escaped_by_exit):
                self.escaped_by_exit[j] += int(hit.sum())
                for agent_type, mask in type_masks.items():
                    count = int((hit & mask).sum())
                    self.escaped_by_type[agent_type] += count
                    self.escaped_by_exit_by_type[agent_type][j] += count
                    self._departed_exit_step_by_type[agent_type][j] += count
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
        self._tom1_changed = self._tom1_changed[keep]
        self._following = self._following[keep]
        self._wandering = self._wandering[keep]
        if self._seen_exits.shape[0] == keep.shape[0]:
            self._seen_exits = self._seen_exits[keep]
        if self._signed_exits.shape[0] == keep.shape[0]:
            self._signed_exits = self._signed_exits[keep]
        self._pack_agents()
        self._t += dt
        for agent_type in AGENT_TYPES:
            if (self._initial_by_type[agent_type] > 0
                    and self.escaped_by_type[agent_type] == self._initial_by_type[agent_type]
                    and self._evacuation_time_by_type[agent_type] is None):
                self._evacuation_time_by_type[agent_type] = self._t
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
        for agent_type, values in stats['by_type'].items():
            self._peak_memory_guided_by_type[agent_type] = max(
                self._peak_memory_guided_by_type[agent_type], values['memory_guided'],
            )
            self._peak_commit_distance_by_type[agent_type] = max(
                self._peak_commit_distance_by_type[agent_type], values['max_commit_distance'],
            )
            self._blind_committed_s_by_type[agent_type] += values['blind_committed'] * dt
            values['peak_memory_guided'] = self._peak_memory_guided_by_type[agent_type]
            values['peak_commit_distance'] = self._peak_commit_distance_by_type[agent_type]
            values['blind_committed_s'] = self._blind_committed_s_by_type[agent_type]
        stats['peak_memory_guided'] = self.peak_memory_guided
        stats['peak_commit_distance'] = self.peak_commit_distance
        stats['blind_committed_s'] = self.blind_committed_s
        self._last_metrics = stats

    def tom0_metrics(self) -> dict:
        n = len(self.agents)
        orders = self._agent_tom_orders if self.has_assigned_tom_orders else self._effective_orders()
        type_masks = _type_masks(orders)
        e = len(self.exits)
        seeing_exit = 0
        seeing_sign = 0
        with_belief = 0
        memory_guided = 0
        blind_committed = 0
        no_belief = n
        mean_uncertainty = 0.0
        mean_age = 0.0
        max_commit_distance = 0.0
        by_type = {
            agent_type: {
                'remaining': int(mask.sum()),
                'escaped': self.escaped_by_type[agent_type],
                'escaped_by_exit': list(self.escaped_by_exit_by_type[agent_type]),
                'departed_exits_step': list(self._departed_exit_step_by_type[agent_type]),
                'seeing_exit': 0, 'seeing_sign': 0, 'with_belief': 0,
                'memory_guided': 0, 'no_belief': int(mask.sum()),
                'tom1_changed': int((self._tom1_changed & mask).sum()),
                'following': int((self._following & mask).sum()),
                'wandering': int((self._wandering & mask).sum()),
                'selected_exits': [int(((self._chosen_exit == j) & mask).sum()) for j in range(e)],
                'switches_step': self._switches_step_by_type[agent_type],
                'switches_total': self._switches_total_by_type[agent_type],
                'agent_seconds': self._agent_seconds_by_type[agent_type],
                'switch_rate': (
                    self._switches_total_by_type[agent_type] / self._agent_seconds_by_type[agent_type]
                    if self._agent_seconds_by_type[agent_type] > 0 else 0.0
                ),
                'blind_committed': 0, 'mean_uncertainty': 0.0,
                'mean_age': 0.0, 'max_commit_distance': 0.0,
                'peak_memory_guided': self._peak_memory_guided_by_type[agent_type],
                'peak_commit_distance': self._peak_commit_distance_by_type[agent_type],
                'blind_committed_s': self._blind_committed_s_by_type[agent_type],
                'evacuation_time': self._evacuation_time_by_type[agent_type],
            }
            for agent_type, mask in type_masks.items()
        }
        for values in by_type.values():
            values['initial'] = values['remaining'] + values['escaped']
        if n > 0 and e > 0 and self._has_belief.shape == (n, e):
            seen_exits = self._seen_exits if self._seen_exits.shape == (n, e) else np.zeros((n, e), dtype=bool)
            signed_exits = self._signed_exits if self._signed_exits.shape == (n, e) else np.zeros((n, e), dtype=bool)
            current_cue = seen_exits | signed_exits
            has_any = self._has_belief.any(axis=1)
            with_belief = int(has_any.sum())
            no_belief = n - with_belief
            blind_committed = int((~current_cue.any(axis=1) & has_any).sum())
            valid = (self._chosen_exit >= 0) & (self._chosen_exit < e)
            for agent_type, mask in type_masks.items():
                values = by_type[agent_type]
                values['with_belief'] = int((has_any & mask).sum())
                values['no_belief'] = values['remaining'] - values['with_belief']
                values['blind_committed'] = int((~current_cue.any(axis=1) & has_any & mask).sum())
            if np.any(valid):
                rows = np.flatnonzero(valid)
                chosen = self._chosen_exit[rows]
                directly_seen = seen_exits[rows, chosen]
                sign_seen = signed_exits[rows, chosen] & ~directly_seen
                seeing_exit = int(directly_seen.sum())
                seeing_sign = int(sign_seen.sum())
                memory_guided = int((~directly_seen & ~sign_seen).sum())
                age = np.maximum(self._t - self._belief_t[rows, chosen], 0.0)
                mean_age = float(age.mean())
                mean_uncertainty = float(uncertainty(age, self.tom0.uncertainty_time).mean())
                commit_dist = np.linalg.norm(self._belief_mu[rows, chosen] - self._pos[rows], axis=1)
                max_commit_distance = float(commit_dist.max())
                for agent_type, mask in type_masks.items():
                    selected = mask[rows]
                    values = by_type[agent_type]
                    values['seeing_exit'] = int((directly_seen & selected).sum())
                    values['seeing_sign'] = int((sign_seen & selected).sum())
                    values['memory_guided'] = int((~directly_seen & ~sign_seen & selected).sum())
                    if np.any(selected):
                        values['mean_age'] = float(age[selected].mean())
                        values['mean_uncertainty'] = float(uncertainty(age[selected], self.tom0.uncertainty_time).mean())
                        values['max_commit_distance'] = float(commit_dist[selected].max())
        escaped = int(sum(self.escaped_by_exit))
        return {
            'n': n,
            'remaining_tom0': int(((orders >= 0) & (orders != 1)).sum()),
            'remaining_tom1': int((orders == 1).sum()),
            'remaining_reactive': int((orders < 0).sum()),
            'initial': self.initial_agent_count,
            'escaped': escaped,
            'escaped_by_exit': list(self.escaped_by_exit),
            'departed_exits_step': [
                sum(self._departed_exit_step_by_type[agent_type][j] for agent_type in AGENT_TYPES)
                for j in range(e)
            ],
            'seeing_exit': seeing_exit,
            'seeing_sign': seeing_sign,
            'with_belief': with_belief,
            'memory_guided': memory_guided,
            'tom1_changed': int(self._tom1_changed.sum()),
            'following': int(self._following.sum()),
            'wandering': int(self._wandering.sum()),
            'selected_exits': [int((self._chosen_exit == j).sum()) for j in range(e)],
            'switches_step': sum(self._switches_step_by_type.values()),
            'switches_total': sum(self._switches_total_by_type.values()),
            'agent_seconds': sum(self._agent_seconds_by_type.values()),
            'switch_rate': (
                sum(self._switches_total_by_type.values()) / sum(self._agent_seconds_by_type.values())
                if sum(self._agent_seconds_by_type.values()) > 0 else 0.0
            ),
            'blind_committed': blind_committed,
            'no_belief': no_belief,
            'mean_uncertainty': mean_uncertainty,
            'mean_age': mean_age,
            'max_commit_distance': max_commit_distance,
            'peak_memory_guided': self.peak_memory_guided,
            'peak_commit_distance': self.peak_commit_distance,
            'blind_committed_s': self.blind_committed_s,
            't': self._t,
            'evacuation_time': self.evacuation_time,
            'by_type': by_type,
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
            self.tom1.memories.clear()
            self._pos = np.empty((0, 2), dtype=float)
            self._vel = np.empty((0, 2), dtype=float)
            self._radii = np.empty(0, dtype=float)
            self._masses = np.empty(0, dtype=float)
            self._taus = np.empty(0, dtype=float)
            self._desired_speeds = np.empty(0, dtype=float)
            self._A = np.empty(0, dtype=float)
            self._B = np.empty(0, dtype=float)
            self._heading = np.empty((0, 2), dtype=float)
            self._agent_tom_orders = np.empty(0, dtype=int)
            self._seen_exits = np.zeros((0, len(self.exits)), dtype=bool)
            self._signed_exits = np.zeros((0, len(self.exits)), dtype=bool)
            self._has_belief = np.zeros((0, len(self.exits)), dtype=bool)
            self._belief_t = np.zeros((0, len(self.exits)), dtype=float)
            self._belief_mu = np.zeros((0, len(self.exits), 2), dtype=float)
            self._chosen_exit = np.empty(0, dtype=int)
            self._tom1_changed = np.empty(0, dtype=bool)
            self._following = np.empty(0, dtype=bool)
            self._wandering = np.empty(0, dtype=bool)
            return
        self._pos = np.array([agent.position for agent in self.agents], dtype=float)
        self._agent_tom_orders = np.array([
            agent.tom_order if agent.tom_order is not None else 0
            for agent in self.agents
        ], dtype=int)
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
        if self._signed_exits.shape != (n, len(self.exits)):
            self._signed_exits = np.zeros((n, len(self.exits)), dtype=bool)
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

    def _effective_orders(self) -> np.ndarray:
        if self.tom_order is None:
            return self._agent_tom_orders
        return np.full(len(self.agents), self.tom_order, dtype=int)

    def _desired_directions(self, pos: np.ndarray, dt: float) -> np.ndarray:
        n = len(pos)
        if not self.exits:
            self._following = np.zeros(n, dtype=bool)
            self._wandering = np.zeros(n, dtype=bool)
            self._switches_step_by_type = {agent_type: 0 for agent_type in AGENT_TYPES}
            return self._heading.copy()

        self._ensure_beliefs()
        previous_choice = self._chosen_exit.copy()
        params = self.tom0
        orders = self._effective_orders()
        seen_exits = np.zeros((n, len(self.exits)), dtype=bool)
        signed_exits = np.zeros_like(seen_exits)

        for sign in self.signs:
            sign_target = np.broadcast_to(np.asarray(sign.position, dtype=float), pos.shape)
            visible = visible_exit_mask(
                pos, self._heading, sign_target, self.obstacles,
                self.view_range, self.fov_rad,
            )
            signed_exits[:, sign.exit_index] |= visible

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
            remember = (seen | signed_exits[:, j]) & (orders >= 0)
            if np.any(remember):
                self._has_belief[remember, j] = True
                self._belief_t[remember, j] = self._t
                self._belief_mu[remember, j] = query.closest_point[remember]

        self._seen_exits = seen_exits
        self._signed_exits = signed_exits
        self._tom1_changed = np.zeros(n, dtype=bool)
        visible_agents = pairwise_visible_mask(
            pos, self._heading, self.obstacles, self.view_range, self.fov_rad,
        )
        reactive = orders < 0
        available_exits = seen_exits | signed_exits
        available_any = available_exits.any(axis=1)
        direction = np.zeros((n, 2), dtype=float)
        chosen = np.full(n, -1, dtype=int)
        personal = np.zeros(n, dtype=bool)
        if np.any(reactive):
            best_distance = np.full(n, np.inf)
            best_closest = np.zeros((n, 2))
            for j, exit_position in enumerate(self.exits):
                query = exit_position.distances(pos)
                closer = available_exits[:, j] & (query.distance < best_distance)
                best_distance = np.where(closer, query.distance, best_distance)
                best_closest = np.where(closer[:, None], query.closest_point, best_closest)
                chosen = np.where(closer & reactive, j, chosen)
            use_reactive = reactive & available_any
            direction[use_reactive] = unit_rows(best_closest[use_reactive] - pos[use_reactive])
            personal |= use_reactive

        if np.any(~reactive):
            age = np.maximum(self._t - self._belief_t, 0.0)
            tom_heading, tom_chosen, occupancy = headings_from_beliefs(
                pos, self._belief_mu, self._has_belief, age, params,
                visible=visible_agents,
            )
            use_tom0 = (orders != 1) & ~reactive & (tom_chosen >= 0)
            direction[use_tom0] = tom_heading[use_tom0]
            chosen[use_tom0] = tom_chosen[use_tom0]
            personal |= use_tom0
        if np.any(orders == 1):
            distance = np.linalg.norm(self._belief_mu - pos[:, None, :], axis=2)
            base_scores = utilities(
                distance, occupancy, age, self._has_belief, params,
            )
            agent_ids = np.array([agent.idx for agent in self.agents], dtype=int)
            tom_heading, tom1_chosen, demand = self.tom1.choose(
                ids=agent_ids,
                pos=pos,
                vel=self._vel,
                mu=self._belief_mu,
                has_belief=self._has_belief,
                base_scores=base_scores,
                visible_agents=visible_agents,
                obstacles=self.obstacles,
                view_range=self.view_range,
                fov_rad=self.fov_rad,
                tom0=params,
                now=self._t,
                observer_mask=orders == 1,
                signs=self.signs,
            )
            use_tom1 = (orders == 1) & (tom1_chosen >= 0)
            direction[use_tom1] = tom_heading[use_tom1]
            chosen[use_tom1] = tom1_chosen[use_tom1]
            personal |= use_tom1
            self._tom1_changed = use_tom1 & (tom1_chosen != tom_chosen)
            if self.score_trace_enabled:
                self._record_score_trace(
                    agent_ids, orders, tom_chosen, tom1_chosen,
                    distance, occupancy, age, demand,
                )
        self._chosen_exit = chosen
        switched = (previous_choice >= 0) & (chosen >= 0) & (previous_choice != chosen)
        metric_orders = self._agent_tom_orders if self.has_assigned_tom_orders else orders
        self._switches_step_by_type = {
            agent_type: int((switched & mask).sum())
            for agent_type, mask in _type_masks(metric_orders).items()
        }

        fallback, can_follow = self._follow_or_wander(pos, dt, visible_agents)
        self._following = ~personal & can_follow
        self._wandering = ~personal & ~can_follow
        return np.where(personal[:, None], direction, fallback)

    def _record_score_trace(
            self, agent_ids: np.ndarray, orders: np.ndarray,
            tom0_chosen: np.ndarray, tom1_chosen: np.ndarray,
            distance: np.ndarray, occupancy: np.ndarray,
            age: np.ndarray, demand: np.ndarray,
    ) -> None:
        eligible = (
            (orders == 1) & (self._has_belief.sum(axis=1) >= 2)
            & (tom0_chosen >= 0) & (tom1_chosen >= 0)
        )
        candidates = np.flatnonzero(eligible)
        if len(candidates) == 0:
            return
        candidate = int(candidates[np.argmin(agent_ids[candidates])])
        candidate_id = int(agent_ids[candidate])
        if self.score_trace_agent_id is None or candidate_id < self.score_trace_agent_id:
            self.score_trace_agent_id = candidate_id
            self.score_trace_rows.clear()
        if not eligible[agent_ids == self.score_trace_agent_id].any():
            return
        row_index = int(np.flatnonzero(agent_ids == self.score_trace_agent_id)[0])
        selected_exit = int(tom1_chosen[row_index])
        tom0_exit = int(tom0_chosen[row_index])
        params = self.tom0
        demand_params = self.tom1.params

        def terms(exit_index: int) -> dict[str, float]:
            values = {
                'distance': float(params.w_distance * distance_penalty(
                    distance[row_index, exit_index], params.distance_scale,
                )),
                'occupancy': float(params.w_occupancy * occupancy_penalty(
                    occupancy[row_index, exit_index], params.occupancy_half_density,
                )),
                'uncertainty': float(params.w_uncertainty * uncertainty(
                    age[row_index, exit_index], params.uncertainty_time,
                )),
                'demand': float(demand_params.demand_weight * demand_penalty(
                    demand[row_index, exit_index], demand_params.demand_half_count,
                )),
            }
            values['base_total'] = values['distance'] + values['occupancy'] + values['uncertainty']
            values['total'] = values['base_total'] + values['demand']
            return values

        selected = terms(selected_exit)
        baseline = terms(tom0_exit)
        self.score_trace_rows.append({
            'status': 'decision', 't': self._t, 'agent_id': self.score_trace_agent_id,
            'selected_exit': selected_exit, 'tom0_exit': tom0_exit,
            **{f'selected_{key}': value for key, value in selected.items()},
            **{f'alternative_{key}': value for key, value in baseline.items()},
            **{f'delta_{key}': selected[key] - baseline[key] for key in selected},
        })

    def _follow_or_wander(
            self,
            pos: np.ndarray,
            dt: float,
            visible: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(pos)
        turn = self._rng.normal(0.0, self.wander_turn * dt, size=n)
        wander = self._rotate_headings(self._heading, turn)
        if n < 2:
            return wander, np.zeros(n, dtype=bool)

        offset = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(offset, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist[~visible] = np.inf
        nearest = np.argmin(dist, axis=1)
        can_follow = np.isfinite(dist[np.arange(n), nearest])
        follow = unit_rows(pos[nearest] - pos)
        return np.where(can_follow[:, None], follow, wander), can_follow

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
