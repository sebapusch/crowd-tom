from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from obstacle import Obstacle

from exit import Exit


# Newtons
F_CUT = 1


class Environment:
    """
    Agents must be passed in constructor as visibility size is computed once in __init__
    """
    def __init__(self, width: int, height: int, agents: list[Agent]) -> None:
        self.width = width
        self.height = height
        self.agents: list[Agent] = agents
        self.exits: list[Exit] = []
        self.obstacles: list[Obstacle] = []
        self._grid = {}
        self._cell_size = self._compute_visibility()

    def add_exit(self, exit_: Exit) -> None:
        self.exits.append(exit_)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def tick(self, dt: float) -> None:
        self._update_grid()
        for agent in self.agents:
            agent.update_acceleration(self)

        remaining_agents = []
        for agent in self.agents:
            previous_position = agent.position.copy()
            agent.update_position(dt)

            if not any(
                    exit_position.is_crossed(previous_position, agent.position)
                    for exit_position in self.exits
            ):
                remaining_agents.append(agent)

        self.agents = remaining_agents


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

    def _update_grid(self) -> None:
        self._grid = {}
        for agent in self.agents:
            key = self._cell_key(agent)

            if not key in self._grid:
                self._grid[key] = [agent]
            else:
                self._grid[key].append(agent)

    def _compute_visibility(self) -> float:
        largest = float('-inf')

        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i == j:
                    continue

                A, B = a.social_repulsion

                dist = a.radius + b.radius - B * log(F_CUT / A)

                if dist > largest:
                    largest = dist

        if largest == float('-inf'):
            raise ValueError('Invalid cell size found')

        return largest

    def _cell_key(self, agent: Agent) -> tuple:
        cell = agent.position // self._cell_size
        return cell[0], cell[1]

