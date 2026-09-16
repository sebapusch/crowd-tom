from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from obstacle import Obstacle

from exit import Exit




class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.exits: list[Exit] = []
        self.obstacles: list[Obstacle] = []

    def add_agent(self, agent) -> None:
        self.agents.append(agent)

    def add_exit(self, exit_: Exit) -> None:
        self.exits.append(exit_)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)