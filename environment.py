import numpy as np
from agent import Agent
from exit import Exit

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.exits = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_exit(self, exit_position):
        self.exits.append(exit_position)