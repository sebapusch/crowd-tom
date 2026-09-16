from random import random

import pygame
import numpy as np
from agent import Agent
from environment import Environment
from exit import Exit
from obstacle import Obstacle, Wall, Circle

Point = tuple[float, float]

width = 900
height = 900

room_width = 15
room_height = 15

scale = width / room_width

pygame.init()
screen = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()

environment = Environment(room_width, room_height)
environment.add_exit(Exit((6.0, 0), (8.0, 0)))

for i in range(20):
    agent_pos = np.array([random() * 15, random() * 15])

    environment.add_agent(Agent(
        social_repulsion=(2e3, 0.08),
        idx=i,
        mass=1.0,
        radius=0.3,
        position=agent_pos,
        velocity=np.zeros(2),
        desired_speed=np.float32(2.0),
        desired_direction=np.zeros(2),
        tau=1.0,
    ))


environment.add_obstacle(Wall((0, 0), (0, 15)))
environment.add_obstacle(Wall((0, 0), (6, 0)))
environment.add_obstacle(Wall((8, 0), (15, 0)))
environment.add_obstacle(Wall((15, 15), (0, 15)))
environment.add_obstacle(Wall((15, 15), (15, 0)))

def draw_environment() -> None:
    pygame.draw.rect(screen, (255, 255, 255), (0, 0, width, height))
    wall_width = 12

    for obj in environment.obstacles:
        if isinstance(obj, Wall):
            pygame.draw.line(screen, (0, 0, 0), obj.start * scale, obj.end * scale, wall_width)
        elif isinstance(obj, Circle):
            pygame.draw.circle(screen, (0, 0, 0), obj.center, obj.radius, wall_width)


def draw_exit(screen, start, end):
    pygame.draw.line(screen, (0, 255, 0), start * scale, end * scale, 20)


def draw_agent(screen, position):
    x, y = position
    pygame.draw.circle(screen, (0, 0, 255), (int(x * scale), int(y * scale)), max(1, int(agent.radius * scale)))


fps = 60

timesteps = 0
running = True

time_scale = 0.5

while running:
    dt = (clock.tick(fps) / 1000) * time_scale

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    for agent in environment.agents:
        agent.tick_acceleration(environment)

    remaining_agents = []
    for agent in environment.agents:
        previous_position = agent.position.copy()
        agent.tick_position(dt)

        if not any(
            exit_position.is_crossed(previous_position, agent.position)
            for exit_position in environment.exits
        ):
            remaining_agents.append(agent)

    environment.agents = remaining_agents

    draw_environment()

    for exit_position in environment.exits:
        draw_exit(screen, exit_position.start, exit_position.end)

    for agent in environment.agents:
        draw_agent(screen, agent.position)

    pygame.display.flip()
    timesteps += 1

pygame.quit()
