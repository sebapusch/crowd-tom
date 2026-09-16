import pygame
import numpy as np
from agent import Agent
from environment import Environment
from exit import Exit

width = 900
height = 900

room_width = 15
room_height = 15

scale = width / room_width

pygame.init()
screen = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()

environment = Environment(room_width, room_height)
environment.add_exit(Exit((6.8, 0), (8.2, 0)))
environment.add_agent(Agent(
    idx=0,
    mass=1.0,
    radius=0.2,
    pos=np.array((4.0, 7.0)),
    vel=np.zeros(2),
    des_speed=0.0,
    des_dir=np.zeros(2),
    tau=1.0,
))

def draw_environment():
    pygame.draw.rect(screen, (255, 255, 255), (0, 0, width, height))
    wall_width = 12
    pygame.draw.line(screen, (0, 0, 0), (0, 0), (width, 0), wall_width)
    pygame.draw.line(screen, (0, 0, 0), (width, 0), (width, height), wall_width)
    pygame.draw.line(screen, (0, 0, 0), (width, height), (0, height), wall_width)
    pygame.draw.line(screen, (0, 0, 0), (0, height), (0, 0), wall_width)


def draw_exit(screen, start, end):
    pygame.draw.line(screen, (0, 255, 0), start * scale, end * scale, 20)


def draw_agent(screen, position):
    x, y = position
    pygame.draw.circle(screen, (0, 0, 255), (int(x * scale), int(y * scale)), 5)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_environment()

    for exit_position in environment.exits:
        draw_exit(screen, exit_position.start, exit_position.end)

    for agent in environment.agents:
        draw_agent(screen, agent.pos)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

