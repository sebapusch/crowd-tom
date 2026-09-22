from random import random
from typing import Callable

import pygame
import numpy as np
from pygame.surface import SurfaceType

from agent import Agent
from environment import Environment
from exit import Exit
from obstacle import Obstacle, Wall, Circle

Point = tuple[float, float]

WALL_WIDTH = 20
EXIT_WIDTH = int(WALL_WIDTH * 1.5)

WINDOW_SIZE = (1000, 1000)

COLORS = {
    'agent': (0, 0, 255),
    'exit': (0, 255, 0),
    'wall': (0, 0, 0),
    'text': (0, 0, 0),
    'debug': (255, 0, 0),
    'background': (255, 255, 255),
}

SPACEBAR_KEY = 32

pygame.font.init()
FONT = pygame.font.SysFont('Comic Sans MS', 30)


def build_environment() -> Environment:
    agents = []
    for i in range(400):
        agent_pos = np.array([random() * 100, random() * 100])

        agents.append(Agent(
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

    environment = Environment(100, 100, agents)

    environment.add_exit(Exit((50.0, 0), (53.0, 0)))

    environment.add_obstacle(Wall((0, 0), (0, 100)))
    environment.add_obstacle(Wall((0, 0), (50, 0)))
    environment.add_obstacle(Wall((53, 0), (100, 0)))
    environment.add_obstacle(Wall((100, 100), (0, 100)))
    environment.add_obstacle(Wall((100, 100), (100, 0)))

    return environment

def draw(
        screen: SurfaceType,
        environment: Environment,
        scale: float,
        time_scale: float,
        debug: bool,
) -> None:
    pygame.draw.rect(screen, COLORS['background'], (0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1]))

    text_scale = FONT.render(f'x{time_scale:.2f}', False, COLORS['text'])
    text_agent = FONT.render(f'number of agents: {len(environment.agents)}', False, COLORS['text'])
    screen.blit(text_scale, (20, 20))
    screen.blit(text_agent, (20, 50))


    for obj in environment.obstacles:
        if isinstance(obj, Wall):
            pygame.draw.line(screen, COLORS['wall'], obj.start * scale, obj.end * scale, WALL_WIDTH)
        elif isinstance(obj, Circle):
            pygame.draw.circle(screen, COLORS['wall'], obj.center * scale, obj.radius * scale, WALL_WIDTH)

    for ext in environment.exits:
        pygame.draw.line(screen, COLORS['exit'], ext.start * scale, ext.end * scale, EXIT_WIDTH)

    for agent in environment.agents:
        x, y = agent.position
        pygame.draw.circle(screen, COLORS['agent'], (int(x * scale), int(y * scale)), max(1, int(agent.radius * scale)))

    if debug:
        for i in range(int(environment.width / environment._cell_size) + 1):
            x = i * environment._cell_size * scale
            pygame.draw.line(screen, COLORS['debug'], (x, 0), (x, environment.height * scale))

        for i in range(int(environment.height / environment._cell_size) + 1):
            y = i * environment._cell_size * scale
            pygame.draw.line(screen, COLORS['debug'], (0, y), (environment.width * scale, y))


def run_simulation(
        reset: Callable[[], Environment],
        fps: int,
) -> None:
    environment = reset()

    scale = WINDOW_SIZE[0] / environment.width

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    time_scale = 1.0
    timesteps = 0
    running = True
    dt = 1.0 / 60
    accumulator = 0.0
    paused = False
    debug = True

    while running:
        frame_time = min(clock.tick(fps) / 1000.0, 0.25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    environment = reset()
                    draw(screen, environment, scale, time_scale, debug)
                elif event.key == pygame.K_MINUS:
                    time_scale = max(0.1, time_scale - 0.1)
                elif event.key == pygame.K_PLUS:
                    time_scale = min(2, time_scale + 0.1)
                elif event.key == pygame.K_d:
                    debug = not debug

        if paused:
            continue

        accumulator += frame_time * time_scale

        while accumulator >= dt:
            environment.tick(dt)
            timesteps += 1
            accumulator -= dt

        draw(screen, environment, scale, time_scale, debug)

        pygame.display.flip()

    pygame.quit()

def main():
    run_simulation(build_environment, fps=60)

if __name__ == '__main__':
    main()
