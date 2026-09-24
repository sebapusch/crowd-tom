import os

# Disable Retina backing-store scaling so pygame is not filling 4x the pixels.
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")

from random import random
from math import cos, sin, pi
from typing import Callable

import pygame
import numpy as np
from pygame.surface import SurfaceType

from agent import Agent
from environment import Environment
from exit import Exit
from obstacle import Wall, Circle, Obstacle

Point = tuple[float, float]

WALL_WIDTH = 20
EXIT_WIDTH = int(WALL_WIDTH * 1.5)

WINDOW_SIZE = (1000, 1000)
MAX_PHYSICS_STEPS = 2
VIEW_RANGE = 40.0
FOV_DEG = 360.0
EXIT_WIDTH_WORLD = 3.0
AGENT_COUNT = 400
AGENT_RADIUS = 0.3
SPAWN_CLEARANCE = 0.05
SPAWN_ATTEMPTS = 10_000


def _position_is_free(
        candidate: np.ndarray,
        radius: float,
        placed: np.ndarray,
        obstacles: list[Obstacle],
) -> bool:
    min_gap = 2.0 * radius + SPAWN_CLEARANCE
    if len(placed) > 0:
        if np.min(np.linalg.norm(placed - candidate, axis=1)) < min_gap:
            return False
    for obstacle in obstacles:
        if float(obstacle.distances(candidate).distance[0]) < radius + SPAWN_CLEARANCE:
            return False
    return True


def _spawn_positions(
        count: int,
        radius: float,
        width: float,
        height: float,
        obstacles: list[Obstacle],
) -> np.ndarray:
    margin = radius + SPAWN_CLEARANCE + 1.0
    positions = np.empty((count, 2), dtype=float)
    placed = 0
    attempts = 0
    max_attempts = SPAWN_ATTEMPTS * count
    while placed < count:
        if attempts >= max_attempts:
            raise RuntimeError(f'Could not place agent {placed + 1} without overlap')
        attempts += 1
        candidate = np.array([
            margin + random() * (width - 2.0 * margin),
            margin + random() * (height - 2.0 * margin),
        ], dtype=float)
        if not _position_is_free(candidate, radius, positions[:placed], obstacles):
            continue
        positions[placed] = candidate
        placed += 1
    min_gap = 2.0 * radius + SPAWN_CLEARANCE
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    if dist.min() < min_gap:
        raise RuntimeError(f'spawn overlap: min dist {dist.min()} < {min_gap}')
    return positions


def build_environment() -> Environment:
    north = ((50.0, 0.0), (50.0 + EXIT_WIDTH_WORLD, 0.0))
    south = ((50.0, 100.0), (50.0 + EXIT_WIDTH_WORLD, 100.0))
    obstacles: list[Obstacle] = [
        Wall((0, 0), (0, 100)),
        Wall((0, 0), north[0]),
        Wall(north[1], (100, 0)),
        Wall((0, 100), south[0]),
        Wall(south[1], (100, 100)),
        Wall((100, 100), (100, 0)),
    ]

    agents = []
    for i, agent_pos in enumerate(_spawn_positions(AGENT_COUNT, AGENT_RADIUS, 100, 100, obstacles)):
        angle = random() * 2.0 * pi
        agents.append(Agent(
            social_repulsion=(2e3, 0.08),
            idx=i,
            mass=1.0,
            radius=AGENT_RADIUS,
            position=np.array(agent_pos, dtype=float, copy=True),
            velocity=np.zeros(2, dtype=float),
            desired_speed=np.float32(2.0),
            desired_direction=np.array([cos(angle), sin(angle)], dtype=float),
            tau=1.0,
        ))

    environment = Environment(
        100,
        100,
        agents,
        view_range=VIEW_RANGE,
        fov_rad=np.deg2rad(FOV_DEG),
    )
    environment.add_exit(Exit(*north))
    environment.add_exit(Exit(*south))
    for obstacle in obstacles:
        environment.add_obstacle(obstacle)

    return environment

COLORS = {
    'agent': (0, 0, 255),
    'exit': (0, 255, 0),
    'wall': (0, 0, 0),
    'text': (0, 0, 0),
    'debug': (255, 0, 0),
    'fov': (0, 170, 255),
    'seen': (0, 200, 120),
    'background': (255, 255, 255),
}


def _agent_sprite(radius_px: int) -> pygame.Surface:
    size = max(2, radius_px * 2)
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(sprite, COLORS['agent'], (size // 2, size // 2), radius_px)
    return sprite


def draw(
        screen: SurfaceType,
        environment: Environment,
        scale: float,
        time_scale: float,
        debug: bool,
        font: pygame.font.Font,
        agent_sprite: pygame.Surface,
        label_cache: dict,
) -> None:
    screen.fill(COLORS['background'])

    key = (f'{time_scale:.2f}', len(environment.agents))
    if key not in label_cache:
        label_cache.clear()
        label_cache[key] = (
            font.render(f'x{time_scale:.2f}', False, COLORS['text']),
            font.render(f'number of agents: {len(environment.agents)}', False, COLORS['text']),
        )
    text_scale, text_agent = label_cache[key]
    screen.blit(text_scale, (20, 20))
    screen.blit(text_agent, (20, 50))

    for obj in environment.obstacles:
        if isinstance(obj, Wall):
            pygame.draw.line(screen, COLORS['wall'], obj.start * scale, obj.end * scale, WALL_WIDTH)
        elif isinstance(obj, Circle):
            pygame.draw.circle(screen, COLORS['wall'], obj.center * scale, obj.radius * scale, WALL_WIDTH)

    for ext in environment.exits:
        pygame.draw.line(screen, COLORS['exit'], ext.start * scale, ext.end * scale, EXIT_WIDTH)

    blit_pos = []
    for agent in environment.agents:
        x, y = agent.position
        blit_pos.append(agent_sprite.get_rect(center=(x * scale, y * scale)))
    if blit_pos:
        screen.blits([(agent_sprite, rect) for rect in blit_pos])

    if debug:
        _draw_perception_debug(screen, environment, scale)

        for i in range(int(environment.width / environment._cell_size) + 1):
            x = i * environment._cell_size * scale
            pygame.draw.line(screen, COLORS['debug'], (x, 0), (x, environment.height * scale))

        for i in range(int(environment.height / environment._cell_size) + 1):
            y = i * environment._cell_size * scale
            pygame.draw.line(screen, COLORS['debug'], (0, y), (environment.width * scale, y))


def _draw_perception_debug(screen: SurfaceType, environment: Environment, scale: float) -> None:
    if len(environment.agents) == 0:
        return

    n_preview = min(8, len(environment.agents))
    half_fov = 0.5 * environment.fov_rad
    full_circle = environment.fov_rad >= 2.0 * pi - 1e-6

    for i in range(n_preview):
        origin = environment._pos[i]
        origin_px = (float(origin[0] * scale), float(origin[1] * scale))
        if full_circle:
            pygame.draw.circle(
                screen,
                COLORS['fov'],
                origin_px,
                int(environment.view_range * scale),
                1,
            )
            continue
        heading = environment._heading[i]
        cone_len = min(8.0, environment.view_range)
        left = np.array([
            heading[0] * cos(-half_fov) - heading[1] * sin(-half_fov),
            heading[0] * sin(-half_fov) + heading[1] * cos(-half_fov),
        ])
        right = np.array([
            heading[0] * cos(half_fov) - heading[1] * sin(half_fov),
            heading[0] * sin(half_fov) + heading[1] * cos(half_fov),
        ])
        pygame.draw.polygon(
            screen,
            COLORS['fov'],
            [
                origin_px,
                (float(origin_px[0] + left[0] * cone_len * scale), float(origin_px[1] + left[1] * cone_len * scale)),
                (float(origin_px[0] + right[0] * cone_len * scale), float(origin_px[1] + right[1] * cone_len * scale)),
            ],
            width=1,
        )

    seen = environment._seen_exits
    if seen.size == 0:
        return
    for i, agent in enumerate(environment.agents):
        for j, ext in enumerate(environment.exits):
            if j >= seen.shape[1] or not seen[i, j]:
                continue
            closest = np.asarray(ext.distances(agent.position).closest_point).reshape(2)
            pygame.draw.line(
                screen,
                COLORS['seen'],
                (float(agent.position[0] * scale), float(agent.position[1] * scale)),
                (float(closest[0] * scale), float(closest[1] * scale)),
                1,
            )


def run_simulation(
        reset: Callable[[], Environment],
        fps: int,
) -> None:
    environment = reset()

    scale = WINDOW_SIZE[0] / environment.width

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 30)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.DOUBLEBUF, vsync=0)
    clock = pygame.time.Clock()
    agent_radius_px = max(1, int(environment.agents[0].radius * scale)) if environment.agents else 1
    sprite = _agent_sprite(agent_radius_px)
    label_cache: dict = {}

    time_scale = 1.0
    timesteps = 0
    running = True
    dt = 1.0 / 60
    accumulator = 0.0
    paused = False
    debug = False

    while running:
        frame_time = min(clock.tick(fps) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    environment = reset()
                    label_cache.clear()
                    draw(screen, environment, scale, time_scale, debug, font, sprite, label_cache)
                elif event.key == pygame.K_MINUS:
                    time_scale = max(0.1, time_scale - 0.1)
                elif event.key == pygame.K_PLUS:
                    time_scale = min(2, time_scale + 0.1)
                elif event.key == pygame.K_d:
                    debug = not debug

        if not paused:
            accumulator += frame_time * time_scale
            steps = 0
            while accumulator >= dt and steps < MAX_PHYSICS_STEPS:
                environment.tick(dt)
                timesteps += 1
                accumulator -= dt
                steps += 1
            if accumulator > dt:
                accumulator = dt

        draw(screen, environment, scale, time_scale, debug, font, sprite, label_cache)
        pygame.display.flip()

    pygame.quit()

def main():
    run_simulation(build_environment, fps=60)

if __name__ == '__main__':
    main()
