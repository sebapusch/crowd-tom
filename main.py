import argparse
import os
from math import pi
from pathlib import Path

os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")

import numpy as np
import pygame
from pygame.surface import SurfaceType

from environment import Environment
from experiment import (
    ExperimentConfig,
    build_environment,
    list_experiments,
    load_experiment,
    save_experiment,
    tom_order_label,
)
from layout import SideExit, nearest_boundary
from obstacle import Circle, Wall
from plots import History, SIM_SIZE, WINDOW_SIZE, draw_plots

WALL_WIDTH = 20
EXIT_WIDTH = int(WALL_WIDTH * 1.5)
MAX_PHYSICS_STEPS = 2
EXPERIMENT_DIR = Path(__file__).parent / 'experiments'
DEFAULT_EXPERIMENT = EXPERIMENT_DIR / 'two-north-one-south.yaml'


def _metrics_path(tom_order: int | None) -> Path:
    if tom_order is None:
        mode = 'mixed'
    elif tom_order < 0:
        mode = 'reactive'
    elif tom_order == 1:
        mode = 'tom-1'
    else:
        mode = 'tom-0'
    return EXPERIMENT_DIR / f'last-metrics-{mode}.csv'


COLORS = {
    'agent': (0, 0, 255),
    'exit': (0, 255, 0),
    'wall': (0, 0, 0),
    'text': (0, 0, 0),
    'debug': (255, 0, 0),
    'fov': (0, 170, 255),
    'seen': (0, 200, 120),
    'memory': (255, 140, 0),
    'background': (255, 255, 255),
    'edit': (180, 0, 180),
    'preview': (120, 80, 180),
}


def _agent_sprite(radius_px: int) -> pygame.Surface:
    size = max(2, radius_px * 2)
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(sprite, COLORS['agent'], (size // 2, size // 2), radius_px)
    return sprite


def _exit_name(environment: Environment, index: int) -> str:
    if index >= len(environment.exits):
        return f'exit {index}'
    x = float(environment.exits[index].start[0])
    y = float(environment.exits[index].start[1])
    if abs(y) < 1e-6:
        return 'NW' if x < environment.width / 2 else 'NE'
    if abs(y - environment.height) < 1e-6:
        return 'south'
    if abs(x) < 1e-6:
        return 'west'
    if abs(x - environment.width) < 1e-6:
        return 'east'
    return f'exit {index}'


def _draw_hud(
        screen: SurfaceType,
        environment: Environment,
        config: ExperimentConfig,
        time_scale: float,
        font: pygame.font.Font,
        edit_mode: bool,
        place_mode: str,
) -> None:
    stats = environment.tom0_metrics()
    if stats['evacuation_time'] is None:
        timer = f"t = {stats['t']:.2f}s"
    else:
        timer = f"all escaped in {stats['evacuation_time']:.2f}s"

    exits = ', '.join(
        f"{_exit_name(environment, i)}={count}"
        for i, count in enumerate(stats['escaped_by_exit'])
    ) or 'none'
    mode = f'EDIT {place_mode}' if edit_mode else 'run'

    lines = [
        f"{config.name}   {tom_order_label(environment.tom_order)}   {mode}",
        timer,
        f"x{time_scale:.2f}   remaining {stats['n']}/{stats['initial']}",
        f"ToM-0 {stats['remaining_tom0']}   ToM-1 {stats['remaining_tom1']}   reactive {stats['remaining_reactive']}",
        f"escaped {stats['escaped']}   ({exits})",
        f"seeing: {stats['seeing']}   know: {stats['with_belief']}   never seen: {stats['no_belief']}",
        f"ToM-0 unseen choice: {stats['memory_guided']}  peak {stats['peak_memory_guided']}",
        f"ToM-1 changed exit: {stats['tom1_changed']}",
        f"blind committed: {stats['blind_committed']}   {stats['blind_committed_s']:.1f} person-s",
    ]
    y = 12
    for line in lines:
        screen.blit(font.render(line, False, COLORS['text']), (12, y))
        y += 24


def draw(
        screen: SurfaceType,
        environment: Environment,
        config: ExperimentConfig,
        scale: float,
        time_scale: float,
        debug: bool,
        font: pygame.font.Font,
        agent_sprite: pygame.Surface,
        history: History,
        edit_mode: bool,
        place_mode: str,
        drag_start: tuple[float, float] | None,
        mouse_world: tuple[float, float] | None,
) -> None:
    screen.fill(COLORS['background'])
    _draw_hud(screen, environment, config, time_scale, font, edit_mode, place_mode)

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

    if edit_mode and drag_start is not None and mouse_world is not None:
        if place_mode == 'circle':
            radius = float(np.linalg.norm(np.array(mouse_world) - np.array(drag_start)))
            pygame.draw.circle(
                screen, COLORS['preview'],
                (drag_start[0] * scale, drag_start[1] * scale),
                max(1, int(radius * scale)), 2,
            )
        else:
            pygame.draw.line(
                screen, COLORS['preview'],
                (drag_start[0] * scale, drag_start[1] * scale),
                (mouse_world[0] * scale, mouse_world[1] * scale), 3,
            )

    if debug:
        _draw_perception_debug(screen, environment, scale)

    draw_plots(screen, history, font, environment.initial_agent_count)


def _draw_perception_debug(screen: SurfaceType, environment: Environment, scale: float) -> None:
    if len(environment.agents) == 0:
        return

    n_preview = min(8, len(environment.agents))
    full_circle = environment.fov_rad >= 2.0 * pi - 1e-6
    for i in range(n_preview):
        origin = environment._pos[i]
        origin_px = (float(origin[0] * scale), float(origin[1] * scale))
        if full_circle:
            pygame.draw.circle(
                screen, COLORS['fov'], origin_px,
                int(environment.view_range * scale), 1,
            )

    seen = environment._seen_exits
    has_belief = getattr(environment, '_has_belief', None)
    if seen.size == 0 and (has_belief is None or has_belief.size == 0):
        return
    for i, agent in enumerate(environment.agents):
        origin = (float(agent.position[0] * scale), float(agent.position[1] * scale))
        for j, ext in enumerate(environment.exits):
            currently_seen = seen.size > 0 and j < seen.shape[1] and seen[i, j]
            remembered = (
                has_belief is not None and has_belief.size > 0
                and j < has_belief.shape[1] and has_belief[i, j]
            )
            if currently_seen:
                closest = np.asarray(ext.distances(agent.position).closest_point).reshape(2)
                target, color = closest, COLORS['seen']
            elif remembered:
                target, color = environment._belief_mu[i, j], COLORS['memory']
            else:
                continue
            pygame.draw.line(
                screen, color, origin,
                (float(target[0] * scale), float(target[1] * scale)), 1,
            )


def _world_from_mouse(pos: tuple[int, int], scale: float) -> tuple[float, float] | None:
    x, y = pos
    if x >= SIM_SIZE or y >= SIM_SIZE:
        return None
    return x / scale, y / scale


def run_simulation(config: ExperimentConfig, catalog: list[Path], fps: int) -> None:
    environment = build_environment(config)
    scale = SIM_SIZE / environment.width

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption('crowd-tom')
    font = pygame.font.SysFont(None, 18)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.DOUBLEBUF, vsync=0)
    clock = pygame.time.Clock()
    sprite = _agent_sprite(max(1, int(config.agent_radius * scale)))
    history = History()
    history.record(environment.tom0_metrics())
    reported_done = False
    time_scale = 1.0
    running = True
    dt = 1.0 / 60
    accumulator = 0.0
    paused = False
    debug = False
    edit_mode = False
    place_mode = 'wall'
    drag_start: tuple[float, float] | None = None
    experiment_index = 0
    if config.source_path in catalog:
        experiment_index = catalog.index(config.source_path)

    def reset() -> None:
        nonlocal environment, reported_done, accumulator, sprite
        environment = build_environment(config)
        sprite = _agent_sprite(max(1, int(config.agent_radius * scale)))
        history.clear()
        history.record(environment.tom0_metrics())
        reported_done = False
        accumulator = 0.0

    while running:
        frame_time = min(clock.tick(fps) / 1000.0, 0.05)
        mouse_world = _world_from_mouse(pygame.mouse.get_pos(), scale)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_MINUS:
                    time_scale = max(0.1, time_scale - 0.1)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    time_scale = min(4.0, time_scale + 0.1)
                elif event.key == pygame.K_d:
                    debug = not debug
                elif event.key == pygame.K_t:
                    if config.tom_proportions is not None:
                        current = environment.tom_order
                        environment.tom_order = (
                            -1 if current is None else 0 if current < 0
                            else 1 if current == 0 else None
                        )
                    else:
                        config.tom_order = (
                            0 if config.tom_order < 0
                            else 1 if config.tom_order == 0
                            else -1
                        )
                        environment.tom_order = config.tom_order
                    history.record(environment.tom0_metrics())
                elif event.key == pygame.K_e:
                    edit_mode = not edit_mode
                    paused = True
                    drag_start = None
                elif event.key == pygame.K_w:
                    place_mode = 'wall'
                elif event.key == pygame.K_c:
                    place_mode = 'circle'
                elif event.key == pygame.K_LEFTBRACKET and catalog:
                    experiment_index = (experiment_index - 1) % len(catalog)
                    config = load_experiment(catalog[experiment_index])
                    reset()
                elif event.key == pygame.K_RIGHTBRACKET and catalog:
                    experiment_index = (experiment_index + 1) % len(catalog)
                    config = load_experiment(catalog[experiment_index])
                    reset()
                elif event.key == pygame.K_f:
                    path = save_experiment(config, EXPERIMENT_DIR / 'last.yaml')
                    print(f'saved experiment {path}')
                elif event.key == pygame.K_g:
                    path = _metrics_path(None if config.tom_proportions else config.tom_order)
                    path.write_text(history.to_csv())
                    print(f'saved metrics {path}')
                elif event.key == pygame.K_BACKSPACE and edit_mode:
                    if config.circles:
                        config.circles.pop()
                        reset()
                    elif config.interior_walls:
                        config.interior_walls.pop()
                        reset()
                    elif config.exits:
                        config.exits.pop()
                        reset()

            elif event.type == pygame.MOUSEBUTTONDOWN and edit_mode and event.button == 1:
                drag_start = _world_from_mouse(event.pos, scale)

            elif event.type == pygame.MOUSEBUTTONUP and edit_mode and event.button == 1 and drag_start:
                end = _world_from_mouse(event.pos, scale)
                drag_start_local = drag_start
                drag_start = None
                if end is None:
                    continue
                travel = float(np.linalg.norm(np.array(end) - np.array(drag_start_local)))
                if travel < 1.0:
                    hit = nearest_boundary(end[0], end[1], config.width, config.height)
                    if hit is not None:
                        side, along = hit
                        config.exits.append(SideExit(side=side, center=along, width=3.0))
                        reset()
                    continue
                if place_mode == 'circle':
                    config.circles.append((drag_start_local, travel))
                else:
                    config.interior_walls.append((drag_start_local, end))
                reset()

        if not paused and not edit_mode:
            accumulator += frame_time * time_scale
            steps = 0
            while accumulator >= dt and steps < MAX_PHYSICS_STEPS:
                environment.tick(dt)
                history.record(environment.tom0_metrics())
                accumulator -= dt
                steps += 1
            if accumulator > dt:
                accumulator = dt

        if environment.evacuation_time is not None and not reported_done:
            stats = environment.tom0_metrics()
            print(
                f"evacuated in {stats['evacuation_time']:.2f}s | "
                f"escaped {stats['escaped_by_exit']} | "
                f"peak memory-guided {stats['peak_memory_guided']}"
            )
            csv_path = _metrics_path(None if config.tom_proportions else config.tom_order)
            csv_path.write_text(history.to_csv())
            reported_done = True

        draw(
            screen, environment, config, scale, time_scale, debug, font, sprite,
            history, edit_mode, place_mode, drag_start, mouse_world,
        )
        pygame.display.flip()

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description='Crowd ToM simulator')
    parser.add_argument(
        'experiment',
        nargs='?',
        default=str(DEFAULT_EXPERIMENT),
        help='YAML experiment file',
    )
    args = parser.parse_args()
    catalog = list_experiments(EXPERIMENT_DIR)
    config = load_experiment(args.experiment)
    run_simulation(config, catalog, fps=60)


if __name__ == '__main__':
    main()
