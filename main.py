import argparse
import os
import re
import statistics
from pathlib import Path

os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")

import numpy as np
import pygame
from pygame.surface import SurfaceType

from environment import Environment
from experiment import (
    ExperimentConfig,
    DEFAULT_TEST_SEED,
    build_environment,
    list_experiments,
    load_experiment,
    save_experiment,
    tom_order_label,
)
from layout import SideExit, nearest_boundary
from obstacle import Circle, Wall
from plots import History, SIM_SIZE, WINDOW_SIZE, draw_plots, save_auc_plot

WALL_WIDTH = 20
EXIT_WIDTH = int(WALL_WIDTH * 1.5)
MAX_PHYSICS_STEPS = 2
CHOICE_COLORS = (
    (180, 35, 55),
    (105, 45, 185),
    (205, 115, 0),
    (0, 130, 135),
    (145, 80, 25),
    (70, 120, 35),
)
EXPERIMENT_DIR = Path(__file__).parent / 'experiments'
DEFAULT_EXPERIMENT = EXPERIMENT_DIR / 'two-north-one-south.yaml'
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / 'results'


def _metrics_path(config: ExperimentConfig, seed: int, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', config.name.strip()).strip('-') or 'untitled'
    return results_dir / name / f'metrics-{seed}.csv'


def _save_metrics(history: History, config: ExperimentConfig, seed: int, results_dir: Path) -> Path:
    path = _metrics_path(config, seed, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(history.to_csv())
    return path


COLORS = {
    'tom0': (0, 80, 220),
    'tom1': (230, 100, 0),
    'reactive': (125, 70, 160),
    'exit': (0, 255, 0),
    'sign': (0, 140, 140),
    'wall': (0, 0, 0),
    'text': (0, 0, 0),
    'background': (255, 255, 255),
    'edit': (180, 0, 180),
    'preview': (120, 80, 180),
}


def _agent_sprite(radius_px: int, color: tuple[int, int, int]) -> pygame.Surface:
    size = max(2, radius_px * 2)
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(sprite, color, (size // 2, size // 2), radius_px)
    return sprite


def _agent_sprites(radius_px: int) -> dict[int, pygame.Surface]:
    return {
        -1: _agent_sprite(radius_px, COLORS['reactive']),
        0: _agent_sprite(radius_px, COLORS['tom0']),
        1: _agent_sprite(radius_px, COLORS['tom1']),
    }


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
        f"seeing exit: {stats['seeing_exit']}   seeing sign: {stats['seeing_sign']}",
        f"know: {stats['with_belief']}   no belief: {stats['no_belief']}",
        f"memory-guided choice: {stats['memory_guided']}  peak {stats['peak_memory_guided']}",
        f"ToM-1 changed exit: {stats['tom1_changed']}",
        f"blind committed: {stats['blind_committed']}   {stats['blind_committed_s']:.1f} person-s",
    ]
    y = 12
    for line in lines:
        screen.blit(font.render(line, False, COLORS['text']), (12, y))
        y += 24
    x = 12
    for label, color in (
        ('ToM-0', COLORS['tom0']),
        ('ToM-1', COLORS['tom1']),
        ('reactive', COLORS['reactive']),
    ):
        pygame.draw.circle(screen, color, (x + 5, y + 7), 5)
        screen.blit(font.render(label, False, COLORS['text']), (x + 15, y))
        x += 95


def draw(
        screen: SurfaceType,
        environment: Environment,
        config: ExperimentConfig,
        scale: float,
        time_scale: float,
        debug: bool,
        font: pygame.font.Font,
        agent_sprites: dict[int, pygame.Surface],
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

    for sign in environment.signs:
        x, y = (int(value * scale) for value in sign.position)
        pygame.draw.polygon(
            screen, COLORS['sign'],
            [(x, y - 9), (x + 9, y), (x, y + 9), (x - 9, y)],
        )
        label = font.render(str(sign.exit_index + 1), True, COLORS['background'])
        screen.blit(label, label.get_rect(center=(x, y)))

    blits = []
    orders = environment._effective_orders()
    for i, agent in enumerate(environment.agents):
        x, y = agent.position
        order = int(orders[i])
        sprite = agent_sprites[1 if order == 1 else -1 if order < 0 else 0]
        blits.append((sprite, sprite.get_rect(center=(x * scale, y * scale))))
    if blits:
        screen.blits(blits)

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
        _draw_choice_debug(screen, environment, scale, font)

    draw_plots(screen, history, font, environment.initial_agent_count)


def _draw_choice_debug(
        screen: SurfaceType,
        environment: Environment,
        scale: float,
        font: pygame.font.Font,
) -> None:
    previous_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(0, 0, SIM_SIZE, SIM_SIZE))
    for j, ext in enumerate(environment.exits):
        color = CHOICE_COLORS[j % len(CHOICE_COLORS)]
        pygame.draw.line(screen, color, ext.start * scale, ext.end * scale, 6)
        center = ((ext.start + ext.end) * 0.5 * scale).astype(int)
        marker = (int(np.clip(center[0], 12, SIM_SIZE - 12)),
                  int(np.clip(center[1], 12, SIM_SIZE - 12)))
        pygame.draw.circle(screen, color, marker, 11)
        label = font.render(str(j + 1), True, COLORS['background'])
        screen.blit(label, label.get_rect(center=marker))

    for i, agent in enumerate(environment.agents):
        chosen = int(environment._chosen_exit[i])
        if not 0 <= chosen < len(environment.exits):
            continue
        direction = environment._heading[i]
        if np.linalg.norm(direction) < 1e-9:
            continue
        color = CHOICE_COLORS[chosen % len(CHOICE_COLORS)]
        origin = agent.position * scale
        tip = origin + direction * 25
        left = tip - direction * 7 + np.array([-direction[1], direction[0]]) * 5
        right = tip - direction * 7 - np.array([-direction[1], direction[0]]) * 5
        pygame.draw.line(screen, color, origin + direction * 5, tip, 2)
        pygame.draw.polygon(screen, color, [tip, left, right])
    screen.set_clip(previous_clip)


def _world_from_mouse(pos: tuple[int, int], scale: float) -> tuple[float, float] | None:
    x, y = pos
    if x >= SIM_SIZE or y >= SIM_SIZE:
        return None
    return x / scale, y / scale


def run_simulation(
        config: ExperimentConfig, catalog: list[Path], fps: int,
        seed: int = DEFAULT_TEST_SEED,
        results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    environment = build_environment(config, seed=seed, initial_state_dir=results_dir / 'initial-states')
    scale = SIM_SIZE / environment.width

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption(f'crowd-tom | seed {seed}')
    font = pygame.font.SysFont(None, 18)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.DOUBLEBUF, vsync=0)
    clock = pygame.time.Clock()
    sprites = _agent_sprites(max(1, int(config.agent_radius * scale)))
    history = History()
    history.record(environment.tom0_metrics())
    reported_done = False
    time_scale = 1.0
    running = True
    dt = 1.0 / 60
    congestion_radius = 5.0
    congestion_totals = [0] * len(environment.exits)
    congestion_peaks = [0] * len(environment.exits)
    congestion_samples = 0
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
        nonlocal environment, reported_done, accumulator, sprites
        nonlocal congestion_totals, congestion_peaks, congestion_samples
        environment = build_environment(config, seed=seed, initial_state_dir=results_dir / 'initial-states')
        congestion_totals = [0] * len(environment.exits)
        congestion_peaks = [0] * len(environment.exits)
        congestion_samples = 0
        sprites = _agent_sprites(max(1, int(config.agent_radius * scale)))
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
                    path = _save_metrics(history, config, seed, results_dir)
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
                        config.signs = [
                            sign for sign in config.signs
                            if sign.exit_index < len(config.exits)
                        ]
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
                positions = np.array([agent.position for agent in environment.agents])
                for i, exit_ in enumerate(environment.exits):
                    if len(positions) == 0:
                        count = 0
                    else:
                        distances = exit_.distances(positions).distance
                        count = int(np.count_nonzero(distances <= congestion_radius))
                    congestion_totals[i] += count
                    congestion_peaks[i] = max(congestion_peaks[i], count)
                congestion_samples += 1
                history.record(environment.tom0_metrics())
                accumulator -= dt
                steps += 1
            if accumulator > dt:
                accumulator = dt

        if environment.evacuation_time is not None and not reported_done:
            stats = environment.tom0_metrics()
            print(
                f"Total evacuation time is {stats['evacuation_time']:.2f}s | "
                f"Exit distribution {stats['escaped_by_exit']}, std {statistics.pstdev(stats['escaped_by_exit'])} | "
                f"peak memory-guided {stats['peak_memory_guided']}"
            )
            times = list(environment.evacuation_times.values())
            if times:
                print(
                    f"Individual evacuation times: "
                    f"mean={statistics.mean(times):.2f}s, "
                    f"median={statistics.median(times):.2f}s, "
                    f"p90={np.percentile(times, 90):.2f}s"
                )
            if congestion_samples:
                print("Exit congestion: ")
                for i, (total, peak) in enumerate(zip(congestion_totals, congestion_peaks)):
                    mean = total / congestion_samples
                    print(f"Exit {i + 1}: avg {mean:.2f}, peak {peak}")
            auc_values = history.auc()
            print(f"AUC ToM-0: {auc_values['ToM-0']:.2f}, ToM-1: {auc_values['ToM-1']:.2f}")
            plots_dir = Path(__file__).resolve().parent / 'plots'
            plots_dir.mkdir(exist_ok=True)
            experiment_name = config.name.strip().lower().replace('-', '_')
            plot_path = plots_dir / f'{experiment_name}_seed_{seed}_auc_plot.png'
            save_auc_plot(history, str(plot_path))
            print(f"Saved AUC plot to {plot_path}")
            csv_path = _save_metrics(history, config, seed, results_dir)
            reported_done = True

        draw(
            screen, environment, config, scale, time_scale, debug, font, sprites,
            history, edit_mode, place_mode, drag_start, mouse_world,
        )
        pygame.display.flip()

    if not reported_done:
        _save_metrics(history, config, seed, results_dir)
    pygame.quit()


def run_seed_range(
        config: ExperimentConfig, start: int, end: int,
        dt: float, max_time: float, results_dir: Path,
) -> list[dict]:
    """Run an inclusive seed range headlessly, one completed run at a time."""
    summaries = []
    for seed in range(start, end + 1):
        environment = build_environment(
            config, seed=seed, initial_state_dir=results_dir / 'initial-states',
        )
        history = History()
        history.record(environment.tom0_metrics())
        next_record = 0.5
        while environment.agents and environment._t < max_time:
            environment.tick(min(dt, max_time - environment._t))
            if environment._t + 1e-9 >= next_record:
                history.record(environment.tom0_metrics())
                next_record += 0.5
        stats = environment.tom0_metrics()
        history.record(stats)
        _save_metrics(history, config, seed, results_dir)
        summaries.append({'seed': seed, 'escaped': stats['escaped'],
                          'remaining': stats['n'], 'time': stats['t']})
        print(
            f"seed {seed}: escaped {stats['escaped']}/{stats['initial']}, "
            f"remaining {stats['n']}, simulated {stats['t']:.3f}s",
            flush=True,
        )
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Crowd ToM simulator')
    parser.add_argument(
        'experiment',
        nargs='?',
        default=str(DEFAULT_EXPERIMENT),
        help='YAML experiment file',
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int, help=f'single run seed (default: {DEFAULT_TEST_SEED})')
    seeds.add_argument('--seed-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='run every seed from START through END, inclusive, headlessly')
    parser.add_argument('--max-time', type=float, default=120.0,
                        help='simulation cutoff in seconds for --seed-range (default: 120)')
    parser.add_argument('--dt', type=float, default=1 / 60,
                        help='physics timestep for --seed-range (default: 1/60)')
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help='directory for metrics and saved initial crowds')
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.seed_range is not None:
        start, end = args.seed_range
        if start < 0 or end < start:
            parser.error('--seed-range requires 0 <= START <= END')
    if not np.isfinite(args.max_time) or args.max_time <= 0:
        parser.error('--max-time must be positive and finite')
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('--dt must be positive and finite')
    catalog = list_experiments(EXPERIMENT_DIR)
    config = load_experiment(args.experiment)
    if args.seed_range is not None:
        run_seed_range(config, start, end, args.dt, args.max_time, args.results_dir)
    else:
        run_simulation(config, catalog, fps=60,
                       seed=args.seed if args.seed is not None else DEFAULT_TEST_SEED,
                       results_dir=args.results_dir)


if __name__ == '__main__':
    main()
