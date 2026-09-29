from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from pathlib import Path
from random import random
from typing import Any

import numpy as np
import yaml

from agent import Agent
from environment import Environment
from exit import Exit
from layout import SideExit, exit_segment, perimeter_walls
from obstacle import Circle, Obstacle, Wall
from tom0 import ToM0Params

SPAWN_CLEARANCE = 0.05
SPAWN_ATTEMPTS = 10_000


def parse_tom_order(value: Any) -> int:
    if value is None or value == 'none' or value == 'reactive':
        return -1
    return int(value)


def tom_order_label(order: int) -> str:
    if order < 0:
        return 'none (reactive)'
    if order == 0:
        return 'ToM-0'
    return f'ToM-{order} (using ToM-0)'


@dataclass
class ExperimentConfig:
    name: str = 'untitled'
    width: float = 100.0
    height: float = 100.0
    tom_order: int = 0
    agent_count: int = 400
    agent_radius: float = 0.3
    desired_speed: float = 2.0
    view_range: float = 40.0
    fov_deg: float = 360.0
    exits: list[SideExit] = field(default_factory=list)
    interior_walls: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    circles: list[tuple[tuple[float, float], float]] = field(default_factory=list)
    source_path: Path | None = None

    def clone(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.name,
            width=self.width,
            height=self.height,
            tom_order=self.tom_order,
            agent_count=self.agent_count,
            agent_radius=self.agent_radius,
            desired_speed=self.desired_speed,
            view_range=self.view_range,
            fov_deg=self.fov_deg,
            exits=list(self.exits),
            interior_walls=list(self.interior_walls),
            circles=list(self.circles),
            source_path=self.source_path,
        )


def _as_point(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def load_experiment(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    agents = raw.get('agents') or {}
    perception = raw.get('perception') or {}
    obstacles = raw.get('obstacles') or {}
    exits: list[SideExit] = []
    for item in raw.get('exits') or []:
        if 'side' in item:
            exits.append(SideExit(
                side=item['side'],
                center=float(item['center']),
                width=float(item.get('width', 3.0)),
                name=item.get('name'),
            ))
            continue
        raise ValueError(f'Exit must have a side and center: {item}')

    interior_walls = []
    for item in obstacles.get('walls') or []:
        if isinstance(item, dict):
            interior_walls.append((_as_point(item['start']), _as_point(item['end'])))
        else:
            interior_walls.append((_as_point(item[0]), _as_point(item[1])))

    circles = []
    for item in obstacles.get('circles') or []:
        center = item['center'] if isinstance(item, dict) else item[0]
        radius = item['radius'] if isinstance(item, dict) else item[1]
        circles.append((_as_point(center), float(radius)))

    return ExperimentConfig(
        name=str(raw.get('name', path.stem)),
        width=float(raw.get('width', 100)),
        height=float(raw.get('height', 100)),
        tom_order=parse_tom_order(raw.get('tom_order', 0)),
        agent_count=int(agents.get('count', raw.get('agent_count', 400))),
        agent_radius=float(agents.get('radius', 0.3)),
        desired_speed=float(agents.get('desired_speed', 2.0)),
        view_range=float(perception.get('view_range', 40.0)),
        fov_deg=float(perception.get('fov_deg', 360.0)),
        exits=exits,
        interior_walls=interior_walls,
        circles=circles,
        source_path=path,
    )


def save_experiment(config: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        'name': config.name,
        'width': config.width,
        'height': config.height,
        'tom_order': None if config.tom_order < 0 else config.tom_order,
        'agents': {
            'count': config.agent_count,
            'radius': config.agent_radius,
            'desired_speed': config.desired_speed,
        },
        'perception': {
            'view_range': config.view_range,
            'fov_deg': config.fov_deg,
        },
        'exits': [
            {
                'side': spec.side,
                'center': spec.center,
                'width': spec.width,
                **({'name': spec.name} if spec.name else {}),
            }
            for spec in config.exits
        ],
        'obstacles': {
            'walls': [
                {'start': list(start), 'end': list(end)}
                for start, end in config.interior_walls
            ],
            'circles': [
                {'center': list(center), 'radius': radius}
                for center, radius in config.circles
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _position_is_free(
        candidate: np.ndarray,
        radius: float,
        placed: np.ndarray,
        obstacles: list[Obstacle],
) -> bool:
    min_gap = 2.0 * radius + SPAWN_CLEARANCE
    if len(placed) > 0 and np.min(np.linalg.norm(placed - candidate, axis=1)) < min_gap:
        return False
    for obstacle in obstacles:
        if float(obstacle.distances(candidate).distance[0]) < radius + SPAWN_CLEARANCE:
            return False
    return True


def spawn_positions(
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
    max_attempts = SPAWN_ATTEMPTS * max(count, 1)
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
    if count >= 2:
        dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        min_gap = 2.0 * radius + SPAWN_CLEARANCE
        if dist.min() < min_gap:
            raise RuntimeError(f'spawn overlap: min dist {dist.min()} < {min_gap}')
    return positions


def build_environment(config: ExperimentConfig) -> Environment:
    side_exits = [Exit(*exit_segment(config.width, config.height, spec)) for spec in config.exits]
    obstacles: list[Obstacle] = list(perimeter_walls(config.width, config.height, config.exits))
    for start, end in config.interior_walls:
        obstacles.append(Wall(start, end))
    for center, radius in config.circles:
        obstacles.append(Circle(center, radius))

    agents = []
    for i, agent_pos in enumerate(spawn_positions(
            config.agent_count,
            config.agent_radius,
            config.width,
            config.height,
            obstacles,
    )):
        angle = random() * 2.0 * pi
        agents.append(Agent(
            social_repulsion=(2e3, 0.08),
            idx=i,
            mass=1.0,
            radius=config.agent_radius,
            position=np.array(agent_pos, dtype=float, copy=True),
            velocity=np.zeros(2, dtype=float),
            desired_speed=np.float32(config.desired_speed),
            desired_direction=np.array([cos(angle), sin(angle)], dtype=float),
            tau=1.0,
        ))

    environment = Environment(
        int(config.width),
        int(config.height),
        agents,
        view_range=config.view_range,
        fov_rad=np.deg2rad(config.fov_deg),
        tom_order=config.tom_order,
    )
    for ext in side_exits:
        environment.add_exit(ext)
    for obstacle in obstacles:
        environment.add_obstacle(obstacle)
    return environment


def list_experiments(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob('*.yaml'))
