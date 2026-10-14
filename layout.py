from __future__ import annotations

from dataclasses import dataclass

from obstacle import Wall


SIDES = ('north', 'south', 'west', 'east')


@dataclass
class SideExit:
    side: str
    center: float
    width: float = 3.0
    name: str | None = None

    def __post_init__(self) -> None:
        side = self.side.lower()
        if side not in SIDES:
            raise ValueError(f'Unknown exit side {self.side!r}')
        self.side = side


def exit_segment(width: float, height: float, spec: SideExit) -> tuple[tuple[float, float], tuple[float, float]]:
    half = spec.width / 2.0
    lo = spec.center - half
    hi = spec.center + half
    if spec.side == 'north':
        lo = max(0.0, lo)
        hi = min(width, hi)
        return (lo, 0.0), (hi, 0.0)
    if spec.side == 'south':
        lo = max(0.0, lo)
        hi = min(width, hi)
        return (lo, height), (hi, height)
    if spec.side == 'west':
        lo = max(0.0, lo)
        hi = min(height, hi)
        return (0.0, lo), (0.0, hi)
    lo = max(0.0, lo)
    hi = min(height, hi)
    return (width, lo), (width, hi)


def _gaps_on_side(specs: list[SideExit], side: str, length: float) -> list[tuple[float, float]]:
    gaps = []
    for spec in specs:
        if spec.side != side:
            continue
        lo = max(0.0, spec.center - spec.width / 2.0)
        hi = min(length, spec.center + spec.width / 2.0)
        if hi > lo:
            gaps.append((lo, hi))
    gaps.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in gaps:
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def _segments_around_gaps(length: float, gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in gaps:
        if lo > cursor:
            pieces.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < length:
        pieces.append((cursor, length))
    return pieces


def perimeter_walls(width: float, height: float, exits: list[SideExit]) -> list[Wall]:
    walls: list[Wall] = []
    north = _segments_around_gaps(width, _gaps_on_side(exits, 'north', width))
    south = _segments_around_gaps(width, _gaps_on_side(exits, 'south', width))
    west = _segments_around_gaps(height, _gaps_on_side(exits, 'west', height))
    east = _segments_around_gaps(height, _gaps_on_side(exits, 'east', height))
    for lo, hi in north:
        walls.append(Wall((lo, 0.0), (hi, 0.0)))
    for lo, hi in south:
        walls.append(Wall((lo, height), (hi, height)))
    for lo, hi in west:
        walls.append(Wall((0.0, lo), (0.0, hi)))
    for lo, hi in east:
        walls.append(Wall((width, lo), (width, hi)))
    return walls


def nearest_boundary(
        x: float,
        y: float,
        width: float,
        height: float,
        threshold: float = 4.0,
) -> tuple[str, float] | None:
    candidates = [
        ('north', y, x),
        ('south', height - y, x),
        ('west', x, y),
        ('east', width - x, y),
    ]
    side, dist, along = min(candidates, key=lambda item: item[1])
    if dist > threshold:
        return None
    return side, along
