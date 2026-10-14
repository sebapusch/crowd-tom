from __future__ import annotations

from dataclasses import dataclass, field

import pygame
from pygame.surface import SurfaceType

PLOT_WIDTH = 360
SIM_SIZE = 1000
WINDOW_SIZE = (SIM_SIZE + PLOT_WIDTH, SIM_SIZE)

PLOT_COLORS = {
    'remaining': (30, 30, 30),
    'escaped': (0, 140, 70),
    'ToM-0': (40, 90, 200),
    'ToM-1': (190, 55, 150),
    'reactive': (190, 110, 20),
    'seeing': (0, 160, 90),
    'belief': (40, 90, 200),
    'memory': (220, 110, 0),
    'none': (140, 140, 140),
    'grid': (210, 210, 210),
    'panel': (245, 245, 245),
}


@dataclass
class History:
    t: list[float] = field(default_factory=list)
    remaining: list[float] = field(default_factory=list)
    escaped: list[float] = field(default_factory=list)
    remaining_tom0: list[float] = field(default_factory=list)
    remaining_tom1: list[float] = field(default_factory=list)
    remaining_reactive: list[float] = field(default_factory=list)
    seeing: list[float] = field(default_factory=list)
    with_belief: list[float] = field(default_factory=list)
    memory_guided: list[float] = field(default_factory=list)
    no_belief: list[float] = field(default_factory=list)

    def record(self, stats: dict) -> None:
        if self.t and stats['t'] < self.t[-1]:
            self.clear()
        if self.t and abs(stats['t'] - self.t[-1]) < 1e-9:
            return
        self.t.append(float(stats['t']))
        self.remaining.append(float(stats['n']))
        self.escaped.append(float(stats['escaped']))
        self.remaining_tom0.append(float(stats['remaining_tom0']))
        self.remaining_tom1.append(float(stats['remaining_tom1']))
        self.remaining_reactive.append(float(stats['remaining_reactive']))
        self.seeing.append(float(stats['seeing']))
        self.with_belief.append(float(stats['with_belief']))
        self.memory_guided.append(float(stats['memory_guided']))
        self.no_belief.append(float(stats['no_belief']))

    def clear(self) -> None:
        self.t.clear()
        self.remaining.clear()
        self.escaped.clear()
        self.remaining_tom0.clear()
        self.remaining_tom1.clear()
        self.remaining_reactive.clear()
        self.seeing.clear()
        self.with_belief.clear()
        self.memory_guided.clear()
        self.no_belief.clear()

    def to_csv(self) -> str:
        lines = ['t,remaining,remaining_tom0,remaining_tom1,remaining_reactive,escaped,seeing,with_belief,memory_guided,no_belief']
        for row in zip(
                self.t, self.remaining, self.remaining_tom0, self.remaining_tom1,
                self.remaining_reactive, self.escaped, self.seeing,
                self.with_belief, self.memory_guided, self.no_belief,
        ):
            lines.append(','.join(f'{value:.4f}' for value in row))
        return '\n'.join(lines) + '\n'


def _polyline(screen: SurfaceType, xs: list[float], ys: list[float], color: tuple[int, int, int], width: int = 2) -> None:
    if len(xs) < 2:
        return
    points = list(zip(xs, ys))
    pygame.draw.lines(screen, color, False, points, width)


def _chart(
        screen: SurfaceType,
        rect: pygame.Rect,
        series: dict[str, list[float]],
        tmax: float,
        ymax: float,
        font: pygame.font.Font,
        title: str,
) -> None:
    pygame.draw.rect(screen, (255, 255, 255), rect)
    pygame.draw.rect(screen, (180, 180, 180), rect, 1)
    screen.blit(font.render(title, False, (20, 20, 20)), (rect.x + 8, rect.y + 4))
    plot = pygame.Rect(rect.x + 36, rect.y + 49, rect.width - 44, rect.height - 61)
    pygame.draw.rect(screen, PLOT_COLORS['grid'], plot, 1)
    if tmax > 0 and ymax > 0:
        for name, values in series.items():
            if len(values) < 2:
                continue
            xs = []
            ys = []
            count = len(values)
            for i, value in enumerate(values):
                t_frac = i / max(count - 1, 1)
                xs.append(plot.x + t_frac * plot.width)
                ys.append(plot.bottom - (value / ymax) * plot.height)
            _polyline(screen, xs, ys, PLOT_COLORS[name])
    legend_y = rect.y + 26
    legend_x = rect.x + 8
    for name in series:
        label = font.render(name, False, PLOT_COLORS[name])
        screen.blit(label, (legend_x, legend_y))
        legend_x += label.get_width() + 12


def draw_plots(
        screen: SurfaceType,
        history: History,
        font: pygame.font.Font,
        initial: int,
) -> None:
    panel = pygame.Rect(SIM_SIZE, 0, PLOT_WIDTH, SIM_SIZE)
    pygame.draw.rect(screen, PLOT_COLORS['panel'], panel)
    pygame.draw.line(screen, (160, 160, 160), (SIM_SIZE, 0), (SIM_SIZE, SIM_SIZE), 2)
    screen.blit(font.render('live metrics', False, (20, 20, 20)), (SIM_SIZE + 16, 16))

    tmax = history.t[-1] if history.t else 1.0
    top = pygame.Rect(SIM_SIZE + 12, 48, PLOT_WIDTH - 24, 280)
    bottom = pygame.Rect(SIM_SIZE + 12, 350, PLOT_WIDTH - 24, 280)
    agent_series = {'remaining': history.remaining}
    if any(history.remaining_tom0):
        agent_series['ToM-0'] = history.remaining_tom0
    if any(history.remaining_tom1):
        agent_series['ToM-1'] = history.remaining_tom1
    if any(history.remaining_reactive):
        agent_series['reactive'] = history.remaining_reactive
    agent_series['escaped'] = history.escaped
    _chart(screen, top, agent_series, tmax, max(initial, 1), font, 'remaining by type')
    _chart(
        screen, bottom,
        {
            'seeing': history.seeing,
            'belief': history.with_belief,
            'memory': history.memory_guided,
            'none': history.no_belief,
        },
        tmax, max(initial, 1), font, 'ToM state',
    )

    help_lines = [
        'Space pause  R reset',
        'T ToM order   [ ] experiment',
        'E edit mode   H help',
        'click-drag wall (edit)',
        'C+drag circle (edit)',
        'click border = add exit',
        'Backspace undo obstacle',
        'F save yaml   G save csv',
        '+ / - speed',
    ]
    y = 660
    for line in help_lines:
        screen.blit(font.render(line, False, (60, 60, 60)), (SIM_SIZE + 16, y))
        y += 22
