from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io

import pygame
from pygame.surface import SurfaceType
from environment import AGENT_TYPES

PLOT_WIDTH = 360
SIM_SIZE = 1000
WINDOW_SIZE = (SIM_SIZE + PLOT_WIDTH, SIM_SIZE)

PLOT_COLORS = {
    'remaining': (30, 30, 30),
    'escaped': (0, 140, 70),
    'ToM-0': (40, 90, 200),
    'ToM-1': (190, 55, 150),
    'reactive': (190, 110, 20),
    'seeing exit': (0, 160, 90),
    'seeing sign': (0, 130, 160),
    'memory': (220, 110, 0),
    'grid': (210, 210, 210),
    'panel': (245, 245, 245),
}


@dataclass
class History:
    type_rows: list[dict] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    remaining: list[float] = field(default_factory=list)
    escaped: list[float] = field(default_factory=list)
    remaining_tom0: list[float] = field(default_factory=list)
    remaining_tom1: list[float] = field(default_factory=list)
    remaining_reactive: list[float] = field(default_factory=list)
    seeing_exit: list[float] = field(default_factory=list)
    seeing_sign: list[float] = field(default_factory=list)
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
        self.seeing_exit.append(float(stats['seeing_exit']))
        self.seeing_sign.append(float(stats['seeing_sign']))
        self.with_belief.append(float(stats['with_belief']))
        self.memory_guided.append(float(stats['memory_guided']))
        self.no_belief.append(float(stats['no_belief']))
        for agent_type in AGENT_TYPES:
            self.type_rows.append({'t': float(stats['t']), 'agent_type': agent_type,
                                   **stats['by_type'][agent_type]})

    def clear(self) -> None:
        self.type_rows.clear()
        self.t.clear()
        self.remaining.clear()
        self.escaped.clear()
        self.remaining_tom0.clear()
        self.remaining_tom1.clear()
        self.remaining_reactive.clear()
        self.seeing_exit.clear()
        self.seeing_sign.clear()
        self.with_belief.clear()
        self.memory_guided.clear()
        self.no_belief.clear()

    def to_csv(self) -> str:
        columns = (
            't', 'agent_type', 'initial', 'remaining', 'escaped', 'seeing_exit',
            'seeing_sign', 'with_belief', 'memory_guided', 'no_belief',
            'tom1_changed', 'blind_committed', 'mean_uncertainty', 'mean_age',
            'max_commit_distance', 'peak_memory_guided', 'peak_commit_distance',
            'blind_committed_s', 'evacuation_time', 'escaped_by_exit',
        )
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in self.type_rows:
            writer.writerow(row)
        return output.getvalue()

    def auc(self) -> dict[str, float]:
        """AUC of the remaining agents over time, for each type."""
        if len(self.t) < 2:
            return {'ToM-0': 0.0, 'ToM-1': 0.0}

        def trapezoidal_auc(values: list[float]) -> float:
            area = 0.0
            for i in range(1, len(values)):
                dt = self.t[i] - self.t[i - 1]
                area += 0.5 * (values[i] + values[i - 1]) * dt
            return area

        return {
            'ToM-0': trapezoidal_auc(self.remaining_tom0),
            'ToM-1': trapezoidal_auc(self.remaining_tom1),
        }


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
    plot = pygame.Rect(rect.x + 48, rect.y + 49, rect.width - 58, rect.height - 92)
    for tick in range(5):
        fraction = tick / 4
        x = round(plot.x + fraction * plot.width)
        y = round(plot.bottom - fraction * plot.height)
        pygame.draw.line(screen, (225, 225, 225), (x, plot.y), (x, plot.bottom), 1)
        pygame.draw.line(screen, (225, 225, 225), (plot.x, y), (plot.right, y), 1)
        x_label = font.render(f'{fraction * tmax:.1f}', False, (70, 70, 70))
        screen.blit(x_label, (x - x_label.get_width() // 2, plot.bottom + 2))
        y_label = font.render(f'{fraction * ymax:.1f}', False, (70, 70, 70))
        screen.blit(y_label, (plot.x - y_label.get_width() - 5, y - y_label.get_height() // 2))
    pygame.draw.rect(screen, PLOT_COLORS['grid'], plot, 1)
    x_title = font.render('time (s)', False, (70, 70, 70))
    screen.blit(x_title, (plot.centerx - x_title.get_width() // 2, plot.bottom + 21))
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
            'seeing exit': history.seeing_exit,
            'seeing sign': history.seeing_sign,
            'memory': history.memory_guided,
        },
        tmax, max(initial, 1), font, 'selected-exit cue',
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


def save_auc_plot(history: History, path: str) -> None:
    """Save the ToM-0 and ToM-1 AUC plot to a file."""
    width, height = 900, 560
    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 24)
    auc_values = history.auc()

    chart_rect = pygame.Rect(30, 30, width - 40, height - 100)
    tmax = history.t[-1] if history.t else 1.0
    ymax = max(max(history.remaining_tom0, default=0), max(history.remaining_tom1, default=0), 1)

    _chart(surface,
           chart_rect,
           {
                'ToM-0': history.remaining_tom0,
                'ToM-1': history.remaining_tom1,
           },
           tmax, ymax, font,
           'Agents remaining over time',
           )

    summary = (f"ToM-0 AUC: {auc_values['ToM-0']:.2f} | "
               f"ToM-1 AUC: {auc_values['ToM-1']:.2f}"
    )

    surface.blit(font.render(summary, False, (20, 20, 20)), (35, height - 50))
    pygame.image.save(surface, path)
