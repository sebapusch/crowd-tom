import numpy as np

from obstacle import Wall, Point, _as_positions


class Exit(Wall):
    def __init__(self, start: Point, end: Point):
        super().__init__(start, end)

    def is_crossed(self, previous_position, current_position) -> bool:
        """Return whether an agent's movement segment crosses this exit."""
        return bool(self.is_crossed_many(previous_position, current_position)[0])

    def is_crossed_many(self, previous_positions, current_positions) -> np.ndarray:
        previous_positions = _as_positions(previous_positions)
        current_positions = _as_positions(current_positions)

        movement = current_positions - previous_positions
        exit_segment = self.end - self.start
        denominator = (
            movement[:, 0] * exit_segment[1]
            - movement[:, 1] * exit_segment[0]
        )

        to_exit = self.start - previous_positions
        safe_denominator = np.where(np.abs(denominator) < 1e-12, 1.0, denominator)
        movement_fraction = (
            to_exit[:, 0] * exit_segment[1]
            - to_exit[:, 1] * exit_segment[0]
        ) / safe_denominator
        exit_fraction = (
            to_exit[:, 0] * movement[:, 1]
            - to_exit[:, 1] * movement[:, 0]
        ) / safe_denominator

        return (
            (np.abs(denominator) >= 1e-12)
            & (movement_fraction >= 0.0)
            & (movement_fraction <= 1.0)
            & (exit_fraction >= 0.0)
            & (exit_fraction <= 1.0)
        )
