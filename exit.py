import numpy as np

from obstacle import Wall, Point


class Exit(Wall):
    def __init__(self, start: Point, end: Point):
        super().__init__(start, end)

    def is_crossed(self, previous_position, current_position) -> bool:
        """Return whether an agent's movement segment crosses this exit."""
        previous_position = np.asarray(previous_position, dtype=float)
        current_position = np.asarray(current_position, dtype=float)

        movement = current_position - previous_position
        exit_segment = self.end - self.start
        denominator = (
            movement[0] * exit_segment[1]
            - movement[1] * exit_segment[0]
        )

        # A parallel movement does not pass through the exit boundary.
        if abs(denominator) < 1e-12:
            return False

        to_exit = self.start - previous_position
        movement_fraction = (
            to_exit[0] * exit_segment[1]
            - to_exit[1] * exit_segment[0]
        ) / denominator
        exit_fraction = (
            to_exit[0] * movement[1]
            - to_exit[1] * movement[0]
        ) / denominator

        return (
            0.0 <= movement_fraction <= 1.0
            and 0.0 <= exit_fraction <= 1.0
        )
