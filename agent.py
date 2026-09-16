from __future__ import annotations

import numpy as np

class Agent:
    def __init__(self,
                 idx: int,
                 mass: float,
                 radius: float,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 des_speed: np.ndarray,
                 des_dir: np.ndarray,
                 tau: float):
        self.idx = idx
        self.mass = mass
        self.radius = radius
        self.tau = tau
        self.pos = pos
        self.vel = vel
        self.des_speed = des_speed
        self.des_dir = des_dir

    def tick(self, t: int, delta_t: int, others: list[Agent]) -> None:
        self.update_desired(others)

        f_drive = self.mass * (self.des_speed * self.des_dir - self.vel) / self.tau



    def update_desired(self, others: list[Agent]) -> None:
        return

    def f_others(self, others: list[Agent]) -> np.ndarray:
        f_others = np.zeros(2)

        for other in others:
            if other.idx == self.idx: continue

            dist = np.linalg.norm(other.pos - self.pos)
            if dist >= other.radius + self.radius:
                continue

            unit_normal = (self.pos - other.pos) / dist


        return f_others


