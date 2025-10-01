from dataclasses import dataclass
from typing import Tuple
import numpy as np
import math

import Level
from Linalg import normalize

# =========================
# Game objects
# =========================
@dataclass
class Player:
    pos: np.ndarray  # x,y,z
    yaw: float
    pitch: float

    def move_and_collide(self, level: Level, dx: float, dz: float, dt: float):
        step = np.array([dx, 0.0, dz], dtype=np.float32) * dt
        # Try x then z separately (AABB vs grid)
        newx = self.pos.copy(); newx[0] += step[0]
        if not self._blocked(level, newx):
            self.pos = newx
        newz = self.pos.copy(); newz[2] += step[2]
        if not self._blocked(level, newz):
            self.pos = newz

    def _blocked(self, level: Level, p: np.ndarray) -> bool:
        radius = 0.25
        cx, cz = int(p[0]), int(p[2])
        # Check surrounding 4 cells for a simple circle-vs-cell test
        for oy in (0,1):
            for ox in (0,1):
                tx, tz = int(math.floor(p[0]+(ox-0.5)*2*radius)), int(math.floor(p[2]+(oy-0.5)*2*radius))
                if level.is_solid(tx, tz):
                    return True
                    # simple square check
                    # if abs((tx+0.5)-p[0]) < 0.5+radius and abs((tz+0.5)-p[2]) < 0.5+radius:
                    #     return True
        return False

@dataclass
class Enemy:
    pos: np.ndarray
    speed: float = 1.5
    aggro_range: float = 6.0

    def update(self, player: Player, dt: float, level: Level):
        to_p = player.pos - self.pos
        dist = float(np.linalg.norm(to_p[[0,2]]))
        if dist < self.aggro_range:
            dir2 = normalize(np.array([to_p[0], 0.0, to_p[2]], dtype=np.float32))
            step = dir2 * self.speed * dt
            trial = self.pos + step
            # crude collision vs walls
            if not level.is_solid(int(trial[0]), int(trial[2])):
                self.pos = trial
