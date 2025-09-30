
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Level:
    grid: List[List[int]]
    width: int
    height: int

    @staticmethod
    def from_strings(rows: List[str]) -> Tuple['Level', Tuple[float,float]]:
        g = []
        spawn = (1.5, 1.5)
        for y, r in enumerate(rows):
            row = []
            for x, ch in enumerate(r):
                if ch == '1': row.append(1)
                elif ch == '2':
                    spawn = (x+0.5, y+0.5)
                    row.append(0)
                else: row.append(0)
            g.append(row)
        h = len(g); w = len(g[0])
        return Level(g, w, h), spawn

    def is_solid(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height: return True
        return self.grid[y][x] == 1


