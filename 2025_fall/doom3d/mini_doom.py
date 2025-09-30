"""
Mini DOOM-Style Demo — Skeleton
Hybrid renderer: CPU (NumPy for transforms) + GPU (moderngl for raster/texture)

Features (MVP):
- Level from simple ASCII grid map (wall/block layout)
- Textured walls/floor/ceiling
- Player WASD + mouse-look (yaw/pitch clamped)
- Collision vs. walls (grid-based)
- Billboard sprites (enemy + pickups) that always face the camera
- Simple enemy AI (idle → chase when in range)
- Weapon sprite rendered as 2D overlay (no depth)

Assets expected:
- assets/texture_wall.png  (tileable)
- assets/texture_floor.png (tileable)
- assets/texture_enemy.png (billboard, alpha)
- assets/texture_weapon.png (2D overlay, alpha)

Note: This is a teaching skeleton. Error handling and edge cases are kept minimal.
"""
# import math
# from dataclasses import dataclass
# from typing import Tuple, Optional, List

import numpy as np
import pygame
import math
# import moderngl

import Renderer
import Config
import Level
#import Player
from Player import Player, Enemy
from Linalg import look_yaw_pitch, perspective, translate, scale, normalize

# =========================
# Config
# =========================
# WIDTH, HEIGHT = 1280, 720
# FOV_DEG = 70.0
# NEAR, FAR = 0.1, 100.0
# SENSITIVITY = 0.15
# MOVE_SPEED = 3.0   # m/s
# CPU_MODE = "CPU_MV"   # We'll let GPU multiply projection; CPU sends MV
# VSYNC = True
# IMAGE_Y_FLIP = True



# =========================
# Simple grid map (1 = wall, 0 = empty)
# =========================
LEVEL_STR = [
    "111111111111",
    "100000000001",
    "101001110001",
    "100000010001",
    "100200010001",
    "100000000001",
    "111111111111",
]
# '2' marks initial player spawn

# =========================
# Main game
# =========================
class Game:
    def __init__(self):

        # Config
        self.cfg = Config()

        pygame.init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        self.render = Renderer()
        self.clock = pygame.time.Clock()

        # Level
        self.level, spawn = Level.from_strings(LEVEL_STR)

        # Player
        self.player = Player(pos=np.array([spawn[0], 1.0, spawn[1]], dtype=np.float32), yaw=0.0, pitch=0.0)

        # One enemy
        self.enemy = Enemy(pos=np.array([spawn[0]+3.0, 1.0, spawn[1]+1.0], dtype=np.float32))

    def run(self):
        cfg = self.cfg
        SENSITIVITY = cfg.sensitivity
        MOVE_SPEED = cfg.move_speed

        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                elif e.type == pygame.MOUSEMOTION:
                    dx, dy = e.rel
                    self.player.yaw   += dx * SENSITIVITY * 0.01
                    self.player.pitch += dy * SENSITIVITY * 0.01
                    self.player.pitch = float(np.clip(self.player.pitch, -1.2, 1.2))

            # Input
            keys = pygame.key.get_pressed()
            vx = 0.0; vz = 0.0
            # forward/back in camera space
            fwd = np.array([math.sin(self.player.yaw), 0.0, math.cos(self.player.yaw)], dtype=np.float32)
            right = np.array([fwd[2], 0.0, -fwd[0]], dtype=np.float32)
            if keys[pygame.K_w]: vz += 1.0
            if keys[pygame.K_s]: vz -= 1.0
            if keys[pygame.K_a]: vx -= 1.0
            if keys[pygame.K_d]: vx += 1.0
            move = normalize(vx*right + vz*fwd) * MOVE_SPEED if (vx!=0 or vz!=0) else np.array([0,0,0], dtype=np.float32)
            self.player.move_and_collide(self.level, move[0], move[2], dt)

            # Camera matrix
            eye = self.player.pos
            V, forward, cam_right = look_yaw_pitch(eye, self.player.yaw, self.player.pitch)

            # Update enemy
            self.enemy.update(self.player, dt, self.level)

            # Render
            self.render.begin_frame()
            self.render.draw_floor(V)
            # draw walls
            for y in range(self.level.height):
                for x in range(self.level.width):
                    if self.level.grid[y][x] == 1:
                        self.render.draw_wall_block(V, x, y)

            # sprites (enemy at y=1.0 center, size 1m x 1.8m)
            self.render.draw_sprite(V, (float(self.enemy.pos[0]), 1.0, float(self.enemy.pos[2])), (1.0, 1.8), self.render.tex_enemy)

            # weapon HUD
            self.render.draw_weapon()

            pygame.display.flip()
        pygame.quit()

if __name__ == "__main__":
    Game().run()
