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
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pygame
import moderngl

# =========================
# Config
# =========================
WIDTH, HEIGHT = 1280, 720
FOV_DEG = 70.0
NEAR, FAR = 0.1, 100.0
SENSITIVITY = 0.15
MOVE_SPEED = 3.0   # m/s
CPU_MODE = "CPU_MV"   # We'll let GPU multiply projection; CPU sends MV
VSYNC = True
IMAGE_Y_FLIP = True

# =========================
# Math helpers
# =========================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0, 0,                 0],
        [0,        f, 0,                 0],
        [0,        0, (far+near)*nf,     2*far*near*nf],
        [0,        0, -1,                0],
    ], dtype=np.float32)


def look_yaw_pitch(eye: np.ndarray, yaw: float, pitch: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yaw around Y, pitch around X
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = np.array([sy*cp, -sp, cy*cp], dtype=np.float32)  # right-handed
    target = eye + forward
    up = np.array([0,1,0], dtype=np.float32)
    # Build lookAt
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.identity(4, dtype=np.float32)
    t[:3, 3] = -eye
    V = m @ t
    return V, f, s


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.identity(4, dtype=np.float32)
    M[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return M


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.diag([sx, sy, sz, 1.0]).astype(np.float32)

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

# =========================
# Renderer (walls: instanced cubes; floor/ceiling: large planes; sprites: billboards)
# =========================
WALL_VS = """
#version 330
uniform mat4 uP;
uniform mat4 uV;
uniform mat4 uM;  // per draw

in vec3 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main(){
    gl_Position = uP * uV * uM * vec4(in_pos,1.0);
    v_uv = in_uv;
}
"""

SPRITE_VS = """
#version 330
uniform mat4 uP;
uniform mat4 uV;
uniform vec3 uCenter;   // world position of sprite center
uniform vec2 uSize;     // width,height in world units

// A unit quad in NDC-like local space: (-0.5,-0.5) .. (0.5,0.5)
in vec2 in_unit;
in vec2 in_uv;

out vec2 v_uv;

void main(){
    // Build billboard in world space using camera right/up from uV
    mat3 R;
    R[0] = vec3(uV[0][0], uV[1][0], uV[2][0]); // camera right (world)
    R[1] = vec3(uV[0][1], uV[1][1], uV[2][1]); // camera up (world)
    R[2] = vec3(uV[0][2], uV[1][2], uV[2][2]); // camera -forward

    vec3 right = R[0];
    vec3 up    = R[1];

    vec2 halfs = 0.5 * uSize;
    vec3 world_pos = uCenter + right * (in_unit.x * uSize.x) + up * (in_unit.y * uSize.y);
    gl_Position = uP * uV * vec4(world_pos, 1.0);
    v_uv = in_uv;
}
"""

TEX_FS = """
#version 330
uniform sampler2D uTex;
in vec2 v_uv;
out vec4 fragColor;
void main(){
    vec4 c = texture(uTex, v_uv);
    if(c.a < 0.1) discard;
    fragColor = c;
}
"""

HUD_VS = """
#version 330
// Render in screen space (0..1) then map to NDC
in vec2 in_pos; // (0..1)
in vec2 in_uv;
out vec2 v_uv;
void main(){
    vec2 ndc = in_pos * 2.0 - 1.0;
    ndc.y = -ndc.y; // flip Y for screen coords
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

HUD_FS = TEX_FS

class GL:
    def __init__(self):
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        if VSYNC:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Programs
        self.prog_wall   = self.ctx.program(vertex_shader=WALL_VS,   fragment_shader=TEX_FS)
        self.prog_sprite = self.ctx.program(vertex_shader=SPRITE_VS, fragment_shader=TEX_FS)
        self.prog_hud    = self.ctx.program(vertex_shader=HUD_VS,    fragment_shader=HUD_FS)

        # Uniform handles
        self.uP_wall = self.prog_wall["uP"]; self.uV_wall = self.prog_wall["uV"]; self.uM_wall = self.prog_wall["uM"]
        self.uP_spr  = self.prog_sprite["uP"]; self.uV_spr  = self.prog_sprite["uV"]; self.uCenter = self.prog_sprite["uCenter"]; self.uSize = self.prog_sprite["uSize"]

        # Geometry buffers
        self._init_wall_geo()
        self._init_sprite_geo()
        self._init_hud_geo()

        # Projection
        aspect = WIDTH / HEIGHT
        self.P = perspective(FOV_DEG, aspect, NEAR, FAR)

        # Textures
        self.white = self._make_white()
        self.tex_wall  = self._load_texture("assets/texture_wall.png", allow_alpha=False) or self.white
        self.tex_floor = self._load_texture("assets/texture_floor.png", allow_alpha=False) or self.white
        self.tex_enemy = self._load_texture("assets/texture_enemy.png", allow_alpha=True) or self.white
        self.tex_weapon= self._load_texture("assets/texture_weapon.png", allow_alpha=True) or self.white


    def _make_white(self):
        t = self.ctx.texture((1,1), 4, data=bytes([255,255,255,255]))
        t.build_mipmaps(); t.filter=(moderngl.LINEAR, moderngl.LINEAR); t.wrap = ('repeat', 'repeat')
        return t

    def _load_texture(self, path: str, allow_alpha: bool) -> Optional[moderngl.Texture]:
        try:
            surf = pygame.image.load(path)
            has_alpha = surf.get_masks()[3] != 0
            if allow_alpha and has_alpha:
                surf = surf.convert_alpha(); mode="RGBA"; comps=4
            else:
                surf = surf.convert(); mode="RGB"; comps=3
            if IMAGE_Y_FLIP:
                surf = pygame.transform.flip(surf, False, True)
            img = pygame.image.tostring(surf, mode, False)
            tex = self.ctx.texture(surf.get_size(), comps, img)
            tex.build_mipmaps(); tex.filter = ('linear_mipmap_linear', 'linear'); tex.wrap = ('repeat', 'repeat')
            return tex
        except Exception as e:
            print(f"[WARN] texture load failed: {path}: {e}")
            return None

    def _init_wall_geo(self):
        # Unit cube (size 1) centered at origin, with tiling UVs for walls
        # We'll scale/translate per tile via uM
        verts = []
        def quad(a,b,c,d, uv_scale=(1,1)):
            ax,ay,az = a; bx,by,bz=b; cx,cy,cz=c; dx,dy,dz=d
            us,vs = uv_scale
            verts.extend([ax,ay,az, 0,0,  bx,by,bz, us,0,  cx,cy,cz, us,vs])
            verts.extend([ax,ay,az, 0,0,  cx,cy,cz, us,vs, dx,dy,dz, 0,vs])
        # Cube faces (only need 4 for walls; floor/ceil separate)
        # +X
        quad((0.5,0.0,-0.5),(0.5,1.0,-0.5),(0.5,1.0,0.5),(0.5,0.0,0.5), (1,1))
        # -X
        quad((-0.5,0.0,0.5),(-0.5,1.0,0.5),(-0.5,1.0,-0.5),(-0.5,0.0,-0.5), (1,1))
        # +Z
        quad((-0.5,0.0,0.5),(-0.5,1.0,0.5),(0.5,1.0,0.5),(0.5,0.0,0.5), (1,1))
        # -Z
        quad((0.5,0.0,-0.5),(0.5,1.0,-0.5),(-0.5,1.0,-0.5),(-0.5,0.0,-0.5), (1,1))
        self.vbo_wall = self.ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
        self.vao_wall = self.ctx.vertex_array(self.prog_wall, [(self.vbo_wall, '3f 2f', 'in_pos', 'in_uv')])

        # Floor/Ceiling plane (tile many times)
        s = 100.0
        fs = 50.0
        floor = np.array([
            -s,0,-s,   0,0,   s,0,-s,   fs,0,   s,0, s,   fs,fs,
            -s,0,-s,   0,0,   s,0, s,   fs,fs,  -s,0, s,   0,fs,
        ], dtype=np.float32)
        self.vbo_floor = self.ctx.buffer(floor.tobytes())
        self.vao_floor = self.ctx.vertex_array(self.prog_wall, [(self.vbo_floor,'3f 2f','in_pos','in_uv')])

    def _init_sprite_geo(self):
        # Unit quad for sprites in local space: corners (-0.5,-0.5) .. (0.5,0.5)
        data = np.array([
            -0.5,-0.5,   0,0,
             0.5,-0.5,   1,0,
             0.5, 0.5,   1,1,
            -0.5,-0.5,   0,0,
             0.5, 0.5,   1,1,
            -0.5, 0.5,   0,1,
        ], dtype=np.float32)
        self.vbo_sprite = self.ctx.buffer(data.tobytes())
        self.vao_sprite = self.ctx.vertex_array(self.prog_sprite, [(self.vbo_sprite,'2f 2f','in_unit','in_uv')])

    def _init_hud_geo(self):
        # Fullscreen-aligned quad area for weapon (bottom center)
        w, h = 0.4, 0.4
        x0 = 0.5 - w/2; x1 = 0.5 + w/2
        y0 = 0.95 - h;  y1 = 0.95
        data = np.array([
            x0,y0, 0,1,  x1,y0, 1,1,  x1,y1, 1,0,
            x0,y0, 0,1,  x1,y1, 1,0,  x0,y1, 0,0,
        ], dtype=np.float32)
        self.vbo_hud = self.ctx.buffer(data.tobytes())
        self.vao_hud = self.ctx.vertex_array(self.prog_hud, [(self.vbo_hud,'2f 2f','in_pos','in_uv')])

    def begin_frame(self):
        self.ctx.viewport = (0,0,WIDTH,HEIGHT)
        self.ctx.clear(0.05,0.08,0.1)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_floor(self, V):
        self.tex_floor.use(0)
        self.prog_wall['uP'].write(self.P.tobytes())
        self.prog_wall['uV'].write(V.tobytes())
        self.prog_wall['uM'].write(np.identity(4, dtype=np.float32).tobytes())
        self.vao_floor.render()

    def draw_wall_block(self, V, wx, wy):
        self.tex_wall.use(0)
        self.prog_wall['uP'].write(self.P.tobytes())
        self.prog_wall['uV'].write(V.tobytes())
        M = translate(wx+0.5, 0.0, wy+0.5) @ scale(1.0,1.0,1.0)
        self.prog_wall['uM'].write(M.tobytes())
        self.vao_wall.render()

    def draw_sprite(self, V, center: Tuple[float,float,float], size: Tuple[float,float], tex: moderngl.Texture):
        tex.use(0)
        self.prog_sprite['uP'].write(self.P.tobytes())
        self.prog_sprite['uV'].write(V.tobytes())
        self.prog_sprite['uCenter'].write(np.array(center, dtype=np.float32).tobytes())
        self.prog_sprite['uSize'].write(np.array(size, dtype=np.float32).tobytes())
        self.vao_sprite.render()

    def draw_weapon(self):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.tex_weapon.use(0)
        self.vao_hud.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

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
                    # simple square check
                    if abs((tx+0.5)-p[0]) < 0.5+radius and abs((tz+0.5)-p[2]) < 0.5+radius:
                        return True
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

# =========================
# Main game
# =========================
class Game:
    def __init__(self):
        pygame.init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.gl = GL()
        self.clock = pygame.time.Clock()

        # Level
        self.level, spawn = Level.from_strings(LEVEL_STR)

        # Player
        self.player = Player(pos=np.array([spawn[0], 1.0, spawn[1]], dtype=np.float32), yaw=0.0, pitch=0.0)

        # One enemy
        self.enemy = Enemy(pos=np.array([spawn[0]+3.0, 1.0, spawn[1]+1.0], dtype=np.float32))

    def run(self):
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
            self.gl.begin_frame()
            self.gl.draw_floor(V)
            # draw walls
            for y in range(self.level.height):
                for x in range(self.level.width):
                    if self.level.grid[y][x] == 1:
                        self.gl.draw_wall_block(V, x, y)

            # sprites (enemy at y=1.0 center, size 1m x 1.8m)
            self.gl.draw_sprite(V, (float(self.enemy.pos[0]), 1.0, float(self.enemy.pos[2])), (1.0, 1.8), self.gl.tex_enemy)

            # weapon HUD
            self.gl.draw_weapon()

            pygame.display.flip()
        pygame.quit()

if __name__ == "__main__":
    Game().run()
