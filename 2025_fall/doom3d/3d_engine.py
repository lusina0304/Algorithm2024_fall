"""
Hybrid Renderer Skeleton
- CPU-side math (via NumPy): model/view/projection matrix multiplications
- GPU-side rasterization: via OpenGL (moderngl) inside a pygame window

Two modes are provided:
 1) CPU_MVP: CPU computes full clip-space position (P*V*M*position). Vertex shader is a pass-through.
 2) CPU_MV: CPU computes only M*V and sends a projection matrix to the shader (slightly more GPU usage).

This is an educational skeleton designed for 2–3rd year undergrad courses.
Fill in the TODOs for OBJ loading, camera movement, and game logic.
"""

import sys
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import pygame
import numpy as np
import moderngl

# =========================
# Config
# =========================
WIDTH, HEIGHT = 1280, 720
FOV_DEG = 70.0
NEAR = 0.1
FAR = 100.0
CPU_MODE = "CPU_MVP"  # "CPU_MVP" or "CPU_MV"
VSYNC = True

# =========================
# Math helpers (CPU via NumPy)
# =========================

def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < eps else v / n


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0, 0,                 0],
        [0,        f, 0,                 0],
        [0,        0, (far+near)*nf,     2*far*near*nf],
        [0,        0, -1,                0],
    ], dtype=np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, normalize(up)))
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.identity(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def rotate_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return m

# =========================
# Data structures
# =========================
@dataclass
class Mesh:
    # Interleaved per-vertex data: [x, y, z, w, u, v, nx, ny, nz]
    vbo: moderngl.Buffer
    n_vertices: int
    texture: Optional[moderngl.Texture] = None

# =========================
# Minimal OBJ loader (positions/uvs/normals) → interleaved array
# =========================

def load_obj_interleaved(ctx: moderngl.Context, path: str, texture: Optional[moderngl.Texture] = None) -> Mesh:
    # NOTE: Keep it simple — triangulated OBJ recommended.
    # TODO: Replace with a robust parser as needed.
    positions, uvs, normals = [], [], []
    indices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()[:4]
                positions.append((float(x), float(y), float(z)))
            elif line.startswith('vt '):
                parts = line.split()
                u, v = float(parts[1]), float(parts[2])
                uvs.append((u, 1.0 - v))  # flip V for typical image origin
            elif line.startswith('vn '):
                _, x, y, z = line.split()[:4]
                normals.append((float(x), float(y), float(z)))
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # face like: v/vt/vn
                f_inds = []
                for p in parts:
                    vv = p.split('/')
                    vi = int(vv[0]) - 1
                    ti = int(vv[1]) - 1 if len(vv) > 1 and vv[1] else 0
                    ni = int(vv[2]) - 1 if len(vv) > 2 and vv[2] else 0
                    # indices.append((vi, ti, ni))
                    f_inds.append((vi, ti, ni))
                indices.append(f_inds)

    def intoVert(face_ind, ind):
        for i in ind:
            (vi, ti, ni) = face_ind[i]
            x, y, z = positions[vi]
            u, v = uvs[ti] if uvs else (0.0, 0.0)
            nx, ny, nz = normals[ni] if normals else (0.0, 1.0, 0.0)
            # w will be filled later (clip-space), init 1.0
            verts.extend([x, y, z, 1.0, u, v, nx, ny, nz])

    vert_cnt = 0
    verts = []
    for f_inds in indices:
        intoVert(f_inds, [0, 1, 2])
        vert_cnt += 3

        if len(f_inds) > 3:
            intoVert(f_inds, [0, 2, 3])
            vert_cnt += 3

    vbo = ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
    return Mesh(vbo=vbo, n_vertices=vert_cnt, texture=texture)

# =========================
# Shaders
# =========================
PASS_THROUGH_VS = """
#version 330
in vec4 in_pos_clip;     // already in clip-space (x,y,z,w)
in vec2 in_uv;
in vec3 in_nrm;

out vec2 v_uv;
out vec3 v_nrm;

void main() {
    gl_Position = in_pos_clip;
    v_uv = in_uv;
    v_nrm = in_nrm; // for simple lighting if desired
}
"""

MV_VS = """
#version 330
uniform mat4 uP;

in vec4 in_pos_mv;       // already model-view transformed (w=1)
in vec2 in_uv;
in vec3 in_nrm;

out vec2 v_uv;
out vec3 v_nrm;

void main() {
    gl_Position = uP * in_pos_mv;
    v_uv = in_uv;
    v_nrm = in_nrm;
}
"""

FS = """
#version 330
uniform sampler2D uTex;
uniform vec3 uLightDir = normalize(vec3(0.5, 1.0, 0.3));

in vec2 v_uv;
in vec3 v_nrm;

out vec4 fragColor;

void main() {
    vec3 n = normalize(v_nrm);
    float diff = clamp(dot(n, normalize(uLightDir)), 0.0, 1.0);
    vec3 base = texture(uTex, v_uv).rgb;
    vec3 color = base * (0.2 + 0.8 * diff);
    fragColor = vec4(color, 1.0);
}
"""

# =========================
# Renderer
# =========================
class Renderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if VSYNC:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Hybrid CPU(pygame/NumPy) + GPU(moderngl)")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Choose program based on mode
        if CPU_MODE == "CPU_MVP":
            self.prog = self.ctx.program(vertex_shader=PASS_THROUGH_VS, fragment_shader=FS)
            self.layout = [
                ("in_pos_clip", 4, "f4"),
                ("in_uv", 2, "f4"),
                ("in_nrm", 3, "f4"),
            ]
        else:
            self.prog = self.ctx.program(vertex_shader=MV_VS, fragment_shader=FS)
            self.uP = self.prog["uP"]
            self.layout = [
                ("in_pos_mv", 4, "f4"),
                ("in_uv", 2, "f4"),
                ("in_nrm", 3, "f4"),
            ]

        # Placeholder white texture if none bound
        self.white = self.ctx.texture((1, 1), 3, data=bytes([255, 255, 255]))
        self.white.build_mipmaps()

        # # Load texture from image file (example: assets/texture.png)
        # try:
        #     surface = pygame.image.load("doom3d/assets/diffColor.tga").convert()
        #     img_data = pygame.image.tostring(surface, "RGB", 1)
        #     tex = self.ctx.texture(surface.get_size(), 3, img_data)
        #     tex.build_mipmaps()
        #     tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        #     self.texture = tex
        # except Exception as e:
        #     print("Texture load failed, using white texture", e)
        #     self.texture = self.white

        IMAGE_Y_FLIP = False

        # Load texture from image file (example: assets/texture.png)
        try:
            surf = pygame.image.load("doom3d/assets/diffColor.tga")
            if surf.get_masks()[3] != 0:
                surf = surf.convert_alpha()
                mode = "RGBA"; comps = 4
            else:
                surf = surf.convert()
                mode = "RGB"; comps = 3
            if IMAGE_Y_FLIP:
                surf = pygame.transform.flip(surf, False, True)
            img_data = pygame.image.tostring(surf, mode, False)
            tex = self.ctx.texture(surf.get_size(), comps, img_data)
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            # tex.wrap = (moderngl.REPEAT, moderngl.REPEAT)
            # moderngl.Texture.w
            self.texture = tex
        except Exception as e:
            print("Texture load failed, using white texture", e)
            self.texture = self.white

        # Geometry (replace path)
        # TODO: load your texture via pygame.image and create moderngl.Texture
        self.mesh = load_obj_interleaved(self.ctx, path="doom3d/assets/dog.obj",texture=self.texture)

        # Create VAO with dynamic VBO (we will update vertices each frame after CPU transforms)
        self.vbo = self.ctx.buffer(reserve=self.mesh.vbo.size)
        self.vao = self._make_vao(self.vbo)

        # Camera
        self.eye = np.array([0.0, 1.5, 4.0], dtype=np.float32)
        self.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.aspect = WIDTH / HEIGHT
        self.P = perspective(FOV_DEG, self.aspect, NEAR, FAR)

        # Model state
        self.theta = 0.0

        # CPU source vertex array (will be transformed per-frame)
        self.src = np.frombuffer(self.mesh.vbo.read(), dtype=np.float32).copy().reshape(-1, 9)
        # columns: [x,y,z,w,u,v,nx,ny,nz]

    def _make_vao(self, vbo: moderngl.Buffer) -> moderngl.VertexArray:
        # Build format string and attributes dynamically
        fmt_parts = []
        attrs = []
        offset = 0
        for name, comps, fmt in self.layout:
            fmt_parts.append(f"{comps}{fmt[0]}")
            attrs.append((name, offset))
            offset += comps * 4  # bytes
        fmt = " ".join(fmt_parts)
        return self.ctx.vertex_array(self.prog, [(vbo, fmt, *[a[0] for a in attrs])])

    def update(self, dt: float):
        # Simple spin
        self.theta += dt * 0.7

    def draw(self):
        self.ctx.viewport = (0, 0, WIDTH, HEIGHT)
        self.ctx.clear(0.05, 0.08, 0.1)

        V = look_at(self.eye, self.target, self.up)
        M = rotate_y(self.theta) @ translate(0, 1.0, 0)

        # CPU transform stage
        pos = self.src[:, 0:4]  # (N,4)
        uv = self.src[:, 4:6]
        nrm = self.src[:, 6:9]

        if CPU_MODE == "CPU_MVP":
            MVP = self.P @ V @ M
            pos_clip = (MVP @ pos.T).T  # (N,4)
            interleaved = np.concatenate([pos_clip, uv, nrm], axis=1).astype(np.float32)
        else:  # CPU_MV
            MV = V @ M
            pos_mv = (MV @ pos.T).T
            interleaved = np.concatenate([pos_mv, uv, nrm], axis=1).astype(np.float32)

        # Update VBO
        self.vbo.write(interleaved.tobytes())

        # Bind texture
        (self.mesh.texture or self.white).use(location=0)

        # If using MV shader, upload projection
        if CPU_MODE == "CPU_MV":
            self.uP.write(self.P.tobytes())

        self.vao.render(mode=moderngl.TRIANGLES, vertices=self.mesh.n_vertices)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            dt = clock.tick(60) / 1000.0
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    Renderer(WIDTH, HEIGHT).run()
