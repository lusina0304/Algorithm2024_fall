import pygame
import moderngl
import numpy as np
from typing import Optional, Tuple

import Config
from Linalg import perspective, translate, scale
# from Config import WIDTH, HEIGHT, FOV_DEG, NEAR, FAR, VSYNC, IMAGE_Y_FLIP
# from Config import width as WIDTH, height as HEIGHT, fov_deg as FOV_DEG, near as NEAR, far as FAR, vsync as VSYNC, image_y_flip as IMAGE_Y_FLIP

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

class Renderer:
    def __init__(self, config:Config):
        self.cfg = config

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        if config.vsync:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        pygame.display.set_mode((config.width, config.height), pygame.OPENGL | pygame.DOUBLEBUF)
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
        aspect = config.width / config.height
        self.P = perspective( config.fov_deg, aspect, config.near, config.far)

        # Textures assets\texture\wall
        self.white = self._make_white()
        self.tex_wall  = self._load_texture("assets/texture/2.png", allow_alpha=False) or self.white
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
            if self.cfg.image_y_flip:
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
        self.ctx.viewport = (0,0,self.cfg.width,self.cfg.height)
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
