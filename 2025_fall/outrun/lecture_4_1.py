import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import numpy.typing as npt
from PIL import Image
import math

# 화면 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
EPSILON = 1e-15

# 맵 (0: 빈 공간, 1-3: 벽 타입)
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Vertex Shader
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;
uniform vec3 tintColor;

void main()
{
    vec4 texColor = texture(texture1, TexCoord);
    FragColor = texColor * vec4(tintColor, 1.0);
}
"""

# Line Vertex Shader (위치와 색상을 VBO에서 입력받음)
LINE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(aPos, 1.0);
    vertexColor = aColor;
}
"""

# Line Fragment Shader (정점 색상 사용)
LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);
}
"""

# HUD Vertex Shader (2D orthographic projection)
HUD_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 projection;
uniform mat4 model;

void main()
{
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# HUD Fragment Shader
HUD_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;
uniform float alpha;

void main()
{
    vec4 texColor = texture(texture1, TexCoord);
    FragColor = vec4(texColor.rgb, texColor.a * alpha);
}
"""

def scale_matrix(s_x: float, s_y: float, s_z: float) -> npt.NDArray:
    return np.array([
        [s_x, 0.0, 0.0, 0.0],
        [0.0, s_y, 0.0, 0.0],
        [0.0, 0.0, s_z, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], np.float32)

def translation_matrix(t_x: float, t_y: float, t_z: float) -> npt.NDArray:
    return np.array([
        [1.0, 0.0, 0.0, t_x],
        [0.0, 1.0, 0.0, t_y],
        [0.0, 0.0, 1.0, t_z],
        [0.0, 0.0, 0.0, 1.0]
    ], np.float32)

def rotation_x_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], np.float32)

def rotation_y_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], np.float32)

def rotation_z_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector
    else:
        return vector / norm
    
def create_view_matrix_lookat(eye, at, up):
    """뷰 행렬 생성 (look-at 방식)

    Args:
        eye: 카메라 위치 (3D 벡터)
        at: 카메라가 보는 지점 (3D 벡터)
        up: 카메라 상향 벡터 (3D 벡터)

    Returns:
        4x4 뷰 행렬
    """
    eye = np.array(eye, dtype=np.float32)
    at = np.array(at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # 카메라 좌표계 구성
    # z축: eye에서 at으로 가는 방향의 반대 (카메라는 -z 방향을 바라봄)
    z_axis = normalize(eye - at)

    # x축: up과 z축의 외적
    x_axis = normalize(np.cross(up, z_axis))

    # y축: z축과 x축의 외적
    y_axis = np.cross(z_axis, x_axis)

    # 뷰 행렬 구성 (회전 + 이동)
    view = np.array([
        [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, eye)],
        [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, eye)],
        [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return view

def model_matrix(
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    rot_x_rad: float = 0.0,
    rot_y_rad: float = 0.0,
    rot_z_rad: float = 0.0,
    trans_x: float = 0.0,
    trans_y: float = 0.0,
    trans_z: float = 0.0
) -> npt.NDArray:
    model = scale_matrix(scale_x, scale_y, scale_z)
    model @= rotation_z_matrix(rot_z_rad) @ rotation_y_matrix(rot_y_rad) @ rotation_x_matrix(rot_x_rad)    
    model @= translation_matrix(trans_x, trans_y, trans_z)

    return model

def projection_matrix(
    fov_y_rad: float = 0.0,
    aspect_ratio: float = 4/3,
    z_near: float = 0.1,
    z_far: float = 10.0
) -> npt.NDArray:
    assert aspect_ratio > 0.0, "aspect_ratio must be > 0"
    assert 0 <= z_near < z_far, "z_near must be >= 0 and z_far must be > z_near"
    factor_y = 1 / (np.tan(fov_y_rad / 2))

    return np.array([
        [factor_y / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, factor_y, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_near - z_far), (2 * z_near * z_far) / (z_near - z_far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

def create_ortho_matrix(left, right, bottom, top, near, far):
    """직교 투영 행렬 생성 (2D용)"""
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
def load_texture(filepath):
    """PNG 파일로부터 텍스처 로드"""
    image = Image.open(filepath)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture

def create_cube_vao():
    """큐브 VAO 생성 (벽용)"""
    vertices = np.array([
        # 위치              텍스처 좌표
        # 앞면
        -0.5, -0.5,  0.5,  0.0, 0.0,
         0.5, -0.5,  0.5,  1.0, 0.0,
         0.5,  0.5,  0.5,  1.0, 1.0,
        -0.5,  0.5,  0.5,  0.0, 1.0,
        # 뒷면
        -0.5, -0.5, -0.5,  1.0, 0.0,
        -0.5,  0.5, -0.5,  1.0, 1.0,
         0.5,  0.5, -0.5,  0.0, 1.0,
         0.5, -0.5, -0.5,  0.0, 0.0,
        # 왼쪽면
        -0.5, -0.5, -0.5,  0.0, 0.0,
        -0.5, -0.5,  0.5,  1.0, 0.0,
        -0.5,  0.5,  0.5,  1.0, 1.0,
        -0.5,  0.5, -0.5,  0.0, 1.0,
        # 오른쪽면
         0.5, -0.5, -0.5,  1.0, 0.0,
         0.5,  0.5, -0.5,  1.0, 1.0,
         0.5,  0.5,  0.5,  0.0, 1.0,
         0.5, -0.5,  0.5,  0.0, 0.0,
        # 윗면
        -0.5,  0.5, -0.5,  0.0, 1.0,
        -0.5,  0.5,  0.5,  0.0, 0.0,
         0.5,  0.5,  0.5,  1.0, 0.0,
         0.5,  0.5, -0.5,  1.0, 1.0,
        # 아랫면
        -0.5, -0.5, -0.5,  0.0, 0.0,
         0.5, -0.5, -0.5,  1.0, 0.0,
         0.5, -0.5,  0.5,  1.0, 1.0,
        -0.5, -0.5,  0.5,  0.0, 1.0,
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,      # 앞
        4, 5, 6, 6, 7, 4,      # 뒤
        8, 9, 10, 10, 11, 8,   # 왼
        12, 13, 14, 14, 15, 12, # 오
        16, 17, 18, 18, 19, 16, # 위
        20, 21, 22, 22, 23, 20  # 아래
    ], dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # 위치
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # 텍스처 좌표
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    return VAO

def create_quad_vao():
    """쿼드 VAO 생성 (바닥용)"""
    vertices = np.array([
        # 위치           텍스처 좌표
        -0.5, -0.5, 0.0, 0.0, 0.0,
         0.5, -0.5, 0.0, 1.0, 0.0,
         0.5,  0.5, 0.0, 1.0, 1.0,
        -0.5,  0.5, 0.0, 0.0, 1.0,
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    return VAO

def create_axis():
    vertices = np.array([
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 위치
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # 색상
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    return VAO

class Terrain:
    def __init__(self, grid):
        self.grid = grid
        self.cube_vao = create_cube_vao()
        self.quad_vao = create_quad_vao()

        # 텍스처 로드
        self.wall_textures = {
            1: load_texture("textures/wall1.png"),
            2: load_texture("textures/wall2.png"),
            3: load_texture("textures/wall3.png"),
        }
        self.floor_texture = load_texture("textures/floor.png")

    def is_wall(self, x, z):
        """주어진 좌표에 벽이 있는지 확인"""
        grid_x = int(x + 0.5)
        grid_z = int(z + 0.5)

        # 맵 범위를 벗어나면 벽으로 간주
        if grid_z < 0 or grid_z >= len(self.grid) or grid_x < 0 or grid_x >= len(self.grid[0]):
            return True

        return self.grid[grid_z][grid_x] != 0

    def can_move_to(self, x, z, radius=0.3):
        """주어진 좌표로 이동 가능한지 확인 (원형 충돌)"""
        # 플레이어의 경계를 확인하기 위해 4방향 체크
        if self.is_wall(x + radius, z):
            return False
        if self.is_wall(x - radius, z):
            return False
        if self.is_wall(x, z + radius):
            return False
        if self.is_wall(x, z - radius):
            return False
        # 대각선도 체크
        if self.is_wall(x + radius * 0.7, z + radius * 0.7):
            return False
        if self.is_wall(x + radius * 0.7, z - radius * 0.7):
            return False
        if self.is_wall(x - radius * 0.7, z + radius * 0.7):
            return False
        if self.is_wall(x - radius * 0.7, z - radius * 0.7):
            return False
        return True

    def draw(self, shader):
        model_loc = glGetUniformLocation(shader, "model")
        tint_loc = glGetUniformLocation(shader, "tintColor")

        # 바닥 그리기
        glBindVertexArray(self.quad_vao)
        glBindTexture(GL_TEXTURE_2D, self.floor_texture)
        glUniform3f(tint_loc, 1.0, 1.0, 1.0)

        # 바닥을 XZ 평면에 그리기
        model = translation_matrix(len(self.grid[0]) / 2 - 0.5, 0, len(self.grid) / 2 - 0.5) @ \
            rotation_x_matrix(-math.pi / 2) @ \
            scale_matrix(len(self.grid), len(self.grid), len(self.grid))
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # 벽 그리기
        glBindVertexArray(self.cube_vao)
        for z in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                wall_type = self.grid[z][x]
                if wall_type != 0:
                    glBindTexture(GL_TEXTURE_2D, self.wall_textures[wall_type])
                    glUniform3f(tint_loc, 1.0, 1.0, 1.0)
                    model = translation_matrix(x, 0.5, z)
                    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

class Player:
    def __init__(self, x, y, z, terrain=None):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = 0
        self.speed = 0.1
        self.terrain = terrain
        self.radius = 0.3
        self.hp = 100
        self.max_hp = 100

    def set_terrain(self, terrain):
        """지형 참조 설정"""
        self.terrain = terrain

    def move(self, move_front, move_right):
        move = [move_right, 0, move_front, 0]
        delta = rotation_y_matrix(self.yaw) @ np.array(move)

        new_x = self.x + delta[0]
        new_z = self.z + delta[2]

        # 지형 충돌 체크
        if self.terrain:
            # X축 이동 체크
            if self.terrain.can_move_to(new_x, self.z, self.radius):
                self.x = new_x

            # Z축 이동 체크 (독립적으로)
            if self.terrain.can_move_to(self.x, new_z, self.radius):
                self.z = new_z
        else:
            # 지형이 없으면 그냥 이동
            self.x = new_x
            self.z = new_z

    def rotate(self, yaw_delta):
        self.yaw += yaw_delta

    def get_view_matrix(self):
        eye = [self.x, self.y, self.z]
        up = [0, 1, 0]
        view_dir = [0, 0, -1, 1]
        at = (rotation_y_matrix(self.yaw) @ np.array(view_dir))[:3] + np.array(eye)
        return create_view_matrix_lookat(eye, at, up)

class Enemy:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.quad_vao = create_quad_vao()
        self.frame = 0
        self.hp = 50
        self.max_hp = 50
        self.radius = 0.5  # 충돌 판정용 반지름

        # 상태 관리
        self.is_hit = False  # 피격 상태
        self.hit_frame = 0  # 피격 애니메이션 프레임
        self.is_dying = False  # 죽는 중
        self.death_frame = 0  # 죽음 애니메이션 프레임
        self.is_dead = False  # 완전히 죽음

        # 텍스처 로드
        self.textures = {
            'walk_1': load_texture("textures/enemy_walk_1.png"),
            'walk_2': load_texture("textures/enemy_walk_2.png"),
            'walk_3': load_texture("textures/enemy_walk_3.png"),
            'walk_4': load_texture("textures/enemy_walk_4.png"),
            'hit': load_texture("textures/enemy_hit_1.png"),
            'die_1': load_texture("textures/enemy_die_1.png"),
            'die_2': load_texture("textures/enemy_die_2.png"),
        }

    def update(self):
        self.frame += 1

        # 피격 상태 처리
        if self.is_hit:
            self.hit_frame += 1
            if self.hit_frame >= 10:  # 10프레임 동안 피격 텍스처 표시
                self.is_hit = False
                self.hit_frame = 0

        # 죽음 애니메이션 처리
        if self.is_dying:
            self.death_frame += 1
            if self.death_frame >= 30:  # die_1을 15프레임 동안 표시
                self.is_dead = True
                self.is_dying = False

    def take_damage(self, damage):
        """데미지를 받음"""
        if self.hp <= 0:
            return

        self.hp -= damage

        if self.hp <= 0:
            self.hp = 0
            # 죽음 애니메이션 시작
            self.is_dying = True
            self.death_frame = 0
        else:
            # 피격 애니메이션 시작
            self.is_hit = True
            self.hit_frame = 0

    def is_alive(self):
        """생존 여부 확인"""
        return self.hp > 0

    def draw(self, camera_x, camera_z, shader):
        model_loc = glGetUniformLocation(shader, "model")
        tint_loc = glGetUniformLocation(shader, "tintColor")

        glBindVertexArray(self.quad_vao)

        # 상태에 따라 텍스처 선택
        if self.is_dead:
            # 완전히 죽었을 때 - die_2 텍스처 고정
            texture = self.textures['die_2']
        elif self.is_dying:
            # 죽는 중 - die_1과 die_2 애니메이션
            if self.death_frame < 15:
                texture = self.textures['die_1']
            else:
                texture = self.textures['die_2']
        elif self.is_hit:
            # 피격 상태 - hit 텍스처
            texture = self.textures['hit']
        else:
            # 일반 상태 - 걷기 애니메이션
            walk_textures = ['walk_1', 'walk_2', 'walk_3', 'walk_4']
            texture_index = int(self.frame / 15) % len(walk_textures)
            texture = self.textures[walk_textures[texture_index]]

        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform3f(tint_loc, 1.0, 1.0, 1.0)

        angle = math.atan2(camera_x - self.x, camera_z - self.z)

        model = translation_matrix(self.x, 0.5, self.z) @ \
            rotation_y_matrix(angle) @ \
            scale_matrix(1, 1, 1)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

class Gun:
    def __init__(self):
        self.quad_vao = create_quad_vao()
        self.frame = 0
        self.is_shooting = False
        self.shoot_frame = 0
        self.damage = 25  # 총알 데미지

        # 발사 속도 제한
        self.cooldown_time = 30  # 30프레임 (60fps 기준 0.5초)
        self.cooldown_counter = 0  # 현재 쿨다운 카운터

        # 텍스처 로드
        self.textures = {
            0: load_texture("textures/gun_idle.png"),
            1: load_texture("textures/gun_shoot1.png"),
            2: load_texture("textures/gun_shoot2.png")
        }

        # 사운드 로드
        try:
            self.shoot_sound = pygame.mixer.Sound("sounds/shotgun.wav")
        except:
            print("Warning: Could not load wav")
            self.shoot_sound = None

    def ray_plane_intersection(self, ray_origin, ray_dir, plane_point, plane_normal):
        """레이와 평면의 교차점을 계산

        Args:
            ray_origin: 레이의 시작점 (3D 벡터)
            ray_dir: 레이의 방향 (정규화된 3D 벡터)
            plane_point: 평면 위의 한 점 (3D 벡터)
            plane_normal: 평면의 법선 벡터 (정규화된 3D 벡터)

        Returns:
            교차점 (3D 벡터) 또는 None (평행하거나 뒤쪽)
        """
        denom = np.dot(plane_normal, ray_dir)

        # 평면과 레이가 평행하거나 레이가 평면 뒤를 향함
        if abs(denom) < EPSILON:
            return None

        # 교차점까지의 거리 계산
        t = np.dot(plane_normal, plane_point - ray_origin) / denom

        # 레이의 시작점 뒤에 있으면 교차하지 않음
        if t < 0:
            return None

        # 교차점 계산
        intersection = ray_origin + ray_dir * t
        return intersection, t

    def shoot(self, player, enemies):
        """총을 발사하고 적의 평면과의 교차점을 계산"""
        # 쿨다운 체크
        if self.cooldown_counter > 0:
            return

        self.is_shooting = True
        self.shoot_frame = 0
        self.cooldown_counter = self.cooldown_time  # 쿨다운 시작

        # 사운드 재생
        if self.shoot_sound:
            self.shoot_sound.play()

        # 레이의 시작점 (플레이어의 눈 높이)
        ray_origin = np.array([player.x, player.y, player.z], dtype=np.float32)

        # 레이의 방향 (플레이어의 시선 방향)
        view_dir = rotation_y_matrix(player.yaw) @ np.array([0, 0, -1, 0])
        ray_dir = normalize(view_dir[:3])

        # 가장 가까운 적을 찾기 위한 변수
        closest_enemy = None
        closest_distance = float('inf')
        closest_hit_point = None

        # 모든 적과의 충돌 체크
        for enemy in enemies:
            if not enemy.is_alive():
                continue

            # 적의 빌보드 평면 정의
            # 적의 위치 (평면 위의 한 점)
            enemy_pos = np.array([enemy.x, 0.5, enemy.z], dtype=np.float32)

            # 적의 평면 법선 벡터 (항상 카메라를 향함)
            # 카메라에서 적으로의 방향
            to_enemy = enemy_pos - ray_origin
            to_enemy[1] = 0  # y축 제거 (수평만)
            plane_normal = normalize(to_enemy)

            # 레이와 평면의 교차 계산
            result = self.ray_plane_intersection(ray_origin, ray_dir, enemy_pos, plane_normal)

            if result is None:
                continue

            intersection, distance = result

            # 교차점이 적의 스프라이트 범위 내에 있는지 확인
            # 적 중심으로부터의 오프셋 계산
            offset = intersection - enemy_pos

            # 적의 로컬 좌표계로 변환
            # 적은 항상 카메라를 향하므로, 적의 오른쪽 벡터 계산
            enemy_right = np.array([-plane_normal[2], 0, plane_normal[0]], dtype=np.float32)
            enemy_up = np.array([0, 1, 0], dtype=np.float32)

            # 로컬 좌표로 투영
            local_x = np.dot(offset, enemy_right)
            local_y = np.dot(offset, enemy_up)

            # 스프라이트 크기는 1x1 (scale_matrix(1, 1, 1))
            # 쿼드는 -0.5 ~ 0.5 범위
            half_width = 0.5
            half_height = 0.5

            # 교차점이 스프라이트 범위 내에 있는지 확인
            if (abs(local_x) <= half_width and abs(local_y) <= half_height):
                # 가장 가까운 적인지 확인
                if distance < closest_distance:
                    closest_distance = distance
                    closest_enemy = enemy
                    closest_hit_point = intersection

        # 가장 가까운 적에게 데미지
        if closest_enemy is not None:
            closest_enemy.take_damage(self.damage)

    def update(self):
        # 발사 애니메이션 업데이트
        if self.is_shooting:
            self.shoot_frame += 1
            if self.shoot_frame >= 10:
                self.is_shooting = False
                self.shoot_frame = 0

        # 쿨다운 감소
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

    def draw(self, hud_shader):
        glUseProgram(hud_shader)
        glDisable(GL_DEPTH_TEST)

        model_loc = glGetUniformLocation(hud_shader, "model")
        proj_loc = glGetUniformLocation(hud_shader, "projection")
        alpha_loc = glGetUniformLocation(hud_shader, "alpha")

        ortho_projection = create_ortho_matrix(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, -1, 1)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, ortho_projection)
        glUniform1f(alpha_loc, 1.0)

        size = (300, 300)
        position = ((SCREEN_WIDTH / 2), size[1] / 2)

        gun_x = position[0]
        gun_y = position[1]

        if self.is_shooting and self.shoot_frame < 5:
            texture_index = 1
        elif self.is_shooting:
            texture_index = 2
        else:
            texture_index = 0

        glBindTexture(GL_TEXTURE_2D, self.textures[texture_index])

        model = translation_matrix(gun_x, gun_y, 0) @ scale_matrix(size[0], size[1], 1)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glEnable(GL_DEPTH_TEST)

def main():
    pygame.init()
    pygame.mixer.init()  # 사운드 시스템 초기화
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple Cube")

    # OpenGL 설정
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glClearColor(0.2, 0.3, 0.3, 1.0)

    # 셰이더 컴파일
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # 라인 셰이더 컴파일
    line_shader = compileProgram(
        compileShader(LINE_VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(LINE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    
    # HUD 셰이더 컴파일
    hud_shader = compileProgram(
        compileShader(HUD_VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(HUD_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # 축 VAO 생성 (축은 클래스로 분리하지 않음)
    axis_vao = create_axis()

    # 유니폼 위치
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")

    # 유니폼 위치 (라인 셰이더)
    line_mvp_loc = glGetUniformLocation(line_shader, "mvp")

    clock = pygame.time.Clock()
    running = True

    # 지형 생성 (내부에서 VAO와 텍스처 생성)
    terrain = Terrain(MAP)

    # 플레이어 생성 및 지형 참조 설정
    player = Player(3, 0.8, 5, terrain)

    # 적 생성 (내부에서 VAO와 텍스처 생성)
    enemies = [
        Enemy(len(MAP[0]) / 2 - 0.5, len(MAP) / 2 - 0.5),
        Enemy(10, 10),
        Enemy(5, 8)
    ]

    # 총 생성 (내부에서 VAO와 텍스처 생성)
    gun = Gun()

    fov = math.radians(60)
    aspect = SCREEN_WIDTH / SCREEN_HEIGHT
    near, far = 0.1, 50.0

    f = 1.0 / math.tan(fov / 2.0)

    frame = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    gun.shoot(player, enemies)

        move_front = 0
        move_right = 0

        # 키보드 입력으로 플레이어 이동
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  # 앞으로
            move_front -= player.speed
        if keys[pygame.K_s]:  # 뒤로
            move_front += player.speed
        if keys[pygame.K_a]:  # 왼쪽으로
            move_right -= player.speed
        if keys[pygame.K_d]:  # 오른쪽으로
            move_right += player.speed

        if keys[pygame.K_LEFT]:
            player.rotate(0.05)
        if keys[pygame.K_RIGHT]:
            player.rotate(-0.05)

        frame += 1

        # 플레이어 이동
        player.move(move_front, move_right)

        # 뷰 행렬 생성
        view = player.get_view_matrix()

        projection = projection_matrix(fov, aspect, near, far )

        # 업데이트
        for enemy in enemies:
            enemy.update()
        gun.update()

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)

        terrain.draw(shader)

        # 모든 적 그리기 (죽은 적도 die_2 텍스처로 표시)
        for enemy in enemies:
            enemy.draw(player.x, player.z, shader)

        # HUD
        gun.draw(hud_shader)

        # 축 렌더링 (라인 셰이더 사용)
        glUseProgram(line_shader)
        glBindVertexArray(axis_vao)
        glLineWidth(3.0)

        # 축은 원점에 고정
        axis_model = model_matrix()
        axis_mvp = projection @ view @ axis_model

        glUniformMatrix4fv(line_mvp_loc, 1, GL_TRUE, axis_mvp)
        glDrawArrays(GL_LINES, 0, 6)  # 3개의 축, 각 2개의 정점

        model = scale_matrix(3, 3, 3)
        axis_mvp = projection @ view @ model

        glUniformMatrix4fv(line_mvp_loc, 1, GL_TRUE, axis_mvp)
        glDrawArrays(GL_LINES, 0, 6)  # 3개의 축, 각 2개의 정점


        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()



