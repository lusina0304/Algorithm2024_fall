import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from matrix_utils import create_model_matrix, \
create_view_matrix, create_projection_matrix, \
create_ortho_matrix, normalize
import math

# 화면 설정
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

# 맵 (0: 빈 공간, 1-3: 벽 타입)
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 2, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 3, 0, 0, 3, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
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
    if (texColor.a < 0.1)
        discard;
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
    //if (texColor.a < 0.1)
    //    discard;
    FragColor = vec4(texColor.rgb, texColor.a * alpha);
}
"""

class Player:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = 0.0
        self.pitch = 0.0
        self.speed = 0.1

    def move(self, keys):
        dx = 0
        dz = 0

        # 화살표로 시점 회전
        if keys[pygame.K_LEFT]:
            self.yaw -= 1
        if keys[pygame.K_RIGHT]:
            self.yaw += 1
        if keys[pygame.K_UP]:
            self.pitch += 1
            self.pitch = min(69, self.pitch)
        if keys[pygame.K_DOWN]:
            self.pitch -= 1
            self.pitch = max(-69, self.pitch)

        if keys[pygame.K_w]:
            dx += math.sin(math.radians(self.yaw))
            dz -= math.cos(math.radians(self.yaw))
        if keys[pygame.K_s]:
            dx -= math.sin(math.radians(self.yaw))
            dz += math.cos(math.radians(self.yaw))
        if keys[pygame.K_a]:
            dx -= math.cos(math.radians(self.yaw))
            dz -= math.sin(math.radians(self.yaw))
        if keys[pygame.K_d]:
            dx += math.cos(math.radians(self.yaw))
            dz += math.sin(math.radians(self.yaw))

        new_x = self.x + dx * self.speed
        new_z = self.z + dz * self.speed

        map_x = int(new_x)
        map_z = int(new_z)

        if 0 <= map_x < len(MAP[0]) and 0 <= map_z < len(MAP):
            if MAP[map_z][map_x] == 0:
                self.x = new_x
                self.z = new_z

    def get_view_matrix(self):
        return create_view_matrix((self.x, self.y, self.z), self.yaw, self.pitch)

class Gun:
    def __init__(self, textures, fire_duration=0.3, size=(300, 300)):
        """
        총 클래스
        :param textures: [idle_texture, fire_texture1, fire_texture2] 순서의 텍스처 리스트
        :param fire_duration: 총 발사 애니메이션 지속 시간 (초)
        :param position: 화면상 총의 위치 (x, y)
        :param size: 총 스프라이트 크기 (width, height)
        """
        self.textures = textures
        self.fire_duration = fire_duration  # 초 단위
        self.position = ((SCREEN_WIDTH / 2), size[1] / 2)
        self.size = size

        self.is_firing = False
        self.fire_timer = 0.0  # 발사 시작 후 경과 시간
        self.current_frame = 0  # 0: idle, 1: fire1, 2: fire2

    def shoot(self):
        """총 발사 시작"""
        if not self.is_firing:
            self.is_firing = True
            self.fire_timer = 0.0

    def update(self, delta_time):
        """
        애니메이션 업데이트
        :param delta_time: 프레임 간 경과 시간 (초)
        """
        if self.is_firing:
            self.fire_timer += delta_time

            # 발사 지속 시간을 3개 프레임으로 나눔 (idle 포함 안함)
            # fire1, fire2를 번갈아 보여줌
            frame_duration = self.fire_duration / 2.0  # 2개의 발사 프레임

            if self.fire_timer < frame_duration:
                self.current_frame = 1  # fire1
            elif self.fire_timer < self.fire_duration:
                self.current_frame = 2  # fire2
            else:
                # 발사 애니메이션 종료
                self.is_firing = False
                self.fire_timer = 0.0
                self.current_frame = 0  # idle
        else:
            self.current_frame = 0  # idle

    def draw(self, quad_vao, hud_shader, ortho_projection):
        """
        HUD에 총 그리기
        :param quad_vao: 쿼드 VAO
        :param hud_shader: HUD 셰이더 프로그램
        :param ortho_projection: 직교 투영 행렬
        """
        glUseProgram(hud_shader)
        glDisable(GL_DEPTH_TEST)

        # Uniform 위치 가져오기
        model_loc = glGetUniformLocation(hud_shader, "model")
        proj_loc = glGetUniformLocation(hud_shader, "projection")
        alpha_loc = glGetUniformLocation(hud_shader, "alpha")

        # Orthographic projection 설정
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, ortho_projection)
        glUniform1f(alpha_loc, 1.0)

        # 총 위치 설정
        gun_x = self.position[0]
        gun_y = self.position[1]
        gun_width = self.size[0]
        gun_height = self.size[1]

        # 발사 중일 때 리코일 효과
        recoil_offset = 0
        if self.is_firing:
            # 발사 초반에 더 강한 리코일
            recoil_strength = 1.0 - (self.fire_timer / self.fire_duration)
            recoil_offset = -30.0 * recoil_strength

        # 현재 프레임에 맞는 텍스처 바인딩
        glBindTexture(GL_TEXTURE_2D, self.textures[self.current_frame])

        # 모델 행렬 생성 (2D 평면)
        model = create_model_matrix(
            gun_x,
            gun_y + recoil_offset,
            0.0,
            scale_x=gun_width,
            scale_y=gun_height,
            scale_z=1.0
        )
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

        # 쿼드 렌더링
        glBindVertexArray(quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glEnable(GL_DEPTH_TEST)

class Enemy:
    def __init__(self, x, y, z, textures, hp=100, speed=0.05, size=0.6, anim_speed=0.2):
        """
        적 클래스
        :param x, y, z: 3D 공간상 위치
        :param textures: 걷기 애니메이션 텍스처 리스트
        :param hp: 체력
        :param speed: 이동 속도
        :param size: 스프라이트 크기
        :param anim_speed: 애니메이션 속도 (초 단위로 프레임 전환 시간)
        """
        self.x = x
        self.y = size / 2
        self.z = z
        self.textures = textures
        self.hp = hp
        # self.max_hp = hp
        self.speed = speed
        self.size = size
        self.anim_speed = anim_speed
        self.yaw = 0.0;

        self.alive = True
        self.current_frame = 0
        self.anim_timer = 0.0

    def take_damage(self, damage):
        """피해 입기"""
        if self.alive:
            self.hp -= damage
            if self.hp <= 0:
                self.hp = 0
                self.alive = False

    def update(self, delta_time, player_pos, game_map):
        """
        적 업데이트 (AI, 애니메이션)
        :param delta_time: 프레임 간 경과 시간 (초)
        :param player_pos: 플레이어 위치 (x, y, z)
        :param game_map: 맵 데이터 (충돌 체크용)
        """
        if not self.alive:
            return

        # 애니메이션 업데이트
        self.anim_timer += delta_time
        if self.anim_timer >= self.anim_speed:
            self.anim_timer = 0.0
            self.current_frame = (self.current_frame + 1) % len(self.textures)

        # 플레이어를 향해 이동
        dx = player_pos[0] - self.x
        dz = player_pos[2] - self.z
        distance = math.sqrt(dx * dx + dz * dz)

        # 일정 거리 이상일 때만 추적
        if distance > 0.5:
            # 정규화된 방향으로 이동

            # dx /= distance
            # dz /= distance

            # new_x = self.x + dx * self.speed
            # new_z = self.z + dz * self.speed

            # # 맵 충돌 체크
            # map_x = int(new_x)
            # map_z = int(new_z)

            # if 0 <= map_x < len(game_map[0]) and 0 <= map_z < len(game_map):
            #     if game_map[map_z][map_x] == 0:
            #         self.x = new_x
            #         self.z = new_z

            direction = np.array(player_pos, np.float32) - np.array((self.x, self.y, self.z), np.float32)
            direction[1] = 0.0
            direction = normalize(direction)        
            self.yaw = math.atan2(direction[2], direction[0])

    def draw(self, quad_vao, shader):
        """
        적을 빌보드로 그리기 (항상 플레이어를 향함)
        :param quad_vao: 쿼드 VAO
        :param shader: 메인 셰이더 프로그램
        :param player_yaw: 플레이어의 yaw 각도
        """
        if not self.alive:
            return

        # Uniform 위치 가져오기
        model_loc = glGetUniformLocation(shader, "model")
        tint_loc = glGetUniformLocation(shader, "tintColor")

        # 현재 애니메이션 프레임 텍스처 바인딩
        glBindTexture(GL_TEXTURE_2D, self.textures[self.current_frame])
        glUniform3f(tint_loc, 1.0, 1.0, 1.0)
        
        # 빌보드 효과: 항상 플레이어를 향하도록 회전
        model = create_model_matrix(
            self.x,
            self.y,
            self.z,
            scale=self.size,
            rotate_y=math.degrees(self.yaw) - 90
        )
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

        # 쿼드 렌더링
        glBindVertexArray(quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

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

def draw_axis():
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

def draw_terrain_from_grid(grid, cube_vao, quad_vao, wall_textures, floor_texture, shader):
    """2D 그리드로부터 3D 지형 생성 및 렌더링"""
    model_loc = glGetUniformLocation(shader, "model")
    tint_loc = glGetUniformLocation(shader, "tintColor")

    # 바닥 그리기
    glBindVertexArray(quad_vao)
    glBindTexture(GL_TEXTURE_2D, floor_texture)
    glUniform3f(tint_loc, 1.0, 1.0, 1.0)

    # 바닥을 XZ 평면에 그리기
    model = create_model_matrix(len(grid[0]) / 2, 0, len(grid) / 2, scale=len(grid) * 2, rotate_x=-90)
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # 벽 그리기
    glBindVertexArray(cube_vao)
    for z in range(len(grid)):
        for x in range(len(grid[0])):
            wall_type = grid[z][x]
            if wall_type != 0:
                glBindTexture(GL_TEXTURE_2D, wall_textures[wall_type])
                glUniform3f(tint_loc, 1.0, 1.0, 1.0)
                model = create_model_matrix(x, 0.5, z, scale=1.0)
                glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

def check_hit(player, enemies):
    """적 명중 체크"""
    dir_x = math.sin(math.radians(player.yaw))
    dir_z = -math.cos(math.radians(player.yaw))

    for enemy in enemies:
        if not enemy.alive:
            continue

        dx = enemy.x - player.x
        dz = enemy.z - player.z
        distance = math.sqrt(dx * dx + dz * dz)

        if distance < 0.1:
            continue

        dx /= distance
        dz /= distance

        dot = dx * dir_x + dz * dir_z

        if dot > 0.95 and distance < 10:
            enemy.alive = False
            return True

    return False

def main():
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D World Generator")

    # OpenGL 설정
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.53, 0.81, 0.92, 1.0)  # 하늘색

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

    # VAO 생성
    cube_vao = create_cube_vao()
    quad_vao = create_quad_vao()
    axis_vao = draw_axis()

    # 텍스처 로드
    wall_textures = {
        1: load_texture("textures/wall1.png"),
        2: load_texture("textures/wall2.png"),
        3: load_texture("textures/wall3.png"),
    }
    floor_texture = load_texture("textures/floor.png")
    enemy_textures = [
        load_texture("textures/enemy_walk_1.png"),
        load_texture("textures/enemy_walk_2.png"),
        load_texture("textures/enemy_walk_3.png"),
        load_texture("textures/enemy_walk_4.png")
    ]
    gun_textures = [
        load_texture("textures/gun_idle.png"),
        load_texture("textures/gun_shoot1.png"),
        load_texture("textures/gun_shoot2.png")
    ]

    player = Player(1.5, 0.5, 1.5)

    # Gun 인스턴스 생성 (발사 시간 0.3초)
    gun = Gun(gun_textures, fire_duration=0.3, size=(300, 300))

    # Enemy 인스턴스 생성
    enemies = [
        Enemy(3.5, 0.5, 3.5, enemy_textures, hp=100, speed=0.02, size=0.6, anim_speed=0.2),
        Enemy(5.5, 0.5, 5.5, enemy_textures, hp=100, speed=0.02, size=0.8, anim_speed=0.2),
        Enemy(2.5, 0.5, 5.5, enemy_textures, hp=100, speed=0.02, size=0.9, anim_speed=0.2),
    ]

    # model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    # tint_loc = glGetUniformLocation(shader, "tintColor")
    

    # 유니폼 위치 (라인 셰이더)
    line_mvp_loc = glGetUniformLocation(line_shader, "mvp")

    # 투영 행렬
    aspect = SCREEN_WIDTH / SCREEN_HEIGHT
    near, far = 0.1, 50.0
    projection = create_projection_matrix(60, aspect, near, far)

    # HUD용 직교 투영 행렬 (2D, 화면 좌표계)
    ortho_projection = create_ortho_matrix(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, -1, 1)

    clock = pygame.time.Clock()
    running = True

    while running:
        # Delta time 계산 (초 단위)
        delta_time = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 키보드 입력 처리
        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            gun.shoot()

        player.move(keys)

        # Gun 애니메이션 업데이트
        gun.update(delta_time)

        # Enemy 업데이트
        player_pos = (player.x, player.y, player.z)
        for enemy in enemies:
            enemy.update(delta_time, player_pos, MAP)

        view = player.get_view_matrix()

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)

        draw_terrain_from_grid(MAP, cube_vao, quad_vao, wall_textures, floor_texture, shader)

        # 적 그리기 (빌보드) - 거리순 정렬 (먼 것부터)
        glBindVertexArray(quad_vao)
        # enemies_sorted = sorted(enemies,
        #                        key=lambda e: (e.x - player.x)**2 + (e.z - player.z)**2,
        #                        reverse=True)
        for enemy in enemies:
            enemy.draw(quad_vao, shader)

        # 축 렌더링 (라인 셰이더 사용)
        glUseProgram(line_shader)
        glBindVertexArray(axis_vao)
        glLineWidth(3.0)

        # 축은 원점에 고정
        axis_model = create_model_matrix(0, 0, 0, scale=3.0)
        axis_mvp = projection @ view @ axis_model

        glUniformMatrix4fv(line_mvp_loc, 1, GL_TRUE, axis_mvp)
        glDrawArrays(GL_LINES, 0, 6)  # 3개의 축, 각 2개의 정점

        # HUD에 총 그리기
        gun.draw(quad_vao, hud_shader, ortho_projection)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
