import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import math
import numpy as np
from PIL import Image
from matrix_utils import create_model_matrix, create_view_matrix, create_projection_matrix

# 화면 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

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
    FragColor = texColor * vec4(tintColor, 1.0);
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

class Enemy:
    def __init__(self, x, z):
        self.x = x
        self.y = 0.5
        self.z = z
        self.alive = True
        self.anim_frame = 0
        self.anim_counter = 0

    def update(self):
        if self.alive:
            self.anim_counter += 1
            if self.anim_counter >= 10:
                self.anim_counter = 0
                self.anim_frame = (self.anim_frame + 1) % 4

def create_default_texture():
    """기본 텍스처 생성 (체크 패턴)"""
    size = 64
    texture_data = np.zeros((size, size, 4), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // 16 + j // 16) % 2 == 0:
                texture_data[i, j] = [200, 200, 200, 255]
            else:
                texture_data[i, j] = [100, 100, 100, 255]

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    return texture

def load_texture(filepath):
    """PNG 파일로부터 텍스처 로드"""
    try:
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
    except FileNotFoundError:
        print(f"Warning: Texture file {filepath} not found. Using default.")
        return create_default_texture()

def create_cube_vao():
    """큐브 VAO 생성"""
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
    """쿼드 VAO 생성 (스프라이트용)"""
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
    pygame.display.set_caption("Simple Doom - OpenGL Shaders")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    # OpenGL 설정
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.53, 0.81, 0.92, 1.0)

    # 셰이더 컴파일
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # VAO 생성
    cube_vao = create_cube_vao()
    quad_vao = create_quad_vao()

    # 텍스처 로드
    wall_textures = {
        1: load_texture("textures/wall1.png"),
        2: load_texture("textures/wall2.png"),
        3: load_texture("textures/wall3.png"),
    }
    floor_texture = load_texture("textures/floor.png")
    enemy_textures = [
        load_texture("textures/enemy1.png"),
        load_texture("textures/enemy2.png"),
        load_texture("textures/enemy3.png"),
        load_texture("textures/enemy4.png")
    ]
    gun_textures = [
        load_texture("textures/gun.png"),
        load_texture("textures/gun_fire.png")
    ]

    # 투영 행렬
    projection = create_projection_matrix(60, SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 50.0)

    player = Player(1.5, 0.5, 1.5)
    enemies = [
        Enemy(4.5, 3.5),
        Enemy(5.5, 5.5),
        Enemy(2.5, 4.5),
    ]

    shooting = False
    shoot_frame = 0
    mouse_sensitivity = 0.2

    # 디버그 모드 (전지적 시점)
    debug_camera = True

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    debug_camera = not debug_camera
            elif event.type == pygame.MOUSEMOTION:
                player.yaw += event.rel[0] * mouse_sensitivity
                player.pitch -= event.rel[1] * mouse_sensitivity
                player.pitch = max(-89, min(89, player.pitch))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    shooting = True
                    shoot_frame = 0
                    check_hit(player, enemies)

        keys = pygame.key.get_pressed()
        player.move(keys)

        for enemy in enemies:
            enemy.update()

        if shooting:
            shoot_frame += 1
            if shoot_frame >= 10:
                shooting = False
                shoot_frame = 0

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        # 디버그 카메라 (전지적 시점)
        if debug_camera:
            map_center_x = len(MAP[0]) / 2
            map_center_z = len(MAP) / 2
            view = create_view_matrix((map_center_x, 10, map_center_z + 10), 0, -45)
        else:
            view = player.get_view_matrix()

        # 유니폼 위치
        model_loc = glGetUniformLocation(shader, "model")
        view_loc = glGetUniformLocation(shader, "view")
        proj_loc = glGetUniformLocation(shader, "projection")
        tint_loc = glGetUniformLocation(shader, "tintColor")

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

        # 바닥 그리기
        glBindVertexArray(quad_vao)
        glBindTexture(GL_TEXTURE_2D, floor_texture)
        glUniform3f(tint_loc, 1.0, 1.0, 1.0)

        # 바닥을 XZ 평면에 그리기 위해 X축으로 -90도 회전
        model = create_model_matrix(len(MAP[0]) / 2, 0, len(MAP) / 2, scale=len(MAP) * 2, rotate_x=-90)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # 벽 그리기
        glBindVertexArray(cube_vao)
        for z in range(len(MAP)):
            for x in range(len(MAP[0])):
                wall_type = MAP[z][x]
                if wall_type != 0:
                    glBindTexture(GL_TEXTURE_2D, wall_textures[wall_type])
                    glUniform3f(tint_loc, 1.0, 1.0, 1.0)
                    model = create_model_matrix(x, 0.5, z, scale=1.0)
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        # 적 그리기 (빌보드)
        glBindVertexArray(quad_vao)
        for enemy in enemies:
            if enemy.alive:
                glBindTexture(GL_TEXTURE_2D, enemy_textures[enemy.anim_frame])
                glUniform3f(tint_loc, 1.0, 1.0, 1.0)
                model = create_model_matrix(enemy.x, enemy.y, enemy.z, scale=0.6, rotate_y=-player.yaw)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # 총 HUD (디버그 모드에서는 표시 안함)
        if not debug_camera:
            glDisable(GL_DEPTH_TEST)

            # 카메라 앞에 총을 고정
            gun_distance = 0.3  # 카메라로부터의 거리
            gun_offset_x = 0.15  # 오른쪽으로
            gun_offset_y = -0.2  # 아래로

            # 플레이어 시선 방향 계산
            yaw_rad = math.radians(player.yaw)
            pitch_rad = math.radians(player.pitch)

            forward_x = math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)

            right_x = math.cos(yaw_rad)
            right_z = math.sin(yaw_rad)

            gun_x = player.x + forward_x * gun_distance + right_x * gun_offset_x
            gun_y = player.y + forward_y * gun_distance + gun_offset_y
            gun_z = player.z + forward_z * gun_distance + right_z * gun_offset_x

            if shooting and shoot_frame % 2 == 0:
                glBindTexture(GL_TEXTURE_2D, gun_textures[1])
                offset = -0.02
            else:
                glBindTexture(GL_TEXTURE_2D, gun_textures[0])
                offset = 0

            glUniform3f(tint_loc, 1.0, 1.0, 1.0)
            model = create_model_matrix(gun_x, gun_y + offset, gun_z, scale=0.15, rotate_y=-player.yaw)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
