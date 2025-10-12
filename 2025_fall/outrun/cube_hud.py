import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from matrix_utils import create_model_matrix, create_view_matrix, create_projection_matrix, create_ortho_matrix
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
            self.pitch = min(89, self.pitch)
        if keys[pygame.K_DOWN]:
            self.pitch -= 1
            self.pitch = max(-89, self.pitch)

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
        load_texture("textures/gun.png"),
        load_texture("textures/gun_fire.png")
    ]

    player = Player(1.5, 0.5, 1.5)
    enemies = [
        Enemy(4.5, 3.5),
        Enemy(5.5, 5.5),
        Enemy(2.5, 4.5),
    ]

    shooting = False
    shoot_frame = 0
    mouse_sensitivity = 0.2

    # 카메라 설정
    map_center_x = len(MAP[0]) / 2
    map_center_z = len(MAP) / 2
    camera_x = map_center_x
    camera_y = 5
    camera_z = map_center_z + 5
    camera_yaw = 0
    camera_pitch = -45
    # camera_speed = 0.1


    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    tint_loc = glGetUniformLocation(shader, "tintColor")
    

    # 유니폼 위치 (라인 셰이더)
    line_mvp_loc = glGetUniformLocation(line_shader, "mvp")

    # 투영 행렬
    fov = math.radians(60)
    aspect = SCREEN_WIDTH / SCREEN_HEIGHT
    near, far = 0.1, 50.0

    f = 1.0 / math.tan(fov / 2.0)
    projection = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)



    clock = pygame.time.Clock()
    running = True

    # 회전 각도 초기화
    rotation_angle = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 키보드 입력 처리
        keys = pygame.key.get_pressed()

        # # WASD로 이동
        # if keys[pygame.K_w]:
        #     camera_z += camera_speed
        # if keys[pygame.K_s]:
        #     camera_z -= camera_speed
        # if keys[pygame.K_a]:
        #     camera_x += camera_speed
        # if keys[pygame.K_d]:
        #     camera_x -= camera_speed

        # # QE로 높이 조절
        # if keys[pygame.K_q]:
        #     camera_y -= camera_speed
        # if keys[pygame.K_e]:
        #     camera_y += camera_speed

        # # 화살표로 시점 회전
        # if keys[pygame.K_LEFT]:
        #     camera_yaw -= 1
        # if keys[pygame.K_RIGHT]:
        #     camera_yaw += 1
        # if keys[pygame.K_UP]:
        #     camera_pitch += 1
        #     camera_pitch = min(89, camera_pitch)
        # if keys[pygame.K_DOWN]:
        #     camera_pitch -= 1
        #     camera_pitch = max(-89, camera_pitch)

        # # 회전 각도 업데이트 (시간에 따라 증가)
        # rotation_angle += 1.0  # 매 프레임마다 1도씩 증가

        if keys[pygame.K_SPACE]:
            shooting = True
            shoot_frame = 0
            check_hit(player, enemies)

        player.move(keys)

        for enemy in enemies:
            enemy.update()
        
        if shooting:
            shoot_frame += 1
            if shoot_frame >= 10:
                shooting = False
                shoot_frame = 0

        # # 뷰 행렬 업데이트
        # camera_pos = (camera_x, camera_y, camera_z)
        # view = create_view_matrix(camera_pos, camera_yaw, camera_pitch)

        view = player.get_view_matrix()

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)

        draw_terrain_from_grid(MAP, cube_vao, quad_vao, wall_textures, floor_texture, shader)

        # 적 그리기 (빌보드)
        glBindVertexArray(quad_vao)
        for enemy in enemies:
            if enemy.alive:
                glBindTexture(GL_TEXTURE_2D, enemy_textures[enemy.anim_frame])
                glUniform3f(tint_loc, 1.0, 1.0, 1.0)
                model = create_model_matrix(enemy.x, enemy.y, enemy.z, scale=0.6, rotate_y=-player.yaw)
                glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


        # 축 렌더링 (라인 셰이더 사용)
        glUseProgram(line_shader)
        glBindVertexArray(axis_vao)
        glLineWidth(3.0)

        # 축은 원점에 고정
        axis_model = create_model_matrix(0, 0, 0, scale=3.0)
        axis_mvp = projection @ view @ axis_model

        glUniformMatrix4fv(line_mvp_loc, 1, GL_TRUE, axis_mvp)
        glDrawArrays(GL_LINES, 0, 6)  # 3개의 축, 각 2개의 정점


        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
