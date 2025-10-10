import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from matrix_utils import create_model_matrix, create_view_matrix, create_projection_matrix

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
        print(f"Warning: Texture file {filepath} not found.")
        return create_default_texture()

def create_default_texture():
    """기본 텍스처 생성"""
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

def generate_terrain_from_grid(grid, cube_vao, quad_vao, wall_textures, floor_texture, shader):
    """2D 그리드로부터 3D 지형 생성 및 렌더링"""
    model_loc = glGetUniformLocation(shader, "model")
    tint_loc = glGetUniformLocation(shader, "tintColor")

    # 바닥 그리기
    glBindVertexArray(quad_vao)
    glBindTexture(GL_TEXTURE_2D, floor_texture)
    glUniform3f(tint_loc, 1.0, 1.0, 1.0)

    # 바닥을 XZ 평면에 그리기
    model = create_model_matrix(len(grid[0]) / 2, 0, len(grid) / 2, scale=len(grid) * 2, rotate_x=-90)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
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
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

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

    # 투영 행렬
    projection = create_projection_matrix(60, SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 50.0)

    # 카메라 설정 (전지적 시점)
    map_center_x = len(MAP[0]) / 2
    map_center_z = len(MAP) / 2
    camera_x = map_center_x
    camera_y = 10
    camera_z = map_center_z + 10
    camera_yaw = 0
    camera_pitch = -45
    camera_speed = 0.1

    # 유니폼 위치
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 키보드 입력 처리
        keys = pygame.key.get_pressed()

        # WASD로 이동 (XZ 평면)
        if keys[pygame.K_w]:
            camera_z -= camera_speed
        if keys[pygame.K_s]:
            camera_z += camera_speed
        if keys[pygame.K_a]:
            camera_x -= camera_speed
        if keys[pygame.K_d]:
            camera_x += camera_speed

        # QE로 높이 조절
        if keys[pygame.K_q]:
            camera_y -= camera_speed
        if keys[pygame.K_e]:
            camera_y += camera_speed

        # 화살표로 시점 회전
        if keys[pygame.K_LEFT]:
            camera_yaw -= 1
        if keys[pygame.K_RIGHT]:
            camera_yaw += 1
        if keys[pygame.K_UP]:
            camera_pitch += 1
            camera_pitch = min(89, camera_pitch)
        if keys[pygame.K_DOWN]:
            camera_pitch -= 1
            camera_pitch = max(-89, camera_pitch)

        # 뷰 행렬 업데이트
        view = create_view_matrix((camera_x, camera_y, camera_z), camera_yaw, camera_pitch)

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

        # 지형 생성 및 렌더링
        generate_terrain_from_grid(MAP, cube_vao, quad_vao, wall_textures, floor_texture, shader)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
