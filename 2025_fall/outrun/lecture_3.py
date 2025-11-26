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
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
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

def draw_terrain_from_grid(grid, cube_vao, quad_vao, wall_textures, floor_texture, shader):
    """2D 그리드로부터 3D 지형 생성 및 렌더링"""
    model_loc = glGetUniformLocation(shader, "model")
    tint_loc = glGetUniformLocation(shader, "tintColor")

    # 바닥 그리기
    glBindVertexArray(quad_vao)
    glBindTexture(GL_TEXTURE_2D, floor_texture)
    glUniform3f(tint_loc, 1.0, 1.0, 1.0)

    # 바닥을 XZ 평면에 그리기
    model = translation_matrix(len(grid[0]) / 2 - 0.5 , 0, len(grid) / 2 - 0.5) @ \
        rotation_x_matrix(-math.pi / 2) @ \
        scale_matrix(len(grid),len(grid), len(grid))
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
                model = translation_matrix(x, 0.5, z)
                glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)


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


def main():
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple Cube")

    # OpenGL 설정
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

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

    # VAO 생성
    cube_vao = create_cube_vao()
    quad_vao = create_quad_vao()
    axis_vao = create_axis()

    # 텍스처 로드
    wall_textures = {
        1: load_texture("textures/wall1.png"),
        2: load_texture("textures/wall2.png"),
        3: load_texture("textures/wall3.png"),
    }
    floor_texture = load_texture("textures/floor.png")


    # 유니폼 위치
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    
    # 유니폼 위치 (라인 셰이더)
    line_mvp_loc = glGetUniformLocation(line_shader, "mvp")


    clock = pygame.time.Clock()
    running = True

    # 카메라 위치
    camera_x = 0
    camera_y = 2
    camera_z = 5

    # 카메라 이동 속도
    camera_speed = 0.1

    fov = math.radians(60)
    aspect = SCREEN_WIDTH / SCREEN_HEIGHT
    near, far = 0.1, 50.0

    f = 1.0 / math.tan(fov / 2.0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 키보드 입력으로 카메라 이동
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  # 앞으로
            camera_z -= camera_speed
        if keys[pygame.K_s]:  # 뒤로
            camera_z += camera_speed
        if keys[pygame.K_a]:  # 왼쪽으로
            camera_x -= camera_speed
        if keys[pygame.K_d]:  # 오른쪽으로
            camera_x += camera_speed
        if keys[pygame.K_q]:  # 위로
            camera_y += camera_speed
        if keys[pygame.K_e]:  # 아래로
            camera_y -= camera_speed

        # MVP 행렬 생성

        # 카메라는 항상 원점을 바라봄
        eye = [camera_x, camera_y, camera_z]
        at = [0, 0, 0]  # 원점을 바라봄
        up = [0, 1, 0]  # Y축이 위쪽
        view = create_view_matrix_lookat(eye, at, up)

        projection = projection_matrix(fov, aspect, near, far )

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)
        
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)
        
        draw_terrain_from_grid(MAP, cube_vao, quad_vao, wall_textures, floor_texture, shader)


        # 큐브 그리기
        glBindVertexArray(cube_vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

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



