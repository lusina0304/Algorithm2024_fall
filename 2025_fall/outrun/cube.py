import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import numpy.typing as npt
import math

# 화면 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Vertex Shader
VERTEX_SHADER = """
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

# Fragment Shader
FRAGMENT_SHADER = """
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
    ])

def translation_matrix(t_x: float, t_y: float, t_z: float) -> npt.NDArray:
    return np.array([
        [1.0, 0.0, 0.0, t_x],
        [0.0, 1.0, 0.0, t_y],
        [0.0, 0.0, 1.0, t_z],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotation_x_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def rotation_y_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def rotation_z_matrix(rad) -> npt.NDArray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

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

def create_cube_vao():
    """색상이 있는 큐브 VAO 생성"""
    vertices = np.array([
        # 위치              색상
        # 앞면 (빨강)
        -0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
         0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
         0.5,  0.5,  0.5,  1.0, 0.0, 0.0,
        -0.5,  0.5,  0.5,  1.0, 0.0, 0.0,
        # 뒷면 (초록)
        -0.5, -0.5, -0.5,  0.0, 1.0, 0.0,
        -0.5,  0.5, -0.5,  0.0, 1.0, 0.0,
         0.5,  0.5, -0.5,  0.0, 1.0, 0.0,
         0.5, -0.5, -0.5,  0.0, 1.0, 0.0,
        # 왼쪽면 (파랑)
        -0.5, -0.5, -0.5,  0.0, 0.0, 1.0,
        -0.5, -0.5,  0.5,  0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
        -0.5,  0.5, -0.5,  0.0, 0.0, 1.0,
        # 오른쪽면 (노랑)
         0.5, -0.5, -0.5,  1.0, 1.0, 0.0,
         0.5,  0.5, -0.5,  1.0, 1.0, 0.0,
         0.5,  0.5,  0.5,  1.0, 1.0, 0.0,
         0.5, -0.5,  0.5,  1.0, 1.0, 0.0,
        # 윗면 (자홍)
        -0.5,  0.5, -0.5,  1.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,  1.0, 0.0, 1.0,
         0.5,  0.5,  0.5,  1.0, 0.0, 1.0,
         0.5,  0.5, -0.5,  1.0, 0.0, 1.0,
        # 아랫면 (청록)
        -0.5, -0.5, -0.5,  0.0, 1.0, 1.0,
         0.5, -0.5, -0.5,  0.0, 1.0, 1.0,
         0.5, -0.5,  0.5,  0.0, 1.0, 1.0,
        -0.5, -0.5,  0.5,  0.0, 1.0, 1.0,
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

    # 위치 속성
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # 색상 속성
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

    # VAO 생성
    cube_vao = create_cube_vao()

    # 유니폼 위치
    mvp_loc = glGetUniformLocation(shader, "mvp")

    clock = pygame.time.Clock()
    running = True

    camera_x = camera_y = camera_z = 0
    camera_z = 3

    fov = math.radians(45)
    aspect = SCREEN_WIDTH / SCREEN_HEIGHT
    near, far = 0.1, 50.0

    f = 1.0 / math.tan(fov / 2.0)

    rotation = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 회전
        rotation += 1

        # MVP 행렬 생성
        model = model_matrix(rot_y_rad = math.radians(rotation))
        view = translation_matrix(-camera_x, -camera_y, -camera_z)
        projection = projection_matrix(fov, aspect, near, far )

        mvp = projection @ view @ model

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)

        # 큐브 그리기
        glBindVertexArray(cube_vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()



