import numpy as np
import math
import glfw # 창 관리 및 입력 처리를 위해 사용
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# --- MVP 행렬 생성 함수 (이전 코드에서 재사용) ---

def identity_matrix():
    return np.identity(4, dtype=np.float32)

def model_matrix(time, scale_factor=0.5):
    angle_x = time * 0.5
    angle_y = time * 0.8
    cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
    cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)

    rotation_x = identity_matrix()
    rotation_x[1, 1] = cos_x
    rotation_x[1, 2] = -sin_x
    rotation_x[2, 1] = sin_x
    rotation_x[2, 2] = cos_x

    rotation_y = identity_matrix()
    rotation_y[0, 0] = cos_y
    rotation_y[0, 2] = sin_y
    rotation_y[2, 0] = -sin_y
    rotation_y[2, 2] = cos_y

    scale = identity_matrix()
    scale[0, 0] = scale_factor
    scale[1, 1] = scale_factor
    scale[2, 2] = scale_factor

    model = rotation_y @ rotation_x @ scale
    return model

def view_matrix(camera_position, target_position, up_vector):
    # 'Look At' 구현
    z_axis = target_position - camera_position
    z_axis = z_axis / np.linalg.norm(z_axis)

    up_vector = up_vector / np.linalg.norm(up_vector)
    
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    
    # 뷰 행렬 R^T * T(-pos)
    R = identity_matrix()
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    T = identity_matrix()
    T[:3, 3] = -camera_position
    
    view = R @ T
    return view

def orthogonal_projection_matrix(left, right, bottom, top, near, far):
    sx = 2 / (right - left)
    tx = -(right + left) / (right - left)
    sy = 2 / (top - bottom)
    ty = -(top + bottom) / (top - bottom)
    sz = -2 / (far - near) 
    tz = -(far + near) / (far - near)

    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = sx
    projection[1, 1] = sy
    projection[2, 2] = sz
    projection[0, 3] = tx
    projection[1, 3] = ty
    projection[2, 3] = tz
    projection[3, 3] = 1.0
    return projection

# --- 큐브 정점, 색상, 인덱스 정의 ---

# 큐브의 정점 데이터 (3차원 좌표 + 색상 RGB)
# 6면을 렌더링하기 위해 36개의 정점 사용 (각 면은 2개의 삼각형, 6개의 정점)
# 각 면에 다른 색상을 지정하여 회전을 쉽게 확인할 수 있도록 했습니다.
CUBE_VERTICES = np.array([
    # 정점 위치 (x, y, z) | 색상 (r, g, b)
    # 앞면 (파랑)
    -1.0, -1.0,  1.0, 0.0, 0.0, 1.0, # 0
     1.0, -1.0,  1.0, 0.0, 0.0, 1.0, # 1
     1.0,  1.0,  1.0, 0.0, 0.0, 1.0, # 2
     1.0,  1.0,  1.0, 0.0, 0.0, 1.0, # 2
    -1.0,  1.0,  1.0, 0.0, 0.0, 1.0, # 3
    -1.0, -1.0,  1.0, 0.0, 0.0, 1.0, # 0
    
    # 뒷면 (빨강)
    -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, # 4
     1.0, -1.0, -1.0, 1.0, 0.0, 0.0, # 5
     1.0,  1.0, -1.0, 1.0, 0.0, 0.0, # 6
     1.0,  1.0, -1.0, 1.0, 0.0, 0.0, # 6
    -1.0,  1.0, -1.0, 1.0, 0.0, 0.0, # 7
    -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, # 4

    # 윗면 (초록)
    -1.0,  1.0,  1.0, 0.0, 1.0, 0.0, # 3
     1.0,  1.0,  1.0, 0.0, 1.0, 0.0, # 2
     1.0,  1.0, -1.0, 0.0, 1.0, 0.0, # 6
     1.0,  1.0, -1.0, 0.0, 1.0, 0.0, # 6
    -1.0,  1.0, -1.0, 0.0, 1.0, 0.0, # 7
    -1.0,  1.0,  1.0, 0.0, 1.0, 0.0, # 3

    # 아랫면 (노랑)
    -1.0, -1.0,  1.0, 1.0, 1.0, 0.0, # 0
     1.0, -1.0,  1.0, 1.0, 1.0, 0.0, # 1
     1.0, -1.0, -1.0, 1.0, 1.0, 0.0, # 5
     1.0, -1.0, -1.0, 1.0, 1.0, 0.0, # 5
    -1.0, -1.0, -1.0, 1.0, 1.0, 0.0, # 4
    -1.0, -1.0,  1.0, 1.0, 1.0, 0.0, # 0

    # 오른쪽 면 (자홍)
     1.0, -1.0,  1.0, 1.0, 0.0, 1.0, # 1
     1.0, -1.0, -1.0, 1.0, 0.0, 1.0, # 5
     1.0,  1.0, -1.0, 1.0, 0.0, 1.0, # 6
     1.0,  1.0, -1.0, 1.0, 0.0, 1.0, # 6
     1.0,  1.0,  1.0, 1.0, 0.0, 1.0, # 2
     1.0, -1.0,  1.0, 1.0, 0.0, 1.0, # 1

    # 왼쪽 면 (하늘)
    -1.0, -1.0,  1.0, 0.0, 1.0, 1.0, # 0
    -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, # 4
    -1.0,  1.0, -1.0, 0.0, 1.0, 1.0, # 7
    -1.0,  1.0, -1.0, 0.0, 1.0, 1.0, # 7
    -1.0,  1.0,  1.0, 0.0, 1.0, 1.0, # 3
    -1.0, -1.0,  1.0, 0.0, 1.0, 1.0, # 0
], dtype=np.float32)

# --- 쉐이더 소스 코드 ---

# 정점 쉐이더: MVP 변환을 적용하고 색상을 프래그먼트 쉐이더로 전달합니다.
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;     // 정점 위치 (VBO 데이터에서 0번째 속성)
layout (location = 1) in vec3 aColor;   // 정점 색상 (VBO 데이터에서 1번째 속성)

out vec3 vColor;                       // 프래그먼트 쉐이더로 전달할 색상

uniform mat4 u_mvp;                    // MVP 행렬

void main()
{
    gl_Position = u_mvp * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

# 프래그먼트 쉐이더: 최종 색상을 결정합니다.
FRAGMENT_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vColor, 1.0);
}
"""

# --- OpenGL 초기화 및 렌더링 함수 ---

def create_shader_program():
    """쉐이더를 컴파일하고 쉐이더 프로그램을 생성합니다."""
    vertex = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    return compileProgram(vertex, fragment)

def init_opengl(window):
    """OpenGL 상태 설정 및 큐브 버퍼를 초기화합니다."""
    # 쉐이더 프로그램 생성
    shader_program = create_shader_program()
    glUseProgram(shader_program)
    
    # 깊이 테스트 활성화 (큐브의 앞면이 뒷면을 가리도록)
    glEnable(GL_DEPTH_TEST)
    
    # VAO (Vertex Array Object) 생성
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    # VBO (Vertex Buffer Object) 생성 및 데이터 전송
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, CUBE_VERTICES.nbytes, CUBE_VERTICES, GL_STATIC_DRAW)
    
    # 정점 위치 속성 설정 (aPos)
    # location=0, 3개의 float, stride=6 * float size (3 pos + 3 color), offset=0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * CUBE_VERTICES.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    # 정점 색상 속성 설정 (aColor)
    # location=1, 3개의 float, stride=6 * float size, offset=3 * float size
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * CUBE_VERTICES.itemsize, ctypes.c_void_p(3 * CUBE_VERTICES.itemsize))
    glEnableVertexAttribArray(1)

    # uniform 위치 가져오기
    mvp_loc = glGetUniformLocation(shader_program, "u_mvp")

    return shader_program, mvp_loc

def render_cube(time, shader_program, mvp_loc, width, height):
    """큐브를 렌더링합니다."""
    
    glClearColor(0.2, 0.3, 0.3, 1.0) # 배경색 설정
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # 색상 및 깊이 버퍼 클리어

    # 1. MVP 행렬 계산
    camera_position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
    target_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 직교 투영 파라미터: 화면 비율을 고려
    aspect_ratio = width / height
    scale = 2.0 
    left, right = -scale * aspect_ratio, scale * aspect_ratio
    bottom, top = -scale, scale
    near, far = 0.1, 100.0

    M = model_matrix(time)
    V = view_matrix(camera_position, target_position, up_vector)
    P = orthogonal_projection_matrix(left, right, bottom, top, near, far)
    MVP = P @ V @ M

    # 2. MVP 행렬을 쉐이더 uniform 변수에 전달
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, MVP)

    # 3. 큐브 드로잉
    # 36개의 정점 (6면 * 2삼각형 * 3정점)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    
# --- 메인 루프 ---

def main():
    # GLFW 초기화
    if not glfw.init():
        return
    
    # OpenGL 3.3 Core Profile 설정
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "PyOpenGL Rotating Cube (Orthogonal)", None, None)
    
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    # OpenGL 초기화 및 버퍼 설정
    shader_program, mvp_loc = init_opengl(window)

    # 메인 루프
    while not glfw.window_should_close(window):
        # 현재 시간 가져오기 (애니메이션에 사용)
        time = glfw.get_time()
        
        # 렌더링
        render_cube(time, shader_program, mvp_loc, WINDOW_WIDTH, WINDOW_HEIGHT)

        # 버퍼 교체 및 이벤트 폴링
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 정리
    glfw.terminate()

if __name__ == "__main__":
    main()