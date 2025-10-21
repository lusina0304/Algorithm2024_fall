import numpy as np
import math

EPSILON = 1e-15

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector
    else:
        return vector / norm
    
# def get_angle_from_direction(direction):
#     math.atan2(  )

def create_translation_matrix(x, y, z):
    """이동 행렬 생성"""
    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix

def create_rotation_x_matrix(angle):
    """X축 회전 행렬 생성 (각도)"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def create_rotation_y_matrix(angle):
    """Y축 회전 행렬 생성 (각도)"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def create_rotation_z_matrix(angle):
    """Z축 회전 행렬 생성 (각도)"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def create_scale_matrix(sx, sy, sz):
    """스케일 행렬 생성"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def create_model_matrix(x, y, z, scale=1.0, scale_x=None, scale_y=None, scale_z=None, rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
    """모델 행렬 생성 (이동, 회전, 스케일)"""
    # OpenGL 변환 순서: 이동 -> 회전 -> 스케일 (오른쪽에서 왼쪽으로)
    model = np.identity(4, dtype=np.float32)

    # 스케일 (개별 축 스케일이 지정되지 않으면 균등 스케일 사용)
    sx = scale_x if scale_x is not None else scale
    sy = scale_y if scale_y is not None else scale
    sz = scale_z if scale_z is not None else scale

    if sx != 1.0 or sy != 1.0 or sz != 1.0:
        model = create_scale_matrix(sx, sy, sz) @ model

    # 회전 (Y -> X -> Z 순서로 적용)
    if rotate_y != 0.0:
        model = create_rotation_y_matrix(rotate_y) @ model
    if rotate_x != 0.0:
        model = create_rotation_x_matrix(rotate_x) @ model
    if rotate_z != 0.0:
        model = create_rotation_z_matrix(rotate_z) @ model

    # 이동
    model = create_translation_matrix(x, y, z) @ model

    return model

def create_view_matrix(camera_pos, yaw, pitch):
    """뷰 행렬 생성 (카메라 위치와 회전)"""
    view = np.identity(4, dtype=np.float32)

    # 이동 (카메라 반대 방향으로)
    view = create_translation_matrix(-camera_pos[0], -camera_pos[1], -camera_pos[2]) @ view

    # 회전 (카메라 회전의 역)
    view = create_rotation_y_matrix(-yaw) @ view
    view = create_rotation_x_matrix(-pitch) @ view
    
    return view

def create_projection_matrix(fov_degrees, aspect_ratio, near, far):
    """원근 투영 행렬 생성"""
    fov = math.radians(fov_degrees)
    f = 1.0 / math.tan(fov / 2.0)
    projection = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

    return projection

def create_ortho_matrix(left, right, bottom, top, near, far):
    """직교 투영 행렬 생성 (2D용)"""
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
