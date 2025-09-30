import math
import numpy as np

# =========================
# Math helpers
# =========================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0, 0,                 0],
        [0,        f, 0,                 0],
        [0,        0, (far+near)*nf,     2*far*near*nf],
        [0,        0, -1,                0],
    ], dtype=np.float32)


def look_yaw_pitch(eye: np.ndarray, yaw: float, pitch: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yaw around Y, pitch around X
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = np.array([sy*cp, -sp, cy*cp], dtype=np.float32)  # right-handed
    target = eye + forward
    up = np.array([0,1,0], dtype=np.float32)
    # Build lookAt
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.identity(4, dtype=np.float32)
    t[:3, 3] = -eye
    V = m @ t
    return V, f, s


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.identity(4, dtype=np.float32)
    M[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return M


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.diag([sx, sy, sz, 1.0]).astype(np.float32)