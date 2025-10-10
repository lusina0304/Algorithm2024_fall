"""Camera and MVP matrix implementation."""
import numpy as np
import math


class Camera:
    """FPS camera with view and projection matrices."""

    def __init__(self, position=None, fov=45.0, aspect_ratio=16/9, near=0.1, far=100.0):
        """
        Initialize camera.

        Args:
            position: Camera position [x, y, z]
            fov: Field of view in degrees
            aspect_ratio: Screen width / height
            near: Near clipping plane
            far: Far clipping plane
        """
        self.position = np.array(position if position else [0.0, 1.0, 5.0], dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Euler angles
        self.yaw = -90.0  # Looking towards -Z
        self.pitch = 0.0

        # Projection parameters
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

        self.update_camera_vectors()

    def get_view_matrix(self):
        """Calculate and return view matrix."""
        return self.look_at(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        """Calculate and return perspective projection matrix."""
        return self.perspective(self.fov, self.aspect_ratio, self.near, self.far)

    def get_mvp_matrix(self, model_matrix=None):
        """
        Calculate Model-View-Projection matrix.

        Args:
            model_matrix: 4x4 model transformation matrix

        Returns:
            4x4 MVP matrix
        """
        if model_matrix is None:
            model_matrix = np.identity(4, dtype=np.float32)

        view = self.get_view_matrix()
        projection = self.get_projection_matrix()

        mvp = projection @ view @ model_matrix
        return mvp

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """
        Process mouse movement for camera rotation.

        Args:
            xoffset: Mouse movement in X
            yoffset: Mouse movement in Y
            constrain_pitch: Limit pitch to avoid camera flip
        """
        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            self.pitch = max(-89.0, min(89.0, self.pitch))

        self.update_camera_vectors()

    def process_keyboard(self, direction, velocity):
        """
        Move camera based on keyboard input.

        Args:
            direction: 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT'
            velocity: Movement speed
        """
        if direction == 'FORWARD':
            self.position += self.front * velocity
        if direction == 'BACKWARD':
            self.position -= self.front * velocity
        if direction == 'LEFT':
            self.position -= self.right * velocity
        if direction == 'RIGHT':
            self.position += self.right * velocity

    def update_camera_vectors(self):
        """Update camera vectors based on yaw and pitch."""
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)

        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)

    @staticmethod
    def look_at(position, target, up):
        """
        Create view matrix using look-at.

        Args:
            position: Camera position
            target: Point to look at
            up: Up vector

        Returns:
            4x4 view matrix
        """
        z = position - target
        z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)

        view = np.identity(4, dtype=np.float32)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = z
        view[:3, 3] = [-np.dot(x, position), -np.dot(y, position), -np.dot(z, position)]

        return view

    @staticmethod
    def perspective(fov, aspect, near, far):
        """
        Create perspective projection matrix.

        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            4x4 projection matrix
        """
        f = 1.0 / math.tan(math.radians(fov) / 2.0)

        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2.0 * far * near) / (near - far)
        projection[3, 2] = -1.0

        return projection

    @staticmethod
    def ortho(left, right, bottom, top, near, far):
        """
        Create orthographic projection matrix for 2D sprites/HUD.

        Args:
            left, right, bottom, top: Screen bounds
            near, far: Depth bounds

        Returns:
            4x4 orthographic matrix
        """
        ortho_matrix = np.identity(4, dtype=np.float32)
        ortho_matrix[0, 0] = 2.0 / (right - left)
        ortho_matrix[1, 1] = 2.0 / (top - bottom)
        ortho_matrix[2, 2] = -2.0 / (far - near)
        ortho_matrix[0, 3] = -(right + left) / (right - left)
        ortho_matrix[1, 3] = -(top + bottom) / (top - bottom)
        ortho_matrix[2, 3] = -(far + near) / (far - near)

        return ortho_matrix
