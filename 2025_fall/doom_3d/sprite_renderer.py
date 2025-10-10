"""2D sprite renderer for enemies, gun, and HUD."""
from OpenGL.GL import *
import numpy as np
import ctypes


class SpriteRenderer:
    """Renders 2D sprites with alpha blending."""

    def __init__(self, shader_program):
        """
        Initialize sprite renderer.

        Args:
            shader_program: Compiled OpenGL shader program
        """
        self.shader_program = shader_program
        self.vao = None
        self.vbo_vertices = None
        self.vbo_uvs = None
        self._init_quad()

    def _init_quad(self):
        """Initialize a quad for sprite rendering."""
        # Quad vertices (2D)
        vertices = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5
        ], dtype=np.float32)

        # UV coordinates
        uvs = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)

        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Vertex buffer
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # UV buffer
        self.vbo_uvs = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uvs)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def render_sprite(self, texture_id, x, y, width, height, alpha=1.0, ortho_matrix=None):
        """
        Render a 2D sprite.

        Args:
            texture_id: OpenGL texture ID
            x, y: Screen position
            width, height: Sprite dimensions
            alpha: Transparency (0-1)
            ortho_matrix: Orthographic projection matrix
        """
        if ortho_matrix is None:
            ortho_matrix = np.identity(4, dtype=np.float32)

        glUseProgram(self.shader_program)

        # Create transformation matrix
        model = np.identity(4, dtype=np.float32)
        model[0, 0] = width
        model[1, 1] = height
        model[0, 3] = x
        model[1, 3] = y

        mvp = ortho_matrix @ model

        # Set uniforms
        mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp.T)

        alpha_loc = glGetUniformLocation(self.shader_program, "alpha")
        glUniform1f(alpha_loc, alpha)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        tex_loc = glGetUniformLocation(self.shader_program, "spriteSampler")
        glUniform1i(tex_loc, 0)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        glDisable(GL_BLEND)
        glUseProgram(0)

    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.vbo_vertices:
            glDeleteBuffers(1, [self.vbo_vertices])
        if self.vbo_uvs:
            glDeleteBuffers(1, [self.vbo_uvs])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])


class Sprite:
    """Represents a sprite with position and animation."""

    def __init__(self, textures, x=0, y=0, width=64, height=64):
        """
        Initialize sprite.

        Args:
            textures: List of texture IDs for animation frames
            x, y: Position
            width, height: Dimensions
        """
        self.textures = textures if isinstance(textures, list) else [textures]
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.current_frame = 0
        self.animation_speed = 0.1
        self.animation_timer = 0.0
        self.alpha = 1.0

    def update(self, delta_time):
        """Update sprite animation."""
        if len(self.textures) > 1:
            self.animation_timer += delta_time
            if self.animation_timer >= self.animation_speed:
                self.animation_timer = 0.0
                self.current_frame = (self.current_frame + 1) % len(self.textures)

    def get_current_texture(self):
        """Get current animation frame texture."""
        return self.textures[self.current_frame]

    def render(self, renderer, ortho_matrix=None):
        """Render sprite using renderer."""
        renderer.render_sprite(
            self.get_current_texture(),
            self.x, self.y,
            self.width, self.height,
            self.alpha,
            ortho_matrix
        )
