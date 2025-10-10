"""3D terrain renderer using OpenGL."""
from OpenGL.GL import *
import numpy as np
import ctypes


class TerrainRenderer:
    """Renders 3D terrain with textures."""

    def __init__(self, shader_program):
        """
        Initialize terrain renderer.

        Args:
            shader_program: Compiled OpenGL shader program
        """
        self.shader_program = shader_program
        self.vao = None
        self.vbo_vertices = None
        self.vbo_uvs = None
        self.vbo_normals = None
        self.vertex_count = 0
        self.texture_id = None

    def load_terrain(self, vertices, uvs, normals):
        """
        Load terrain mesh data into OpenGL buffers.

        Args:
            vertices: Vertex positions array
            uvs: UV coordinates array
            normals: Normal vectors array
        """
        self.vertex_count = len(vertices)

        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Vertex buffer
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # UV buffer
        self.vbo_uvs = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uvs)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        # Normal buffer
        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def set_texture(self, texture_id):
        """Set terrain texture."""
        self.texture_id = texture_id

    def render(self, mvp_matrix, model_matrix, camera_pos, light_pos=(10, 10, 10), light_color=(1, 1, 1)):
        """
        Render terrain.

        Args:
            mvp_matrix: Model-View-Projection matrix
            model_matrix: Model transformation matrix
            camera_pos: Camera position for lighting
            light_pos: Light source position
            light_color: Light color RGB
        """
        if self.vao is None or self.vertex_count == 0:
            return

        glUseProgram(self.shader_program)

        # Set uniforms
        mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_matrix.T)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix.T)

        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        glUniform3f(view_pos_loc, *camera_pos)

        light_pos_loc = glGetUniformLocation(self.shader_program, "lightPos")
        glUniform3f(light_pos_loc, *light_pos)

        light_color_loc = glGetUniformLocation(self.shader_program, "lightColor")
        glUniform3f(light_color_loc, *light_color)

        # Bind texture
        if self.texture_id:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            tex_loc = glGetUniformLocation(self.shader_program, "textureSampler")
            glUniform1i(tex_loc, 0)

        # Draw
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

        glUseProgram(0)

    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.vbo_vertices:
            glDeleteBuffers(1, [self.vbo_vertices])
        if self.vbo_uvs:
            glDeleteBuffers(1, [self.vbo_uvs])
        if self.vbo_normals:
            glDeleteBuffers(1, [self.vbo_normals])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
