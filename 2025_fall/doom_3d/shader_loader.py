"""Shader loading and compilation utilities."""
from OpenGL.GL import *


def load_shader(vertex_path, fragment_path):
    """
    Load and compile shaders.

    Args:
        vertex_path: Path to vertex shader
        fragment_path: Path to fragment shader

    Returns:
        int: Compiled shader program ID
    """
    # Read shader source
    with open(vertex_path, 'r') as f:
        vertex_code = f.read()

    with open(fragment_path, 'r') as f:
        fragment_code = f.read()

    # Compile vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_code)
    glCompileShader(vertex_shader)

    # Check compilation
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(vertex_shader).decode()
        raise RuntimeError(f"Vertex shader compilation failed:\n{error}")

    # Compile fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_code)
    glCompileShader(fragment_shader)

    # Check compilation
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(fragment_shader).decode()
        raise RuntimeError(f"Fragment shader compilation failed:\n{error}")

    # Link program
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check linking
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader program linking failed:\n{error}")

    # Clean up shaders (they're linked into program now)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program
