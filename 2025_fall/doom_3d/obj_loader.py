"""OBJ file loader for 3D terrain."""
import numpy as np


class OBJLoader:
    """Loads .obj files and returns vertex data."""

    @staticmethod
    def load(filename):
        """
        Load OBJ file and return vertices, UVs, normals.

        Returns:
            tuple: (vertices, uvs, normals) as numpy arrays
        """
        vertices = []
        uvs = []
        normals = []

        vertex_indices = []
        uv_indices = []
        normal_indices = []

        try:
            with open(filename, 'r') as file:
                for line in file:
                    if line.startswith('v '):
                        # Vertex
                        parts = line.split()[1:]
                        vertices.append([float(x) for x in parts])
                    elif line.startswith('vt '):
                        # UV coordinate
                        parts = line.split()[1:]
                        uvs.append([float(x) for x in parts[:2]])
                    elif line.startswith('vn '):
                        # Normal
                        parts = line.split()[1:]
                        normals.append([float(x) for x in parts])
                    elif line.startswith('f '):
                        # Face
                        parts = line.split()[1:]
                        for part in parts:
                            indices = part.split('/')
                            vertex_indices.append(int(indices[0]) - 1)
                            if len(indices) > 1 and indices[1]:
                                uv_indices.append(int(indices[1]) - 1)
                            if len(indices) > 2 and indices[2]:
                                normal_indices.append(int(indices[2]) - 1)
        except FileNotFoundError:
            print(f"Error: Could not find file {filename}")
            return None, None, None

        # Build final arrays
        final_vertices = []
        final_uvs = []
        final_normals = []

        for i in range(len(vertex_indices)):
            final_vertices.append(vertices[vertex_indices[i]])

            if uv_indices and i < len(uv_indices):
                final_uvs.append(uvs[uv_indices[i]])
            else:
                final_uvs.append([0.0, 0.0])

            if normal_indices and i < len(normal_indices):
                final_normals.append(normals[normal_indices[i]])
            else:
                final_normals.append([0.0, 1.0, 0.0])

        return (np.array(final_vertices, dtype=np.float32),
                np.array(final_uvs, dtype=np.float32),
                np.array(final_normals, dtype=np.float32))
