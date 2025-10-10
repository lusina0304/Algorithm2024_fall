"""Texture loader using PIL and OpenGL."""
from OpenGL.GL import *
from PIL import Image
import numpy as np


class TextureLoader:
    """Handles texture loading for OpenGL."""

    @staticmethod
    def load_texture(filename):
        """
        Load texture from image file.

        Args:
            filename: Path to image file

        Returns:
            int: OpenGL texture ID
        """
        try:
            img = Image.open(filename)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            # Texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Upload texture
            if img.mode == 'RGB':
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height,
                           0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            elif img.mode == 'RGBA':
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height,
                           0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            else:
                print(f"Unsupported image mode: {img.mode}")
                return None

            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

            return texture_id

        except Exception as e:
            print(f"Error loading texture {filename}: {e}")
            return None

    @staticmethod
    def load_sprite(filename):
        """
        Load sprite texture with alpha channel.

        Args:
            filename: Path to sprite image file

        Returns:
            tuple: (texture_id, width, height)
        """
        try:
            img = Image.open(filename).convert('RGBA')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height,
                       0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

            glBindTexture(GL_TEXTURE_2D, 0)

            return texture_id, img.width, img.height

        except Exception as e:
            print(f"Error loading sprite {filename}: {e}")
            return None, 0, 0
