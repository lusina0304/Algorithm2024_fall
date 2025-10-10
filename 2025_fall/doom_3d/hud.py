"""HUD (Heads-Up Display) system."""
import pygame
from OpenGL.GL import *


class HUD:
    """HUD for displaying game information."""

    def __init__(self, screen_width, screen_height):
        """
        Initialize HUD.

        Args:
            screen_width: Screen width
            screen_height: Screen height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize pygame font
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Player stats
        self.health = 100
        self.ammo = 50
        self.kills = 0

    def update_stats(self, health=None, ammo=None, kills=None):
        """Update HUD statistics."""
        if health is not None:
            self.health = max(0, health)
        if ammo is not None:
            self.ammo = max(0, ammo)
        if kills is not None:
            self.kills = kills

    def render_text(self, text, x, y, font=None, color=(255, 255, 255)):
        """
        Render text to screen using pygame.

        Args:
            text: Text to render
            x, y: Screen position
            font: Font to use (default: self.font)
            color: Text color RGB
        """
        if font is None:
            font = self.font

        # Render text to surface
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)

        # Save OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # Setup for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Create texture from text
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_surface.get_width(),
                    text_surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Render quad with text texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        glColor4f(1, 1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + text_surface.get_width(), y)
        glTexCoord2f(1, 1)
        glVertex2f(x + text_surface.get_width(), y + text_surface.get_height())
        glTexCoord2f(0, 1)
        glVertex2f(x, y + text_surface.get_height())
        glEnd()

        glDeleteTextures([texture])

        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def render(self):
        """Render HUD elements."""
        # Health color changes based on value
        health_color = (0, 255, 0) if self.health > 50 else (255, 255, 0) if self.health > 25 else (255, 0, 0)

        # Render health
        self.render_text(f"Health: {int(self.health)}", 20, 20, color=health_color)

        # Render ammo
        self.render_text(f"Ammo: {self.ammo}", 20, 60, color=(255, 255, 0))

        # Render kills
        self.render_text(f"Kills: {self.kills}", 20, 100, color=(255, 255, 255))

        # Crosshair
        self.draw_crosshair()

    def draw_crosshair(self):
        """Draw simple crosshair in center of screen."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        size = 10

        glPushAttrib(GL_ALL_ATTRIB_BITS)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(1, 1, 1, 0.8)
        glLineWidth(2)

        # Draw crosshair
        glBegin(GL_LINES)
        # Horizontal line
        glVertex2f(center_x - size, center_y)
        glVertex2f(center_x + size, center_y)
        # Vertical line
        glVertex2f(center_x, center_y - size)
        glVertex2f(center_x, center_y + size)
        glEnd()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()
