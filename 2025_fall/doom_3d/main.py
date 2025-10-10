"""Main game loop for Doom 3D."""
import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import os
import time

from camera import Camera
from terrain_renderer import TerrainRenderer
from sprite_renderer import SpriteRenderer
from obj_loader import OBJLoader
from texture_loader import TextureLoader
from shader_loader import load_shader
from input_handler import InputHandler
from enemy import Enemy, EnemyManager
from weapon import Weapon
from hud import HUD


class Game:
    """Main game class."""

    def __init__(self, width=1280, height=720):
        """Initialize game."""
        self.width = width
        self.height = height
        self.running = True

        # Initialize Pygame
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Doom 3D")

        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # Initialize camera
        self.camera = Camera(position=[0, 1.6, 5], aspect_ratio=width/height)

        # Load shaders
        self.terrain_shader = load_shader('shaders/terrain.vert', 'shaders/terrain.frag')
        self.sprite_shader = load_shader('shaders/sprite.vert', 'shaders/sprite.frag')

        # Initialize renderers
        self.terrain_renderer = TerrainRenderer(self.terrain_shader)
        self.sprite_renderer = SpriteRenderer(self.sprite_shader)

        # Initialize systems
        self.input_handler = InputHandler(self.camera, width, height)
        self.enemy_manager = EnemyManager()
        self.hud = HUD(width, height)

        # Game state
        self.player_health = 100
        self.kills = 0

        # Load assets
        self._load_assets()

        # Clock for delta time
        self.clock = pygame.time.Clock()
        self.last_time = time.time()

    def _load_assets(self):
        """Load game assets (terrain, textures, sprites)."""
        print("Loading assets...")

        # Load terrain
        if os.path.exists('assets/models/terrain.obj'):
            vertices, uvs, normals = OBJLoader.load('assets/models/terrain.obj')
            if vertices is not None:
                self.terrain_renderer.load_terrain(vertices, uvs, normals)

                # Load terrain texture
                if os.path.exists('assets/textures/terrain.png'):
                    terrain_tex = TextureLoader.load_texture('assets/textures/terrain.png')
                    self.terrain_renderer.set_texture(terrain_tex)
        else:
            # Create simple floor plane if no terrain exists
            self._create_default_terrain()

        # Load weapon sprites
        self.weapon_textures = []
        weapon_files = ['gun_idle.png', 'gun_shoot1.png', 'gun_shoot2.png']
        for wf in weapon_files:
            path = f'assets/sprites/{wf}'
            if os.path.exists(path):
                tex, _, _ = TextureLoader.load_sprite(path)
                self.weapon_textures.append(tex)

        if not self.weapon_textures:
            # Create placeholder
            self.weapon_textures = [self._create_placeholder_texture()]

        self.weapon = Weapon(self.weapon_textures, damage=25)

        # Load enemy sprites
        self._load_enemies()

        print("Assets loaded!")

    def _create_default_terrain(self):
        """Create a simple floor plane."""
        size = 20
        vertices = np.array([
            -size, 0, -size,
             size, 0, -size,
             size, 0,  size,
            -size, 0, -size,
             size, 0,  size,
            -size, 0,  size
        ], dtype=np.float32).reshape(-1, 3)

        uvs = np.array([
            0, 0,
            10, 0,
            10, 10,
            0, 0,
            10, 10,
            0, 10
        ], dtype=np.float32).reshape(-1, 2)

        normals = np.array([
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        ], dtype=np.float32).reshape(-1, 3)

        self.terrain_renderer.load_terrain(vertices, uvs, normals)

        # Create simple texture
        self.terrain_renderer.set_texture(self._create_placeholder_texture())

    def _create_placeholder_texture(self):
        """Create a simple colored texture."""
        from PIL import Image
        img = Image.new('RGB', (64, 64), color=(100, 100, 100))
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE,
                    np.array(img, dtype=np.uint8))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture_id

    def _load_enemies(self):
        """Load enemy sprites and create enemies."""
        # Load enemy animation frames
        enemy_anims = {
            'idle': [],
            'walk': [],
            'hit': [],
            'die': []
        }

        # Try to load enemy sprites
        for state in ['idle', 'walk', 'hit', 'die']:
            for i in range(1, 5):  # Assume 4 frames per animation
                path = f'assets/sprites/enemy_{state}_{i}.png'
                if os.path.exists(path):
                    tex, _, _ = TextureLoader.load_sprite(path)
                    enemy_anims[state].append(tex)

        # If no sprites found, use placeholder
        if not any(enemy_anims.values()):
            placeholder = self._create_placeholder_texture()
            for key in enemy_anims:
                enemy_anims[key] = [placeholder]

        # Create some enemies
        enemy_positions = [
            [5, 1, 0],
            [-5, 1, 0],
            [0, 1, -5],
            [3, 1, -3]
        ]

        for pos in enemy_positions:
            enemy = Enemy(pos, enemy_anims)
            self.enemy_manager.add_enemy(enemy)

    def update(self, delta_time):
        """Update game state."""
        # Update input
        self.input_handler.update(delta_time)

        # Handle shooting
        if self.input_handler.is_shooting():
            if self.weapon.shoot():
                # Ray cast to check hit
                hit_enemy = self.enemy_manager.check_hit(
                    self.camera.position,
                    self.camera.front
                )
                if hit_enemy:
                    hit_enemy.take_damage(self.weapon.damage)
                    if not hit_enemy.alive:
                        self.kills += 1

        # Update weapon
        self.weapon.update(delta_time)

        # Update enemies
        self.enemy_manager.update(delta_time, self.camera.position)

        # Update HUD
        self.hud.update_stats(health=self.player_health, kills=self.kills)

    def render(self):
        """Render frame."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render 3D terrain
        model_matrix = np.identity(4, dtype=np.float32)
        mvp = self.camera.get_mvp_matrix(model_matrix)

        self.terrain_renderer.render(
            mvp, model_matrix,
            self.camera.position,
            light_pos=[10, 10, 10],
            light_color=[1, 1, 1]
        )

        # Render enemies (billboards)
        self._render_enemies()

        # Render weapon (2D overlay)
        self._render_weapon()

        # Render HUD
        self.hud.render()

        pygame.display.flip()

    def _render_enemies(self):
        """Render enemy sprites as billboards."""
        # Simple billboard rendering
        ortho = Camera.ortho(-10, 10, -10, 10, -100, 100)

        for enemy in self.enemy_manager.enemies:
            texture = enemy.get_current_texture()
            if texture is None:
                continue

            # Calculate screen position (simplified billboard)
            # In a full implementation, you'd project 3D position to screen space
            # For now, we'll use a simple approach

            # Calculate vector from camera to enemy
            to_enemy = enemy.position - self.camera.position
            distance = np.linalg.norm(to_enemy)

            if distance > 0:
                # Project onto camera space
                forward = self.camera.front
                right = self.camera.right
                up = self.camera.up

                # Local coordinates
                local_x = np.dot(to_enemy, right)
                local_y = np.dot(to_enemy, up)
                local_z = np.dot(to_enemy, forward)

                if local_z > 0:  # In front of camera
                    # Simple perspective projection
                    screen_x = (local_x / local_z) * 5
                    screen_y = (local_y / local_z) * 5

                    # Scale based on distance
                    scale = 2.0 / distance

                    self.sprite_renderer.render_sprite(
                        texture,
                        screen_x, screen_y,
                        enemy.width * scale, enemy.height * scale,
                        1.0, ortho
                    )

    def _render_weapon(self):
        """Render weapon sprite at bottom center of screen."""
        ortho = Camera.ortho(0, self.width, 0, self.height, -1, 1)

        weapon_texture = self.weapon.get_current_texture()
        if weapon_texture:
            weapon_width = 400
            weapon_height = 400
            weapon_x = self.width / 2
            weapon_y = 100

            self.sprite_renderer.render_sprite(
                weapon_texture,
                weapon_x, weapon_y,
                weapon_width, weapon_height,
                1.0, ortho
            )

    def run(self):
        """Main game loop."""
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            # Limit delta time to avoid huge jumps
            delta_time = min(delta_time, 0.1)

            # Handle events
            events = pygame.event.get()
            if not self.input_handler.process_events(events):
                self.running = False

            # Update
            self.update(delta_time)

            # Render
            self.render()

            # Cap framerate
            self.clock.tick(60)

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.terrain_renderer.cleanup()
        self.sprite_renderer.cleanup()
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
