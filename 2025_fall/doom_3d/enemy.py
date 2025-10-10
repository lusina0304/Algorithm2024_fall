"""Enemy AI and animation system."""
import numpy as np
import math


class Enemy:
    """Enemy character with AI and animations."""

    # Animation states
    IDLE = 0
    WALKING = 1
    HIT = 2
    DYING = 3
    DEAD = 4

    def __init__(self, position, animations):
        """
        Initialize enemy.

        Args:
            position: [x, y, z] position in world
            animations: Dict of animation name -> list of texture IDs
                       {'idle': [...], 'walk': [...], 'hit': [...], 'die': [...]}
        """
        self.position = np.array(position, dtype=np.float32)
        self.animations = animations
        self.state = self.IDLE
        self.current_frame = 0
        self.animation_timer = 0.0
        self.animation_speed = 0.15

        self.health = 100
        self.speed = 2.0
        self.alive = True

        # For billboard rendering
        self.width = 1.5
        self.height = 2.0

    def update(self, delta_time, player_pos):
        """
        Update enemy AI and animation.

        Args:
            delta_time: Time since last frame
            player_pos: Player position [x, y, z]
        """
        if not self.alive:
            return

        # Update animation
        self.animation_timer += delta_time
        current_anim = self._get_current_animation()

        if self.animation_timer >= self.animation_speed:
            self.animation_timer = 0.0
            self.current_frame += 1

            # Handle state transitions
            if self.state == self.HIT:
                if self.current_frame >= len(current_anim):
                    self.state = self.IDLE if self.health > 0 else self.DYING
                    self.current_frame = 0
            elif self.state == self.DYING:
                if self.current_frame >= len(current_anim):
                    self.state = self.DEAD
                    self.alive = False
                    self.current_frame = len(current_anim) - 1
            else:
                self.current_frame %= len(current_anim)

        # AI behavior
        if self.state in [self.IDLE, self.WALKING]:
            self._ai_behavior(delta_time, player_pos)

    def _ai_behavior(self, delta_time, player_pos):
        """Simple AI: move towards player."""
        direction = player_pos - self.position
        distance = np.linalg.norm(direction[:2])  # Only XZ plane

        if distance > 1.5:  # Keep some distance
            direction = direction / np.linalg.norm(direction)
            self.position += direction * self.speed * delta_time
            self.state = self.WALKING
        else:
            self.state = self.IDLE

    def take_damage(self, damage):
        """
        Apply damage to enemy.

        Args:
            damage: Amount of damage
        """
        if not self.alive or self.state == self.DYING:
            return

        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.state = self.DYING
        else:
            self.state = self.HIT

        self.current_frame = 0
        self.animation_timer = 0.0

    def _get_current_animation(self):
        """Get current animation frames based on state."""
        if self.state == self.IDLE:
            return self.animations.get('idle', [])
        elif self.state == self.WALKING:
            return self.animations.get('walk', [])
        elif self.state == self.HIT:
            return self.animations.get('hit', [])
        elif self.state in [self.DYING, self.DEAD]:
            return self.animations.get('die', [])
        return []

    def get_current_texture(self):
        """Get current texture for rendering."""
        anim = self._get_current_animation()
        if anim and self.current_frame < len(anim):
            return anim[self.current_frame]
        return None

    def get_billboard_facing(self, camera_pos):
        """
        Calculate rotation to face camera (billboard).

        Args:
            camera_pos: Camera position

        Returns:
            float: Rotation angle in radians
        """
        direction = camera_pos - self.position
        angle = math.atan2(direction[0], direction[2])
        return angle


class EnemyManager:
    """Manages multiple enemies."""

    def __init__(self):
        """Initialize enemy manager."""
        self.enemies = []

    def add_enemy(self, enemy):
        """Add enemy to manager."""
        self.enemies.append(enemy)

    def update(self, delta_time, player_pos):
        """Update all enemies."""
        for enemy in self.enemies:
            enemy.update(delta_time, player_pos)

        # Remove dead enemies after a delay
        # For now, keep them for visual effect

    def get_alive_enemies(self):
        """Get list of alive enemies."""
        return [e for e in self.enemies if e.alive]

    def check_hit(self, ray_origin, ray_direction, max_distance=50.0):
        """
        Check if ray hits any enemy.

        Args:
            ray_origin: Ray starting point
            ray_direction: Ray direction (normalized)
            max_distance: Maximum hit distance

        Returns:
            Enemy or None
        """
        closest_enemy = None
        closest_distance = max_distance

        for enemy in self.get_alive_enemies():
            # Simple sphere collision
            to_enemy = enemy.position - ray_origin
            projection = np.dot(to_enemy, ray_direction)

            if projection > 0:
                closest_point = ray_origin + ray_direction * projection
                distance_to_enemy = np.linalg.norm(closest_point - enemy.position)

                if distance_to_enemy < 0.5 and projection < closest_distance:
                    closest_distance = projection
                    closest_enemy = enemy

        return closest_enemy
