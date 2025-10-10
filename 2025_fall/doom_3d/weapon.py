"""Weapon system for shooting."""
import time


class Weapon:
    """Player weapon (gun) with shooting mechanics."""

    def __init__(self, textures, damage=25):
        """
        Initialize weapon.

        Args:
            textures: List of texture IDs [idle, shooting_frame1, shooting_frame2, ...]
            damage: Damage per shot
        """
        self.textures = textures
        self.damage = damage

        # Animation
        self.current_frame = 0
        self.is_shooting = False
        self.shoot_animation_time = 0.3  # Duration of shoot animation
        self.shoot_start_time = 0

        # Fire rate
        self.fire_rate = 0.5  # Seconds between shots
        self.last_shot_time = 0

    def shoot(self):
        """Trigger weapon shooting."""
        current_time = time.time()

        if current_time - self.last_shot_time >= self.fire_rate:
            self.is_shooting = True
            self.shoot_start_time = current_time
            self.last_shot_time = current_time
            self.current_frame = 1
            return True

        return False

    def update(self, delta_time):
        """Update weapon animation."""
        if self.is_shooting:
            elapsed = time.time() - self.shoot_start_time

            # Animate through shooting frames
            if elapsed < self.shoot_animation_time:
                # Cycle through shooting frames
                frame_index = int((elapsed / self.shoot_animation_time) * (len(self.textures) - 1))
                self.current_frame = min(frame_index + 1, len(self.textures) - 1)
            else:
                # Return to idle
                self.is_shooting = False
                self.current_frame = 0

    def get_current_texture(self):
        """Get current weapon texture."""
        return self.textures[self.current_frame]

    def can_shoot(self):
        """Check if weapon can shoot."""
        return time.time() - self.last_shot_time >= self.fire_rate
