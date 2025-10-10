"""Input handling for keyboard and mouse."""
import pygame


class InputHandler:
    """Handles keyboard and mouse input for FPS controls."""

    def __init__(self, camera, screen_width, screen_height):
        """
        Initialize input handler.

        Args:
            camera: Camera instance to control
            screen_width: Window width
            screen_height: Window height
        """
        self.camera = camera
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Mouse settings
        self.mouse_captured = True
        self.first_mouse = True
        self.last_x = screen_width / 2
        self.last_y = screen_height / 2

        # Movement settings
        self.movement_speed = 5.0
        self.mouse_sensitivity = 0.1

        # Key states
        self.keys_pressed = set()

        # Shooting
        self.shoot_requested = False

        # Capture mouse
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def process_events(self, events):
        """
        Process pygame events.

        Args:
            events: List of pygame events
        """
        self.shoot_requested = False

        for event in events:
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)

                # ESC to quit
                if event.key == pygame.K_ESCAPE:
                    return False

            if event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)

            if event.type == pygame.MOUSEMOTION:
                self._process_mouse_movement(event.pos)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.shoot_requested = True

        return True

    def _process_mouse_movement(self, pos):
        """Process mouse movement for camera rotation."""
        x, y = pos

        if self.first_mouse:
            self.last_x = x
            self.last_y = y
            self.first_mouse = False

        xoffset = x - self.last_x
        yoffset = self.last_y - y  # Reversed

        self.last_x = x
        self.last_y = y

        self.camera.process_mouse_movement(xoffset, yoffset)

    def update(self, delta_time):
        """
        Update camera position based on keyboard input.

        Args:
            delta_time: Time since last frame
        """
        velocity = self.movement_speed * delta_time

        if pygame.K_w in self.keys_pressed:
            self.camera.process_keyboard('FORWARD', velocity)
        if pygame.K_s in self.keys_pressed:
            self.camera.process_keyboard('BACKWARD', velocity)
        if pygame.K_a in self.keys_pressed:
            self.camera.process_keyboard('LEFT', velocity)
        if pygame.K_d in self.keys_pressed:
            self.camera.process_keyboard('RIGHT', velocity)

    def is_shooting(self):
        """Check if player requested to shoot."""
        return self.shoot_requested

    def release_mouse(self):
        """Release mouse capture."""
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        self.mouse_captured = False

    def capture_mouse(self):
        """Capture mouse."""
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.mouse_captured = True
        self.first_mouse = True
