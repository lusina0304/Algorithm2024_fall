"""Create simple test assets for the game."""
from PIL import Image, ImageDraw
import os


def create_terrain_obj():
    """Create a simple terrain OBJ file."""
    os.makedirs('assets/models', exist_ok=True)

    with open('assets/models/terrain.obj', 'w') as f:
        # Create a simple floor with some variation
        size = 20
        segments = 10
        step = size * 2 / segments

        vertices = []
        uvs = []
        faces = []

        # Generate vertices
        for i in range(segments + 1):
            for j in range(segments + 1):
                x = -size + i * step
                z = -size + j * step
                y = 0
                vertices.append(f"v {x} {y} {z}\n")

                u = i / segments
                v = j / segments
                uvs.append(f"vt {u} {v}\n")

        f.writelines(vertices)
        f.writelines(uvs)
        f.write("vn 0 1 0\n")  # Normal pointing up

        # Generate faces
        for i in range(segments):
            for j in range(segments):
                # Two triangles per quad
                v1 = i * (segments + 1) + j + 1
                v2 = v1 + 1
                v3 = v1 + segments + 1
                v4 = v3 + 1

                faces.append(f"f {v1}/{v1}/1 {v2}/{v2}/1 {v3}/{v3}/1\n")
                faces.append(f"f {v2}/{v2}/1 {v4}/{v4}/1 {v3}/{v3}/1\n")

        f.writelines(faces)

    print("Created terrain.obj")


def create_terrain_texture():
    """Create a simple terrain texture."""
    os.makedirs('assets/textures', exist_ok=True)

    # Create checkerboard pattern
    size = 512
    tile_size = 64
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)

    for i in range(0, size, tile_size):
        for j in range(0, size, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                color = (80, 80, 80)
            else:
                color = (100, 100, 100)
            draw.rectangle([i, j, i + tile_size, j + tile_size], fill=color)

    img.save('assets/textures/terrain.png')
    print("Created terrain.png")


def create_gun_sprites():
    """Create simple gun sprites."""
    os.makedirs('assets/sprites', exist_ok=True)

    # Gun idle
    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 150, 156, 200], fill=(60, 60, 60, 255))  # Body
    draw.rectangle([110, 180, 146, 190], fill=(40, 40, 40, 255))  # Barrel
    img.save('assets/sprites/gun_idle.png')

    # Gun shoot 1
    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 155, 156, 205], fill=(60, 60, 60, 255))  # Recoil
    draw.rectangle([110, 185, 146, 195], fill=(40, 40, 40, 255))
    draw.ellipse([140, 175, 160, 195], fill=(255, 200, 0, 200))  # Muzzle flash
    img.save('assets/sprites/gun_shoot1.png')

    # Gun shoot 2
    img.save('assets/sprites/gun_shoot2.png')

    print("Created gun sprites")


def create_enemy_sprites():
    """Create simple enemy sprites."""
    os.makedirs('assets/sprites', exist_ok=True)

    # Enemy idle
    img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([44, 20, 84, 60], fill=(200, 100, 100, 255))  # Head
    draw.rectangle([50, 60, 78, 100], fill=(150, 50, 50, 255))  # Body
    img.save('assets/sprites/enemy_idle_1.png')

    # Enemy walk (slightly different pose)
    img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([44, 18, 84, 58], fill=(200, 100, 100, 255))
    draw.rectangle([50, 58, 78, 98], fill=(150, 50, 50, 255))
    img.save('assets/sprites/enemy_walk_1.png')
    img.save('assets/sprites/enemy_walk_2.png')

    # Enemy hit (flash white)
    img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([44, 20, 84, 60], fill=(255, 200, 200, 255))
    draw.rectangle([50, 60, 78, 100], fill=(255, 150, 150, 255))
    img.save('assets/sprites/enemy_hit_1.png')

    # Enemy die (falling)
    img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 70, 80, 110], fill=(150, 50, 50, 200))
    draw.rectangle([20, 75, 100, 95], fill=(100, 30, 30, 200))
    img.save('assets/sprites/enemy_die_1.png')
    img.save('assets/sprites/enemy_die_2.png')

    print("Created enemy sprites")


if __name__ == "__main__":
    print("Creating test assets...")
    create_terrain_obj()
    create_terrain_texture()
    create_gun_sprites()
    create_enemy_sprites()
    print("\nAll test assets created successfully!")
    print("You can now run: python main.py")
