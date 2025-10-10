"""
간단한 텍스처 이미지 생성 스크립트
실제 게임에서는 이 스크립트 대신 직접 PNG 파일을 만들어 사용할 수 있습니다.
"""
from PIL import Image, ImageDraw
import os

def create_brick_wall(color, filename):
    """벽돌 벽 텍스처 생성"""
    size = 64
    img = Image.new('RGBA', (size, size), color)
    draw = ImageDraw.Draw(img)

    # 어두운 색 (벽돌 선)
    dark_color = tuple(int(c * 0.3) for c in color[:3]) + (255,)

    # 수평선
    for i in range(0, size, 16):
        draw.rectangle([0, i, size, i+2], fill=dark_color)

    # 수직선
    for i in range(0, size, 32):
        draw.rectangle([i, 0, i+2, size], fill=dark_color)
        draw.rectangle([i+16, 0, i+18, size], fill=dark_color)

    img.save(filename)
    print(f"Created {filename}")

def create_floor(filename):
    """바닥 체크무늬 텍스처 생성"""
    size = 64
    img = Image.new('RGBA', (size, size), (100, 100, 100, 255))
    draw = ImageDraw.Draw(img)

    for i in range(0, size, 16):
        for j in range(0, size, 16):
            if (i // 16 + j // 16) % 2 == 0:
                draw.rectangle([i, j, i+16, j+16], fill=(100, 100, 100, 255))
            else:
                draw.rectangle([i, j, i+16, j+16], fill=(80, 80, 80, 255))

    img.save(filename)
    print(f"Created {filename}")

def create_enemy(color, filename):
    """적 스프라이트 텍스처 생성"""
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 원형 스프라이트
    center = size // 2
    draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0, 255))

    img.save(filename)
    print(f"Created {filename}")

def create_gun(color, filename):
    """총 텍스처 생성"""
    width, height = 64, 128
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 총신
    draw.rectangle([20, 30, 44, height], fill=color)
    # 총몸
    draw.rectangle([10, 80, 54, 100], fill=(50, 50, 50, 255))

    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    os.makedirs("textures", exist_ok=True)

    # 벽 텍스처
    create_brick_wall((200, 100, 100, 255), "textures/wall1.png")
    create_brick_wall((100, 200, 100, 255), "textures/wall2.png")
    create_brick_wall((100, 100, 200, 255), "textures/wall3.png")

    # 바닥 텍스처
    create_floor("textures/floor.png")

    # 적 스프라이트 (4 프레임)
    create_enemy((255, 0, 0, 255), "textures/enemy1.png")
    create_enemy((255, 50, 50, 255), "textures/enemy2.png")
    create_enemy((200, 0, 0, 255), "textures/enemy3.png")
    create_enemy((255, 100, 100, 255), "textures/enemy4.png")

    # 총 텍스처 (2 프레임)
    create_gun((80, 80, 80, 255), "textures/gun.png")
    create_gun((255, 255, 0, 255), "textures/gun_fire.png")

    print("\n모든 텍스처가 생성되었습니다!")
    print("원하는 경우 textures/ 폴더의 PNG 파일을 직접 편집할 수 있습니다.")
