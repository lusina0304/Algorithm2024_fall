# Simple Doom - OpenGL FPS Game

Python과 OpenGL Shader를 사용한 간단한 Doom 스타일 FPS 게임입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 텍스처 생성

처음 실행하기 전에 텍스처 파일을 생성해야 합니다:

```bash
python create_textures.py
```

또는 `textures/` 폴더에 직접 PNG 파일을 넣을 수 있습니다:
- `wall1.png`, `wall2.png`, `wall3.png`: 벽 텍스처
- `floor.png`: 바닥 텍스처
- `enemy1.png`, `enemy2.png`, `enemy3.png`, `enemy4.png`: 적 스프라이트 (4 프레임)
- `gun.png`, `gun_fire.png`: 총 스프라이트 (2 프레임)

## 실행

```bash
python game.py
```

## 조작법

- **W/A/S/D**: 전후좌우 이동
- **마우스**: 시점 회전
- **왼쪽 클릭**: 총 발사
- **ESC**: 종료

## 구조

### 파일 구성
- `game.py`: 메인 게임 로직
- `matrix_utils.py`: 선형대수 유틸리티 (행렬 변환)
- `create_textures.py`: 텍스처 생성 스크립트

### 게임 요소
- **3D 지형**: 2D 그리드 기반, OpenGL Shader로 렌더링
- **벽 타입**: MAP 배열에서 1-3은 다른 텍스처의 벽
- **적**: 빌보드 스프라이트, 4프레임 걷기 애니메이션
- **총**: 2D HUD 오버레이, 2프레임 발사 애니메이션
- **Shader**: Vertex/Fragment Shader로 텍스처 매핑

## 맵 수정

`game.py`의 `MAP` 변수를 수정하여 지형을 변경할 수 있습니다:
- 0: 빈 공간
- 1: wall1.png 텍스처 벽
- 2: wall2.png 텍스처 벽
- 3: wall3.png 텍스처 벽
