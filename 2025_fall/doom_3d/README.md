# Doom 3D - Python FPS Game

Python 기반 Doom 스타일 3D FPS 게임입니다.

## 특징

- **3D 지형 렌더링**: OpenGL을 사용한 텍스처 매핑 3D 지형 (OBJ 파일 로드)
- **2D 스프라이트**: 적, 총, HUD를 위한 2D 스프라이트 렌더링
- **MVP 매트릭스**: Python에서 계산된 Model-View-Projection 매트릭스를 셰이더로 전달
- **FPS 컨트롤**:
  - WASD로 이동
  - 마우스로 시점 변경
  - 마우스 클릭으로 사격
- **적 AI**:
  - 플레이어를 추적하는 AI
  - 애니메이션: 걷기, 맞기, 죽기
- **HUD**: 체력, 탄약, 킬 수, 조준점 표시

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
doom_3d/
├── main.py                 # 메인 게임 루프
├── camera.py              # 카메라 및 MVP 매트릭스 계산
├── terrain_renderer.py    # 3D 지형 렌더러
├── sprite_renderer.py     # 2D 스프라이트 렌더러
├── obj_loader.py          # OBJ 파일 로더
├── texture_loader.py      # 텍스처 로더
├── shader_loader.py       # 셰이더 로더
├── input_handler.py       # 입력 처리
├── enemy.py              # 적 AI 및 매니저
├── weapon.py             # 무기 시스템
├── hud.py                # HUD 시스템
├── shaders/
│   ├── terrain.vert      # 지형 버텍스 셰이더
│   ├── terrain.frag      # 지형 프래그먼트 셰이더
│   ├── sprite.vert       # 스프라이트 버텍스 셰이더
│   └── sprite.frag       # 스프라이트 프래그먼트 셰이더
└── assets/
    ├── models/           # 3D 모델 (.obj 파일)
    │   └── terrain.obj
    ├── textures/         # 텍스처 이미지
    │   └── terrain.png
    └── sprites/          # 스프라이트 이미지
        ├── gun_idle.png
        ├── gun_shoot1.png
        ├── gun_shoot2.png
        ├── enemy_idle_1.png
        ├── enemy_walk_1.png
        ├── enemy_hit_1.png
        └── enemy_die_1.png
```

## 실행

```bash
python main.py
```

## 조작법

- **W/A/S/D**: 이동
- **마우스**: 시점 변경
- **마우스 좌클릭**: 사격
- **ESC**: 종료

## 에셋 준비

게임이 완전히 작동하려면 다음 에셋이 필요합니다:

1. **3D 지형**: `assets/models/terrain.obj` (또는 기본 평면 사용)
2. **지형 텍스처**: `assets/textures/terrain.png`
3. **총 스프라이트**: `assets/sprites/gun_*.png`
4. **적 스프라이트**: `assets/sprites/enemy_*.png`

에셋이 없으면 게임이 플레이스홀더를 사용합니다.

## 기술 스택

- **Python 3.x**
- **PyGame**: 윈도우 및 입력 처리
- **PyOpenGL**: 3D 렌더링
- **NumPy**: 행렬 계산
- **Pillow**: 이미지 로딩

## 렌더링 파이프라인

1. **Python**에서 MVP 매트릭스 계산 (camera.py)
2. **OpenGL Shader**로 매트릭스 전달
3. **Vertex Shader**에서 정점 변환
4. **Fragment Shader**에서 조명 및 텍스처 적용
5. 화면에 렌더링

## 라이센스

Educational/Learning purposes
