from dataclasses import dataclass

@dataclass
class Config:
    width: int = 1280
    height: int = 720
    fov_deg: float = 70.0
    near: float = 0.1
    far: float = 100.0
    sensitivity: float = 0.15
    move_speed: float = 3.0
    cpu_mode: str = "CPU_MV"
    vsync: bool = True
    image_y_flip: bool = True
