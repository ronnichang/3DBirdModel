from dataclasses import dataclass

@dataclass
class Config:
    colmap_bin: str = r"..\TOOLS\colmap-x64-windows-nocuda\bin\colmap.exe" 
    use_gpu: bool = False
    matcher: str = "sequential"         # sequential fits arc captures
    camera_model: str = "SIMPLE_RADIAL" # good for phone
    single_camera: bool = True
