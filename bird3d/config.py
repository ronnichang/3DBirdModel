from dataclasses import dataclass


@dataclass
class Config:
    colmap_bin: str = r"..\TOOLS\colmap-x64-windows-nocuda\bin\colmap.exe" 
    use_gpu: bool = False
    matcher: str = "sequential"         # sequential fits arc captures
    camera_model: str = "SIMPLE_RADIAL" # good for phone
    single_camera: bool = True

    dense_max_image_size: int = 1200
    dense_geom_consistency: bool = False
    dense_patchmatch_cache_size_gb: int = 4
    dense_fusion_cache_size_gb: int = 4
    dense_backend: str = "openmvs"  # "colmap" or "openmvs"
    openmvs_bin: str = r"..\TOOLS\openMVS\bin"
    openmvs_resolution_level: int = 2