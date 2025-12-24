from dataclasses import dataclass

@dataclass
class Config:
    # COLMAP
    colmap_bin: str = "colmap"
    use_gpu: bool = True
    matcher: str = "sequential"         # "sequential" usually safer for arc captures
    camera_model: str = "SIMPLE_RADIAL" # good default for phone images
    single_camera: bool = True

    # Dense/mesh
    dense_max_image_size: int = 2000
    poisson_depth: int = 10

    # QA filtering
    min_laplacian_var: float = 60.0     # blur threshold
    max_glare_fraction: float = 0.03    # drop frames with too much specular glare
    drop_worst_blur_fraction: float = 0.15
    drop_worst_glare_fraction: float = 0.15

    # Masking
    masks_mode: str = "none"            # "none" or "yolo"
    yolo_model: str = "yolov8n-seg.pt"  # downloaded automatically on first run
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5

    # Mask refinements
    glare_masking: bool = True
    glare_v_thresh: int = 245
    glare_s_thresh: int = 60
    glare_dilate_ksize: int = 9
    bird_mask_dilate_ksize: int = 5     # expand bird mask slightly
    bird_mask_dilate_iters: int = 1

    # Optional: reduce shelf-frame influence by ignoring borders
    border_ignore_px: int = 0           # set e.g. 20-60 if shelf frame sits near image edges

    # Robustness knobs (still “existing methods”)
    sift_max_features: int = 12000
    sift_matching_max_ratio: float = 0.75
    guided_matching: bool = True
