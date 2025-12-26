import subprocess
from pathlib import Path

from .io_utils import ensure_clean_dir

def run(cmd):
    print("\n>>", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"Command failed (code={p.returncode})")
    return p.stdout

def run_sfm_sparse(
    *,
    images_dir: Path,
    work_dir: Path,
    colmap_bin: str,
    use_gpu: bool,
    matcher: str,
    camera_model: str,
    single_camera: bool,
) -> Path:
    """
    Runs: feature_extractor -> matcher -> mapper
    Returns: sparse model directory (BIN), e.g. work_dir/sparse/0
    """
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    ensure_clean_dir(sparse_dir)

    # Feature extraction
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
    ]
    if not use_gpu:
        cmd += ["--FeatureExtraction.use_gpu", "0"]
    run(cmd)

    # Matching
    if matcher == "sequential":
        cmd = [colmap_bin, "sequential_matcher", "--database_path", str(db_path)]
    elif matcher == "exhaustive":
        cmd = [colmap_bin, "exhaustive_matcher", "--database_path", str(db_path)]
    else:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

    if not use_gpu:
        cmd += ["--FeatureMatching.use_gpu", "0"]
    run(cmd)

    # Mapping
    run([
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ])

    model0 = sparse_dir / "0"
    if not model0.exists():
        subs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not subs:
            raise RuntimeError("No sparse model produced. Likely insufficient matches/overlap.")
        model0 = subs[0]

    return model0
