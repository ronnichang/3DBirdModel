import shutil
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


def run_dense_fused_pointcloud(
    *,
    colmap_bin: str,
    images_dir: Path,
    sparse_model_dir: Path,
    dense_workspace_dir: Path,
    fused_out_ply: Path,
    max_image_size: int = 1200,
    geom_consistency: bool = False,
    patchmatch_cache_size_gb: int = 4,
    fusion_cache_size_gb: int = 4,
    clean: bool = False,
    resume: bool = True,
):
    """
    Produce a dense colored point cloud (fused.ply) from a sparse SfM model.

    Dense pipeline (COLMAP workspace_format=COLMAP):
      1) image_undistorter
      2) patch_match_stereo
      3) stereo_fusion
    """
    dense_workspace_dir = Path(dense_workspace_dir)
    fused_out_ply = Path(fused_out_ply)
    fused_out_ply.parent.mkdir(parents=True, exist_ok=True)

    # Final sentinel (fast skip)
    if resume and (not clean) and fused_out_ply.exists() and fused_out_ply.stat().st_size > 100_000:
        print(f"[SKIP] Dense fused point cloud exists: {fused_out_ply}")
        return fused_out_ply

    if clean:
        ensure_clean_dir(dense_workspace_dir)
    dense_workspace_dir.mkdir(parents=True, exist_ok=True)

    # 1) Undistort (creates COLMAP dense workspace: images/, sparse/, stereo/)
    undist_images_dir = dense_workspace_dir / "images"
    if (not resume) or clean or (not undist_images_dir.exists()) or (len(list(undist_images_dir.glob("*"))) == 0):
        cmd = [
            colmap_bin, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_model_dir),
            "--output_path", str(dense_workspace_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ]
        run(cmd)
    else:
        print("[SKIP] image_undistorter (dense workspace already exists)")

    # 2) Patch-match stereo (creates stereo/depth_maps/*.bin etc.)
    depth_maps_dir = dense_workspace_dir / "stereo" / "depth_maps"
    have_depth = depth_maps_dir.exists() and any(depth_maps_dir.glob("*.bin"))
    if (not resume) or clean or (not have_depth):
        cmd = [
            colmap_bin, "patch_match_stereo",
            "--workspace_path", str(dense_workspace_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", str(max_image_size),
            "--PatchMatchStereo.geom_consistency", "true" if geom_consistency else "false",
            "--PatchMatchStereo.cache_size", str(patchmatch_cache_size_gb),
        ]
        run(cmd)
    else:
        print("[SKIP] patch_match_stereo (depth_maps already exist)")

    # 3) Stereo fusion -> fused point cloud
    fused_in_workspace = dense_workspace_dir / "fused.ply"
    input_type = "geometric" if geom_consistency else "photometric"
    if (not resume) or clean or (not fused_in_workspace.exists()) or fused_in_workspace.stat().st_size < 100_000:
        cmd = [
            colmap_bin, "stereo_fusion",
            "--workspace_path", str(dense_workspace_dir),
            "--workspace_format", "COLMAP",
            "--input_type", input_type,
            "--output_path", str(fused_in_workspace),
            "--StereoFusion.max_image_size", str(max_image_size),
            "--StereoFusion.cache_size", str(fusion_cache_size_gb),
        ]
        run(cmd)
    else:
        print("[SKIP] stereo_fusion (fused.ply already exists in workspace)")

    if not fused_in_workspace.exists():
        raise RuntimeError("Dense fusion did not produce fused.ply")

    shutil.copy2(fused_in_workspace, fused_out_ply)
    print(f"[OK] Dense fused point cloud: {fused_out_ply}")
    return fused_out_ply
