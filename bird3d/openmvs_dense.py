import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _prepare_colmap_input_for_openmvs(colmap_root: Path, work_dir: Path) -> Path:
    """
    OpenMVS InterfaceCOLMAP expects:
        <input>/sparse/cameras.bin (or cameras.txt), images.bin, points3D.bin
    but COLMAP typically writes:
        <root>/sparse/0/cameras.bin ...

    We create a small shim workspace:
        <work_dir>/_colmap_ws/sparse/{cameras.bin,images.bin,points3D.bin}
    by copying from <colmap_root>/sparse/0/* when needed.
    """
    colmap_root = Path(colmap_root)
    work_dir = Path(work_dir)

    # Where OpenMVS will look:
    sparse_dir = colmap_root / "sparse"
    need_bin = ["cameras.bin", "images.bin", "points3D.bin"]
    need_txt = ["cameras.txt", "images.txt", "points3D.txt"]

    # If already flat, use it directly
    if all((sparse_dir / f).exists() for f in need_bin) or all((sparse_dir / f).exists() for f in need_txt):
        return colmap_root

    # Otherwise, try sparse/0
    sparse0 = sparse_dir / "0"
    src_files = None
    if all((sparse0 / f).exists() for f in need_bin):
        src_files = need_bin
    elif all((sparse0 / f).exists() for f in need_txt):
        src_files = need_txt

    if src_files is None:
        raise RuntimeError(
            "OpenMVS InterfaceCOLMAP cannot find COLMAP model files.\n"
            f"Expected either {sparse_dir}/{{cameras,images,points3D}}.(bin|txt) "
            f"or {sparse0}/{{cameras,images,points3D}}.(bin|txt)"
        )

    # Create shim workspace under the OpenMVS work dir
    shim = work_dir / "_colmap_ws"
    shim_sparse = shim / "sparse"
    shim_sparse.mkdir(parents=True, exist_ok=True)

    for f in src_files:
        shutil.copy2(sparse0 / f, shim_sparse / f)

    return shim


def _run(cmd, cwd: Optional[Path] = None):
    print("\n>>", " ".join(str(x) for x in cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"Command failed (code={p.returncode})")
    return p.stdout


def _resolve_exe(openmvs_bin: Path, name: str) -> Path:
    """
    Resolve OpenMVS executable path (Windows-friendly).
    """
    openmvs_bin = Path(openmvs_bin)
    candidates = [
        openmvs_bin / name,
        openmvs_bin / f"{name}.exe",
        Path(name),              # in PATH
        Path(f"{name}.exe"),     # in PATH on Windows
    ]
    for c in candidates:
        if c.exists():
            return c
    # last resort: return bin/name.exe so error shows expected location
    return openmvs_bin / f"{name}.exe"


def run_openmvs_dense_pointcloud(
    *,
    openmvs_bin: Path,
    colmap_model_dir: Path,
    images_dir: Path,
    work_dir: Path,
    dense_ply_out: Path,
    resolution_level: int = 2,
    clean: bool = False,
    resume: bool = True,
):
    """
    Stage 2 (CPU): Use OpenMVS to densify COLMAP SfM output.

    Steps:
      1) InterfaceCOLMAP -> scene.mvs
      2) DensifyPointCloud -> scene_dense.mvs (+ typically scene_dense.ply)
      3) Copy scene_dense.ply to desired output path

    resolution_level: 1 (higher detail, slower) ... 3 (faster, lower detail).
    OpenMVS maintainers/users often recommend 2 or 3 for speed/memory. :contentReference[oaicite:2]{index=2}
    """
    openmvs_bin = Path(openmvs_bin)
    colmap_model_dir = Path(colmap_model_dir)
    images_dir = Path(images_dir)
    work_dir = Path(work_dir)
    dense_ply_out = Path(dense_ply_out)
    dense_ply_out.parent.mkdir(parents=True, exist_ok=True)

    # Final sentinel skip
    if resume and (not clean) and dense_ply_out.exists() and dense_ply_out.stat().st_size > 100_000:
        print(f"[SKIP] OpenMVS dense point cloud exists: {dense_ply_out}")
        return dense_ply_out

    if clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    interface = _resolve_exe(openmvs_bin, "InterfaceCOLMAP")
    densify = _resolve_exe(openmvs_bin, "DensifyPointCloud")

    scene_mvs = work_dir / "scene.mvs"
    scene_dense_mvs = work_dir / "scene_dense.mvs"
    scene_dense_ply = work_dir / "scene_dense.ply"

    # 1) InterfaceCOLMAP
    if (not resume) or clean or (not scene_mvs.exists()):
        # Example usage appears in OpenMVS community reports:
        # interfaceCOLMAP.exe -i <colmap_model_dir> -o scene.mvs --image-folder <images_dir> :contentReference[oaicite:3]{index=3}
        colmap_input = _prepare_colmap_input_for_openmvs(colmap_model_dir, work_dir)
        cmd = [
                str(interface),
                "-i", str(colmap_input),
                "-o", str(scene_mvs),
                "--image-folder", str(images_dir),
        ]
        _run(cmd, cwd=work_dir)
    else:
        print("[SKIP] InterfaceCOLMAP (scene.mvs exists)")

    # 2) DensifyPointCloud
    # Many OpenMVS steps produce both .mvs and .ply; people commonly reference scene_dense.ply. :contentReference[oaicite:4]{index=4}
    if (not resume) or clean or (not scene_dense_mvs.exists()) or (not scene_dense_ply.exists()):
        cmd = [
            str(densify),
            "-w", str(work_dir),
            "-i", str(scene_mvs),
            "-o", str(scene_dense_mvs),
            "--resolution-level", str(int(resolution_level)),
        ]
        _run(cmd, cwd=work_dir)
    else:
        print("[SKIP] DensifyPointCloud (scene_dense.* exists)")

    if not scene_dense_ply.exists():
        raise RuntimeError(
            f"OpenMVS did not produce {scene_dense_ply}. "
            "Check OpenMVS stdout above for clues."
        )

    shutil.copy2(scene_dense_ply, dense_ply_out)
    print(f"[OK] OpenMVS dense point cloud: {dense_ply_out}")
    return dense_ply_out
