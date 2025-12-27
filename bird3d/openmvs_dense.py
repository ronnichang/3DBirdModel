import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd, cwd: Path | None = None):
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
        cmd = [
            str(interface),
            "-i", str(colmap_model_dir),
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
