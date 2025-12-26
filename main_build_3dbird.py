#!/usr/bin/env python3
import argparse
from pathlib import Path

from bird3d.config import Config
from bird3d.io_utils import list_bird_folders, list_images, ensure_clean_dir, copy_images
from bird3d.colmap_sfm import run_sfm_sparse
from bird3d.metrics_sfm import export_model_to_txt, count_registered_images


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Path to Proj-3Dbird")
    ap.add_argument("--data_dir", default="data", help="Relative path under project (default: data)")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--colmap_bin", default=None, help="Path to colmap executable if not on PATH")
    return ap.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project).resolve()
    data_dir = (project_dir / args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cfg = Config()
    
    cb = Path(cfg.colmap_bin)
    if not cb.is_absolute():
        cfg.colmap_bin = str((project_dir / cb).resolve())    
    
    cfg.matcher = args.matcher
    if args.colmap_bin:
        cfg.colmap_bin = args.colmap_bin

    work_root = project_dir / "_work"
    out_root = project_dir / "_outputs"
    work_root.mkdir(exist_ok=True)
    out_root.mkdir(exist_ok=True)

    bird_dirs = list_bird_folders(data_dir)
    if not bird_dirs:
        raise RuntimeError(f"No bird folders under: {data_dir}")

    for bird_dir in bird_dirs:
        bird = bird_dir.name
        print("\n==============================")
        print(f"Bird: {bird}")
        print("==============================")

        imgs = list_images(bird_dir)
        print(f"Found {len(imgs)} images")
        if len(imgs) < 15:
            print("Skipping (too few images).")
            continue

        # Work dirs
        bird_work = work_root / bird
        ensure_clean_dir(bird_work)

        clean_dir = bird_work / "images_clean"
        copy_images(imgs, clean_dir)

        # Run SfM (sparse)
        sparse_bin = run_sfm_sparse(
            images_dir=clean_dir,
            work_dir=bird_work,
            colmap_bin=cfg.colmap_bin,
            use_gpu=cfg.use_gpu,
            matcher=cfg.matcher,
            camera_model=cfg.camera_model,
            single_camera=cfg.single_camera,
        )

        # Export to TXT for easy metric parsing
        sparse_txt = out_root / bird / "sparse_txt"
        export_model_to_txt(cfg.colmap_bin, sparse_bin, sparse_txt)

        registered = count_registered_images(sparse_txt / "images.txt")
        print(f"[OK] Registered images: {registered} / {len(imgs)}")
        print(f"TXT model: {sparse_txt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
