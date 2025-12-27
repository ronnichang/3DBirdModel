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
    ap.add_argument("--clean", action="store_true", help="Delete _work/<bird> and recompute")
    ap.add_argument("--resume", action="store_true",
                help="Skip birds that already have outputs in _outputs/<bird>/sparse_txt/")
    return ap.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project).resolve()
    data_dir = (project_dir / args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cfg = Config()
    
    # Choose colmap path: CLI override wins, else config default
    colmap_path = args.colmap_bin if args.colmap_bin else cfg.colmap_bin
    cb = Path(colmap_path)
    if not cb.is_absolute():
        cb = (project_dir / cb).resolve()
    cfg.colmap_bin = str(cb)

    cfg.matcher = args.matcher

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
        if args.clean:
            ensure_clean_dir(bird_work)
        else:
            bird_work.mkdir(parents=True, exist_ok=True)

        sparse_txt = out_root / bird / "sparse_txt"
        images_txt = sparse_txt / "images.txt"
        model0 = bird_work / "sparse" / "0"
        if args.resume and (not args.clean) and images_txt.exists():
            registered = count_registered_images(images_txt)
            print(f"[SKIP] Existing result. [OK] Registered images: {registered} / {len(imgs)}")
            print(f"TXT model: {sparse_txt}")
            continue
        elif args.resume and (not args.clean) and model0.exists():
            # Re-export TXT from existing binary model (fast)
            export_model_to_txt(cfg.colmap_bin, model0, sparse_txt)
            registered = count_registered_images(images_txt)
            print(f"[SKIP] Re-exported TXT. [OK] Registered images: {registered} / {len(imgs)}")
            print(f"TXT model: {sparse_txt}")
            continue        

        clean_dir = bird_work / "images_clean"
        exts = {".jpg", ".jpeg", ".png"}
        existing_imgs = []
        if clean_dir.exists():
            existing_imgs = [p for p in clean_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in exts]
        if (not clean_dir.exists()) or (len(existing_imgs) != len(imgs)):
            copy_images(imgs, clean_dir)
        else:
            print("[SKIP] images_clean already prepared")

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
        export_model_to_txt(cfg.colmap_bin, sparse_bin, sparse_txt)

        registered = count_registered_images(sparse_txt / "images.txt")
        print(f"[OK] Registered images: {registered} / {len(imgs)}")
        print(f"TXT model: {sparse_txt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
