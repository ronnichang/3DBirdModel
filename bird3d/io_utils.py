import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def list_bird_folders(data_dir: Path):
    return [p for p in sorted(data_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".")]

def list_images(images_dir: Path):
    return [p for p in sorted(images_dir.iterdir())
            if p.is_file() and p.suffix in IMAGE_EXTS]

def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_images(image_paths, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        shutil.copy2(p, dst_dir / p.name)
