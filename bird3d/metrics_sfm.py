import subprocess
from pathlib import Path
import re

_IMG_RE = re.compile(r".*\.(jpg|jpeg|png|tif|tiff|bmp|webp)$", re.IGNORECASE)


def export_model_to_txt(colmap_bin: str, sparse_bin_dir: Path, out_txt_dir: Path):
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "model_converter",
        "--input_path", str(sparse_bin_dir),
        "--output_path", str(out_txt_dir),
        "--output_type", "TXT",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError("model_converter failed")


def count_registered_images(images_txt: Path) -> int:
    """
    Count registered images from COLMAP sparse TXT model images.txt.

    images.txt format is pairs of lines per image:
      - header line ends with the image filename
      - next line contains 2D points (numbers), no filename
    So we count only lines where the last token is an image filename.
    """
    if not images_txt.exists():
        return 0
    
    n = 0
    with images_txt.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and _IMG_RE.match(parts[-1]):
                n += 1
    return n
