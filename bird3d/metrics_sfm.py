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


def registered_images_model_analyzer(colmap_bin: str, model_dir: Path) -> int:
    """
    Use COLMAP's model_analyzer to count registered images.
    Works on a model directory (BIN or TXT). If parsing fails, raise.
    """
    cmd = [
        colmap_bin, "model_analyzer",
        "--path", str(model_dir),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError("model_analyzer failed")

    # Typical line contains: "Registered images: <N>"
    m = re.search(r"Registered images\s*:\s*(\d+)", p.stdout)
    if not m:
        # If COLMAP output format changes, this will show you what to match.
        print(p.stdout)
        raise RuntimeError("Could not parse registered images from model_analyzer output")

    return int(m.group(1))


def registered_images_best_effort(
    colmap_bin: str,
    model_dir: Path,
    images_txt_fallback: Path = None,
) -> int:
    """
    Prefer COLMAP model_analyzer; fall back to parsing images.txt.
    """
    try:
        return registered_images_model_analyzer(colmap_bin, model_dir)
    except Exception:
        if images_txt_fallback is not None:
            return count_registered_images(images_txt_fallback)
        return 0


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
