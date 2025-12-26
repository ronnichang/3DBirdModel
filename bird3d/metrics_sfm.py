import subprocess
from pathlib import Path

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
    Count image entries in COLMAP images.txt (TXT model).
    """
    if not images_txt.exists():
        return 0
    count = 0
    with images_txt.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                count += 1
    return count
