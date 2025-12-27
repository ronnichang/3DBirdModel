from pathlib import Path
from bird3d.metrics_sfm import count_registered_images


def main():
    out_root = Path("_outputs")
    if not out_root.exists():
        raise FileNotFoundError("_outputs not found. Run from project root.")

    for bird_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        images_txt = bird_dir / "sparse_txt" / "images.txt"
        if not images_txt.exists():
            print(f"[SKIP] {bird_dir.name}: no sparse_txt/images.txt")
            continue
        n = count_registered_images(images_txt)
        print(f"{bird_dir.name}: Registered images = {n}")


if __name__ == "__main__":
    main()
