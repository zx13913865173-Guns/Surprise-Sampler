
    python -m src.evaluation.laplacian_variance --images outputs/images/original --out outputs/scores/lapvar_original.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src import utils


def _lapvar(path: Path) -> float:
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    gray = np.asarray(Image.open(path).convert("L"), dtype="float64")
    return float(ndimage.laplace(gray).var())


def compute(image_dir: Path, out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in sorted(image_dir.glob("*.png")):
        parts = p.stem.split("__")
        rows.append({
            "image": p.name,
            "sampler": parts[1] if len(parts) > 1 else "unknown",
            "prompt_id": parts[2] if len(parts) > 2 else "unknown",
            "laplacian_variance": _lapvar(p),
        })
    utils.ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "sampler", "prompt_id", "laplacian_variance"])
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {out_csv}（{len(rows)} 张）")
    return rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉普拉斯方差（描述性）")
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images", "original")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "lapvar_original.csv")))
    args = parser.parse_args(argv)
    compute(Path(args.images), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
