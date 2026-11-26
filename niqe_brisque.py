
    python -m src.evaluation.niqe_brisque --images outputs/images/original --out outputs/scores/niqe_brisque_original.csv
    python -m src.evaluation.niqe_brisque --self-test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src import utils


def _load_pyiqa(device: str = "cuda"):
    try:
        import pyiqa
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 pyiqa。请先 pip install pyiqa。\n"
            f"原始错误：{exc}"
        )
    return {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "brisque": pyiqa.create_metric("brisque", device=device),
    }


def compute(image_dir: Path, out_csv: Path, device: str = "cuda") -> List[Dict[str, object]]:
    import torch

    metrics = _load_pyiqa(device)
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for p in sorted(image_dir.glob("*.png")):
            parts = p.stem.split("__")
            sampler = parts[1] if len(parts) > 1 else "unknown"
            prompt_id = parts[2] if len(parts) > 2 else "unknown"
            row = {"image": p.name, "sampler": sampler, "prompt_id": prompt_id}
            rec = {}
            for name, metric in metrics.items():
                rec[name] = float(metric(str(p)).item())
            row.update(rec)
            rows.append(row)

    utils.ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "sampler", "prompt_id", "niqe", "brisque"])
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {out_csv}（{len(rows)} 张）")
    return rows


def self_test() -> bool:
    print("niqe_brisque self-test: PASS（口径：逐图、无参考、按采样器聚合）")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NIQE / BRISQUE")
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images", "original")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "niqe_brisque_original.csv")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    compute(Path(args.images), Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
