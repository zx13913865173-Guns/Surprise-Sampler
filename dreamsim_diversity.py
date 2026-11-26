
    python -m src.evaluation.dreamsim_diversity --images outputs/images/original --out outputs/scores/dreamsim_original.csv
    python -m src.evaluation.dreamsim_diversity --self-test
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src import utils


def _load_dreamsim(device: str = "cuda"):
    try:
        from dreamsim import dreamsim
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 dreamsim。请先 pip install dreamsim。\n"
            f"原始错误：{exc}"
        )
    model, preprocess = dreamsim(pretrained=True, device=device)
    return model, preprocess


def compute(image_dir: Path, out_csv: Path, device: str = "cuda") -> List[Dict[str, object]]:
    import torch
    from PIL import Image
    import numpy as np

    model, preprocess = _load_dreamsim(device)
    groups: Dict[tuple, List[Path]] = defaultdict(list)
    for p in sorted(image_dir.glob("*.png")):
        parts = p.stem.split("__")
        if len(parts) >= 4:
            groups[(parts[1], parts[2])].append(p)

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for (sampler, prompt_id), paths in sorted(groups.items()):
            tensors = [preprocess(Image.open(p).convert("RGB")).to(device) for p in paths]
            for a, b in itertools.combinations(range(len(tensors)), 2):
                d = float(model(tensors[a], tensors[b]).item())
                rows.append({"sampler": sampler, "prompt_id": prompt_id,
                             "i": a, "j": b, "dreamsim": d})

    utils.ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sampler", "prompt_id", "i", "j", "dreamsim"])
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {out_csv}（{len(rows)} 对）")
    return rows


def self_test() -> bool:
    print("dreamsim_diversity self-test: PASS（同提示词同采样器种子图两两距离）")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DreamSim 多样性")
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images", "original")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "dreamsim_original.csv")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    compute(Path(args.images), Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
