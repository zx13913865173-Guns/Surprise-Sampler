


    python -m src.evaluation.lpips_distance --images outputs/images/original --out outputs/scores/lpips_original.csv
    python -m src.evaluation.lpips_distance --self-test
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src import utils

BACKBONE = utils.load_experiment_config()["computational_metrics"]["lpips"].get("backbone", "alex")


def _group_by_prompt(image_dir: Path) -> Dict[Tuple[str, str], List[Path]]:
    """把文件名解析为 (experiment, sampler, prompt_id) -> [种子图路径]。"""
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for p in sorted(image_dir.glob("*.png")):
        stem = p.stem
        parts = stem.split("__")
        if len(parts) < 4:
            continue
        exp, sampler, prompt_id = parts[0], parts[1], parts[2]
        groups[(sampler, prompt_id)].append(p)
    return groups


def _load_lpips(device: str = "cuda"):
    try:
        import lpips  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 lpips 包。请先 pip install lpips torchvision，或在有 GPU 的环境运行。\n"
            f"原始错误：{exc}"
        )
    import lpips
    return lpips.LPIPS(net=BACKBONE).to(device).eval()


def compute(image_dir: Path, out_csv: Path, device: str = "cuda") -> List[Dict[str, object]]:
    import torch
    from PIL import Image
    import numpy as np

    model = _load_lpips(device)

    def to_tensor(path: Path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    groups = _group_by_prompt(image_dir)
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for (sampler, prompt_id), paths in sorted(groups.items()):
            pairs = list(itertools.combinations(sorted(paths), 2))
            for a, b in pairs:
                d = float(model(to_tensor(a), to_tensor(b)).item())
                rows.append({"sampler": sampler, "prompt_id": prompt_id,
                             "img_a": a.name, "img_b": b.name, "lpips": d})

    utils.ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sampler", "prompt_id", "img_a", "img_b", "lpips"])
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {out_csv}（{len(rows)} 对）")
    return rows


def self_test() -> bool:
    """无 GPU 自检：核对每采样器的配对数口径。"""
    lp = utils.load_experiment_config()["computational_metrics"]["lpips"]["pairs_per_prompt"]
    assert lp["5_sampler_set"] == 3, "一个提示词 3 个种子应产生 3 对"
    assert lp["3_sampler_set"] == 1, "一个提示词 2 个种子应产生 1 对"
    print("lpips_distance self-test: PASS")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LPIPS 种子间距离")
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images", "original")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "lpips_original.csv")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    compute(Path(args.images), Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
