

    python -m src.evaluation.fid --images outputs/images/original --reference data/coco2017_val_1000 \
        --out outputs/scores/fid_original.csv
    python -m src.evaluation.fid --self-test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src import utils

FID_SAMPLERS = utils.load_experiment_config()["computational_metrics"]["fid"]["samplers"]


def compute(image_dir: Path, reference_dir: Path, out_csv: Path, device: str = "cuda"):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 pytorch-fid。请先 pip install pytorch-fid。\n"
            f"原始错误：{exc}"
        )

    rows: List[Dict[str, object]] = []
    for sampler in FID_SAMPLERS:
        gen_paths = sorted(image_dir.glob(f"*__{sampler}__*.png"))
        if not gen_paths:
            print(f"  跳过 {sampler}：未找到图像")
            continue
        # pytorch-fid 按目录计算，这里把该采样器的图软链到临时目录
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            for p in gen_paths:
                os.symlink(p.resolve(), Path(tmp) / p.name)
            fid = calculate_fid_given_paths(
                [tmp, str(reference_dir)], batch_size=50, device=device, dims=2048
            )
        rows.append({"sampler": sampler, "n_images": len(gen_paths), "fid": float(fid)})
        print(f"  {sampler}: FID = {fid:.1f} ({len(gen_paths)} 张)")

    utils.ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sampler", "n_images", "fid"])
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {out_csv}")


def self_test() -> bool:
    assert FID_SAMPLERS == ["euler_a", "dpmpp_2m_karras", "ddim"], \
        "FID 只对三个代表性采样器计算"
    print("fid self-test: PASS（仅 3 个采样器，集合级，描述性）")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="集合级 FID")
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images", "original")))
    parser.add_argument("--reference", default=str(utils.repo_path("data", "coco2017_val_1000")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "fid_original.csv")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    compute(Path(args.images), Path(args.reference), Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
