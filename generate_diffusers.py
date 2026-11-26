"""第 2 阶段：用 Diffusers 生成论文分析的全部图像。

论文 2.3 节：SDXL Base 1.0，Diffusers v0.25.0，CFG 7.5，30 步，1024x1024，
主评分集种子 42，负向提示词 "low quality, blurry, distorted, ugly, bad anatomy"。

采样器 -> 调度器映射见 configs/samplers.yaml。实验 B 的连续噪声尺度操纵
（0 / 0.25 / 0.5 / 0.75 / 1.0）实现为 EulerAncestralDiscreteScheduler.step 的
``s_noise`` 参数：尺度 0 即确定性 Euler 极限，尺度 1.0 即标准全量祖先噪声注入。

用法：
    python -m src.generation.generate_diffusers --experiment original experiment_a
    python -m src.generation.generate_diffusers --experiment experiment_b --limit 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src import utils


def build_scheduler(sampler_key: str):
    """按 configs/samplers.yaml 实例化对应调度器。"""
    import diffusers

    spec = utils.sampler_table()[sampler_key]
    cls = getattr(diffusers, spec["scheduler"])
    return cls(**spec.get("kwargs", {}))


def load_pipeline(model_id: str, sampler_key: str, device: str = "cuda"):
    """加载 SDXL 管线并替换调度器。"""
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.scheduler = build_scheduler(sampler_key)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_one(pipe, row: Dict[str, Any], device: str = "cuda"):
    """按一行清单生成一张图；返回 PIL.Image。"""
    import torch

    kwargs = dict(
        prompt=row["prompt"],
        negative_prompt=row.get("negative_prompt"),
        num_inference_steps=row["steps"],
        guidance_scale=row["cfg_scale"],
        width=row["resolution"][0],
        height=row["resolution"][1],
        generator=torch.Generator(device=device).manual_seed(row["seed"]),
    )
    # 实验 B：祖先噪声尺度 -> EulerAncestralDiscreteScheduler 的 s_noise
    if row.get("noise_scale") is not None:
        kwargs["s_noise"] = float(row["noise_scale"])
    out = pipe(**kwargs)
    return out.images[0]


def run(experiments: List[str], manifests_dir: Path, out_dir: Path,
        limit: int | None = None, device: str = "cuda") -> int:
    rows: List[Dict[str, Any]] = []
    for name in experiments:
        rows.extend(utils.read_jsonl(manifests_dir / f"{name}.jsonl"))
    if limit:
        rows = rows[:limit]

    print(f"待生成 {len(rows)} 张，设备 {device}")
    written = 0
    current_sampler = None
    pipe = None
    for row in rows:
        img_dir = utils.ensure_dir(out_dir / row["experiment"])
        img_path = img_dir / f"{row['image_id']}.png"
        if img_path.exists():
            continue
        if row["sampler"] != current_sampler:
            pipe = load_pipeline(row["model_id"], row["sampler"], device=device)
            current_sampler = row["sampler"]
        image = generate_one(pipe, row, device=device)
        image.save(img_path)
        written += 1
        if written % 50 == 0:
            print(f"  已生成 {written} 张 ...")
    print(f"完成：新生成 {written} 张，输出目录 {out_dir}")
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diffusers 图像生成")
    parser.add_argument("--experiment", nargs="+", default=["original", "experiment_a",
                                                            "experiment_b", "experiment_c"])
    parser.add_argument("--manifests", default=str(utils.repo_path("outputs", "manifests")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "images")))
    parser.add_argument("--limit", type=int, default=None, help="只生成前 N 张（冒烟测试）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    run(args.experiment, Path(args.manifests), Path(args.out),
        limit=args.limit, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
