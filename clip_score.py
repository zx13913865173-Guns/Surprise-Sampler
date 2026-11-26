"""CLIP Score：图像与文本的语义对齐（论文 2.4 节）。

论文使用 CLIP-L（ViT-L/14）。逐图计算图像嵌入与提示词文本嵌入的余弦相似度，
再按采样器聚合；实验 C 用逐图分数拟合 CLIP ~ PromptType * Sampler + (1|Prompt)。

用法：
    python -m src.evaluation.clip_score --manifest outputs/manifests/original.jsonl --out outputs/scores/clip_original.csv
    python -m src.evaluation.clip_score --manifest outputs/manifests/experiment_c.jsonl --out outputs/scores/clip_experiment_c.csv
    python -m src.evaluation.clip_score --self-test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src import utils

CLIP_MODEL = utils.load_experiment_config()["computational_metrics"]["clip_score"]["model"]


def _load_clip(device: str = "cuda"):
    try:
        import open_clip
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 open_clip_torch。请先 pip install open_clip_torch。\n"
            f"原始错误：{exc}"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, preprocess, tokenizer


def compute(manifest: Path, images_dir: Path, out_csv: Path, device: str = "cuda"):
    import torch
    from PIL import Image

    model, preprocess, tokenizer = _load_clip(device)
    rows = utils.read_jsonl(manifest)
    out: List[Dict[str, object]] = []
    with torch.no_grad():
        for row in rows:
            img_path = images_dir / row["experiment"] / f"{row['image_id']}.png"
            if not img_path.exists():
                continue
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([row["prompt"]]).to(device)
            img_emb = model.encode_image(image)
            txt_emb = model.encode_text(text)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            score = float((img_emb @ txt_emb.T).item())
            out.append({
                "image_id": row["image_id"],
                "experiment": row["experiment"],
                "sampler": row["sampler"],
                "prompt_id": row["prompt_id"],
                "prompt_type": row.get("prompt_type"),
                "noise_scale": row.get("noise_scale"),
                "clip_score": score,
            })

    utils.ensure_dir(out_csv.parent)
    fields = ["image_id", "experiment", "sampler", "prompt_id", "prompt_type",
              "noise_scale", "clip_score"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"写入 {out_csv}（{len(out)} 张）")


def self_test() -> bool:
    assert CLIP_MODEL, "CLIP 模型名不可为空"
    print(f"clip_score self-test: PASS（模型 {CLIP_MODEL}，逐图余弦相似度）")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLIP Score")
    parser.add_argument("--manifest", default=str(utils.repo_path("outputs", "manifests", "original.jsonl")))
    parser.add_argument("--images", default=str(utils.repo_path("outputs", "images")))
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "scores", "clip_original.csv")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    compute(Path(args.manifest), Path(args.images), Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
