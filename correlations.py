"""计算指标与人类判断的相关分析（论文 2.8 / 3.6 节）。

论文报告（N = 192 学生评分集，Pearson，Benjamini-Hochberg 校正）：
    惊喜度 ~ CLIP         r = 0.183，p_BH = 0.011
    启发性 ~ CLIP         r = 0.319，r2 = 10.2%，p_BH < 0.001
    惊喜度 ~ 启发性       r = 0.214，r2 = 4.6%，p_BH = 0.004

用法：
    python -m src.analysis.correlations --ratings outputs/ratings/student_image_means.csv \
        --clip outputs/scores/clip_original.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import utils


def pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    r, p = stats.pearsonr(x, y)
    n = len(x)
    # 用 t 近似给出解析 p（与 scipy 一致，便于核对）
    t = r * np.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.inf
    p_analytic = 2 * stats.t.sf(abs(t), df=n - 2)
    return {"r": float(r), "r2": float(r ** 2), "n": n, "t": float(t),
            "p": float(p), "p_analytic": float(p_analytic)}


def run(ratings_path: Path, clip_path: Path, out_dir: Path) -> None:
    ratings = pd.read_csv(ratings_path)
    if "image_id" in ratings.columns:
        ratings["key"] = ratings["image_id"].astype(str)
    if "sampler" not in ratings.columns:
        raise SystemExit("评分表需含 sampler 列")

    clip = pd.read_csv(clip_path) if clip_path.exists() else None
    rows = []

    if clip is not None:
        # 逐图 CLIP 需与评分一一对应；这里按 sampler+prompt 匹配
        clip_key = clip.copy()
        clip_key["pkey"] = clip_key["sampler"].astype(str) + "__" + clip_key["prompt_id"].astype(str)
        clip_mean = clip_key.groupby("pkey")["clip_score"].mean()

        def match(row):
            k = f"{row['sampler']}__{row.get('prompt_id', '')}"
            return clip_mean.get(k, np.nan)

        ratings["clip_score"] = ratings.apply(match, axis=1)

        for dim in ("surprise", "inspirational", "aesthetic"):
            if dim in ratings.columns:
                d = ratings[[dim, "clip_score"]].dropna()
                if len(d) > 2:
                    res = pearson(d[dim].to_numpy(), d["clip_score"].to_numpy())
                    rows.append({"relation": f"{dim}~CLIP", **res})

    if {"surprise", "inspirational"}.issubset(ratings.columns):
        d = ratings[["surprise", "inspirational"]].dropna()
        res = pearson(d["surprise"].to_numpy(), d["inspirational"].to_numpy())
        rows.append({"relation": "surprise~inspirational", **res})

    out = pd.DataFrame(rows)
    if len(out):
        # 论文的相关分析对三个检验做 BH 校正：surprise~CLIP、inspirational~CLIP、
        # surprise~inspirational。其余探索性关系（如 aesthetic~CLIP）不进入校正族，
        # 单列 p_uncorrected，避免改变报告口径。
        family = ["surprise~CLIP", "inspirational~CLIP", "surprise~inspirational"]
        out["p_uncorrected"] = out["p"]
        out["p_BH"] = np.nan
        mask = out["relation"].isin(family)
        if mask.sum() >= 2:
            out.loc[mask, "p_BH"] = multipletests(out.loc[mask, "p"], method="fdr_bh")[1]
        out.loc[~mask, "p_BH"] = out.loc[~mask, "p"]
    utils.ensure_dir(out_dir)
    out.to_csv(out_dir / "correlations.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"写入 {out_dir / 'correlations.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="相关分析（BH 校正）")
    parser.add_argument("--ratings", default=str(utils.repo_path("outputs", "ratings", "student_image_means.csv")))
    parser.add_argument("--clip", default=str(utils.repo_path("outputs", "scores", "clip_original.csv")))
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)
    run(Path(args.ratings), Path(args.clip), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
