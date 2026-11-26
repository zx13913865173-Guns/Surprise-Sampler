"""采样器级指标聚合与对齐矩阵（论文 3.1 / 3.3 节）。

输入：outputs/scores/ 下的逐图指标 CSV（lpips、niqe、brisque、clip、dreamsim、lapvar）
      以及可选的 outputs/scores/fid_original.csv（集合级）。
输出：outputs/tables/sampler_metrics.csv        每采样器一行，各指标均值
      outputs/tables/alignment_matrix.csv       指标 x 学生评分维度 的定性对齐矩阵

用法：
    python -m src.analysis.sampler_metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src import utils

PER_IMAGE_METRICS = {
    "lpips": "lpips",
    "niqe_brisque": None,   # 特殊处理（同文件两列）
    "clip": "clip_score",
    "dreamsim": "dreamsim",
    "lapvar": "laplacian_variance",
}


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def aggregate_sampler_metrics(scores_dir: Path) -> pd.DataFrame:
    """把所有逐图指标按采样器聚合成一张表。"""
    frames: Dict[str, pd.Series] = {}

    p = _read_if_exists(scores_dir / "lpips_original.csv")
    if p is not None:
        frames["LPIPS_inter_seed"] = p.groupby("sampler")["lpips"].mean()
    p = _read_if_exists(scores_dir / "niqe_brisque_original.csv")
    if p is not None:
        frames["NIQE"] = p.groupby("sampler")["niqe"].mean()
        frames["BRISQUE"] = p.groupby("sampler")["brisque"].mean()
    p = _read_if_exists(scores_dir / "clip_original.csv")
    if p is not None:
        frames["CLIP_Score"] = p.groupby("sampler")["clip_score"].mean()
    p = _read_if_exists(scores_dir / "dreamsim_original.csv")
    if p is not None:
        frames["DreamSim_diversity"] = p.groupby("sampler")["dreamsim"].mean()
    p = _read_if_exists(scores_dir / "lapvar_original.csv")
    if p is not None:
        frames["Laplacian_var"] = p.groupby("sampler")["laplacian_variance"].mean()

    table = pd.DataFrame(frames)
    f = _read_if_exists(scores_dir / "fid_original.csv")
    if f is not None:
        table["FID"] = f.set_index("sampler")["fid"]

    # 按论文顺序排列采样器
    order = list(utils.sampler_table().keys())
    table = table.reindex([s for s in order if s in table.index])
    return table


def qualitative_direction(values: pd.Series) -> str:
    """把一列指标的采样器排序压缩为正/弱/反向三类，用于对齐矩阵着色。"""
    if values.isna().all():
        return "n/a"
    ranks = values.rank()
    spread = ranks.max() - ranks.min()
    if spread <= 1:
        return "weak"
    return "positive"


def alignment_matrix(table: pd.DataFrame, student_means: pd.DataFrame | None) -> pd.DataFrame:
    """构造 指标 x 学生维度 的定性对齐矩阵。

    student_means 需含 sampler 索引与 5 个学生维度列；缺失时仅输出指标列。
    """
    cols = []
    dims = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]
    if student_means is not None and set(dims).issubset(student_means.columns):
        for dim in dims:
            cols.append(student_means[dim])
    metric_cols = [c for c in table.columns]
    out = pd.DataFrame(index=metric_cols)
    for dim in (student_means.columns if student_means is not None else []):
        if dim not in dims:
            continue
        corr = {}
        for m in metric_cols:
            s = table[m].dropna()
            d = student_means[dim].reindex(s.index).dropna()
            common = s.index.intersection(d.index)
            if len(common) >= 3 and s.loc[common].std() > 0 and d.loc[common].std() > 0:
                corr[m] = float(s.loc[common].corr(d.loc[common]))
            else:
                corr[m] = float("nan")
        out[dim] = pd.Series(corr)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="采样器级指标聚合")
    parser.add_argument("--scores", default=str(utils.repo_path("outputs", "scores")))
    parser.add_argument("--student-means",
                        default=str(utils.repo_path("outputs", "ratings", "student_image_means.csv")))
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)

    scores_dir = Path(args.scores)
    out_dir = utils.ensure_dir(Path(args.out_dir))

    table = aggregate_sampler_metrics(scores_dir)
    table.to_csv(out_dir / "sampler_metrics.csv")
    print("采样器级指标表：")
    print(table.round(3).to_string())
    print(f"写入 {out_dir / 'sampler_metrics.csv'}")

    sm_path = Path(args.student_means)
    student_means = None
    if sm_path.exists():
        raw = pd.read_csv(sm_path)
        dims = [c for c in ("surprise", "inspirational", "usability", "aesthetic", "clarity")
                if c in raw.columns]
        # 图像级表里 sampler 会重复，先按采样器聚合成均值再对齐
        student_means = raw.groupby("sampler", as_index=True)[dims].mean()

    matrix = alignment_matrix(table, student_means)
    matrix.to_csv(out_dir / "alignment_matrix.csv")
    print(f"写入 {out_dir / 'alignment_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
