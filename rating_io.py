"""评分表的生成与整理。

生成两类空白评分表（供线上实现或线下抄录）：
    outputs/ratings/student_sheet_template.csv   192 图 x 5 维度，4 位评分者
    outputs/ratings/expert_sheet_template.csv     90 图 x (惊喜度 + 3 项创造价值)，8 位评分者

并把原始长表评分聚合为图像级均值，输出分析所需的宽表。

用法：
    python -m src.rating.rating_io --make-templates
    python -m src.rating.rating_io --aggregate outputs/ratings/student_raw.csv --out outputs/ratings/student_image_means.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import utils

STUDENT_DIMS = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]
EXPERT_VALUE_ITEMS = ["originality", "artistic_value", "exhibitability"]


def _sampler_order() -> List[str]:
    return list(utils.sampler_table().keys())


def make_templates(out_dir: Path) -> Dict[str, Path]:
    """生成学生与专家的空白评分表模板。"""
    utils.ensure_dir(out_dir)

    # 学生表：8 采样器 x 24 图 = 192
    student_rows = []
    orig_samplers = ["euler_a", "dpmpp_2m_karras", "lms", "heun", "ddim",
                     "dpm2_a", "kdpm2_ancestral", "unipc"]
    for sampler in orig_samplers:
        for i in range(1, 25):
            row = {"image_id": f"seed42__{sampler}__p{i:02d}", "sampler": sampler, "prompt_id": f"p{i:02d}"}
            for dim in STUDENT_DIMS:
                for r in range(1, 5):
                    row[f"{dim}_r{r}"] = ""
            student_rows.append(row)
    student_path = out_dir / "student_sheet_template.csv"
    with open(student_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(student_rows[0].keys()))
        w.writeheader()
        w.writerows(student_rows)

    # 专家表：3 尺度 x 30 = 90
    expert_rows = []
    for scale in ("0.0", "0.5", "1.0"):
        for i in range(1, 31):
            row = {"image_id": f"expB__euler_a__scale{scale}__{i:02d}",
                   "noise_scale": scale, "prompt_id": f"b{i:02d}"}
            row["surprise"] = ""
            for item in EXPERT_VALUE_ITEMS:
                row[item] = ""
            expert_rows.append(row)
    expert_path = out_dir / "expert_sheet_template.csv"
    with open(expert_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(expert_rows[0].keys()))
        w.writeheader()
        w.writerows(expert_rows)

    print(f"写入 {student_path}（{len(student_rows)} 行）")
    print(f"写入 {expert_path}（{len(expert_rows)} 行）")
    return {"student": student_path, "expert": expert_path}


def aggregate(raw_csv: Path, out_csv: Path,
              subject_col: str = "image_id",
              value_cols: List[str] | None = None) -> pd.DataFrame:
    """把长表评分聚合为图像级均值。

    长表需含 image_id、rater_id 以及各维度列（学生）或 surprise/originality/... （专家）。
    专家创造价值 = 三项的均值，再跨评分者取均值。
    """
    df = pd.read_csv(raw_csv)
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in {subject_col, "rater_id", "sampler", "prompt_id", "noise_scale"}]

    if set(EXPERT_VALUE_ITEMS).issubset(df.columns):
        df["creative_value"] = df[EXPERT_VALUE_ITEMS].mean(axis=1)

    grouped = df.groupby(subject_col)[value_cols + (["creative_value"] if "creative_value" in df.columns else [])].mean()
    grouped = grouped.reset_index()
    utils.ensure_dir(out_csv.parent)
    grouped.to_csv(out_csv, index=False)
    print(f"写入 {out_csv}（{len(grouped)} 行，图像级均值）")
    return grouped


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评分表模板与聚合")
    parser.add_argument("--make-templates", action="store_true")
    parser.add_argument("--aggregate", default=None, help="原始长表 CSV 路径")
    parser.add_argument("--out", default=str(utils.repo_path("outputs", "ratings", "image_means.csv")))
    args = parser.parse_args(argv)

    if args.make_templates:
        make_templates(utils.output_dir("ratings"))
    if args.aggregate:
        aggregate(Path(args.aggregate), Path(args.out))
    if not args.make_templates and not args.aggregate:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
