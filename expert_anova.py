
    python -m src.analysis.expert_anova --ratings outputs/ratings/expert_image_means.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src import utils

CONSTRUCTS = ["surprise", "creative_value"]
SCALES = [0.0, 0.5, 1.0]


def omega_or_eta_2(groups: List[np.ndarray]) -> float:
    """单因素 eta^2 = SS_between / SS_total。"""
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_vals - grand) ** 2).sum()
    return float(ss_between / ss_total)


def one_way(groups: List[np.ndarray]) -> Dict[str, float]:
    F, p = stats.f_oneway(*groups)
    return {"F": float(F), "p": float(p), "eta2": omega_or_eta_2(groups),
            "df1": len(groups) - 1, "df2": int(sum(len(g) for g in groups) - len(groups))}


def tukey(df: pd.DataFrame, value_col: str, group_col: str = "scale") -> pd.DataFrame:
    res = pairwise_tukeyhsd(df[value_col], df[group_col])
    rows = []
    for row in res.summary().data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        rows.append({"group1": g1, "group2": g2, "meandiff": float(meandiff),
                     "p_adj": float(p_adj), "reject": bool(reject)})
    return pd.DataFrame(rows)


def run(ratings_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(ratings_path)
    df = df.rename(columns={"noise_scale": "scale"})
    df["scale"] = df["scale"].astype(float)

    utils.ensure_dir(out_dir)
    summary = []
    for construct in CONSTRUCTS:
        if construct not in df.columns:
            continue
        groups = [df.loc[df["scale"] == s, construct].dropna().to_numpy() for s in SCALES]
        res = one_way(groups)
        summary.append({"construct": construct, **res})
        print(f"{construct:<14} means = "
              f"{[round(g.mean(), 2) for g in groups]}  "
              f"F({res['df1']},{res['df2']}) = {res['F']:.2f}, p = {res['p']:.3f}, "
              f"eta2 = {res['eta2']:.3f}")

        tk = tukey(df, construct)
        tk.to_csv(out_dir / f"expert_tukey_{construct}.csv", index=False)
        print(tk.to_string(index=False))

    pd.DataFrame(summary).to_csv(out_dir / "expert_anova.csv", index=False)
    print(f"写入 {out_dir / 'expert_anova.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="专家方差分析与 Tukey HSD")
    parser.add_argument("--ratings", default=str(utils.repo_path("outputs", "ratings", "expert_image_means.csv")))
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)
    run(Path(args.ratings), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
