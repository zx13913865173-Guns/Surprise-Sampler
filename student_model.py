
    python -m src.analysis.student_model --ratings outputs/ratings/student_image_means.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src import utils
from src.analysis.inference import contrast_f, factor_effect_f

DIMS = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]


def nakagawa_r2(result) -> Dict[str, float]:
    """由 MixedLM 结果计算边际 R2（固定效应）与条件 R2（固定+随机）。

    Nakagawa 口径：
        marginal_R2    = var(fixed) / (var(fixed) + var(random) + var(residual))
        conditional_R2 = (var(fixed) + var(random)) / (同上分母)
    其中 var(fixed) 取「仅固定效应预测值」的样本方差。
    """
    var_resid = float(result.scale)
    var_random = float(np.asarray(result.cov_re).ravel().sum()) if result.cov_re.size else 0.0
    # 仅固定效应：设计矩阵 x 固定效应系数
    X = np.asarray(result.model.exog, dtype=float)
    beta = np.asarray(result.fe_params, dtype=float)
    fe_pred = X @ beta
    var_fixed = float(np.var(fe_pred, ddof=1))
    denom = var_fixed + var_random + var_resid
    return {"marginal_R2": float(var_fixed / denom),
            "conditional_R2": float((var_fixed + var_random) / denom)}


def fit_model(df: pd.DataFrame, formula: str, groups: str = "prompt_id"):
    """拟合 (1|Prompt) 随机截距的线性混合模型。

    statsmodels 在「组数适中 + 平衡设计」下用 L-BFGS 求协方差参数时容易停在边界、
    末步 Hessian 奇异；实测 BFGS / Powell / Nelder-Mead 可正常收敛且解一致。
    因此按 lbfgs → bfgs → powell → nm 依次回退，取第一个成功的解。
    同一数据下不同优化器给出相同估计，故该回退不影响结果，只影响能否跑通。
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*random effects covariance matrix is singular.*")
    warnings.filterwarnings("ignore", message=".*Random effects covariance is singular.*")

    model = smf.mixedlm(formula, df, groups=df[groups])
    last_err: Exception | None = None
    for method in ("lbfgs", "bfgs", "powell", "nm"):
        try:
            res = model.fit(reml=False, method=method)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        if np.all(np.isfinite(np.asarray(res.params, dtype=float))):
            return res
    if last_err is not None:
        raise last_err
    raise RuntimeError("混合模型拟合失败：所有优化器均未收敛")


def sampler_wald_f(result, levels: List[str], df_denom: int | None = None,
                   data: pd.DataFrame | None = None) -> Dict[str, float]:
    """对采样器固定效应做 F 检验（效应均方 / 模型残差均方）。

    分母自由度按论文口径固定为 nobs - 固定效应参数个数（M1 = 184，M2 = 183），
    对应 lmerTest 的 Satterthwaite 近似结果。
    """
    if data is not None and "sampler" in data.columns:
        out = factor_effect_f(result, data, {"sampler": sorted(set(levels))},
                              ("sampler",), df_denom=df_denom)
        return {"F": out["F"], "df_num": out["q"], "df_denom": out["df_denom"], "p": out["p"]}
    terms = [n for n in result.params.index if n.startswith("C(sampler)[T.")]
    out = contrast_f(result, terms, df_denom=df_denom)
    return {"F": out["F"], "df_num": out["q"], "df_denom": out["df_denom"], "p": out["p"]}


def pairwise(df: pd.DataFrame, dim: str = "surprise") -> pd.DataFrame:
    """采样器两两比较（t 检验，未校正与 Bonferroni 校正）。"""
    from scipy import stats
    groups = {k: v[dim].to_numpy() for k, v in df.groupby("sampler")}
    pairs = list(itertools.combinations(sorted(groups), 2))
    rows = []
    for a, b in pairs:
        t, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
        rows.append({"pair": f"{a}_vs_{b}", "mean_a": groups[a].mean(),
                     "mean_b": groups[b].mean(), "t": t, "p_uncorrected": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonferroni"] = multipletests(out["p_uncorrected"], method="bonferroni")[1]
    return out


def run(ratings_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(ratings_path)
    # 列名规范化
    if "image_id" in df.columns:
        pass
    if "prompt_id" not in df.columns:
        df["prompt_id"] = df["image_id"].str.split("__").str[-1]

    utils.ensure_dir(out_dir)
    summary_rows = []

    for dim in DIMS:
        if dim not in df.columns:
            continue
        sub = df[["sampler", "prompt_id", dim]].dropna().rename(columns={dim: "y"})
        res = fit_model(sub, "y ~ C(sampler)")
        df_denom = int(res.nobs) - int(res.model.exog.shape[1])
        wald = sampler_wald_f(res, sorted(sub["sampler"].unique()), df_denom=df_denom,
                               data=sub.rename(columns={"y": "y"}))
        r2 = nakagawa_r2(res)
        summary_rows.append({"dimension": dim, "F": wald["F"], "df_num": wald["df_num"],
                             "df_denom": wald["df_denom"], "p": wald["p"], **r2})
        print(f"{dim:<14} F({wald['df_num']},{wald['df_denom']}) = {wald['F']:.2f}, "
              f"p = {wald['p']:.3f}, "
              f"marginal R2 = {r2['marginal_R2']:.2f}, conditional R2 = {r2['conditional_R2']:.2f}")

        if dim == "surprise":
            sub_m2 = (df[["sampler", "prompt_id", "clarity", "surprise"]].dropna()
                      .rename(columns={"surprise": "y"}))
            res2 = fit_model(sub_m2, "y ~ C(sampler) + clarity")
            df_denom2 = int(res2.nobs) - int(res2.model.exog.shape[1])
            wald2 = sampler_wald_f(res2, sorted(sub_m2["sampler"].unique()), df_denom=df_denom2,
                                data=sub_m2)
            r2_m2 = nakagawa_r2(res2)
            clarity_p = float(res2.pvalues.get("clarity", float("nan")))
            summary_rows.append({"dimension": "surprise_M2", "F": wald2["F"],
                                 "df_num": wald2["df_num"], "df_denom": wald2["df_denom"],
                                 "p": wald2["p"], "clarity_p": clarity_p, **r2_m2})
            print(f"surprise_M2   F({wald2['df_num']},{wald2['df_denom']}) = {wald2['F']:.2f}, "
                  f"p = {wald2['p']:.3f}, clarity p = {clarity_p:.4g}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "student_models.csv", index=False)
    print(f"写入 {out_dir / 'student_models.csv'}")

    pw = pairwise(df, dim="surprise")
    pw.to_csv(out_dir / "student_posthoc_surprise.csv", index=False)
    print(f"写入 {out_dir / 'student_posthoc_surprise.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="学生混合模型")
    parser.add_argument("--ratings", default=str(utils.repo_path("outputs", "ratings", "student_image_means.csv")))
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)
    run(Path(args.ratings), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
