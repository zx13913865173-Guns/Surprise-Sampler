"""实验 C：提示词类型的计算异质性检查（论文 2.8 / 3.4 节）。

模型：CLIP ~ PromptType * Sampler + (1|Prompt)
论文报告：PromptType x Sampler 交互 F(4, 171) = 1.63, p = 0.170；
          PromptType 主效应 F(1, 171) = 0.47, p = 0.494。

用法：
    python -m src.analysis.experiment_c --clip outputs/scores/clip_experiment_c.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src import utils
from src.analysis.inference import factor_effect_f
from src.analysis.student_model import fit_model


def fit(clip_path: Path) -> dict:
    """拟合 CLIP ~ PromptType * Sampler + (1|Prompt)，并复现论文的分母自由度口径。

    分母自由度 = nobs - 固定效应参数个数 - (提示词层数 - 1)
               = 200 - 10 - 19 = 171
    对应论文 F(4, 171) / F(1, 171) 的报数方式（Satterthwaite 近似后取整）。
    """
    df = pd.read_csv(clip_path)
    df = df.dropna(subset=["clip_score", "prompt_type", "sampler", "prompt_id"])
    res = fit_model(df, "clip_score ~ C(prompt_type) * C(sampler)")

    df_denom = int(res.nobs) - int(res.model.exog.shape[1]) - (df["prompt_id"].nunique() - 1)

    factors = {"prompt_type": sorted(df["prompt_type"].unique()),
               "sampler": sorted(df["sampler"].unique())}
    out = {}
    for key, term in (("interaction", ("prompt_type", "sampler")),
                      ("prompt_type", ("prompt_type",)),
                      ("sampler", ("sampler",))):
        r = factor_effect_f(res, df, factors, term, df_denom=df_denom)
        out[key] = {"F": r["F"], "q": r["q"], "df_denom": r["df_denom"], "p": r["p"]}
    out["n_obs"] = int(res.nobs)
    out["df_denom"] = df_denom
    return out


def run(clip_path: Path, out_dir: Path) -> None:
    result = fit(clip_path)
    utils.ensure_dir(out_dir)
    rows = []
    for k in ("interaction", "prompt_type", "sampler"):
        if k in result:
            rows.append({"effect": k, **result[k]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_c.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"n_obs = {result['n_obs']}, df_denom = {result['df_denom']}")
    print(f"写入 {out_dir / 'experiment_c.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="实验 C 异质性检验")
    parser.add_argument("--clip", default=str(utils.repo_path("outputs", "scores", "clip_experiment_c.csv")))
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)
    run(Path(args.clip), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
