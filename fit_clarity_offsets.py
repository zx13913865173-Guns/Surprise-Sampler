
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402
from src.analysis import student_model  # noqa: E402
from scripts.demo_scores_from_reference import (  # noqa: E402
    CLARITY_NOISE_SD, CLARITY_OFFSETS, PROMPT_SIGMA, add_prompt_effect, prompt_delta,
)

T_MIN = 3.4            # β 的 t 值下限（p < 0.001）
DDIM_GAP = 0.35        # DDIM 与 Euler a 的清晰度差距下限
DELTA_MAX = 0.45       # 偏移绝对值上限


def load_ratings() -> tuple[list[str], dict[str, np.ndarray]]:
    path = utils.repo_path("outputs", "ratings", "student_image_means.csv")
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}；请先运行 python scripts/demo_scores_from_reference.py")
    df = pd.read_csv(path)
    # 与 demo 脚本用同一采样器顺序，保证随机化流一致、数字可比
    samplers = list(utils.sampler_table().keys())
    sur = {s: df[df.sampler == s].sort_values("prompt_id")["surprise"].to_numpy()
           for s in samplers}
    return samplers, sur


def evaluate(samplers, sur, delta, within_scale, within_sd):
    """按 demo 脚本完全相同的随机化设定构造清晰度并拟合 M2，返回 (F, beta, t)。"""
    rng = np.random.default_rng([20261120, int(round(within_scale * 1e6))])
    rows = []
    for k, s in enumerate(samplers):
        dev = sur[s] - sur[s].mean()
        sd_s = dev.std(ddof=0)
        dev = dev / sd_s if sd_s > 0 else dev
        base = 3.5 + delta[k]
        clar = add_prompt_effect(
            np.clip(base + within_scale * within_sd * dev
                    + rng.normal(0, CLARITY_NOISE_SD, len(sur[s])), 1, 5),
            prompt_delta(len(sur[s]), PROMPT_SIGMA, rng), base, 1, 5)
        for i in range(len(sur[s])):
            rows.append({"sampler": s, "prompt_id": f"p{i + 1:02d}", "y": sur[s][i],
                         "clarity": float(clar[i])})
    frame = pd.DataFrame(rows)
    res = student_model.fit_model(frame, "y ~ C(sampler) + clarity")
    F = student_model.sampler_wald_f(res, samplers, df_denom=len(frame) - 9,
                                     data=frame)["F"]
    beta = float(res.params["clarity"])
    return F, beta, beta / float(res.bse["clarity"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()

    samplers, sur = load_ratings()
    within_sd = float(np.mean([(sur[s] - sur[s].mean()).std(ddof=0) for s in samplers]))
    i_dd, i_ea = samplers.index("ddim"), samplers.index("euler_a")
    zero = np.zeros(len(samplers))
    seed_delta = np.array([CLARITY_OFFSETS[s] for s in samplers])

    print("一、扫描清晰度组内强度（偏移取 0 与 demo 脚本的偏移两种）")
    best = None
    feasible_scales = []
    for within_scale in np.round(np.arange(0.06, 0.19, 0.02), 2):
        for tag, delta in (("δ=0", zero), ("δ=demo", seed_delta)):
            F, beta, t = evaluate(samplers, sur, delta, within_scale, within_sd)
            ok = beta > 0 and t >= T_MIN
            if ok:
                feasible_scales.append(within_scale)
                if best is None or F < best[0]:
                    best = (F, beta, t, delta.copy(), within_scale)
            print(f"  组内强度 {within_scale:.2f} {tag:>7s}：F = {F:6.3f}，"
                  f"β = {beta:+.3f}，t = {t:+5.1f}，β 显著为正 = {ok}")

    if not feasible_scales:
        print("\n在扫描范围内没有「β 为正且 p < 0.001」的设定；"
              "可放宽 T_MIN 或扩大扫描范围。")
        return 1

    base_scale = min(feasible_scales)
    print(f"\n二、在最小可行组内强度 {base_scale:.2f} 上扫描偏移缩放系数 k"
          f"（δ = k × demo 偏移；k = 1 即论文格局：DDIM 最高、Euler a 低 0.35 以上）")
    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        delta = k * seed_delta
        F, beta, t = evaluate(samplers, sur, delta, base_scale, within_sd)
        ok = beta > 0 and t >= T_MIN
        if ok and (best is None or F < best[0]):
            best = (F, beta, t, delta.copy(), base_scale)
        print(f"  k = {k:4.2f}：F = {F:6.3f}，β = {beta:+.3f}，t = {t:+5.1f}，"
              f"β 显著为正 = {ok}")

    F, beta, t, delta, found_scale = best
    table = pd.DataFrame({
        "sampler": samplers,
        "clarity_delta": np.round(delta, 3),
        "clarity_mean": np.round(3.5 + delta, 3),
    }).sort_values("clarity_mean", ascending=False)
    out = utils.ensure_dir(utils.repo_path("outputs", "tables")) / "clarity_offsets.csv"
    table.to_csv(out, index=False)
    print(f"\n可行解：组内强度 {found_scale:.2f}，F(7,183) = {F:.3f}"
          f"（论文 2.18，合成数据下界约 2.6，见 docs/known_inconsistencies.md）")
    print(f"清晰度系数 β = {beta:+.3f}（t = {t:.1f}，p < 0.001）")
    print(table.to_string(index=False))
    print(f"\n写出 {out}")
    print("将 clarity_delta 抄入 scripts/demo_scores_from_reference.py 的 "
          "CLARITY_OFFSETS 即可固定该构造。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
