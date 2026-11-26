"""用论文报告值合成演示数据，使第 3、4 阶段在没有 GPU、没有原始数据时也能跑通。

设计原则：以「让复现脚本算出论文报告的统计量」为目标反解生成参数；
凡是无法同时满足的约束，如实打印达成值与目标值的差异，不做掩饰。

精确复现（二分／网格反解生成参数，使复现脚本算出的值等于报告值）
    - 组均值：学生惊喜度 8 档、专家惊喜度与创造价值 3 档，严格等于报告值
    - 专家一元方差分析：惊喜度 F(2,87) = 4.12、创造价值 F(2,87) = 4.18
    - 学生 M1：Surprise ~ Sampler + (1|Prompt)，F(7,184) = 2.41，df2 = 192 - 8
    - 学生 M2：加入 clarity 协变量，F(7,183) = 2.18
    - 三类相关：惊喜度~CLIP = 0.183、启发性~CLIP = 0.319、惊喜度~启发性 = 0.214
    - 信度：学生 ICC(2,k) ≈ 0.86、专家 0.88 / 0.90

近似对齐
    - 实验 C：论文未检出类型效应与交互效应，目标 F 属零效应下的抽样结果，
      故搜索随机种子并微调效应幅度，使 F 落在目标值附近（不追求精确相等）
    - 逐评分者的整数打分表：由图像级均值加噪声再四舍五入，噪声按目标 ICC 二分标定

仅为示意（论文未报告，按描述的方向填入占位值）
    - LMS / Heun 的学生惊喜度、启发性 / 审美 / 清晰度的逐采样器均值
    - 逐采样器 LPIPS / NIQE / BRISQUE / DreamSim / Laplacian 方差

已知无法同时满足的约束（详见 docs/known_inconsistencies.md）
    - M2：论文既报采样器效应 F(7,183) = 2.18（比 M1 的 2.41 更小），又报 clarity 协变量
      为正且 p < 0.001，还报 DDIM 清晰度最高而 Euler a 更低。三者互斥：正系数的协变量会
      吸走组内残差方差、把 F 抬高，要压低 F 就得让清晰度组均值与采样器效应同向，与论文
      格局相反。本脚本在保持论文清晰度格局、且 clarity 系数为正显著的前提下取 F 最小
      （约 3.0，论文 2.18）；把清晰度组均值取齐可低到约 2.3，但那与 Fig.2 矛盾。
    - M1 条件 R2：论文报边际 0.09 / 条件 0.16，即提示词层随机截距占约 7% 的总方差。但
      F(7,184) = 2.41 与 8 个组均值把图像级残差方差锁在约 2.5（组内 SD ≈ 1.6），而每个
      提示词只有 8 张图，提示词层方差低于约 0.56 时在最大似然下会被压到 0（不可识别）；
      把提示词层标准差提到 0.55 时条件 R2 又会升到 0.22。论文报的边际 0.09 与其自身的 F
      和组均值也不能同时成立（0.22/(0.22+0+2.5) ≈ 0.08 < 0.09）。本脚本取 F 精确、
      条件 R2 落到 0.08 的设定。
    - 由 F(7,184) = 2.41 与 8 个组均值反解，学生惊喜度的组内标准差需约 1.6。
      在 1-5 量表上要拿到这个方差，各采样器的图像级均值必然向两端聚集，
      故合成数据分布偏极化、边际 R² 只能到 0.08 左右（论文报 0.09 / 0.16）。
    - 生成有界高方差数据时必须事打乱取值顺序：若保留分块顺序，各采样器的
      取值列会近似有序，人为制造出约 0.7 的跨采样器相关。

用法：
    python scripts/demo_scores_from_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402
from src.analysis import student_model  # noqa: E402
from src.analysis.inference import factor_effect_f  # noqa: E402
from src.rating import reliability  # noqa: E402

RNG = np.random.default_rng(42)

STUDENT_DIMS = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]

# 提示词随机截距的标准差：必须明显大于「残差标准差 / sqrt(每个提示词的图像数)」，
# 否则提示词层方差在最大似然下被压到 0、条件 R2 会等于边际 R2（论文报 0.09 / 0.16）。
PROMPT_SIGMA = 0.45
CLARITY_NOISE_SD = 0.35      # 清晰度的组内评分噪声（用于反解组均值偏移）

# 论文报告的学生惊喜度均值；LMS / Heun 未逐项报告，取插值（assumed）
SURPRISE_MEANS = {
    "euler_a": 3.9, "dpmpp_2m_karras": 3.4, "ddim": 2.3,
    "dpm2_a": 3.7, "kdpm2_ancestral": 3.8, "unipc": 3.2,
    "lms": 3.5, "heun": 3.3,
}
# 清晰度组均值：论文只报告 DDIM 最高（3.8）且 Euler a 明显低于它（Fig. 2 / 3.7 节）。
# 其余采样器的清晰度均值未报告，这里取「约束优化得到的最小 F 解」：
# DDIM = 3.5 + 0.295 ≈ 3.80（对上报告的 3.8），Euler a 比 DDIM 低 0.35 以上。
# 优化过程见 scripts/fit_clarity_offsets.py，结论与不可同时满足的约束见
# docs/known_inconsistencies.md。
CLARITY_OFFSETS = {
    "ddim": 0.295, "lms": 0.163, "dpm2_a": 0.145, "kdpm2_ancestral": -0.037,
    "heun": -0.053, "euler_a": -0.056, "unipc": -0.202, "dpmpp_2m_karras": -0.255,
}
INSPIRATIONAL_MEANS = {
    "dpmpp_2m_karras": 3.7, "kdpm2_ancestral": 3.6, "dpm2_a": 3.5,
    "euler_a": 3.2, "unipc": 3.3, "lms": 3.1, "heun": 3.0, "ddim": 2.9,
}
AESTHETIC_MEANS = {
    "dpmpp_2m_karras": 3.8, "kdpm2_ancestral": 3.7, "ddim": 3.3,
    "dpm2_a": 3.6, "euler_a": 3.1, "unipc": 3.4, "lms": 3.2, "heun": 3.1,
}
USABILITY_MEANS = {
    "dpmpp_2m_karras": 3.6, "kdpm2_ancestral": 3.5, "dpm2_a": 3.4,
    "unipc": 3.2, "euler_a": 3.0, "ddim": 2.9, "lms": 3.1, "heun": 3.0,
}

LPIPS_MEANS = {"euler_a": 0.416, "dpmpp_2m_karras": 0.222, "ddim": 0.228,
               "dpm2_a": 0.381, "kdpm2_ancestral": 0.394, "unipc": 0.231,
               "lms": 0.250, "heun": 0.240}
NIQE_MEANS = {"euler_a": 5.8, "ddim": 4.3, "dpm2_a": 5.5, "kdpm2_ancestral": 5.6,
              "unipc": 4.7, "dpmpp_2m_karras": 4.4, "lms": 4.6, "heun": 4.5}
BRISQUE_MEANS = {"euler_a": 38.2, "ddim": 28.7, "dpm2_a": 36.4, "kdpm2_ancestral": 37.1,
                 "unipc": 31.2, "dpmpp_2m_karras": 29.0, "lms": 30.5, "heun": 30.0}
DREAMSIM_MEANS = {"euler_a": 0.56, "dpm2_a": 0.52, "kdpm2_ancestral": 0.54,
                  "unipc": 0.38, "dpmpp_2m_karras": 0.37, "ddim": 0.36,
                  "lms": 0.40, "heun": 0.39}
FID_VALUES = {"euler_a": 152.4, "dpmpp_2m_karras": 85.1, "ddim": 86.2}


# ---------------------------------------------------------------------------
# 有界生成与反解工具
# ---------------------------------------------------------------------------
def bounded_group(mu: float, n: int, target_sd: float, lo: float, hi: float,
                  rng: np.random.Generator, jitter: float = 0.05) -> np.ndarray:
    """在 [lo, hi] 内生成 n 个数，样本均值严格等于 mu、样本标准差逼近 target_sd。

    有界区间内要拿到大方差，取值必然向两端聚集，因此用「三点分布 + 抖动」构造：
    三份质量分别放在 lo、mu、hi 上，使其矩恰为 (mu, target_sd)，解为
        u = s^2 / ((mu-lo)(hi-lo))、v = s^2 / ((hi-mu)(hi-lo))、w = 1 - u - v
    可行条件 s^2 <= (mu-lo)(hi-mu)（Bernoulli 上界）。生成后先随机打乱，
    再做「再中心化—再缩放—截断」迭代，把两个矩校正到精确值。

    注意：打乱是必需的——若保留分块顺序，各采样器的取值列会近似有序，
    人为制造出约 0.7 的跨采样器相关，使混合模型把方差错分给提示词层。
    """
    span = hi - lo
    sd_max = float(np.sqrt((mu - lo) * (hi - mu)))
    sd = float(min(target_sd, 0.999 * sd_max))
    u = sd ** 2 / ((mu - lo) * span)
    v = sd ** 2 / ((hi - mu) * span)
    w = 1.0 - u - v
    if w < 0:  # 数值兜底：目标略超上界时按比例缩回
        scale = (1.0 - 1e-9) / (u + v)
        u, v, w = u * scale, v * scale, 0.0

    counts = rng.multinomial(n, [u, w, v])
    x = np.concatenate([np.full(counts[0], lo), np.full(counts[1], mu),
                        np.full(counts[2], hi)])
    x = x + rng.uniform(-jitter * span, jitter * span, size=n)
    rng.shuffle(x)
    np.clip(x, lo, hi, out=x)
    for _ in range(600):
        x = x - x.mean() + mu
        sd_now = x.std(ddof=0)
        if sd_now > 1e-12:
            x = mu + (x - mu) * (sd / sd_now)
        np.clip(x, lo, hi, out=x)
        if abs(x.mean() - mu) < 1e-12 and abs(x.std(ddof=0) - sd) < 1e-10:
            break
    return x - x.mean() + mu


def add_prompt_effect(values: np.ndarray, delta: np.ndarray, mu: float,
                      lo: float, hi: float, iters: int = 60) -> np.ndarray:
    """叠加提示词层面的公共偏移，再截断回量表范围并恢复组均值。"""
    y = np.clip(values + delta, lo, hi)
    for _ in range(iters):
        y = y - y.mean() + mu
        np.clip(y, lo, hi, out=y)
        if abs(y.mean() - mu) < 1e-12:
            break
    return y


def bisect_knob(fn: Callable[[float], float], target: float, lo: float, hi: float,
                iters: int = 22, tol: float = 1e-3) -> tuple[float, float]:
    """对标量关系二分反解 fn(knob) = target（fn 单调即可，不必递增）。"""
    f_lo, f_hi = fn(lo), fn(hi)
    if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
        return (lo if abs(f_lo - target) < abs(f_hi - target) else hi,
                min(f_lo, f_hi, key=lambda v: abs(v - target)))
    for _ in range(iters):
        mid = (lo + hi) / 2
        v = fn(mid)
        if abs(v - target) <= tol:
            return mid, v
        if (f_lo < f_hi) == (v < target):
            lo, f_lo = mid, v
        else:
            hi, f_hi = mid, v
    mid = (lo + hi) / 2
    return mid, fn(mid)


def scan_best(fn: Callable[[float], float], target: float, lo: float, hi: float,
              n: int = 41, refine: int = 4) -> tuple[float, float]:
    """在 [lo, hi] 上粗扫再在最优网格点附近细扫，返回使 |fn - target| 最小的参数与取值。

    统计量关于参数是「跳变」的（整数打分、有界分布的多项式抽样都会引入跳变），
    二分可能收敛到跳变点两侧而偏离目标，故这里直接取最优网格点；
    细扫把可达精度提高到细扫步长对应的跳幅。
    """
    best = None
    for knob in np.linspace(lo, hi, n):
        v = float(fn(float(knob)))
        if best is None or abs(v - target) < abs(best[1] - target):
            best = (float(knob), v)
    step = (hi - lo) / max(n - 1, 1)
    lo2, hi2 = max(lo, best[0] - step), min(hi, best[0] + step)
    for knob in np.linspace(lo2, hi2, refine * 4 + 1):
        v = float(fn(float(knob)))
        if abs(v - target) < abs(best[1] - target):
            best = (float(knob), v)
    return best


def prompt_delta(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(0, 1, size=n)
    d -= d.mean()
    if d.std(ddof=0) > 0:
        d *= sigma / d.std(ddof=0)
    return d


def _calibrate_rater_noise(image_means: np.ndarray, n_raters: int, target_icc: float,
                           lo: int, hi: int, seed: int) -> tuple[np.ndarray, float]:
    """二分评分者噪声标准差，使整数原始打分算出的 ICC(2,k) 接近目标值。

    原始打分取整数，ICC 关于噪声标准差是跳变函数，二分只能收敛到跳变点附近；
    因此每轮都留下「与目标差距最小」的那份打分矩阵，而不是返回最后一步的结果。
    同一噪声标准差必须给出同一份矩阵，故 RNG 由 seed 与噪声标准差共同决定。
    """
    def gen(sd: float) -> np.ndarray:
        rng = np.random.default_rng([seed, int(round(sd * 1e4))])
        out = np.empty((len(image_means), n_raters))
        for r in range(n_raters):
            out[:, r] = np.clip(np.round(image_means + rng.normal(0, sd, len(image_means))),
                                lo, hi)
        return out

    lo_sd, hi_sd, best = 0.05, 3.0, None
    for _ in range(18):
        mid = (lo_sd + hi_sd) / 2
        mat = gen(mid)
        icc = reliability.icc_2k(mat)
        if best is None or abs(icc - target_icc) < abs(best[1] - target_icc):
            best = (mat, icc)
        if icc > target_icc:
            lo_sd = mid
        else:
            hi_sd = mid
    return best


# ---------------------------------------------------------------------------
# 专家：3 尺度 x 30 图，一元方差分析
# ---------------------------------------------------------------------------
def one_way_f(groups: List[np.ndarray]) -> float:
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return float((ssb / (k - 1)) / (ssw / (n - k)))


def build_expert(ref: dict):
    exp = ref["expert"]
    means = exp["image_level_means"]
    scales = [0.0, 0.5, 1.0]

    s_means = [means["surprise"][f"scale_{s}"] for s in scales]
    v_means = [means["creative_value"][f"scale_{s}"] for s in scales]

    def make(sd: float, means_list: List[float], lo: float, hi: float) -> List[np.ndarray]:
        return [bounded_group(m, 30, sd, lo, hi, RNG) for m in means_list]

    out = {}
    for key, means_list, target, lo, hi in (
            ("surprise", s_means, exp["anova"]["surprise"]["F"], 1.0, 5.0),
            ("creative_value", v_means, exp["anova"]["creative_value"]["F"], 1.0, 7.0)):
        knob, got = scan_best(
            lambda sd: one_way_f(make(sd, means_list, lo, hi)), target, 0.4, 1.9)
        groups = make(knob, means_list, lo, hi)
        print(f"  专家{key}：网格搜索组内 SD 到 {knob:.3f}，"
              f"实得 F({len(groups) - 1},{sum(len(g) for g in groups) - len(groups)})"
              f" = {got:.3f}（目标 {target}）")
        out[key] = groups

    rows = []
    for si, s in enumerate(scales):
        for i in range(30):
            rows.append({
                "image_id": f"expB__euler_a__scale{s}__{i + 1:02d}",
                "noise_scale": s, "prompt_id": f"b{i + 1:02d}",
                "surprise": round(float(out["surprise"][si][i]), 3),
                "creative_value": round(float(out["creative_value"][si][i]), 3),
            })
    image_means = pd.DataFrame(rows)

    mat_s, icc_s = _calibrate_rater_noise(
        image_means["surprise"].to_numpy(), 8, ref["reliability"]["expert_surprise_icc_2k"],
        1, 5, seed=20261130)
    mat_v, icc_v = _calibrate_rater_noise(
        image_means["creative_value"].to_numpy(), 8, ref["reliability"]["expert_value_icc_2k"],
        1, 7, seed=20261131)
    print(f"  专家原始打分标定：惊喜度 ICC={icc_s:.3f}"
          f"（目标 {ref['reliability']['expert_surprise_icc_2k']}），"
          f"创造价值 ICC={icc_v:.3f}（目标 {ref['reliability']['expert_value_icc_2k']}）")

    # 创造价值三题项 alpha 也按报告值标定（题项噪声越大、alpha 越低）。
    # 标定必须用与最终打分表完全相同的生成方式，否则算出来的 alpha 与打分表不一致：
    # 这里固定「题项 = 该评分者的整数量表分 + 题项噪声（保留两位）」，
    # alpha 在 720 行 x 3 题的矩阵上计算，与 report_reliability 口径一致。
    target_alpha = ref["reliability"]["cronbach_alpha"]["expert_creative_value"]

    def item_matrices(noise_sd: float) -> np.ndarray:
        rng = np.random.default_rng([20261132, int(round(noise_sd * 1e4))])
        mats = np.empty((len(mat_v), 8, 3))
        for rid in range(8):
            for j in range(3):
                mats[:, rid, j] = np.round(
                    np.clip(mat_v[:, rid] + rng.normal(0, noise_sd, len(mat_v)), 1, 7), 2)
        return mats

    def alpha_at(noise_sd: float) -> float:
        return reliability.cronbach_alpha(item_matrices(noise_sd).reshape(-1, 3))

    noise_sd, alpha_got = scan_best(alpha_at, target_alpha, 0.2, 3.0, n=41)
    items = item_matrices(noise_sd)
    print(f"  创造价值题项噪声标定到 {noise_sd:.3f}，三题项 alpha = {alpha_got:.3f}"
          f"（目标 {target_alpha}）")

    raw_rows = []
    for rid in range(8):
        base_rs = mat_s[:, rid]
        for k, (_, r) in enumerate(image_means.iterrows()):
            raw_rows.append({
                "image_id": r["image_id"], "noise_scale": r["noise_scale"],
                "prompt_id": r["prompt_id"], "rater_id": f"E{rid + 1:02d}",
                "surprise": int(base_rs[k]),
                "originality": float(items[k, rid, 0]),
                "artistic_value": float(items[k, rid, 1]),
                "exhibitability": float(items[k, rid, 2]),
            })
    return image_means, pd.DataFrame(raw_rows)


# ---------------------------------------------------------------------------
# 学生：8 采样器 x 24 提示词，混合模型
# ---------------------------------------------------------------------------
def build_student(ref: dict):
    samplers = list(utils.sampler_table().keys())
    levels = sorted(samplers)
    n = 24
    delta = prompt_delta(n, PROMPT_SIGMA, RNG)

    def frame(values: Dict[str, np.ndarray], extra: Dict[str, np.ndarray] | None = None):
        rows = []
        idx = 0
        for s in samplers:
            for i in range(n):
                row = {"image_id": f"seed42__{s}__p{i + 1:02d}", "sampler": s,
                       "prompt_id": f"p{i + 1:02d}", "surprise": float(values[s][i])}
                for k, v in (extra or {}).items():
                    row[k] = float(v[idx])
                rows.append(row)
                idx += 1
        return pd.DataFrame(rows)

    def surprise_data(sd: float) -> Dict[str, np.ndarray]:
        """与 sd 一一对应的确定性生成：二分过程中重复调用同一 sd 必须得到同一份数据。"""
        rng = np.random.default_rng([20261119, int(round(sd * 1e6))])
        return {s: add_prompt_effect(bounded_group(SURPRISE_MEANS[s], n, sd, 1, 5, rng),
                                     delta, SURPRISE_MEANS[s], 1, 5) for s in samplers}

    def fit_F1(sd: float) -> float:
        d = frame(surprise_data(sd))[["sampler", "prompt_id", "surprise"]].rename(
            columns={"surprise": "y"})
        res = student_model.fit_model(d, "y ~ C(sampler)")
        return student_model.sampler_wald_f(
            res, levels, df_denom=int(res.nobs) - int(res.model.exog.shape[1]),
            data=d)["F"]

    target_f1 = ref["student_model"]["M1"]["F"]
    knob, got = scan_best(fit_F1, target_f1, 0.6, 2.6)
    surprise = surprise_data(knob)
    print(f"  学生 M1：网格搜索组内 SD 到 {knob:.3f}，实得 F = {got:.3f}（目标 {target_f1}）")
    within_sd = knob          # 惊喜度的组内标准差，用于给清晰度定标

    # 启发性：与惊喜度的相关调到 0.214（混合组内离差，保持组均值不变）
    target_r_ws = ref["correlations_N192"]["surprise_inspirational"]["r"]
    rng_i = np.random.default_rng(20261121)
    inner = {s: bounded_group(INSPIRATIONAL_MEANS[s], n, knob * 0.9, 1, 5, rng_i)
             for s in samplers}

    def mixed_insp(lam: float) -> Dict[str, np.ndarray]:
        out = {}
        for s in samplers:
            dw = inner[s] - inner[s].mean()
            dx = surprise[s] - surprise[s].mean()
            dev = (1 - lam) * dw + lam * dx
            if dev.std(ddof=0) > 0:
                dev = dev * (dw.std(ddof=0) / dev.std(ddof=0))
            out[s] = INSPIRATIONAL_MEANS[s] + dev
        return out

    def corr_ws(lam: float) -> float:
        f = frame(surprise, {"inspirational": np.concatenate(
            [mixed_insp(lam)[s] for s in samplers])})
        return float(np.corrcoef(f["surprise"], f["inspirational"])[0, 1])

    lam, got_r = bisect_knob(corr_ws, target_r_ws, 0.0, 0.99, iters=30, tol=1e-4)
    insp = mixed_insp(lam)
    print(f"  启发性混合系数 {lam:.4f}，实得 r(surprise, inspirational) = {got_r:.4f}"
          f"（目标 {target_r_ws}）")

    rng_o = np.random.default_rng(20261122)
    other = {}
    for name, table in (("usability", USABILITY_MEANS), ("aesthetic", AESTHETIC_MEANS)):
        other[name] = np.concatenate([
            bounded_group(table[s], n, knob * 0.9, 1, 5, rng_o) for s in samplers])

    df = frame(surprise, {"inspirational": np.concatenate([insp[s] for s in samplers]),
                          **other})

    def clarity_data(a_scale: float) -> Dict[str, np.ndarray]:
        """a_scale 为「组内波动跟随惊喜度」的强度（单位：惊喜度的组内标准差）。"""
        rng = np.random.default_rng([20261120, int(round(a_scale * 1e6))])
        clar = {}
        for s in samplers:
            dev = surprise[s] - surprise[s].mean()
            sd_s = dev.std(ddof=0)
            dev = dev / sd_s if sd_s > 0 else dev
            clar[s] = add_prompt_effect(
                np.clip(3.5 + CLARITY_OFFSETS[s] + a_scale * within_sd * dev
                        + rng.normal(0, CLARITY_NOISE_SD, n), 1, 5),
                prompt_delta(n, PROMPT_SIGMA, rng), 3.5 + CLARITY_OFFSETS[s], 1, 5)
        return clar

    target_f2 = ref["student_model"]["M2"]["F"]
    # 论文 M2 的两个报数在合成数据上互斥：既要清晰度组均值按 Fig.2 的格局
    # （DDIM 最高 3.8、Euler a 明显更低），又要清晰度协变量「为正且 p<0.001」，
    # 那么把 clarity 放进模型只会让采样器效应 F 变大（协变量吸走组内残差方差，
    # 而它与采样器均值反向，调整后组均值差距被拉大），不可能从 2.41 降到 2.18。
    # 因此在「系数显著为正 + DDIM 清晰度最高且 Euler a 至少低 0.35」的约束下，
    # 用约束优化找 F 的最小值，再对组内强度做一维搜索取最小 F（推导见
    # docs/known_inconsistencies.md，脚本 scripts/fit_clarity_offsets.py 可复现）。
    best2 = None
    for a_scale in np.linspace(0.06, 0.16, 11):
        clar_c = clarity_data(a_scale)
        d2 = frame(surprise, {"clarity": np.concatenate([clar_c[s] for s in samplers])})
        d2 = d2[["sampler", "prompt_id", "surprise", "clarity"]].rename(
            columns={"surprise": "y"})
        res2 = student_model.fit_model(d2, "y ~ C(sampler) + clarity")
        F2 = student_model.sampler_wald_f(
            res2, levels, df_denom=int(res2.nobs) - int(res2.model.exog.shape[1]),
            data=d2)["F"]
        coef2 = float(res2.params.get("clarity", float("nan")))
        t2 = coef2 / float(res2.bse.get("clarity", float("nan")))
        if coef2 <= 0 or t2 < 3.4:      # 约束：系数为正且 p < 0.001
            continue
        if best2 is None or F2 < best2[1]:
            best2 = (a_scale, F2, coef2, t2)
    if best2 is None:
        raise RuntimeError("未找到满足「清晰度系数为正且显著」的构造")
    a_scale, got2, coef2, t2 = best2
    print(f"  学生 M2：组内强度 {a_scale:.3f}（约束优化的最小 F），实得 F = {got2:.3f}"
          f"（目标 {target_f2}，合成数据无法达到，见 docs/known_inconsistencies.md），"
          f"clarity 系数 = {coef2:+.3f}（t = {t2:.1f}，p < 0.001）")

    clar = clarity_data(a_scale)
    df["clarity"] = np.concatenate([clar[s] for s in samplers])

    # CLIP：用正交基构造，令 corr(CLIP, surprise) = 0.183、corr(CLIP, inspirational) = 0.319
    target_r_sc = ref["correlations_N192"]["surprise_clip"]["r"]
    target_r_wc = ref["correlations_N192"]["inspirational_clip"]["r"]
    s_vec = (df["surprise"] - df["surprise"].mean()) / df["surprise"].std()
    w_vec = (df["inspirational"] - df["inspirational"].mean()) / df["inspirational"].std()
    rho = float(np.corrcoef(s_vec, w_vec)[0, 1])

    e1 = s_vec.to_numpy() / np.linalg.norm(s_vec.to_numpy())
    tmp = w_vec.to_numpy() - (w_vec.to_numpy() @ e1) * e1
    e2 = tmp / np.linalg.norm(tmp)
    tmp3 = np.random.default_rng(20261123).normal(0, 1, size=e1.shape)
    tmp3 = tmp3 - (tmp3 @ e1) * e1 - (tmp3 @ e2) * e2
    e3 = tmp3 / np.linalg.norm(tmp3)

    p = target_r_sc
    q = (target_r_wc - rho * p) / np.sqrt(1 - rho ** 2)
    rr = float(np.sqrt(max(0.0, 1 - p ** 2 - q ** 2)))
    c_unit = p * e1 + q * e2 + rr * e3
    df["clip_score"] = np.round(0.30 + 0.03 * c_unit, 5)
    print(f"  CLIP 构造校验：r(surprise, CLIP) = "
          f"{np.corrcoef(df['surprise'], df['clip_score'])[0, 1]:.4f}（目标 {target_r_sc}），"
          f"r(inspirational, CLIP) = "
          f"{np.corrcoef(df['inspirational'], df['clip_score'])[0, 1]:.4f}"
          f"（目标 {target_r_wc}）")

    # 逐评分者长表：4 位学生，整数打分，噪声标定到报告 ICC
    mat, icc = _calibrate_rater_noise(
        df["surprise"].to_numpy(), 4, ref["reliability"]["student_icc_2k"], 1, 5,
        seed=20261133)
    print(f"  学生原始打分标定：惊喜度 ICC={icc:.3f}"
          f"（目标 {ref['reliability']['student_icc_2k']}）")

    raw_rows = []
    for rid in range(4):
        col = mat[:, rid]
        for k, (_, r) in enumerate(df.iterrows()):
            raw_rows.append({
                "image_id": r["image_id"], "sampler": r["sampler"],
                "prompt_id": r["prompt_id"], "rater_id": f"S{rid + 1:02d}",
                "surprise": int(col[k]),
                "inspirational": int(np.clip(round(r["inspirational"] + RNG.normal(0, 0.8)), 1, 5)),
                "usability": int(np.clip(round(r["usability"] + RNG.normal(0, 0.85)), 1, 5)),
                "aesthetic": int(np.clip(round(r["aesthetic"] + RNG.normal(0, 0.8)), 1, 5)),
                "clarity": int(np.clip(round(r["clarity"] + RNG.normal(0, 0.7)), 1, 5)),
            })
    return df, pd.DataFrame(raw_rows)


# ---------------------------------------------------------------------------
# 实验 C：CLIP ~ PromptType * Sampler + (1|Prompt)
# ---------------------------------------------------------------------------
def build_experiment_c(ref: dict) -> pd.DataFrame:
    """实验 C：论文未检出提示词类型效应与交互效应，故合成数据中真实效应为 0。

    目标值 F(1,171) = 0.47、F(4,171) = 1.63 属零效应下的抽样结果，无法靠调参精确命中，
    因此改为「搜索随机种子」：在若干噪声实现中挑一个使零效应下的两个 F 都低于目标值，
    再分别二分类型效应与交互效应的幅度，把 F 抬到目标值附近。
    """
    samplers = ["euler_a", "dpmpp_2m_karras", "lms", "heun", "ddim"]
    types = ["embodied_sensory", "relational_process"]
    n_prompt = 20
    n_obs = n_prompt * len(types) * len(samplers)

    target_f_inter = ref["experiment_c"]["prompttype_x_sampler"]["F"]
    target_f_type = ref["experiment_c"]["prompttype_main"]["F"]
    samp_eff = dict(zip(samplers, np.linspace(-0.01, 0.01, len(samplers))))
    inter_pattern = np.array([[-1, 1, -1, 1, -1], [1, -1, 1, -1, 1]], dtype=float)

    def gen(seed: int, a_type: float, a_inter: float) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.012, n_obs)
        delta = prompt_delta(n_prompt, 0.02, rng)
        rows, k = [], 0
        for i in range(n_prompt):
            for ti, t in enumerate(types):
                for si, sm in enumerate(samplers):
                    rows.append({
                        "image_id": f"experiment_c__{sm}__c{i + 1:02d}__s42",
                        "experiment": "experiment_c", "sampler": sm,
                        "prompt_id": f"c{i + 1:02d}", "prompt_type": t, "noise_scale": "",
                        "clip_score": float(0.30 + samp_eff[sm]
                                            + a_type * (1 if ti else -1)
                                            + a_inter * inter_pattern[ti, si]
                                            + delta[i] + noise[k]),
                    })
                    k += 1
        return pd.DataFrame(rows)

    def fit(d: pd.DataFrame) -> dict:
        res = student_model.fit_model(d, "clip_score ~ C(prompt_type) * C(sampler)")
        df2 = int(res.nobs) - int(res.model.exog.shape[1]) - (n_prompt - 1)
        factors = {"prompt_type": sorted(types), "sampler": sorted(samplers)}
        out = {}
        for key, term in (("interaction", ("prompt_type", "sampler")),
                          ("prompt_type", ("prompt_type",))):
            out[key] = factor_effect_f(res, d, factors, term, df_denom=df2)["F"]
        out["df2"] = df2
        return out

    seed, found = None, False
    for cand in range(300):
        f = fit(gen(cand, 0.0, 0.0))
        if f["prompt_type"] < target_f_type and f["interaction"] < target_f_inter:
            seed, found = cand, True
            break
    if not found:
        seed = 0
        print(f"  实验 C：300 个种子内未找到零效应下两 F 均低于目标值的实现，改用种子 {seed}")

    a_type = a_inter = 0.0
    for _ in range(4):   # 两个幅度弱耦合，交替二分几轮即可收敛
        a_type, _ = bisect_knob(
            lambda a: fit(gen(seed, a, a_inter))["prompt_type"], target_f_type,
            0.0, 0.20, iters=12, tol=2e-3)
        a_inter, _ = bisect_knob(
            lambda a: fit(gen(seed, a_type, a))["interaction"], target_f_inter,
            0.0, 0.20, iters=12, tol=2e-3)

    d = gen(seed, a_type, a_inter)
    f = fit(d)
    print(f"  实验 C：选定种子 {seed}、类型幅度 {a_type:.4f}、交互幅度 {a_inter:.4f}；"
          f"实得交互 F(4,{f['df2']}) = {f['interaction']:.3f}（目标 {target_f_inter}），"
          f"类型 F(1,{f['df2']}) = {f['prompt_type']:.3f}（目标 {target_f_type}）")
    return d


# ---------------------------------------------------------------------------
# 计算指标与信度汇总
# ---------------------------------------------------------------------------
def write_scores(student_means: pd.DataFrame, exp_c: pd.DataFrame) -> None:
    """写出计算指标 CSV，使 sampler_metrics 聚合后得到报告均值。"""
    scores = utils.ensure_dir(utils.repo_path("outputs", "scores"))
    samplers = list(utils.sampler_table().keys())
    rng = np.random.default_rng(20261124)

    rows = []
    for s in samplers:
        raw = rng.normal(0, 0.02, size=72); raw -= raw.mean()
        v = LPIPS_MEANS[s] + raw
        k = 0
        for i in range(1, 25):
            for a, b in ((1, 2), (1, 3), (2, 3)):
                rows.append({"sampler": s, "prompt_id": f"p{i:02d}",
                             "img_a": f"seed{a}", "img_b": f"seed{b}",
                             "lpips": round(float(v[k]), 4)})
                k += 1
    pd.DataFrame(rows).to_csv(scores / "lpips_original.csv", index=False)

    rows = []
    for s in samplers:
        raw_n = rng.normal(0, 0.25, size=72); raw_n -= raw_n.mean()
        raw_b = rng.normal(0, 2.0, size=72); raw_b -= raw_b.mean()
        k = 0
        for i in range(1, 25):
            for seed in (42, 43, 44):
                rows.append({"image": f"original__{s}__p{i:02d}__s{seed}.png",
                             "sampler": s, "prompt_id": f"p{i:02d}",
                             "niqe": round(float(NIQE_MEANS[s] + raw_n[k]), 4),
                             "brisque": round(float(BRISQUE_MEANS[s] + raw_b[k]), 4)})
                k += 1
    pd.DataFrame(rows).to_csv(scores / "niqe_brisque_original.csv", index=False)

    rows = []
    for s in samplers:
        raw = rng.normal(0, 0.02, size=72); raw -= raw.mean()
        v = DREAMSIM_MEANS[s] + raw
        k = 0
        for i in range(1, 25):
            for a, b in ((0, 1), (0, 2), (1, 2)):
                rows.append({"sampler": s, "prompt_id": f"p{i:02d}", "i": a, "j": b,
                             "dreamsim": round(float(v[k]), 4)})
                k += 1
    pd.DataFrame(rows).to_csv(scores / "dreamsim_original.csv", index=False)

    rows = []
    for s in samplers:
        for i in range(1, 25):
            rows.append({"image": f"original__{s}__p{i:02d}__s42.png", "sampler": s,
                         "prompt_id": f"p{i:02d}",
                         "laplacian_variance": round(float(rng.normal(120, 30)), 2)})
    pd.DataFrame(rows).to_csv(scores / "lapvar_original.csv", index=False)

    pd.DataFrame([{"sampler": k, "n_images": 24, "fid": v} for k, v in FID_VALUES.items()]) \
        .to_csv(scores / "fid_original.csv", index=False)

    clip_rows = [{
        "image_id": r["image_id"], "experiment": "original", "sampler": r["sampler"],
        "prompt_id": r["prompt_id"], "prompt_type": "", "noise_scale": "",
        "clip_score": r["clip_score"],
    } for _, r in student_means.iterrows()]
    pd.DataFrame(clip_rows).to_csv(scores / "clip_original.csv", index=False)

    exp_c.to_csv(scores / "clip_experiment_c.csv", index=False)


def report_reliability(student_raw: pd.DataFrame, expert_raw: pd.DataFrame) -> None:
    print("\n信度（由合成原始长表计算，用于对齐论文报告值）：")
    pivot = student_raw.pivot_table(index="image_id", columns="rater_id", values="surprise")
    rep = reliability.reliability_report(pivot.to_numpy(), "student surprise")
    print(f"  学生惊喜度  alpha={rep['cronbach_alpha']:.3f}  ICC(2,k)={rep['icc_2k']:.3f}"
          f"  （论文 alpha=0.89, ICC=0.86）")

    pivot_e = expert_raw.pivot_table(index="image_id", columns="rater_id", values="surprise")
    rep_e = reliability.reliability_report(pivot_e.to_numpy(), "expert surprise")
    print(f"  专家惊喜度  alpha={rep_e['cronbach_alpha']:.3f}  ICC(2,k)={rep_e['icc_2k']:.3f}"
          f"  （论文 ICC=0.88）")

    items = expert_raw[["originality", "artistic_value", "exhibitability"]].to_numpy()
    print(f"  创造价值三题项 alpha={reliability.cronbach_alpha(items):.3f}（论文 0.87）")


def main() -> int:
    ref = utils.load_reference_results()
    ratings_dir = utils.ensure_dir(utils.repo_path("outputs", "ratings"))

    print("合成专家评分 ...")
    expert_means, expert_raw = build_expert(ref)
    expert_means.to_csv(ratings_dir / "expert_image_means.csv", index=False)
    expert_raw.to_csv(ratings_dir / "expert_raw.csv", index=False)

    print("合成学生评分 ...")
    student_means, student_raw = build_student(ref)
    student_means.to_csv(ratings_dir / "student_image_means.csv", index=False)
    student_raw.to_csv(ratings_dir / "student_raw.csv", index=False)

    print("合成计算指标 ...")
    print("  实验 C：")
    exp_c = build_experiment_c(ref)
    write_scores(student_means, exp_c)

    print("\n组均值核对：")
    print("  专家惊喜度 =", {k: round(v, 3) for k, v in
                          expert_means.groupby("noise_scale")["surprise"].mean().items()})
    print("  专家创造价值 =", {k: round(v, 3) for k, v in
                           expert_means.groupby("noise_scale")["creative_value"].mean().items()})
    print("  学生惊喜度 =", {k: round(v, 3) for k, v in
                          student_means.groupby("sampler")["surprise"].mean().items()})

    report_reliability(student_raw, expert_raw)

    print(f"\n写出：{ratings_dir}/{{student,expert}}_*.csv、"
          f"{utils.repo_path('outputs', 'scores')}/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
