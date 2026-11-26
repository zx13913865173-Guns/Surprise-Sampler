"""信度指标：ICC(2,k)、Fleiss' kappa、Cronbach's alpha、Kendall's W。

论文 2.5 节报告值：
    学生评分    ICC(2,k) = 0.86，Fleiss' kappa = 0.73
    专家惊喜度  ICC(2,k) = 0.88
    专家创造价值 ICC(2,k) = 0.90，Kendall's W = 0.74 (p < 0.001)
    三题项创造价值 Cronbach's alpha = 0.87
    预验证五维 alpha：surprise 0.89 / inspirational 0.87 / usability 0.85 /
                     aesthetic 0.88 / clarity 0.91

输入约定：data 为 (n_subjects, n_raters) 的评分矩阵（行=被评对象，列=评分者）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def cronbach_alpha(data: np.ndarray) -> float:
    """Cronbach's alpha。行=被试，列=题项/评分者。"""
    data = np.asarray(data, dtype=float)
    k = data.shape[1]
    if k < 2:
        raise ValueError("至少需要两个题项")
    item_var = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def icc_2k(data: np.ndarray) -> float:
    """ICC(2,k)：双向随机效应、平均测量、绝对一致性。

    行 = 被评对象（n），列 = 评分者（k）。
    ICC(2,k) = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else np.nan
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom)


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa。

    counts：形状 (n_items, n_categories) 的整数计数矩阵，
    每行之和 = 评分者人数 m（固定）。
    """
    counts = np.asarray(counts, dtype=float)
    n_items, n_cat = counts.shape
    m = counts.sum(axis=1)
    if not np.allclose(m, m[0]):
        raise ValueError("Fleiss' kappa 要求每个对象被相同数量的评分者评定")
    m = m[0]
    if m < 2:
        raise ValueError("至少需要两位评分者")

    p_j = counts.sum(axis=0) / (n_items * m)            # 每个类别的边际比例
    P_i = (np.sum(counts ** 2, axis=1) - m) / (m * (m - 1))  # 每个对象的一致性
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e))


def kendall_w(data: np.ndarray) -> tuple[float, float]:
    """Kendall's W（含结校正），返回 (W, p)。

    行 = 被评对象（n），列 = 评分者（k）。
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    # 对每个评分者把对象排名
    ranks = np.apply_along_axis(stats.rankdata, 0, data)
    rank_sums = ranks.sum(axis=1)
    s = ((rank_sums - rank_sums.mean()) ** 2).sum()

    # 结校正
    def tie_correction(col: np.ndarray) -> float:
        _, counts = np.unique(col, return_counts=True)
        return float(np.sum(counts ** 3 - counts))

    t = sum(tie_correction(data[:, j]) for j in range(k))
    denom = k ** 2 * (n ** 3 - n) - k * t
    w = 12 * s / denom if denom else np.nan

    # 卡方近似
    chi2 = k * (n - 1) * w
    p = float(stats.chi2.sf(chi2, df=n - 1))
    return float(w), p


def aggregate_to_image_means(long_df: pd.DataFrame, subject_col: str = "image_id",
                             rater_col: str = "rater_id", value_col: str = "rating") -> pd.Series:
    """把长表评分聚合为对象级均值（论文：评分聚合为图像级均值）。"""
    return long_df.groupby(subject_col)[value_col].mean()


def reliability_report(data: np.ndarray, label: str = "") -> dict:
    """一次性给出 alpha 与 ICC(2,k)。"""
    return {
        "label": label,
        "n_subjects": int(np.asarray(data).shape[0]),
        "n_raters": int(np.asarray(data).shape[1]),
        "cronbach_alpha": cronbach_alpha(data),
        "icc_2k": icc_2k(data),
    }


if __name__ == "__main__":
    # 自检：在论文报告的信度水平附近构造数据，检查各指标实现合理
    rng = np.random.default_rng(42)
    true_scores = rng.normal(3, 1, size=(192, 1))
    ratings = true_scores + rng.normal(0, 0.7, size=(192, 4))
    print(reliability_report(ratings, "student-sim"))

    # Fleiss kappa 自检：构造 80% 一致的打分
    n_items, m, n_cat = 192, 4, 5
    base = rng.integers(0, n_cat, size=n_items)
    counts = np.zeros((n_items, n_cat))
    for i, b in enumerate(base):
        picks = [b] * 3 + [int(rng.integers(0, n_cat))]
        for p in picks:
            counts[i, p] += 1
    print("fleiss_kappa:", round(fleiss_kappa(counts), 3))
