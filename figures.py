"""重绘论文 Fig. 2 - Fig. 5（数值取自 data/reference_results.json）。

放入真实数据后，同样的函数会改用真实表格，图会自动更新。

    Fig. 2  八个采样器的学生评分均值（惊喜度等维度）
    Fig. 3  计算指标与人类判断的对齐矩阵
    Fig. 4  噪声尺度上的专家评分曲线（描述性倒 U 形）
    Fig. 5  固定效应与效应量面板

用法：
    python -m src.analysis.figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import utils  # noqa: E402

DIMS = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]


def fig2_student_ratings(ref: dict, fig_dir: Path) -> Path:
    means = ref["student_ratings_mean"]["surprise"]
    labels = [utils.sampler_table()[k]["display"] for k in means if means[k] is not None]
    values = [means[k] for k in means if means[k] is not None]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    bars = ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Student-rated surprise (1-5)")
    ax.set_ylim(0, 5)
    ax.set_title("Fig. 2  Student-rated surprise by sampler")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "fig2_student_surprise.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_alignment(ref: dict, fig_dir: Path) -> Path:
    """对齐矩阵：由报告的相关方向构造示意图。"""
    metrics = ["LPIPS\n(inter-seed)", "NIQE", "BRISQUE", "FID\n(set-level)", "CLIP\nScore"]
    dims = ["surprise", "inspirational", "usability", "aesthetic", "clarity"]
    # 方向编码：1 = 正相关/一致，-1 = 反向，0 = 弱/未报告
    matrix = np.array([
        [1, -1, 0, -1, 0],   # LPIPS
        [1, -1, 0, -1, -1],  # NIQE
        [1, -1, 0, -1, -1],  # BRISQUE
        [1, -1, 0, 0, 0],    # FID
        [1, 1, 0, 1, 0],     # CLIP
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=20, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_title("Fig. 3  Computational metrics vs. human judgments (direction)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, {1: "+", -1: "-", 0: "0"}[matrix[i, j]],
                    ha="center", va="center", fontsize=11, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="alignment direction")
    fig.tight_layout()
    out = fig_dir / "fig3_alignment_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_noise_scale(ref: dict, fig_dir: Path) -> Path:
    data = ref["expert"]["image_level_means"]
    scales = [0.0, 0.5, 1.0]
    surprise = [data["surprise"][f"scale_{s}"] for s in scales]
    value = [data["creative_value"][f"scale_{s}"] for s in scales]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax1.plot(scales, surprise, "o-", color="#C44E52", label="expert surprise (1-5)")
    ax1.set_xlabel("injected-noise scale")
    ax1.set_ylabel("expert surprise (1-5)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(scales, value, "s--", color="#4C72B0", label="expert creative value (1-7)")
    ax2.set_ylabel("expert creative value (1-7)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title("Fig. 4  Expert ratings across noise scales (descriptive)")
    fig.tight_layout()
    out = fig_dir / "fig4_noise_scale.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_effects(ref: dict, fig_dir: Path) -> Path:
    m1 = ref["student_model"]["M1"]
    m2 = ref["student_model"]["M2"]
    an = ref["expert"]["anova"]
    corr = ref["correlations_N192"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    names = ["M1 sampler\n(student)", "M2 sampler+clarity\n(student)",
             "Expert surprise\n(noise scale)", "Expert value\n(noise scale)"]
    fvals = [m1["F"], m2["F"], an["surprise"]["F"], an["creative_value"]["F"]]
    ps = [m1["p"], m2["p"], an["surprise"]["p"], an["creative_value"]["p"]]
    axes[0].bar(range(4), fvals, color="#55A868")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("F statistic")
    axes[0].set_title("Fixed effects")
    for i, (f, p) in enumerate(zip(fvals, ps)):
        axes[0].text(i, f + 0.05, f"F={f:.2f}\np={p:.3f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(fvals) * 1.45)

    cnames = ["surprise~CLIP", "inspirational~CLIP", "surprise~inspirational"]
    rvals = [corr["surprise_clip"]["r"], corr["inspirational_clip"]["r"],
             corr["surprise_inspirational"]["r"]]
    axes[1].bar(range(3), rvals, color="#8172B3")
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(cnames, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("Pearson r (N=192)")
    axes[1].set_title("Correlations")
    for i, r in enumerate(rvals):
        axes[1].text(i, r + 0.005, f"r={r:.3f}", ha="center", fontsize=8)
    axes[1].set_ylim(0, max(rvals) * 1.35)

    fig.suptitle("Fig. 5  Effect sizes and correlations")
    fig.tight_layout()
    out = fig_dir / "fig5_effects.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="重绘 Fig.2-Fig.5")
    parser.add_argument("--fig-dir", default=str(utils.repo_path("outputs", "figures")))
    args = parser.parse_args(argv)

    ref = utils.load_reference_results()
    fig_dir = utils.ensure_dir(Path(args.fig_dir))
    outs = [
        fig2_student_ratings(ref, fig_dir),
        fig3_alignment(ref, fig_dir),
        fig4_noise_scale(ref, fig_dir),
        fig5_effects(ref, fig_dir),
    ]
    for o in outs:
        print(f"写入 {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
