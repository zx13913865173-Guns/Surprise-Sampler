"""稳健性检查与效力敏感性讨论（论文 2.7 / 3.5 节）。

论文口径：独立生成集（50 提示词 x 5 采样器），检验
    - 模型版本：SD1.5、SD3.5（各 1 次运行）
    - CFG 尺度：3.0、15.0（各 2 次运行）
    - 步数：20、50（各 2 次运行）
    - 独立集基准：Euler a 与 DPM++ 2M Karras 各 3 次运行
报告值：
    CFG 15 vs 7.5：惊喜度均值变化 -0.4
    步数 50 vs 20（Euler a）：+0.3
    SD1.5：五个采样器上的 Spearman rho = 0.86
    SD3.5：rho = 0.79

本模块不重新生成图像，只把上述报告值落到表格，并对 Spearman 秩相关的
计算方式做一次可跑通的演示（用模拟的采样器均值排序）。

用法：
    python -m src.analysis.robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import utils


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="稳健性检查汇总")
    parser.add_argument("--out-dir", default=str(utils.repo_path("outputs", "tables")))
    args = parser.parse_args(argv)

    ref = utils.load_reference_results()["robustness"]
    rows = [
        {"check": "CFG 15 vs 7.5 (surprise delta)", "value": ref["cfg_15_vs_7_5_surprise_delta"]},
        {"check": "steps 50 vs 20, Euler a (surprise delta)", "value": ref["steps_50_vs_20_euler_a_surprise_delta"]},
        {"check": "SD1.5 Spearman rho (5 samplers)", "value": ref["sd15_spearman_rho_over_5_samplers"]},
        {"check": "SD3.5 Spearman rho (5 samplers)", "value": ref["sd35_spearman_rho_over_5_samplers"]},
    ]
    df = pd.DataFrame(rows)

    # 演示 Spearman 秩相关的计算方式
    rng = np.random.default_rng(42)
    base = np.array([3.9, 3.4, 2.3, 3.7, 3.8])  # 5 个采样器的主实验惊喜度均值
    perturbed = base + rng.normal(0, 0.25, size=base.shape)
    rho, p = stats.spearmanr(base, perturbed)
    print(f"Spearman 演示：rho = {rho:.2f}, p = {p:.3f}（示意，非论文数据）")

    out_dir = utils.ensure_dir(Path(args.out_dir))
    df.to_csv(out_dir / "robustness.csv", index=False)
    print(df.to_string(index=False))
    print(f"写入 {out_dir / 'robustness.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
