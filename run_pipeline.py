
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402
from src.analysis import (  # noqa: E402
    correlations, experiment_c, expert_anova, figures, robustness,
    sampler_metrics, student_model,
)

STEPS = [
    ("采样器级指标聚合", sampler_metrics.main),
    ("专家方差分析与 Tukey HSD", expert_anova.main),
    ("学生混合模型", student_model.main),
    ("相关分析（BH 校正）", correlations.main),
    ("实验 C 异质性检验", experiment_c.main),
    ("稳健性检查汇总", robustness.main),
    ("重绘 Fig.2-Fig.5", figures.main),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="运行第 3、4 阶段流水线")
    parser.add_argument("--with-demo", action="store_true",
                        help="先运行演示数据合成脚本")
    args = parser.parse_args(argv)

    if args.with_demo:
        from scripts import demo_scores_from_reference  # type: ignore
        print("=" * 70)
        print("阶段 3 前置：合成演示数据")
        print("=" * 70)
        demo_scores_from_reference.main()

    failures = []
    for name, fn in STEPS:
        print("\n" + "=" * 70)
        print(f"运行：{name}")
        print("=" * 70)
        try:
            fn([])
        except Exception:  # noqa: BLE001
            failures.append(name)
            traceback.print_exc()

    print("\n" + "=" * 70)
    if failures:
        print(f"流水线结束：{len(STEPS) - len(failures)}/{len(STEPS)} 步成功，"
              f"失败步骤：{', '.join(failures)}")
        return 1
    print(f"流水线结束：{len(STEPS)}/{len(STEPS)} 步全部成功")
    print(f"表格输出目录：{utils.repo_path('outputs', 'tables')}")
    print(f"图件输出目录：{utils.repo_path('outputs', 'figures')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
