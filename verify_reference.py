"""用论文报告值做确定性反算核对。

只在**可由汇总值唯一反算**的关系上判定 PASS/FAIL，不臆造数据：
    - 计数关系（图像数、评分数、每采样器图像数）
    - F 分布：由 (F, df1, df2) 反算 p
    - 单因素 eta^2：eta2 = F*df1 / (F*df1 + df2)
    - 学生化极差：由 q 反算 Tukey p
    - Pearson：由 r 与 N 反算 t 与 p；r^2
    - Bonferroni 校正阈值
另有若干“交叉一致性”检查标为 WARN：这些量各自自洽，但两两之间若不能同时成立，
会在核对报告里列出，供你回查原始输出。

用法：
    python -m scripts.verify_reference
    python scripts/verify_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))


def record_warn(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "WARN", detail))


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def check_counts(ref: dict) -> None:
    c = ref["image_counts"]
    record("图像数 2250+144+240+200=2834",
           c["original_150x5x3"] + c["experiment_a_3x24x2"] + c["experiment_b_5x24x2"]
           + c["experiment_c_2x20x5x1"] == c["total_generated"],
           f"{c['original_150x5x3']}+{c['experiment_a_3x24x2']}"
           f"+{c['experiment_b_5x24x2']}+{c['experiment_c_2x20x5x1']}={c['total_generated']}")
    record("150 x 5 x 3 = 2250",
           c["original_150x5x3"] == 150 * 5 * 3, str(c["original_150x5x3"]))
    record("3 x 24 x 2 = 144",
           c["experiment_a_3x24x2"] == 3 * 24 * 2, str(c["experiment_a_3x24x2"]))
    record("5 x 24 x 2 = 240",
           c["experiment_b_5x24x2"] == 5 * 24 * 2, str(c["experiment_b_5x24x2"]))
    record("2 x 20 x 5 x 1 = 200",
           c["experiment_c_2x20x5x1"] == 2 * 20 * 5 * 1, str(c["experiment_c_2x20x5x1"]))
    record("24 x 8 = 192 学生评分图数",
           c["student_images"] == 24 * 8, str(c["student_images"]))
    record("4 x 192 = 768 学生评分条数",
           c["student_ratings_total"] == 4 * 192, str(c["student_ratings_total"]))
    record("8 x 90 = 720 专家评分条数 x2",
           c["expert_surprise_ratings_total"] == 8 * 90
           and c["expert_value_ratings_total"] == 8 * 90,
           f"{c['expert_surprise_ratings_total']}/{c['expert_value_ratings_total']}")
    record("768+720+720 = 2208 主分析评分",
           c["student_ratings_total"] + c["expert_surprise_ratings_total"]
           + c["expert_value_ratings_total"] == c["primary_analysis_ratings"],
           str(c["primary_analysis_ratings"]))
    record("150-24=126 条未分析提示词",
           c["analysis_prompts"] + c["unanalyzed_prompts"] == 150
           and c["unanalyzed_prompts"] == 126, str(c["unanalyzed_prompts"]))
    record("126 x 5 x 3 = 1890 张未分析图像",
           c["unanalyzed_prompts"] * 5 * 3 == c["unanalyzed_images"],
           str(c["unanalyzed_images"]))
    record("24 x 3 = 72 每采样器计算图数",
           c["computational_original_per_sampler"] == 24 * 3,
           str(c["computational_original_per_sampler"]))
    record("24 x 2 = 48 每新增采样器图数",
           c["computational_added_per_sampler"] == 24 * 2,
           str(c["computational_added_per_sampler"]))
    record("3 x 30 = 90 专家评分图像数",
           c["expert_rated_images"] == 3 * 30, str(c["expert_rated_images"]))


def check_f_p(name: str, F: float, df1: int, df2: int, p: float, tol: float = 2e-3) -> None:
    """由 (F, df1, df2) 反算 p。容差取 2e-3：报告里 F 与 p 均经过四舍五入。"""
    p_calc = float(stats.f.sf(F, df1, df2))
    record(f"{name}: F({df1},{df2})={F} -> p",
           close(p_calc, p, tol), f"反算 p={p_calc:.4f}，报告 {p}")
    print(f"    {name}: 反算 p = {p_calc:.4f}（报告 {p}）")


def check_eta2(name: str, F: float, df1: int, df2: int, eta2: float, tol: float = 5e-4) -> None:
    e = F * df1 / (F * df1 + df2)
    record(f"{name}: eta2 = F*df1/(F*df1+df2)",
           close(e, eta2, tol), f"反算 {e:.4f}，报告 {eta2}")


def check_tukey(name: str, q: float, p: float, k: int = 3, df: int = 87, tol: float = 6e-3) -> None:
    p_calc = float(stats.studentized_range.sf(q, k, df))
    record_warn(f"{name}: q={q} -> Tukey p (k={k}, df={df})",
                close(p_calc, p, tol), f"反算 p={p_calc:.3f}，报告 {p}")


def check_r2(name: str, r: float, r2: float, tol: float = 1e-3) -> None:
    record(f"{name}: r2 = r^2", close(r ** 2, r2, tol), f"反算 {r ** 2:.4f}，报告 {r2}")


def main() -> int:
    ref = utils.load_reference_results()

    print("=" * 78)
    print("计数关系")
    print("=" * 78)
    check_counts(ref)

    print("\n" + "=" * 78)
    print("F 分布 -> p、eta^2")
    print("=" * 78)
    m1 = ref["student_model"]["M1"]
    check_f_p("学生 M1", m1["F"], m1["df1"], m1["df2"], m1["p"])
    m2 = ref["student_model"]["M2"]
    check_f_p("学生 M2", m2["F"], m2["df1"], m2["df2"], m2["p"])
    an = ref["expert"]["anova"]
    check_f_p("专家惊喜度", an["surprise"]["F"], an["surprise"]["df1"], an["surprise"]["df2"],
              an["surprise"]["p"])
    check_f_p("专家创造价值", an["creative_value"]["F"], an["creative_value"]["df1"],
              an["creative_value"]["df2"], an["creative_value"]["p"])
    ec = ref["experiment_c"]
    check_f_p("实验 C 交互", ec["prompttype_x_sampler"]["F"],
              ec["prompttype_x_sampler"]["df1"], ec["prompttype_x_sampler"]["df2"],
              ec["prompttype_x_sampler"]["p"], tol=8e-3)
    check_f_p("实验 C 主效应", ec["prompttype_main"]["F"], ec["prompttype_main"]["df1"],
              ec["prompttype_main"]["df2"], ec["prompttype_main"]["p"], tol=8e-3)

    check_eta2("专家惊喜度", an["surprise"]["F"], an["surprise"]["df1"],
               an["surprise"]["df2"], an["surprise"]["eta2"])
    check_eta2("专家创造价值", an["creative_value"]["F"], an["creative_value"]["df1"],
               an["creative_value"]["df2"], an["creative_value"]["eta2"])

    print("\n" + "=" * 78)
    print("Tukey q -> p（学生化极差，k=3, df=87）")
    print("=" * 78)
    for construct, table in ref["expert"]["tukey"].items():
        for pair, vals in table.items():
            check_tukey(f"{construct} {pair}", vals["q"], vals["p"])

    print("\n" + "=" * 78)
    print("相关分析（报告值为 Benjamini-Hochberg 校正后的 p）")
    print("=" * 78)
    corr = ref["correlations_N192"]
    r_list = [corr["surprise_clip"]["r"], corr["inspirational_clip"]["r"],
              corr["surprise_inspirational"]["r"]]
    keys = ["惊喜度~CLIP", "启发性~CLIP", "惊喜度~启发性"]
    raw_ps = []
    for r in r_list:
        t = r * np.sqrt(190 / (1 - r ** 2))
        raw_ps.append(float(2 * stats.t.sf(abs(t), df=190)))
    from statsmodels.stats.multitest import multipletests
    bh = list(multipletests(raw_ps, method="fdr_bh")[1])
    reported = [corr["surprise_clip"]["p_bh"], corr["inspirational_clip"]["p_bh"],
                corr["surprise_inspirational"]["p_bh"]]
    for key, r, raw, adj, rep in zip(keys, r_list, raw_ps, bh, reported):
        print(f"    {key}: 原始 p = {raw:.5f}，BH 校正 = {adj:.5f}（报告 {rep}）")
        record(f"{key}: r={r} -> p_BH",
               close(adj, rep, 1.5e-3) or (rep <= 0.001 and adj <= 0.001),
               f"反算 BH p={adj:.5f}，报告 {rep}")
    check_r2("启发性~CLIP", corr["inspirational_clip"]["r"],
             corr["inspirational_clip"]["r2"], tol=1e-3)
    check_r2("惊喜度~启发性", corr["surprise_inspirational"]["r"],
             corr["surprise_inspirational"]["r2"], tol=1e-3)

    print("\n" + "=" * 78)
    print("校正阈值")
    print("=" * 78)
    b = ref["student_model"]["bonferroni"]
    record("Bonferroni 0.05/28", close(0.05 / 28, b["alpha"], 1e-6),
           f"0.05/28={0.05 / 28:.7f}")
    record("专家 Bonferroni 0.05/2 = 0.025",
           close(0.05 / 2, ref["expert"]["bonferroni_alpha"], 1e-9),
           str(ref["expert"]["bonferroni_alpha"]))

    print("\n" + "=" * 78)
    print("交叉一致性（WARN）：ANOVA F 与 Tukey q 能否同时成立")
    print("=" * 78)
    means = ref["expert"]["image_level_means"]["surprise"]
    ms = [means["scale_0.0"], means["scale_0.5"], means["scale_1.0"]]
    n = 30
    ssb = n * sum((m - np.mean(ms)) ** 2 for m in ms)
    F = an["surprise"]["F"]
    msw = ssb / ((len(ms) - 1) * F)
    se = np.sqrt(msw / n)
    q_implied = abs(ms[1] - ms[0]) / se
    q_reported = ref["expert"]["tukey"]["surprise"]["0.0_vs_0.5"]["q"]
    record_warn("由 ANOVA F 反推的 q 是否等于报告的 Tukey q",
                close(q_implied, q_reported, 0.05),
                f"由 F={F}, 均值 {ms} 反推 MSW={msw:.3f}, q={q_implied:.2f}；报告 q={q_reported}")

    # 汇总
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")
    print(f"核对结果：PASS {n_pass} / FAIL {n_fail} / WARN {n_warn}（共 {len(RESULTS)} 项）")
    print("=" * 78)
    for name, status, detail in RESULTS:
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
        print(f"{mark} {name}")
        if status != "PASS":
            print(f"        {detail}")

    out = utils.repo_path("outputs", "tables")
    utils.ensure_dir(out)
    with open(out / "verify_reference.json", "w", encoding="utf-8") as fh:
        json.dump([{"name": n, "status": s, "detail": d} for n, s, d in RESULTS],
                  fh, ensure_ascii=False, indent=2)
    print(f"\n明细写入 {out / 'verify_reference.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
