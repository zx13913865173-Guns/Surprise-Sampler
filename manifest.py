
    python -m src.generation.manifest
    python -m src.generation.manifest --verify   # 只核对数量，不写文件
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from src import utils

# 每个实验使用的采样器（与论文 2.1 节一致）
ORIGINAL_SAMPLERS = ["euler_a", "dpmpp_2m_karras", "lms", "heun", "ddim"]
EXPERIMENT_A_SAMPLERS = ["dpm2_a", "kdpm2_ancestral", "unipc"]
EXPERIMENT_C_SAMPLERS = ORIGINAL_SAMPLERS

NOISE_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]

# 论文报告的计数，供断言用
EXPECTED_COUNTS = {
    "original": 2250,
    "experiment_a": 144,
    "experiment_b": 240,
    "experiment_c": 200,
    "total": 2834,
}


def _base_row(experiment: str, sampler_key: str, prompt_id: str, prompt: str,
              seed: int, defaults: Dict[str, Any]) -> Dict[str, Any]:
    samplers = utils.sampler_table()
    return {
        "image_id": f"{experiment}__{sampler_key}__{prompt_id}__s{seed}",
        "experiment": experiment,
        "sampler": sampler_key,
        "sampler_display": samplers[sampler_key]["display"],
        "sampler_family": samplers[sampler_key]["family"],
        "prompt_id": prompt_id,
        "prompt": prompt,
        "seed": seed,
        "noise_scale": None,
        "prompt_type": None,
        "cfg_scale": defaults["cfg_scale"],
        "steps": defaults["steps"],
        "resolution": list(defaults["resolution"]),
        "model_id": defaults["model_id"],
        "negative_prompt": defaults["negative_prompt"],
    }


def build_original() -> List[Dict[str, Any]]:
    """原始集：150 提示词 x 5 采样器 x 3 种子。"""
    defaults = utils.generation_defaults()
    prompts = utils.load_prompts_150()
    seeds = utils.load_experiment_config()["seeds"]["original_seeds"]
    analysis = set(utils.load_analysis_prompts_24())
    rows = []
    for i, prompt in enumerate(prompts, start=1):
        pid = f"p{i:03d}"
        in_analysis = prompt in analysis
        for skey in ORIGINAL_SAMPLERS:
            for seed in seeds:
                row = _base_row("original", skey, pid, prompt, seed, defaults)
                row["in_analysis"] = in_analysis
                rows.append(row)
    return rows


def build_experiment_a() -> List[Dict[str, Any]]:
    """实验 A：新增 3 个祖先/确定性采样器，24 提示词 x 2 种子。"""
    defaults = utils.generation_defaults()
    prompts = utils.load_analysis_prompts_24()
    seeds = utils.load_experiment_config()["seeds"]["added_seeds"]
    rows = []
    for i, prompt in enumerate(prompts, start=1):
        pid = f"a{i:02d}"
        for skey in EXPERIMENT_A_SAMPLERS:
            for seed in seeds:
                row = _base_row("experiment_a", skey, pid, prompt, seed, defaults)
                row["in_analysis"] = True
                rows.append(row)
    return rows


def build_experiment_b() -> List[Dict[str, Any]]:
    """实验 B：Euler a 框架内的连续噪声尺度操纵，24 提示词 x 5 尺度 x 2 种子。"""
    defaults = utils.generation_defaults()
    prompts = utils.load_analysis_prompts_24()
    seeds = utils.load_experiment_config()["seeds"]["added_seeds"]
    expert_scales = {0.0, 0.5, 1.0}
    # 专家评分子集：每个尺度 24 张 seed-42 图 + 6 张 seed-43 图（提示词 1/5/9/13/17/21）
    expert_extra_prompts = {"b01", "b05", "b09", "b13", "b17", "b21"}
    rows = []
    for i, prompt in enumerate(prompts, start=1):
        pid = f"b{i:02d}"
        for scale in NOISE_SCALES:
            for seed in seeds:
                row = _base_row("experiment_b", "euler_a", pid, prompt, seed, defaults)
                row["noise_scale"] = scale
                row["in_analysis"] = True
                # 进入专家评分：尺度属于 {0, 0.5, 1.0}，且为 seed 42 全部图 + seed 43 的 6 个提示词
                row["expert_rated"] = (
                    scale in expert_scales
                    and (seed == 42 or (seed == 43 and pid in expert_extra_prompts))
                )
                rows.append(row)
    return rows


def build_experiment_c() -> List[Dict[str, Any]]:
    """实验 C：20 基础提示词 x 2 类型 x 5 采样器 x 1 种子。"""
    defaults = utils.generation_defaults()
    items = utils.load_prompt_types_20()
    seed = utils.load_experiment_config()["seeds"]["main_rating_seed"]
    rows = []
    for i, item in enumerate(items, start=1):
        pid = f"c{i:02d}"
        for ptype in ("embodied_sensory", "relational_process"):
            for skey in EXPERIMENT_C_SAMPLERS:
                row = _base_row("experiment_c", skey, pid, item[ptype], seed, defaults)
                row["prompt_type"] = ptype
                row["base_prompt"] = item["base"]
                row["in_analysis"] = True
                rows.append(row)
    return rows


BUILDERS = {
    "original": build_original,
    "experiment_a": build_experiment_a,
    "experiment_b": build_experiment_b,
    "experiment_c": build_experiment_c,
}


def build_all() -> Dict[str, List[Dict[str, Any]]]:
    return {name: fn() for name, fn in BUILDERS.items()}


def verify_counts(sets: Dict[str, List[Dict[str, Any]]], verbose: bool = True) -> bool:
    ok = True
    total = 0
    for name, rows in sets.items():
        n = len(rows)
        total += n
        expected = EXPECTED_COUNTS[name]
        flag = "PASS" if n == expected else "FAIL"
        if n != expected:
            ok = False
        if verbose:
            print(f"  {name:<14} {n:>5} / {expected:<5} {flag}")
    flag = "PASS" if total == EXPECTED_COUNTS["total"] else "FAIL"
    if total != EXPECTED_COUNTS["total"]:
        ok = False
    if verbose:
        print(f"  {'total':<14} {total:>5} / {EXPECTED_COUNTS['total']:<5} {flag}")

    # 专家评分图像数必须为 90
    b_rows = sets["experiment_b"]
    expert = [r for r in b_rows if r.get("expert_rated")]
    flag = "PASS" if len(expert) == 90 else "FAIL"
    if len(expert) != 90:
        ok = False
    if verbose:
        print(f"  {'expert_rated':<14} {len(expert):>5} / {90:<5} {flag}")

    # 分析集提示词应为 24，未分析应为 126
    orig_prompts = {r["prompt"] for r in sets["original"]}
    orig_analysis = {r["prompt"] for r in sets["original"] if r["in_analysis"]}
    if verbose:
        print(f"  analysis prompts {len(orig_analysis)} / 24 "
              f"{'PASS' if len(orig_analysis) == 24 else 'FAIL'}")
        print(f"  pool prompts     {len(orig_prompts)} / 150 "
              f"{'PASS' if len(orig_prompts) == 150 else 'FAIL'}")
    if len(orig_analysis) != 24 or len(orig_prompts) != 150:
        ok = False
    return ok


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建生成任务清单")
    parser.add_argument("--verify", action="store_true", help="只核对数量，不写文件")
    parser.add_argument("--out", default=str(utils.output_dir("manifests")),
                        help="清单输出目录")
    args = parser.parse_args(argv)

    print("构建任务清单 ...")
    sets = build_all()
    ok = verify_counts(sets)

    if not args.verify:
        from pathlib import Path
        out = Path(args.out)
        for name, rows in sets.items():
            n = utils.write_jsonl(rows, out / f"{name}.jsonl")
            print(f"  写入 {out / (name + '.jsonl')} ({n} 行)")

    if ok:
        print("计数核对：全部通过。")
        return 0
    print("计数核对：存在不一致，请检查 prompts/ 与 configs/。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
