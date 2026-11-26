"""公共工具：路径解析、配置加载、提示词读取。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

# 仓库根目录 = 本文件所在目录（src/）的上一级
REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "configs"
PROMPT_DIR = REPO_ROOT / "prompts"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"


def repo_path(*parts: str) -> Path:
    """返回仓库内的绝对路径。"""
    p = REPO_ROOT.joinpath(*parts)
    return p


@lru_cache(maxsize=None)
def load_yaml(name: str) -> Dict[str, Any]:
    """按文件名加载 configs/ 下的 YAML（带缓存）。"""
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_experiment_config() -> Dict[str, Any]:
    return load_yaml("experiment.yaml")


def load_sampler_config() -> Dict[str, Any]:
    return load_yaml("samplers.yaml")


@lru_cache(maxsize=None)
def load_prompts_150() -> List[str]:
    data = load_json(PROMPT_DIR / "abstract_prompts_150.json")
    return list(data["prompts"])


@lru_cache(maxsize=None)
def load_analysis_prompts_24() -> List[str]:
    data = load_json(PROMPT_DIR / "analysis_prompts_24.json")
    return list(data["prompts"])


@lru_cache(maxsize=None)
def load_prompt_types_20() -> List[Dict[str, str]]:
    data = load_json(PROMPT_DIR / "prompt_types_20.json")
    return list(data["items"])


def load_reference_results() -> Dict[str, Any]:
    return load_json(DATA_DIR / "reference_results.json")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> int:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def output_dir(kind: str) -> Path:
    """outputs/<kind> 目录，按需创建。"""
    return ensure_dir(OUTPUT_DIR / kind)


# 论文统一的负向提示词与生成参数（单一真源）
def generation_defaults() -> Dict[str, Any]:
    cfg = load_experiment_config()["generation"]
    return {
        "model_id": cfg["model_id"],
        "cfg_scale": cfg["cfg_scale"],
        "steps": cfg["num_inference_steps"],
        "resolution": tuple(cfg["resolution"]),
        "negative_prompt": cfg["negative_prompt"],
    }


def sampler_table() -> Dict[str, Dict[str, Any]]:
    """返回 sampler_key -> {display, family, scheduler, kwargs, groups}。"""
    return load_sampler_config()["samplers"]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
