"""Surprise-Sampler-Repro

论文《When Surprise Does Not Predict Value》的复现包。

按四个阶段组织：
    1. 构建任务清单（无 GPU）        src/generation/manifest.py
    2. 生成图像（需 CUDA）           src/generation/generate_diffusers.py
    3. 计算指标与整理评分           src/evaluation/*, src/rating/*
    4. 统计分析与绘图（无 GPU）     src/analysis/*
"""

__version__ = "1.0.0"
