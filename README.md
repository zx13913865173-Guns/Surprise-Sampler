# When Surprise and Creative Value Show Different Orderings: Computational Metrics and Human Judgments of Diffusion-Generated Art

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official code, generated image datasets, prompt lists, and evaluation scripts for the paper: **"When Surprise and Creative Value Show Different Orderings: Computational Metrics and Human Judgments of Diffusion-Generated Art"** by Dmitry Galkin and Xin Zhang (Tomsk State University).

## 📖 Overview

Computational metrics (FID, CLIP Score, LPIPS, etc.) are routinely used to evaluate and rank diffusion models. However, their relationship to human judgments of creative value remains undiagnosed at the sampler level. 

In this study, we compared **eight SDXL samplers** across 150 abstract prompts, generating a total of **2,834 images**. By combining computational metrics with human ratings (from both digital-art graduate students and professional artists/curators), we found that **higher perceived surprise does not uniformly correspond to higher creative value**. The computational metrics and human ratings provided different orderings across the tested configurations.

## 📂 Repository Structure

*   `prompts/`: Contains the 150 abstract concept prompts used in the main experiment, and the 20 prompts used in Experiment C.
*   `scripts/`: Python scripts for image generation (using `Diffusers`), computational metric calculation (FID, CLIP, LPIPS, NIQE, BRISQUE, DreamSim), and statistical analysis (Mixed models, ANOVA).
*   `data/`:
    *   `student_ratings.csv`: Image-level means for the 192 rated images (4 raters).
    *   `expert_ratings.csv`: Image-level means for the 90 rated images across noise scales 0, 0.5, and 1.0 (8 experts).
    *   `computational_metrics.csv`: Calculated metrics for all evaluated image sets.
*   `figures/`: High-resolution figures included in the paper (e.g., alignment matrices, rating distributions).

## 🛠 Experimental Design Summary

The study utilized `SDXL Base 1.0` via `Diffusers v0.25.0` (CFG scale = 7.5, steps = 30, resolution = 1024x1024).

| Experiment | Description | Configuration | Total Images |
| :--- | :--- | :--- | :--- |
| **Original Set** | Baseline samplers | 150 prompts × 5 samplers × 3 seeds | 2,250 |
| **Experiment A** | Added samplers | 24 prompts × 3 samplers × 2 seeds | 144 |
| **Experiment B** | Continuous noise scale | 24 prompts × 5 scales (0-1.0) × 2 seeds | 240 |
| **Experiment C** | Prompt type exploration | 20 prompts × 2 types × 5 samplers × 1 seed | 200 |
| **Total** | | | **2,834** |

*Note: Human ratings were collected on a subset of seed-42 images (192 images for students; 90 images for experts).*

## 🚀 Quick Start / Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zx13913865173-Guns/Surprise-Sampler.git
   cd Surprise-Sampler
