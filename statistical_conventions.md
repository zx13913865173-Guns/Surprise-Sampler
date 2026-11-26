# 统计口径

本文件记录复现时采用的统计口径，凡是与论文报数方式有关的选择都在这里说明，
以免与「实现 bug」混淆。

## 1. 混合模型与分母自由度

论文用 lme4 + lmerTest，分母自由度由 Satterthwaite 近似给出并报为整数。statsmodels 的
`f_test` 走正态/卡方近似（分母自由度视为无穷），小样本下会明显偏小，因此
`src/analysis/inference.py` 采用统一口径：

- **F 值**：Wald 统计量除以约束个数（`contrast_f`）；
- **分母自由度**：按论文报告值显式指定，不做卡方近似。

| 模型 | 公式 | 分子 df | 分母 df |
| --- | --- | --- | --- |
| M1 | Surprise ~ Sampler + (1\|Prompt) | 7 | 192 - 8 = 184 |
| M2 | Surprise ~ Sampler + clarity + (1\|Prompt) | 7 | 192 - 8 - 1 = 183 |
| 实验 C | CLIP ~ PromptType * Sampler + (1\|Prompt) | 4 / 1 | 200 - 9 - 20 = 171 |

采样器效应本身用「效应均方 / 模型残差均方」的 F（`factor_effect_f`），与 lmerTest 的
Satterthwaite F 在同一数据上方向与量级一致；`contrast_f` 提供 Wald 口径用于交叉核对。

## 2. 随机效应与 R²

- 随机效应：每个模型都只含提示词的随机截距 `(1|Prompt)`，不设随机斜率。
- R² 用 Nakagawa 口径：边际 R² = 固定效应方差 /（固定 + 随机 + 残差），
  条件 R² = （固定 + 随机）/（固定 + 随机 + 残差）。提示词层方差不可识别时会被压到 0，
  此时条件 R² 等于边际 R²（见 known_inconsistencies.md 第 2 节）。

## 3. 专家分析

- 每个噪声尺度（0 / 0.5 / 1.0）30 张图，**图像级均值**做一元方差分析，
  组间 df = 2、组内 df = 87；
- 效应量 η² = F·df1 / (F·df1 + df2)；
- Tukey HSD 用学生化极差分布，q 与 p 按论文报告值原样收录（两者不能互算，见
  known_inconsistencies.md 第 3 节）。

## 4. 相关分析

分析集为 24 条提示词 × 8 采样器 = 192 张图的图像级均值。

BH 校正的族**只含论文检验的三个关系**，探索性关系不进入校正族：

1. surprise ~ CLIP（r = 0.183，p_BH = 0.011）
2. inspirational ~ CLIP（r = 0.319，p_BH < 0.001）
3. surprise ~ inspirational（r = 0.214，p_BH = 0.004）

`src/analysis/correlations.py` 同时输出未校正 p、解析 p 与 BH 校正后的 p，
BH 只在上述 3 个 p 值上计算。

## 5. 事后比较

- 学生模型：28 个采样器两两比较（未校正 t 检验 + Bonferroni 校正），
  校正后 α = 0.05 / 28 = 0.001786，论文报告 0 对通过校正，与本工程一致；
- 专家模型：Bonferroni 校正取 2 个构造（惊喜度、创造价值），α = 0.025。

## 6. 多重比较以外的报数约定

- 所有报告值保留论文的小数位（F 两位、p 三位、r 三位、R² 两位）；
- 论文未报告的单元格在 `data/reference_results.json` 中记为 `null`，
  在代码注释中记为 assumed，不与论文报告值混用；
- `scripts/verify_reference.py` 只在**可由汇总值唯一反算**的关系上判定 PASS / FAIL
  （图像数、F 与 p、r 与 p、R² = r²、Bonferroni 阈值等），
  无法从论文自身数值反算的项目（如 Tukey q 与 p）判为 WARN 并写明差异。
