

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def contrast_f(res, terms: Sequence[str], df_denom: int | None = None) -> dict:
    """对指定参数名的联合线性约束做 F 检验。

    Args:
        res: statsmodels MixedLMResults（或任何含 params / cov_params 的拟合结果）。
        terms: 需要联合检验的参数名列表（每个参数单独约束为 0）。
        df_denom: 分母自由度；None 时用卡方近似（等价于 df_denom = 无穷）。

    Returns:
        {"F": 统计量, "q": 分子自由度, "df_denom": 分母自由度, "p": p 值}
    """
    names = list(res.params.index)
    idx = [names.index(t) for t in terms if t in names]
    q = len(idx)
    if q == 0:
        return {"F": float("nan"), "q": 0, "df_denom": df_denom, "p": float("nan")}

    R = np.zeros((q, len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0

    beta = np.asarray(res.params, dtype=float).reshape(-1, 1)
    cov = np.asarray(res.cov_params(), dtype=float)
    rb = R @ beta
    rvr = R @ cov @ R.T
    wald = float((rb.T @ np.linalg.pinv(rvr) @ rb).ravel()[0])
    F = wald / q
    if df_denom is None:
        p = float(stats.chi2.sf(wald, q))
    else:
        p = float(stats.f.sf(F, q, df_denom))
    return {"F": F, "q": q, "df_denom": df_denom, "p": p}


def terms_starting_with(res, prefixes: Sequence[str], require: Sequence[str] = ()) -> List[str]:
    """按前缀筛选参数名（用于选出某因子的哑变量）。"""
    names = list(res.params.index)
    out = []
    for n in names:
        if any(n.startswith(p) for p in prefixes) and all(r in n for r in require):
            out.append(n)
    return out


def _coef(res, name: str) -> float:
    params = res.params
    return float(params[name]) if name in params.index else 0.0


def _cell_mean_fn(res, factors: Dict[str, List[str]]):
    """按 patsy 的处理对照（treatment coding）编码，返回「水平组合 → 单元均值」函数。

    参考水平取 levels[0]；主效应参数名为 ``C(因子)[T.水平]``；
    两因子交互仅在双方都非参考水平时出现，参数名为
    ``C(因子1)[T.水平1]:C(因子2)[T.水平2]``。
    """
    names = list(factors)
    refs = {f: factors[f][0] for f in names}
    inter_name = None
    if len(names) == 2:
        f1, f2 = names
        inter_name = f"C({f1})[T.LEVEL1]:C({f2})[T.LEVEL2]"

    def cell_mean(combo: Dict[str, str]) -> float:
        mu = _coef(res, "Intercept")
        nonref = []
        for f in names:
            lv = combo[f]
            if lv != refs[f]:
                mu += _coef(res, f"C({f})[T.{lv}]")
                nonref.append((f, lv))
        if inter_name is not None and len(nonref) == 2:
            f1, l1 = nonref[0]
            f2, l2 = nonref[1]
            mu += _coef(res, inter_name.replace("LEVEL1", l1).replace("LEVEL2", l2))
        return mu

    return cell_mean


def factor_effect_f(res, data, factors: Dict[str, List[str]], term: Sequence[str],
                    df_denom: int | None = None) -> dict:
    """按 ANOVA 口径检验指定因子的效应。

    做法：用固定效应参数重建各水平组合的单元均值，再算该效应的平方和与自由度，
    除以模型的残差均方（``res.scale``）得到 F。等价于「效应均方 / 误差均方」，
    与均衡设计下的经典结果一致，但对协变量调整后的模型同样适用。
    """
    cell_mean = _cell_mean_fn(res, factors)
    names = list(factors)
    levels = [factors[f] for f in names]

    # 参数名自检：因子水平必须按 patsy 的字母序传入，否则参考水平会对不上，
    # 单元均值会被静默算错（这类错误不会报错，只会给出错误的 F 值）。
    params = set(res.params.index)
    for f in names:
        nonref = [lv for lv in factors[f][1:]]
        missing = [lv for lv in nonref if f"C({f})[T.{lv}]" not in params]
        if missing and len(nonref) > len(missing):
            raise ValueError(
                f"因子 {f} 的参数名与传入水平不匹配（缺失 {missing}）；"
                f"请按字母序传入水平，使 factors[{f}][0] 为参照水平。")

    grid = list(itertools.product(*levels))
    raw_counts = data.groupby(names).size().to_dict()
    # 单因子时 groupby 的键是标量而非元组
    counts = {k if isinstance(k, tuple) else (k,): v for k, v in raw_counts.items()}
    cells = []
    total_n = 0
    for combo in grid:
        c = dict(zip(names, combo))
        n = int(counts.get(tuple(combo), 0))
        cells.append({"combo": c, "n": n, "mean": cell_mean(c)})
        total_n += n
    if total_n == 0:
        return {"F": float("nan"), "q": 0, "df_denom": df_denom, "p": float("nan")}

    grand = sum(c["n"] * c["mean"] for c in cells) / total_n

    def marg(var: str, lv: str) -> float:
        num = sum(c["n"] * c["mean"] for c in cells if c["combo"][var] == lv)
        den = sum(c["n"] for c in cells if c["combo"][var] == lv)
        return num / den if den else float("nan")

    term = list(term)
    if len(term) == 1:
        var = term[0]
        ss = 0.0
        for lv in factors[var]:
            n_lv = sum(c["n"] for c in cells if c["combo"][var] == lv)
            ss += n_lv * (marg(var, lv) - grand) ** 2
        q = len(factors[var]) - 1
    elif len(term) == 2:
        f1, f2 = term
        ss = 0.0
        for c in cells:
            fitted = marg(f1, c["combo"][f1]) + marg(f2, c["combo"][f2]) - grand
            ss += c["n"] * (c["mean"] - fitted) ** 2
        q = (len(factors[f1]) - 1) * (len(factors[f2]) - 1)
    else:
        raise ValueError("factor_effect_f 目前只支持一因子或两因子效应")

    mse = float(res.scale)
    ms = ss / q if q else float("nan")
    F = ms / mse if mse and mse > 0 else float("nan")
    if df_denom is None:
        p = float(stats.f.sf(F, q, total_n - q))
    else:
        p = float(stats.f.sf(F, q, df_denom))
    return {"F": F, "q": q, "df_denom": df_denom, "p": p,
            "SS": ss, "MS": ms, "MSE": mse}

