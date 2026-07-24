"""
analysis.py
===========
Statistical analysis utilities:
    - bootstrap_ci      : bootstrap confidence intervals for any statistic.
    - fit_power_law     : log-linear fit y ~ x^alpha with bootstrap CI on alpha.
    - fit_exp_decay     : log-linear fit y ~ exp(-beta * x) with bootstrap CI on beta.
    - aggregate_runs    : repeated metric evaluation with mean, std, 95% CI.
"""

import numpy as np
from scipy.stats import linregress


def bootstrap_ci(values, statistic=np.mean, n_boot=1000, ci=95, seed=0):
    """Return (point_estimate, lo, hi) at the requested CI level."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(values[idx])
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return point, lo, hi


def fit_power_law(x, y, n_boot=1000, seed=0):
    """
    Fit y = c * x^alpha by linear regression on log-log data.
    Returns dict with alpha (slope), intercept (log c), R^2, alpha_lo, alpha_hi (95% CI).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"alpha": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "alpha_lo": float("nan"), "alpha_hi": float("nan")}

    logx, logy = np.log(x), np.log(y)
    res = linregress(logx, logy)
    alpha = float(res.slope)
    intercept = float(res.intercept)
    r2 = float(res.rvalue ** 2)

    rng = np.random.default_rng(seed)
    n = len(x)
    boot_alphas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            boot_alphas[b] = alpha
            continue
        r = linregress(logx[idx], logy[idx])
        boot_alphas[b] = r.slope
    alpha_lo = float(np.percentile(boot_alphas, 2.5))
    alpha_hi = float(np.percentile(boot_alphas, 97.5))

    return {"alpha": alpha, "intercept": intercept, "r2": r2,
            "alpha_lo": alpha_lo, "alpha_hi": alpha_hi}


def fit_exp_decay(x, y, n_boot=1000, seed=0):
    """
    Fit y = c * exp(-beta * x) by linear regression on log y vs x.
    Returns dict with beta, intercept, R^2, beta_lo, beta_hi.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"beta": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "beta_lo": float("nan"), "beta_hi": float("nan")}

    logy = np.log(y)
    res = linregress(x, logy)
    beta = float(-res.slope)
    intercept = float(res.intercept)
    r2 = float(res.rvalue ** 2)

    rng = np.random.default_rng(seed)
    n = len(x)
    boot_betas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            boot_betas[b] = beta
            continue
        r = linregress(x[idx], logy[idx])
        boot_betas[b] = -r.slope
    beta_lo = float(np.percentile(boot_betas, 2.5))
    beta_hi = float(np.percentile(boot_betas, 97.5))

    return {"beta": beta, "intercept": intercept, "r2": r2,
            "beta_lo": beta_lo, "beta_hi": beta_hi}


def aggregate_runs(metric_fn, n_runs=30, base_seed=0, **metric_kwargs):
    """
    Run metric_fn n_runs times with different seeds, return mean, std, 95% CI.
    metric_fn must accept a `seed` keyword.
    """
    vals = []
    for r in range(n_runs):
        v = metric_fn(seed=base_seed + r, **metric_kwargs)
        if isinstance(v, dict):
            vals.append(v)
        else:
            vals.append(float(v))

    if isinstance(vals[0], dict):
        keys = vals[0].keys()
        out = {}
        for k in keys:
            arr = np.array([v[k] for v in vals], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out[k] = {"mean": float("nan"), "std": float("nan"),
                          "lo": float("nan"), "hi": float("nan")}
                continue
            mean, lo, hi = bootstrap_ci(arr, statistic=np.mean, n_boot=1000, seed=base_seed)
            out[k] = {"mean": mean, "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                      "lo": lo, "hi": hi}
        return out

    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    mean, lo, hi = bootstrap_ci(arr, statistic=np.mean, n_boot=1000, seed=base_seed)
    return {"mean": mean,
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "lo": lo, "hi": hi, "n": int(arr.size)}
