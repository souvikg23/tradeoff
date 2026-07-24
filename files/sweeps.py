"""
sweeps.py
=========
Top-level sweep runners. Each function loops over (ansatz, noise_preset, n, L)
and calls the metric with aggregate_runs to get mean + 95% CI.

Output: pandas DataFrame with columns
    ansatz, noise, n_qubits, n_layers, n_params, metric, mean, std, lo, hi, n_runs
"""

import time
import pandas as pd
import numpy as np

from metrics import (
    haar_kl_divergence, effective_kernel_rank,
    gradient_variance, function_output_variance, qfim_metrics,
)
from analysis import aggregate_runs
from ansatze import n_params


def _row(ansatz, noise, n, L, metric_name, agg, n_runs, extra=None):
    row = {
        "ansatz": ansatz,
        "noise": noise,
        "n_qubits": n,
        "n_layers": L,
        "n_params": n_params(n, L),
        "metric": metric_name,
        "mean": agg["mean"],
        "std": agg["std"],
        "lo": agg["lo"],
        "hi": agg["hi"],
        "n_runs": n_runs,
    }
    if extra:
        row.update(extra)
    return row


def sweep_haar(ansatze, noises, widths, depths, n_runs=10, n_samples=60, verbose=True):
    rows = []
    t0 = time.time()
    for a in ansatze:
        for noise in noises:
            for n in widths:
                for L in depths:
                    if verbose:
                        print(f"[Haar] ansatz={a} noise={noise} n={n} L={L} "
                              f"elapsed={time.time()-t0:.1f}s")
                    agg = aggregate_runs(
                        haar_kl_divergence, n_runs=n_runs,
                        ansatz_name=a, n_qubits=n, n_layers=L,
                        noise_preset=noise, n_samples=n_samples,
                    )
                    rows.append(_row(a, noise, n, L, "haar_kl", agg, n_runs))
    return pd.DataFrame(rows)


def sweep_kernel(ansatze, noises, widths, depths, n_runs=10, n_data=20, verbose=True):
    rows = []
    t0 = time.time()
    for a in ansatze:
        for noise in noises:
            for n in widths:
                for L in depths:
                    if verbose:
                        print(f"[Kernel] ansatz={a} noise={noise} n={n} L={L} "
                              f"elapsed={time.time()-t0:.1f}s")
                    agg = aggregate_runs(
                        effective_kernel_rank, n_runs=n_runs,
                        ansatz_name=a, n_qubits=n, n_layers=L,
                        noise_preset=noise, n_data=n_data,
                    )
                    rows.append(_row(a, noise, n, L, "kernel_rank", agg, n_runs))
    return pd.DataFrame(rows)


def sweep_gradient(ansatze, noises, widths, depths, n_runs=10, n_samples=50, verbose=True):
    rows = []
    t0 = time.time()
    for a in ansatze:
        for noise in noises:
            for n in widths:
                for L in depths:
                    if verbose:
                        print(f"[Grad] ansatz={a} noise={noise} n={n} L={L} "
                              f"elapsed={time.time()-t0:.1f}s")
                    agg = aggregate_runs(
                        gradient_variance, n_runs=n_runs,
                        ansatz_name=a, n_qubits=n, n_layers=L,
                        noise_preset=noise, n_samples=n_samples,
                    )
                    rows.append(_row(a, noise, n, L, "grad_var", agg, n_runs))
    return pd.DataFrame(rows)


def sweep_function(ansatze, noises, widths, depths, n_runs=10, n_data=30, verbose=True):
    rows = []
    t0 = time.time()
    for a in ansatze:
        for noise in noises:
            for n in widths:
                for L in depths:
                    if verbose:
                        print(f"[Func] ansatz={a} noise={noise} n={n} L={L} "
                              f"elapsed={time.time()-t0:.1f}s")
                    agg = aggregate_runs(
                        function_output_variance, n_runs=n_runs,
                        ansatz_name=a, n_qubits=n, n_layers=L,
                        noise_preset=noise, n_data=n_data,
                    )
                    rows.append(_row(a, noise, n, L, "func_var", agg, n_runs))
    return pd.DataFrame(rows)


def sweep_qfim(ansatze, noises, widths, depths, n_runs=5, n_samples=15,
               n_data_for_eff_dim=200, verbose=True):
    """
    QFIM is the most expensive metric. Default n_runs=5 and n_samples=15 to keep
    it tractable. The metric_tensor call itself averages over many circuit
    evaluations internally.
    """
    rows = []
    t0 = time.time()
    for a in ansatze:
        for noise in noises:
            for n in widths:
                for L in depths:
                    if verbose:
                        print(f"[QFIM] ansatz={a} noise={noise} n={n} L={L} "
                              f"elapsed={time.time()-t0:.1f}s")
                    agg = aggregate_runs(
                        qfim_metrics, n_runs=n_runs,
                        ansatz_name=a, n_qubits=n, n_layers=L,
                        noise_preset=noise, n_samples=n_samples,
                        n_data_for_eff_dim=n_data_for_eff_dim,
                    )
                    rows.append(_row(a, noise, n, L, "qfim_trace", agg["trace_F"], n_runs))
                    rows.append(_row(a, noise, n, L, "eff_dim", agg["eff_dim"], n_runs))
    return pd.DataFrame(rows)


def add_param_efficiency(df_metric, metric_to_use="haar_kl", invert=True):
    """
    Add a parameter-efficiency column to a metric DataFrame.

    For metrics where lower = better (haar_kl), we use 1/(metric * params).
    For metrics where higher = better (kernel_rank, eff_dim), we use metric / params.
    """
    sub = df_metric[df_metric["metric"] == metric_to_use].copy()
    if invert:
        sub["param_efficiency"] = 1.0 / (sub["mean"] * sub["n_params"] + 1e-12)
    else:
        sub["param_efficiency"] = sub["mean"] / sub["n_params"]
    return sub
