"""
run_tqe_upgrade.py
==================
Experiments needed to upgrade the Access paper to TQE level.

What this adds
--------------
1. Heavy-hex topology (approximating IBM Falcon/Eagle hardware):
   - Full noiseless sweep: n=2..10, L=1..8, all 6 metrics, 30 runs
   - Noisy gradient sweep: n=2..6, L=1..8, 4 noise presets, 20 runs

2. Alpha(L) analysis:
   - For ALL 6 topologies, fit alpha(L) at every depth L=1..8
   - This becomes a new figure in the TQE paper
   - Uses existing master_dataset.csv + new heavy-hex data

All results go to outputs_tqe/ so they don't overwrite Access data.

Estimated runtime on Ryzen 7 (parallel, 6 cores):
    ~30-50 minutes for noiseless, ~20 minutes for noisy gradient
    Total: ~1 hour

Usage:
    python run_tqe_upgrade.py
"""

import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "outputs_tqe"
os.makedirs(OUT_DIR, exist_ok=True)
INCREMENTAL_FILE = os.path.join(OUT_DIR, "heavy_hex_cells.csv")
N_PROCESSES = max(1, mp.cpu_count() - 2)

# -----------------------------------------------------------------------
# Worker (same pattern as run_local_parallel.py)
# -----------------------------------------------------------------------
def run_one_cell(task):
    import warnings; warnings.filterwarnings("ignore")
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from metrics import (gradient_variance, haar_kl_divergence,
                         qfim_metrics, effective_kernel_rank,
                         function_output_variance)
    from analysis import aggregate_runs
    from ansatze import n_params

    metric, ansatz, noise, n, L, n_runs = task

    kwargs = dict(ansatz_name=ansatz, n_qubits=n, n_layers=L,
                  noise_preset=noise)

    if metric == "grad_var":
        agg = aggregate_runs(gradient_variance, n_runs=n_runs,
                             n_samples=50, **kwargs)
        return [_row(metric, ansatz, noise, n, L, agg, n_runs)]

    if metric == "haar_kl":
        agg = aggregate_runs(haar_kl_divergence, n_runs=n_runs,
                             n_samples=60, **kwargs)
        return [_row(metric, ansatz, noise, n, L, agg, n_runs)]

    if metric == "kernel_rank":
        # heavy_hex does not yet have a data-encoding variant
        # skip silently
        return []

    if metric == "func_var":
        return []

    if metric == "qfim":
        agg = aggregate_runs(qfim_metrics, n_runs=n_runs,
                             n_samples=15, n_data_for_eff_dim=1000, **kwargs)
        rows = []
        for sub_m, key in [("qfim_trace","trace_F"), ("eff_dim","eff_dim")]:
            rows.append(_row(sub_m, ansatz, noise, n, L,
                             agg[key], n_runs))
        return rows

    raise ValueError(f"Unknown metric: {metric}")


def _row(metric, ansatz, noise, n, L, agg, n_runs):
    from ansatze import n_params
    if isinstance(agg, dict):
        mean, std, lo, hi = agg["mean"], agg["std"], agg["lo"], agg["hi"]
    else:
        mean, std, lo, hi = agg, 0.0, agg, agg
    return {"ansatz": ansatz, "noise": noise,
            "n_qubits": n, "n_layers": L,
            "n_params": n_params(n, L), "metric": metric,
            "mean": mean, "std": std, "lo": lo, "hi": hi,
            "n_runs": n_runs}


# -----------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------
def load_completed():
    if not os.path.exists(INCREMENTAL_FILE):
        return set()
    df = pd.read_csv(INCREMENTAL_FILE)
    return set(
        (r["metric"] if r["metric"] not in ("qfim_trace","eff_dim")
         else "qfim",
         r["ansatz"], r["noise"],
         int(r["n_qubits"]), int(r["n_layers"]))
        for _, r in df.iterrows()
    )


def append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(INCREMENTAL_FILE)
    df.to_csv(INCREMENTAL_FILE, mode="a", header=header, index=False)


# -----------------------------------------------------------------------
# Build task list
# -----------------------------------------------------------------------
def build_tasks():
    tasks = []

    # --- Step 1: Noiseless full sweep for heavy_hex ---
    NOISELESS_METRICS = ["grad_var", "haar_kl", "qfim"]
    for metric in NOISELESS_METRICS:
        n_runs = 10 if metric == "qfim" else 30
        for n in range(2, 11):      # 2..10
            for L in range(1, 9):   # 1..8
                tasks.append((metric, "heavy_hex", "noiseless", n, L, n_runs))

    # --- Step 2: Noisy gradient sweep for heavy_hex ---
    NOISY_PRESETS = ["depolarizing_realistic", "depolarizing_strong",
                     "amplitude_realistic",    "amplitude_strong"]
    for noise in NOISY_PRESETS:
        for n in range(2, 7):       # 2..6 (density matrix cap)
            for L in range(1, 9):
                tasks.append(("grad_var", "heavy_hex", noise, n, L, 20))

    return tasks


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print(f"Using {N_PROCESSES} parallel processes (of {mp.cpu_count()} cores)")

    all_tasks = build_tasks()
    completed = load_completed()
    todo = [t for t in all_tasks
            if (t[0] if t[0] not in ("qfim_trace","eff_dim") else "qfim",
                t[1], t[2], t[3], t[4]) not in completed]

    print(f"Total tasks:       {len(all_tasks)}")
    print(f"Already done:      {len(all_tasks) - len(todo)}")
    print(f"To run:            {len(todo)}")

    if not todo:
        print("All done — nothing to run.")
        _finalize()
        return

    # Sort cheapest first so workers stay busy throughout
    todo.sort(key=lambda t: (t[3] * t[4]))

    t0 = time.time()
    done = 0; failed = 0

    with mp.Pool(N_PROCESSES) as pool:
        for rows in pool.imap_unordered(run_one_cell, todo, chunksize=1):
            try:
                append_rows(rows)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  ERROR: {e}")
            if done % 20 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.001)
                eta  = (len(todo) - done) / max(rate, 0.001)
                print(f"  [{done}/{len(todo)}]  "
                      f"elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  "
                      f"eta={eta:.0f}s ({eta/60:.1f}min)")

    print(f"\nDone. {done} cells completed, {failed} failed.")
    _finalize()


def _finalize():
    if not os.path.exists(INCREMENTAL_FILE):
        print("No data to finalize.")
        return

    df_new = pd.read_csv(INCREMENTAL_FILE)
    df_new = df_new.drop_duplicates(
        subset=["metric","ansatz","noise","n_qubits","n_layers"],
        keep="last")

    # Merge with existing Access data
    master_path = "master_dataset.csv"
    if os.path.exists(master_path):
        df_old = pd.read_csv(master_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(
            subset=["metric","ansatz","noise","n_qubits","n_layers"],
            keep="last")
    else:
        df_all = df_new

    out_path = os.path.join(OUT_DIR, "master_dataset_tqe.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\nMerged dataset: {len(df_all)} rows -> {out_path}")
    print(f"Topologies now: {sorted(df_all['ansatz'].unique())}")

    # Also write heavy_hex only for inspection
    df_new.to_csv(os.path.join(OUT_DIR, "heavy_hex_all.csv"), index=False)
    print(f"Heavy-hex only: {len(df_new)} rows -> "
          f"{OUT_DIR}/heavy_hex_all.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
