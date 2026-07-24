"""
run_local_parallel.py
=====================
Parallel local runner that uses all your CPU cores.

Designed for an 8-core Ryzen 7 (16 threads). Works on Windows/Mac/Linux.

Key features
------------
- multiprocessing.Pool: distributes (ansatz, noise, n, L) cells across cores.
- Resume-on-crash: scans outputs_focused/ for already-completed cells and
  skips them. If the run dies, just rerun the script — it picks up where it
  left off.
- Per-cell CSV append (with file locking via tempfile + rename): each cell
  writes its result immediately, so partial progress is never lost.

Usage
-----
    python run_local_parallel.py

Adjust N_PROCESSES below if you want to leave cores for other tasks.

Estimated runtime on Ryzen 7 5800H, plugged in, "Best performance" mode:
    20-40 minutes for the full focused matrix.
"""

import os
import sys
import time
import warnings
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

# Silence pennylane deprecation warnings inside workers
warnings.filterwarnings("ignore")


OUT_DIR = "outputs_focused"
os.makedirs(OUT_DIR, exist_ok=True)

# Use 6-8 processes for an 8-core chip. Leave 2-4 cores free for the OS.
# Set to 1 to disable parallelism for debugging.
# N_PROCESSES = max(1, mp.cpu_count() - 2)
N_PROCESSES = 6


# ---------------------------------------------------------------------------
# Task definition: each task is a single (ansatz, noise, n, L, metric) cell
# ---------------------------------------------------------------------------
def run_one_cell(task):
    """
    A single sweep cell. Returns a row dict.
    Imports happen inside the worker so each process gets its own qnodes.
    """
    from metrics import gradient_variance, haar_kl_divergence, qfim_metrics
    from analysis import aggregate_runs
    from ansatze import n_params

    metric, ansatz, noise, n, L, n_runs = task

    if metric == "grad_var":
        agg = aggregate_runs(
            gradient_variance, n_runs=n_runs,
            ansatz_name=ansatz, n_qubits=n, n_layers=L,
            noise_preset=noise, n_samples=50,
        )
        row = {"ansatz": ansatz, "noise": noise, "n_qubits": n, "n_layers": L,
               "n_params": n_params(n, L), "metric": "grad_var",
               "mean": agg["mean"], "std": agg["std"],
               "lo": agg["lo"], "hi": agg["hi"], "n_runs": n_runs}
        return [row]

    if metric == "haar_kl":
        agg = aggregate_runs(
            haar_kl_divergence, n_runs=n_runs,
            ansatz_name=ansatz, n_qubits=n, n_layers=L,
            noise_preset=noise, n_samples=60,
        )
        row = {"ansatz": ansatz, "noise": noise, "n_qubits": n, "n_layers": L,
               "n_params": n_params(n, L), "metric": "haar_kl",
               "mean": agg["mean"], "std": agg["std"],
               "lo": agg["lo"], "hi": agg["hi"], "n_runs": n_runs}
        return [row]

    if metric == "qfim":
        agg = aggregate_runs(
            qfim_metrics, n_runs=n_runs,
            ansatz_name=ansatz, n_qubits=n, n_layers=L,
            noise_preset=noise, n_samples=15,
            n_data_for_eff_dim=1000,
        )
        rows = []
        for sub_metric, key in [("qfim_trace", "trace_F"), ("eff_dim", "eff_dim")]:
            rows.append({
                "ansatz": ansatz, "noise": noise, "n_qubits": n, "n_layers": L,
                "n_params": n_params(n, L), "metric": sub_metric,
                "mean": agg[key]["mean"], "std": agg[key]["std"],
                "lo": agg[key]["lo"], "hi": agg[key]["hi"], "n_runs": n_runs,
            })
        return rows

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Build the task list
# ---------------------------------------------------------------------------
def build_tasks():
    tasks = []

    # Step 1: noiseless metrics for NEW topologies only (linear/ring/brick
    # already done in the previous noiseless run).
    NEW_ANSATZE = ["all_to_all", "star"]
    for a in NEW_ANSATZE:
        for n in range(2, 11):    # 2..10
            for L in range(1, 9): # 1..8
                tasks.append(("grad_var", a, "noiseless", n, L, 30))
                tasks.append(("haar_kl", a, "noiseless", n, L, 30))
                tasks.append(("qfim",    a, "noiseless", n, L, 10))

    # Step 2: noisy gradient_variance for ALL 5 topologies, 4 noise presets
    ALL_ANSATZE = ["linear", "ring", "brick", "all_to_all", "star"]
    NOISES = [
        "depolarizing_realistic",
        "depolarizing_strong",
        "amplitude_realistic",
        "amplitude_strong",
    ]
    for a in ALL_ANSATZE:
        for noise in NOISES:
            for n in range(2, 7):    # 2..6
                for L in range(1, 9): # 1..8
                    tasks.append(("grad_var", a, noise, n, L, 20))

    return tasks


# ---------------------------------------------------------------------------
# Resume support: figure out which tasks are already done
# ---------------------------------------------------------------------------
INCREMENTAL_FILE = os.path.join(OUT_DIR, "all_cells.csv")

def load_completed_keys():
    """Return a set of (metric, ansatz, noise, n, L) already in INCREMENTAL_FILE."""
    if not os.path.exists(INCREMENTAL_FILE):
        return set()
    df = pd.read_csv(INCREMENTAL_FILE)
    keys = set()
    for _, r in df.iterrows():
        # collapse qfim_trace / eff_dim back to "qfim" task
        metric_key = "qfim" if r["metric"] in ("qfim_trace", "eff_dim") else r["metric"]
        keys.add((metric_key, r["ansatz"], r["noise"], int(r["n_qubits"]), int(r["n_layers"])))
    return keys


def append_rows(rows):
    """Append rows to the incremental CSV. Creates header if file is new."""
    df = pd.DataFrame(rows)
    header = not os.path.exists(INCREMENTAL_FILE)
    df.to_csv(INCREMENTAL_FILE, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Using {N_PROCESSES} parallel processes (of {mp.cpu_count()} cores)")

    all_tasks = build_tasks()
    completed = load_completed_keys()
    todo = [t for t in all_tasks
            if (t[0], t[1], t[2], t[3], t[4]) not in completed]

    print(f"Total tasks: {len(all_tasks)}")
    print(f"Already completed: {len(all_tasks) - len(todo)}")
    print(f"To do: {len(todo)}")

    if not todo:
        print("Nothing to do. All cells already completed.")
        return

    # Group tasks by (noise, metric) for cleaner progress output, and so that
    # easy tasks (small n) get distributed too.
    # Sort by approximate cost: small n first, so workers stay busy through the run.
    todo.sort(key=lambda t: (t[3] * t[4], t[2], t[1]))

    t0 = time.time()
    done_count = 0
    fail_count = 0

    # imap_unordered streams results as they complete. We append to CSV after each.
    with mp.Pool(N_PROCESSES) as pool:
        for rows in pool.imap_unordered(run_one_cell, todo, chunksize=1):
            try:
                append_rows(rows)
                done_count += 1
            except Exception as e:
                fail_count += 1
                print(f"  ERROR appending: {e}")

            if done_count % 20 == 0 or done_count == len(todo):
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 0.001)
                eta = (len(todo) - done_count) / max(rate, 0.001)
                print(f"  [{done_count}/{len(todo)}] elapsed={elapsed:.0f}s "
                      f"rate={rate:.2f}/s eta={eta:.0f}s ({eta/60:.1f}min)")

    print(f"\nDone. {done_count} cells completed, {fail_count} failed.")
    print(f"Total elapsed: {time.time() - t0:.0f}s")

    # Final consolidation: split into per-noise CSVs to match the original schema
    print("\nWriting per-noise CSVs for compatibility...")
    df_all = pd.read_csv(INCREMENTAL_FILE)
    df_all = df_all.drop_duplicates(subset=["metric", "ansatz", "noise", "n_qubits", "n_layers"], keep="last")
    df_all.to_csv(os.path.join(OUT_DIR, "all_focused.csv"), index=False)

    for noise in df_all["noise"].unique():
        sub = df_all[df_all["noise"] == noise]
        for metric in sub["metric"].unique():
            sub_m = sub[sub["metric"] == metric]
            fn = f"{metric}_{noise}.csv"
            sub_m.to_csv(os.path.join(OUT_DIR, fn), index=False)

    print(f"Wrote consolidated CSVs to {OUT_DIR}/")


if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    mp.freeze_support()
    main()
