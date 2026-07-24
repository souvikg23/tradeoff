"""
run_gap_fill.py
===============
Fills the only remaining gap in the dataset:
    kernel_rank and func_var for all_to_all and star topologies.
    Noiseless only. n=2..10, L=1..8, 30 runs each.

These two metrics require data-encoding variants of the ansatz.
We add those inline here rather than touching ansatze.py.

Estimated runtime: 10-20 minutes on Ryzen 7 with 6 parallel processes.
Output: outputs_gapfill/gapfill.csv (merge into master_dataset.csv when done)
"""

import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


OUT_DIR = "outputs_gapfill"
os.makedirs(OUT_DIR, exist_ok=True)
INCREMENTAL_FILE = os.path.join(OUT_DIR, "gapfill_cells.csv")
N_PROCESSES = max(1, mp.cpu_count() - 2)


# ---------------------------------------------------------------------------
# Data-encoding variants for all_to_all and star
# These are defined here so the worker can import them cleanly
# ---------------------------------------------------------------------------
def _add_data_encoding_variants():
    """Register data-encoding versions of all_to_all and star in ansatze.py's dict."""
    import pennylane as qml
    from ansatze import ANSATZE_WITH_DATA

    def ansatz_all_to_all_with_data(x, params, n_qubits, n_layers):
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            for q in range(n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])

    def ansatz_star_with_data(x, params, n_qubits, n_layers):
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            for q in range(n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            for j in range(1, n_qubits):
                qml.CNOT(wires=[0, j])

    ANSATZE_WITH_DATA["all_to_all"] = ansatz_all_to_all_with_data
    ANSATZE_WITH_DATA["star"] = ansatz_star_with_data


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
def run_one_cell(task):
    import warnings
    warnings.filterwarnings("ignore")

    _add_data_encoding_variants()

    from metrics import effective_kernel_rank, function_output_variance
    from analysis import aggregate_runs
    from ansatze import n_params

    metric, ansatz, n, L, n_runs = task

    if metric == "kernel_rank":
        agg = aggregate_runs(
            effective_kernel_rank, n_runs=n_runs,
            ansatz_name=ansatz, n_qubits=n, n_layers=L,
            noise_preset="noiseless", n_data=20,
        )
    elif metric == "func_var":
        agg = aggregate_runs(
            function_output_variance, n_runs=n_runs,
            ansatz_name=ansatz, n_qubits=n, n_layers=L,
            noise_preset="noiseless", n_data=30,
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "ansatz": ansatz, "noise": "noiseless",
        "n_qubits": n, "n_layers": L,
        "n_params": n_params(n, L), "metric": metric,
        "mean": agg["mean"], "std": agg["std"],
        "lo": agg["lo"], "hi": agg["hi"], "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_completed():
    if not os.path.exists(INCREMENTAL_FILE):
        return set()
    df = pd.read_csv(INCREMENTAL_FILE)
    return set(
        (r["metric"], r["ansatz"], int(r["n_qubits"]), int(r["n_layers"]))
        for _, r in df.iterrows()
    )


def append_row(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(INCREMENTAL_FILE)
    df.to_csv(INCREMENTAL_FILE, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tasks = []
    for metric in ["kernel_rank", "func_var"]:
        for ansatz in ["all_to_all", "star"]:
            for n in range(2, 11):    # 2..10
                for L in range(1, 9): # 1..8
                    tasks.append((metric, ansatz, n, L, 30))

    completed = load_completed()
    todo = [t for t in tasks if (t[0], t[1], t[3], t[4]) not in completed]

    print(f"Gap-fill: {len(tasks)} total cells, {len(todo)} to run")
    print(f"Using {N_PROCESSES} processes")

    if not todo:
        print("Nothing to do — all cells already completed.")
        return

    t0 = time.time()
    done = 0

    with mp.Pool(N_PROCESSES) as pool:
        for row in pool.imap_unordered(run_one_cell, todo, chunksize=1):
            append_row(row)
            done += 1
            if done % 10 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.001)
                eta = (len(todo) - done) / max(rate, 0.001)
                print(f"  [{done}/{len(todo)}] elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s ({eta/60:.1f}min)")

    # Final CSV
    df = pd.read_csv(INCREMENTAL_FILE)
    df.to_csv(os.path.join(OUT_DIR, "gapfill.csv"), index=False)
    print(f"\nDone. {done} cells. Saved to {OUT_DIR}/gapfill.csv")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
