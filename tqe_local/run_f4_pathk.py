"""
run_f4_pathk.py
===============
Runs grad_var for 5 path-k topologies to support Finding 4.

Path-k topology: linear CNOT chain + one extra CNOT(0, k).
k values tested: 2, 3, 4, 5, and n//2 (adaptive).

Why: to make F4 (predictive regression of asymmetry ratio from
d(alpha)/dL) defensible at TQE level. Current data has only
3 distinct x-values. Path-k adds 5 more, giving 8 total.

What is run:
    - Noiseless only (default.qubit, fast)
    - n = 2..10, L = 1..8, 30 runs per cell
    - 5 topologies: path_2, path_3, path_4, path_5, path_half
    - Metric: grad_var only

Estimated time: 3-6 minutes on Ryzen 7 (6 cores, plugged in).
GPU: not needed for this sweep (n<=10 noiseless is fast on CPU).

Output: outputs_f4/pathk_cells.csv and outputs_f4/pathk_all.csv
"""

import os, time, sys, warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT_DIR = "outputs_f4"
os.makedirs(OUT_DIR, exist_ok=True)
INCREMENTAL_FILE = os.path.join(OUT_DIR, "pathk_cells.csv")
N_PROCESSES = max(1, mp.cpu_count() - 2)

# k values. "half" means k = n // 2 (depends on n, handled in worker)
K_VALUES = [2, 3, 4, 5, "half"]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_one_cell(task):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pennylane as qml
    from pennylane import numpy as pnp

    k_label, n, L, n_runs, base_seed = task

    # Resolve actual k value
    k = n // 2 if k_label == "half" else int(k_label)
    # Clamp: k must be >= 2 and < n
    k = max(2, min(k, n - 1))

    # Build the circuit
    dev = qml.device("default.qubit", wires=n)
    n_p = n * L

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        idx = 0
        for _ in range(L):
            for q in range(n):
                qml.RY(params[idx], wires=q)
                idx += 1
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
            if k < n:
                qml.CNOT(wires=[0, k])
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit, argnums=0)

    # Sample gradients
    g0_samples = []
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        params = pnp.array(
            rng.uniform(0, 2 * np.pi, n_p), requires_grad=True
        )
        try:
            g = grad_fn(params)
            g0_samples.append(float(g[0]))
        except Exception:
            continue

    if len(g0_samples) < 2:
        return None

    arr = np.array(g0_samples)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    lo   = float(np.percentile(arr, 2.5))
    hi   = float(np.percentile(arr, 97.5))
    var  = float(np.var(arr))

    return {
        "ansatz":    f"path_{k_label}",
        "k_value":   k,
        "noise":     "noiseless",
        "n_qubits":  n,
        "n_layers":  L,
        "n_params":  n_p,
        "metric":    "grad_var",
        "mean":      var,        # grad variance (not mean of output)
        "std":       std,
        "lo":        lo,
        "hi":        hi,
        "n_runs":    len(g0_samples),
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def load_completed():
    if not os.path.exists(INCREMENTAL_FILE):
        return set()
    df = pd.read_csv(INCREMENTAL_FILE)
    return set(
        (r["ansatz"], int(r["n_qubits"]), int(r["n_layers"]))
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
    print(f"Using {N_PROCESSES} processes (of {mp.cpu_count()} cores)")

    tasks = []
    for k_label in K_VALUES:
        for n in range(2, 11):      # 2..10
            for L in range(1, 9):   # 1..8
                tasks.append((k_label, n, L, 30, 0))

    completed = load_completed()
    todo = []
    for t in tasks:
        k_label, n, L = t[0], t[1], t[2]
        k = n // 2 if k_label == "half" else int(k_label)
        k = max(2, min(k, n - 1))
        ansatz_name = f"path_{k_label}"
        if (ansatz_name, n, L) not in completed:
            todo.append(t)

    print(f"Total tasks:  {len(tasks)}")
    print(f"Already done: {len(tasks) - len(todo)}")
    print(f"To run:       {len(todo)}")

    if not todo:
        print("Nothing to run.")
        _finalize()
        return

    # Cheapest first
    todo.sort(key=lambda t: t[1] * t[2])

    t0 = time.time()
    done = 0; failed = 0

    with mp.Pool(N_PROCESSES) as pool:
        for row in pool.imap_unordered(run_one_cell, todo, chunksize=1):
            if row is not None:
                append_row(row)
                done += 1
            else:
                failed += 1
            if done % 20 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.001)
                eta  = (len(todo) - done) / max(rate, 0.001)
                print(f"  [{done}/{len(todo)}]  "
                      f"elapsed={elapsed:.0f}s  "
                      f"eta={eta:.0f}s ({eta/60:.1f}min)")

    print(f"\nDone. {done} OK, {failed} failed.")
    _finalize()


def _finalize():
    if not os.path.exists(INCREMENTAL_FILE):
        return
    df = pd.read_csv(INCREMENTAL_FILE)
    df = df.drop_duplicates(
        subset=["ansatz","noise","n_qubits","n_layers"], keep="last")
    df.to_csv(os.path.join(OUT_DIR, "pathk_all.csv"), index=False)
    print(f"\nSaved {len(df)} rows to {OUT_DIR}/pathk_all.csv")
    print(f"Topologies: {sorted(df['ansatz'].unique())}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
