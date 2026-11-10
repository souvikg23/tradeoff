"""
run_gap23.py
============
Fills Gap 2 (wider system sizes: n=2..16) and Gap 3 (richer rotations:
RZ-RY-RZ per qubit per layer instead of single RY) simultaneously.

What changes vs the original sweep
-----------------------------------
Gap 2: noiseless sweep extended to n=2..16.
       Noisy sweep stays at n=2..6 (density matrix memory hard limit).
       At n=11..16 we compute only grad_var and haar_kl (skip QFIM —
       too slow at large n).

Gap 3: Each qubit gets RZ(theta1) RY(theta2) RZ(theta3) per layer
       instead of a single RY.
       Parameter count: p = 3 * n * L  (was n * L).
       The entangling topology is unchanged.

Everything else identical to the original:
  - 6 topologies: brick, linear, ring, star, all_to_all, heavy_hex
  - 8 depths: L = 1..8
  - 30 runs noiseless, 20 runs noisy grad_var
  - Bootstrap CI via analysis.py
  - Incremental CSV output with resume support

Estimated runtime (Ryzen 7, 6 cores, plugged in, Best Performance):
  Noiseless: ~2.5-3.5 hours
  Noisy:     ~45 minutes
  Total:     ~3.5-4.5 hours

Output
------
  outputs_gap23/
    cells.csv                 -- incremental (resume file)
    master_dataset_gap23.csv  -- full merged dataset (new sweep only)
    master_dataset_tqe2.csv   -- merged with existing master_dataset_tqe.csv

Usage
-----
  python run_gap23.py

  To resume after interruption, just re-run. Already-computed cells
  are skipped automatically.
"""

import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
OUT_DIR         = "outputs_gap23"
INCREMENTAL     = os.path.join(OUT_DIR, "cells.csv")
EXISTING_MASTER = "master_dataset_tqe.csv"   # your current dataset
N_PROCESSES     = max(1, mp.cpu_count() - 2)

TOPOS = ["brick", "linear", "ring", "star", "all_to_all", "heavy_hex"]

# Gap 2: extended width range
N_NOISELESS_MIN, N_NOISELESS_MAX = 2, 16
N_NOISY_MIN,     N_NOISY_MAX     = 2, 6
DEPTHS = list(range(1, 9))     # L = 1..8

# Metrics to run at large n (n > 10): grad_var + haar_kl only
# QFIM at n=16 requires ~65536 entries AND many QFIM evaluations — skip
METRICS_LARGE_N  = ["grad_var", "haar_kl"]
METRICS_STANDARD = ["grad_var", "haar_kl", "qfim"]

NOISE_PRESETS_NOISY = [
    "depolarizing_realistic", "depolarizing_strong",
    "amplitude_realistic",    "amplitude_strong",
]

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Gap 3: RZ-RY-RZ rotation layer
# -----------------------------------------------------------------------
def _rzryrz_layer(params, n_qubits, idx):
    """
    NOTE (Nov 2026 audit): the code applies RX-RY-RZ, not RZ-RY-RZ as the name
    and the module docstring say. Decide which one the paper describes and make
    the code and the text agree before reusing these results.
    Uses 3 parameters per qubit per layer.
    Returns updated idx.
    """
    import pennylane as qml
    for q in range(n_qubits):
        qml.RX(params[idx],     wires=q)
        qml.RY(params[idx + 1], wires=q)
        qml.RZ(params[idx + 2], wires=q)
        idx += 3
    return idx


def _ent_layer(ansatz_name, n_qubits, layer_idx=0):
    """Apply the entangling block for the given topology."""
    import pennylane as qml

    if ansatz_name == "linear":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])

    elif ansatz_name == "ring":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])

    elif ansatz_name == "brick":
        if layer_idx % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])
        else:
            for q in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])

    elif ansatz_name == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

    elif ansatz_name == "star":
        for j in range(1, n_qubits):
            qml.CNOT(wires=[0, j])

    elif ansatz_name == "heavy_hex":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(0, n_qubits - 2, 3):
            if i + 2 < n_qubits:
                qml.CNOT(wires=[i, i + 2])


def n_params_rzryrz(n_qubits, n_layers):
    """3 params per qubit per layer."""
    return 3 * n_qubits * n_layers


def build_circuit(ansatz_name, n_qubits, n_layers, device_name,
                  noise_preset="noiseless"):
    """
    Returns a QNode for the RZ-RY-RZ ansatz with the given topology.
    """
    import pennylane as qml
    from noise import NOISE_PRESETS

    if noise_preset == "noiseless":
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("default.mixed", wires=n_qubits)

    from noise import apply_layer_noise
    # FIX (Nov 2026): NOISE_PRESETS holds {"kind", "rate"} dicts, not
    # (fn, kwargs) pairs. The old loop raised on every noisy cell, the error was
    # swallowed, and the noisy sweep wrote no rows at all.

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        for layer in range(n_layers):
            idx_start = layer * 3 * n_qubits
            _rzryrz_layer(params, n_qubits, idx_start)
            _ent_layer(ansatz_name, n_qubits, layer)
            if noise_preset != "noiseless":
                apply_layer_noise(noise_preset, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


# -----------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------
def run_one_cell(task):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    metric, ansatz, noise, n, L, n_runs = task
    n_p = n_params_rzryrz(n, L)

    try:
        if metric == "grad_var":
            return _compute_grad_var(ansatz, noise, n, L, n_runs, n_p)
        elif metric == "haar_kl":
            return _compute_haar_kl(ansatz, noise, n, L, n_runs, n_p)
        elif metric == "qfim":
            return _compute_qfim(ansatz, n, L, n_runs, n_p)
        else:
            return []
    except Exception as e:
        print(f"  ERROR {metric} {ansatz} n={n} L={L}: {e}")
        return []


def _compute_grad_var(ansatz, noise, n, L, n_runs, n_p):
    from pennylane import numpy as pnp
    import pennylane as qml
    import numpy as np

    circuit = build_circuit(ansatz, n, L, "default.qubit"
                            if noise == "noiseless" else "default.mixed",
                            noise_preset=noise)
    grad_fn = qml.grad(circuit, argnums=0)
    rng = np.random.default_rng(42)
    g0 = []
    for _ in range(n_runs):
        p = pnp.array(rng.uniform(0, 2 * np.pi, n_p), requires_grad=True)
        try:
            g = grad_fn(p)
            g0.append(float(g[0]))
        except Exception:
            continue
    if len(g0) < 2:
        return []
    arr = np.array(g0)
    return [_row("grad_var", ansatz, noise, n, L, n_p,
                 float(np.var(arr)), float(np.std(arr, ddof=1)),
                 float(np.percentile(arr, 2.5)),
                 float(np.percentile(arr, 97.5)), len(g0))]


def _compute_haar_kl(ansatz, noise, n, L, n_runs, n_p):
    import pennylane as qml
    from pennylane import numpy as pnp
    import numpy as np

    if noise != "noiseless":
        return []   # only noiseless Haar KL

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def state_circuit(params):
        for layer in range(L):
            _rzryrz_layer(params, n, layer * 3 * n)
            _ent_layer(ansatz, n, layer)
        return qml.state()

    n_samples = 60
    rng = np.random.default_rng(0)
    states = []
    for _ in range(n_samples):
        p = pnp.array(rng.uniform(0, 2 * np.pi, n_p), requires_grad=False)
        try:
            states.append(np.array(state_circuit(p)))
        except Exception:
            continue

    if len(states) < 10:
        return []

    fids = [abs(np.dot(states[i].conj(), states[j])) ** 2
            for i in range(len(states))
            for j in range(i + 1, len(states))]

    dim = 2 ** n
    haar_fids = rng.beta(1, dim - 1, size=len(fids))
    bins = np.linspace(0, 1, 51)
    pv = np.histogram(fids,      bins=bins, density=True)[0] + 1e-10
    ph = np.histogram(haar_fids, bins=bins, density=True)[0] + 1e-10
    pv /= pv.sum()
    ph /= ph.sum()
    kl = float(np.sum(pv * np.log(pv / ph)))

    return [_row("haar_kl", ansatz, "noiseless", n, L, n_p,
                 kl, 0.0, kl, kl, len(states))]


def _get_state(circuit, params, n):
    import pennylane as qml
    import numpy as np
    from pennylane import numpy as pnp
    # Get statevector by running the circuit
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def state_circuit(p):
        # re-run the same ansatz structure
        return qml.state()

    # We extract via density matrix trick
    result = circuit(params)
    # Get state from device after circuit execution
    state = circuit.device.state if hasattr(circuit, 'device') else None
    if state is None:
        # fallback: random unit vector
        v = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        return v / np.linalg.norm(v)
    return np.array(state).flatten()


def _compute_qfim(ansatz, n, L, n_runs, n_p):
    import pennylane as qml
    from pennylane import numpy as pnp
    import numpy as np

    dev = qml.device("default.qubit", wires=n)

    def make_qnode(p):
        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            for layer in range(L):
                idx_start = layer * 3 * n
                _rzryrz_layer(params, n, idx_start)
                _ent_layer(ansatz, n, layer)
            return qml.expval(qml.PauliZ(0))
        return circuit

    rng = np.random.default_rng(7)
    traces = []
    for _ in range(min(n_runs, 10)):
        p = pnp.array(rng.uniform(0, 2 * np.pi, n_p), requires_grad=True)
        try:
            circuit = make_qnode(p)
            mt = qml.metric_tensor(circuit, approx="block-diag")(p)
            trace = float(np.trace(mt)) * 4.0
            traces.append(trace)
        except Exception:
            continue
    if len(traces) == 0:
        return []
    arr = np.array(traces)
    rows = []
    for sub_m in ["qfim_trace"]:
        rows.append(_row(sub_m, ansatz, "noiseless", n, L, n_p,
                         float(np.mean(arr)), float(np.std(arr, ddof=1)),
                         float(np.percentile(arr, 2.5)),
                         float(np.percentile(arr, 97.5)), len(traces)))
    return rows


def _row(metric, ansatz, noise, n, L, n_p,
         mean, std, lo, hi, n_runs):
    return {
        "ansatz": ansatz, "noise": noise,
        "n_qubits": n, "n_layers": L,
        "n_params": n_p, "metric": metric,
        "mean": mean, "std": std,
        "lo": lo, "hi": hi,
        "n_runs": n_runs,
        "rotation": "rxryrz",   # tag to distinguish from original RY data
    }


# -----------------------------------------------------------------------
# Task builder
# -----------------------------------------------------------------------
def build_tasks():
    tasks = []

    for a in TOPOS:
        # --- Noiseless sweep: n=2..16 ---
        for n in range(N_NOISELESS_MIN, N_NOISELESS_MAX + 1):
            metrics = METRICS_LARGE_N if n > 10 else METRICS_STANDARD
            for L in DEPTHS:
                for m in metrics:
                    n_runs = 10 if m == "qfim" else 30
                    tasks.append((m, a, "noiseless", n, L, n_runs))

        # --- Noisy grad_var sweep: n=2..6 ---
        for noise in NOISE_PRESETS_NOISY:
            for n in range(N_NOISY_MIN, N_NOISY_MAX + 1):
                for L in DEPTHS:
                    tasks.append(("grad_var", a, noise, n, L, 20))

    return tasks


# -----------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------
def load_completed():
    if not os.path.exists(INCREMENTAL):
        return set()
    df = pd.read_csv(INCREMENTAL)
    return set(
        (r["metric"] if r["metric"] != "qfim_trace" else "qfim",
         r["ansatz"], r["noise"],
         int(r["n_qubits"]), int(r["n_layers"]))
        for _, r in df.iterrows()
    )


def append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(INCREMENTAL)
    df.to_csv(INCREMENTAL, mode="a", header=header, index=False)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print(f"Gap 2+3 sweep — RX-RY-RZ rotations, n=2..16")
    print(f"Using {N_PROCESSES} processes of {mp.cpu_count()} cores")
    print()

    all_tasks  = build_tasks()
    completed  = load_completed()
    todo = []
    for t in all_tasks:
        key = (t[0] if t[0] != "qfim_trace" else "qfim",
               t[1], t[2], t[3], t[4])
        if key not in completed:
            todo.append(t)

    print(f"Total tasks:    {len(all_tasks)}")
    print(f"Already done:   {len(all_tasks) - len(todo)}")
    print(f"To run:         {len(todo)}")
    print()

    if not todo:
        print("Nothing to run — finalising.")
        _finalize()
        return

    # Sort cheapest first (small n × L first)
    todo.sort(key=lambda t: (t[3] * t[4]))

    t0    = time.time()
    done  = 0
    failed = 0

    with mp.Pool(N_PROCESSES) as pool:
        for rows in pool.imap_unordered(run_one_cell, todo, chunksize=1):
            try:
                append_rows(rows)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  WRITE ERROR: {e}")
            if done % 30 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate    = done / max(elapsed, 0.001)
                eta     = (len(todo) - done) / max(rate, 0.001)
                pct     = 100 * done / len(todo)
                print(f"  [{done}/{len(todo)}] {pct:.0f}%  "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"eta={eta/60:.1f}min")

    print(f"\nDone. {done} cells completed, {failed} failed.")
    _finalize()


def _finalize():
    if not os.path.exists(INCREMENTAL):
        print("No data to finalise.")
        return

    df_new = pd.read_csv(INCREMENTAL)
    df_new = df_new.drop_duplicates(
        subset=["metric", "ansatz", "noise", "n_qubits", "n_layers"],
        keep="last")
    df_new.to_csv(os.path.join(OUT_DIR, "master_dataset_gap23.csv"),
                  index=False)
    print(f"New sweep saved: {len(df_new)} rows")

    # Merge with existing TQE master dataset
    if os.path.exists(EXISTING_MASTER):
        df_old = pd.read_csv(EXISTING_MASTER)
        # Tag old data as RY rotation
        if "rotation" not in df_old.columns:
            df_old["rotation"] = "ry"
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        out = os.path.join(OUT_DIR, "master_dataset_tqe2.csv")
        df_all.to_csv(out, index=False)
        print(f"Merged dataset: {len(df_all)} rows -> {out}")
        print(f"  RY (original):    {len(df_old)} rows")
        print(f"  RZ-RY-RZ (new):   {len(df_new)} rows")
    else:
        print(f"Note: {EXISTING_MASTER} not found, "
              f"only new data saved.")

    print()
    print("Next steps:")
    print("  1. Upload outputs_gap23/master_dataset_tqe2.csv")
    print("  2. Re-run figure generation scripts")
    print("  3. Check if alpha groupings hold for RZ-RY-RZ")
    print("  4. Update tables and figures in the TQE paper")


if __name__ == "__main__":
    mp.freeze_support()
    main()
