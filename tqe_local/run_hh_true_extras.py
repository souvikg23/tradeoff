"""
run_hh_true_extras.py
=====================
Adds the real IBM heavy-hex subgraphs (heavy_hex_true) to the two noiseless
experiments that were run before heavy_hex_true existed:

  1. Rotation richness + large n (same code as run_gap23.py):
     RX-RY-RZ rotations, noiseless, n = 2..16, L = 1..8, grad_var, 30 samples.
  2. VQE validation (same code as run_gap4_vqe.py):
     TFIM, p = 24 RY parameters, configs wide/square/deep, 15 seeds.

Nothing else changes: the circuits, seeds, sample counts and optimiser are
those of the original scripts. Only the entangling block for
"heavy_hex_true" is added (CNOT on every edge of the Falcon-27 BFS subgraph,
as in ansatze.py).

Usage (inside the project venv, from this folder):
    python run_hh_true_extras.py              # run both (resumes if stopped)
    python run_hh_true_extras.py --procs 8
    python run_hh_true_extras.py --analyze    # print the numbers for the paper

Outputs: outputs_hh_true/rot_cells.csv and outputs_hh_true/vqe_results.csv
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import run_gap23 as G23
import run_gap4_vqe as V
from ansatze import _falcon_subgraph

OUT = os.path.join(HERE, "outputs_hh_true")
ROT = os.path.join(OUT, "rot_cells.csv")
VQE = os.path.join(OUT, "vqe_results.csv")
NAME = "heavy_hex_true"


# --- add heavy_hex_true to both scripts' entangling functions -------------
# Done at import time so worker processes (spawned on Windows) get it too.
def _falcon_block(n):
    import pennylane as qml
    for c, t in _falcon_subgraph(n):
        qml.CNOT(wires=[c, t])


_orig23 = G23._ent_layer
_orig4 = V.ent_layer


def _ent23(name, n, layer_idx=0):
    if name == NAME:
        _falcon_block(n)
    else:
        _orig23(name, n, layer_idx)


def _ent4(name, n, layer_idx=0):
    if name == NAME:
        _falcon_block(n)
    else:
        _orig4(name, n, layer_idx)


G23._ent_layer = _ent23
V.ent_layer = _ent4


# --- workers ---------------------------------------------------------------
def rot_task(task):
    rows = G23.run_one_cell(task)
    return task, rows


def vqe_task(task):
    return task, V.run_one_task(task)


def _done_rot():
    if not os.path.exists(ROT):
        return set()
    d = pd.read_csv(ROT)
    return set(zip(d.n_qubits.astype(int), d.n_layers.astype(int)))


def _done_vqe():
    if not os.path.exists(VQE):
        return set()
    d = pd.read_csv(VQE)
    return set(zip(d.config, d.seed.astype(int)))


def _append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run(procs):
    os.makedirs(OUT, exist_ok=True)
    rdone, vdone = _done_rot(), _done_vqe()
    rot = [("grad_var", NAME, "noiseless", n, L, 30)
           for n in range(2, 17) for L in range(1, 9) if (n, L) not in rdone]
    rot.sort(key=lambda t: t[3] * t[4])
    vqe = [(NAME, c, n, L, s) for c, n, L in V.CONFIGS
           for s in range(V.N_SEEDS) if (c, s) not in vdone]
    print(f"rotation cells to run: {len(rot)} (done {len(rdone)}); "
          f"VQE runs to run: {len(vqe)} (done {len(vdone)}); {procs} processes", flush=True)
    failed = []
    t0 = time.time()
    with mp.Pool(procs) as pool:
        for i, (task, row) in enumerate(pool.imap_unordered(vqe_task, vqe), 1):
            if row.get("error_final") is None:
                failed.append(("vqe", task, row.get("error_msg")))
            else:
                _append(VQE, [row])
            if i % 5 == 0 or i == len(vqe):
                print(f"[VQE {i}/{len(vqe)}] {(time.time()-t0)/60:.1f} min", flush=True)
        for i, (task, rows) in enumerate(pool.imap_unordered(rot_task, rot), 1):
            if not rows:
                failed.append(("rot", task, "no rows returned"))
            else:
                _append(ROT, rows)
            if i % 5 == 0 or i == len(rot):
                print(f"[rotation {i}/{len(rot)}] {(time.time()-t0)/60:.1f} min", flush=True)
    if failed:
        print(f"\n{len(failed)} FAILED tasks (rerun to retry; send this list back):")
        for f in failed:
            print("  ", f)
    else:
        print("\nAll tasks finished. Now run:  python run_hh_true_extras.py --analyze")


def _fit(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def analyze():
    from scipy import stats
    if os.path.exists(ROT):
        d = pd.read_csv(ROT)
        g = {(int(n), int(L)): m for n, L, m in zip(d.n_qubits, d.n_layers, d["mean"])}
        try:
            al = [_fit(range(3, 11), [g[(n, L)] for n in range(3, 11)]) for L in range(1, 9)]
            dadl = stats.linregress(range(2, 9), al[1:]).slope
            a8 = al[-1]
            b8 = _fit(range(2, 9), [g[(8, L)] for L in range(2, 9)])
            print(f"RX-RY-RZ heavy_hex_true: d(alpha)/dL = {dadl:.3f}, alpha(L=8) = {a8:.2f}, "
                  f"beta(n=8) = {b8:.2f}, |alpha/beta| = {abs(a8/b8):.2f}")
            print(f"  amplification vs RY (-0.300): {dadl/-0.300:.2f}x")
            a16 = _fit(range(3, 17), [g[(n, 8)] for n in range(3, 17)])
            print(f"  width exponent at L=8: n<=10 {a8:.2f}, n<=16 {a16:.2f}; "
                  f"G(n=16, L=8) = {g[(16, 8)]:.2e}")
        except KeyError as e:
            print("rotation sweep incomplete, missing cell", e)
    if os.path.exists(VQE):
        v = pd.read_csv(VQE)
        w = v[v.config == "wide"].error_final
        dd = v[v.config == "deep"].error_final
        print(f"VQE heavy_hex_true: wide {w.mean():.3f}, square "
              f"{v[v.config=='square'].error_final.mean():.3f}, deep {dd.mean():.4f}; "
              f"gap {w.mean()-dd.mean():.3f}, Welch p = {stats.ttest_ind(w, dd, equal_var=False).pvalue:.2g} "
              f"(n = {len(w)}, {len(dd)})")


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        run(a.procs)
