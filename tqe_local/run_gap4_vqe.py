"""
run_gap4_vqe.py
===============
Gap 4: Downstream task validation.

Tests whether the regime classifier (depth-dominated / balanced /
width-dominated) predicts real training outcomes on a VQE problem.

Hamiltonian: Transverse-field Ising model
  H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)
  J = 1.0, h = 0.5

The key experiment
------------------
Fixed parameter budget p = 24 (n * L = 24 for RY ansatz).
Three configurations tested for each topology:

  Config A: n=8, L=3  (wide and shallow)
  Config B: n=6, L=4  (square)
  Config C: n=4, L=6  (narrow and deep)

Prediction from the regime classifier:
  Brick  (depth-dom,  ratio=0.48): Config C wins
  Linear (depth-dom,  ratio=0.71): Config C wins
  HH     (balanced,   ratio=1.24): Configs similar, Config B or C wins
  Star   (width-dom,  ratio=1.32): Config C wins, Config A fails
  Ring   (width-dom,  ratio=5.82): Config C wins, Config A completely fails

Metric: energy error at convergence = E_final - E_ground
Lower = better. Ground state energy computed by exact diagonalization.

Optimizer: Adam, lr=0.05, 200 steps.
Seeds: 10 per configuration.

Output: outputs_gap4/vqe_results.csv

Estimated runtime: ~20-40 minutes on Ryzen 7, 6 cores.
"""

import os, time, warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT_DIR = "outputs_gap4"
os.makedirs(OUT_DIR, exist_ok=True)

J = 1.0
H_FIELD = 0.5
N_STEPS = 300
LR = 0.05
N_SEEDS = 15
N_PROCESSES = max(1, mp.cpu_count() - 2)

CONFIGS = [
    ("wide",   8, 3),   # Config A: n=8, L=3
    ("square", 6, 4),   # Config B: n=6, L=4
    ("deep",   4, 6),   # Config C: n=4, L=6
]
TOPOS = ["brick", "linear", "heavy_hex", "star", "ring"]


# -----------------------------------------------------------------------
# Exact ground state energy via sparse diagonalization
# -----------------------------------------------------------------------
def exact_ground_energy(n, J=1.0, h=0.5):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    I = sp.eye(2, format='csr')

    def kron_op(op, i, n):
        ops = [I] * n
        ops[i] = op
        result = ops[0]
        for o in ops[1:]:
            result = sp.kron(result, o, format='csr')
        return result

    H = sp.csr_matrix((2 ** n, 2 ** n), dtype=float)
    for i in range(n - 1):
        H -= J * kron_op(Z, i, n).dot(kron_op(Z, i + 1, n))
    for i in range(n):
        H -= h * kron_op(X, i, n)

    return eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]


# -----------------------------------------------------------------------
# Entangling layers (same as main study, RY rotation)
# -----------------------------------------------------------------------
def ent_layer(ansatz_name, n, layer_idx=0):
    import pennylane as qml
    if ansatz_name == "brick":
        if layer_idx % 2 == 0:
            for q in range(0, n - 1, 2): qml.CZ(wires=[q, q + 1])
        else:
            for q in range(1, n - 1, 2): qml.CZ(wires=[q, q + 1])
    elif ansatz_name == "linear":
        for q in range(n - 1): qml.CNOT(wires=[q, q + 1])
    elif ansatz_name == "ring":
        for q in range(n - 1): qml.CNOT(wires=[q, q + 1])
        if n > 1: qml.CNOT(wires=[n - 1, 0])
    elif ansatz_name == "star":
        for j in range(1, n): qml.CNOT(wires=[0, j])
    elif ansatz_name == "heavy_hex":
        for i in range(n - 1): qml.CNOT(wires=[i, i + 1])
        for i in range(0, n - 2, 3):
            if i + 2 < n: qml.CNOT(wires=[i, i + 2])
    elif ansatz_name == "all_to_all":
        for i in range(n):
            for j in range(i + 1, n): qml.CNOT(wires=[i, j])


# -----------------------------------------------------------------------
# Ising Hamiltonian observable via PennyLane
# -----------------------------------------------------------------------
def ising_hamiltonian(n, J=1.0, h=0.5):
    import pennylane as qml
    coeffs, obs = [], []
    for i in range(n - 1):
        coeffs.append(-J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, obs)


# -----------------------------------------------------------------------
# Single VQE run
# -----------------------------------------------------------------------
def run_vqe(ansatz_name, n, L, seed, J=1.0, h=0.5,
            n_steps=N_STEPS, lr=LR):
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n)
    H = ising_hamiltonian(n, J, h)
    n_p = n * L  # RY rotation, single param per qubit per layer

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        for layer in range(L):
            for q in range(n):
                qml.RY(params[layer * n + q], wires=q)
            ent_layer(ansatz_name, n, layer)
        return qml.expval(H)

    rng = np.random.default_rng(seed)
    params = pnp.array(
        rng.uniform(0, 2 * np.pi, n_p), requires_grad=True
    )

    # Adam optimizer
    opt = qml.AdamOptimizer(stepsize=lr)
    energies = []
    for step in range(n_steps):
        params, energy = opt.step_and_cost(circuit, params)
        energies.append(float(energy))

    return float(energies[-1]), energies


# -----------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------
def run_one_task(task):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np

    ansatz, config_name, n, L, seed = task

    try:
        E_ground = exact_ground_energy(n, J, H_FIELD)
        E_final, energies = run_vqe(ansatz, n, L, seed, J, H_FIELD)
        error = E_final - E_ground
        # Also record best energy reached (in case of non-convergence)
        E_best = min(energies) - E_ground

        return {
            "ansatz":      ansatz,
            "config":      config_name,
            "n_qubits":    n,
            "n_layers":    L,
            "n_params":    n * L,
            "seed":        seed,
            "E_ground":    E_ground,
            "E_final":     E_final,
            "error_final": error,
            "error_best":  E_best,
            "converged":   error < 0.1,
        }
    except Exception as e:
        return {
            "ansatz": ansatz, "config": config_name,
            "n_qubits": n, "n_layers": L, "n_params": n*L,
            "seed": seed, "E_ground": None, "E_final": None,
            "error_final": None, "error_best": None,
            "converged": False, "error_msg": str(e),
        }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def build_tasks():
    tasks = []
    for ansatz in TOPOS:
        for config_name, n, L in CONFIGS:
            for seed in range(N_SEEDS):
                tasks.append((ansatz, config_name, n, L, seed))
    return tasks


def main():
    print("Gap 4: Downstream VQE Validation")
    print(f"Hamiltonian: Ising n=4/6/8, J={J}, h={H_FIELD}")
    print(f"Optimizer: Adam lr={LR}, {N_STEPS} steps, {N_SEEDS} seeds")
    print(f"Topologies: {TOPOS}")
    print(f"Using {N_PROCESSES} processes")
    print()

    tasks = build_tasks()
    print(f"Total runs: {len(tasks)}")

    t0 = time.time()
    results = []

    with mp.Pool(N_PROCESSES) as pool:
        for i, row in enumerate(
                pool.imap_unordered(run_one_task, tasks, chunksize=1)):
            results.append(row)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (len(tasks) - i - 1) / max(rate, 0.001)
                print(f"  [{i+1}/{len(tasks)}]  "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"eta={eta/60:.1f}min")

    df = pd.DataFrame(results)
    out = os.path.join(OUT_DIR, "vqe_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    _summarize(df)


def _summarize(df):
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("Mean energy error at convergence (lower = better)")
    print("="*70)
    print(f"\n{'Topology':>12}  {'Wide A':>10}  {'Square B':>10}  "
          f"{'Deep C':>10}  Prediction  Match?")
    print("-"*70)

    preds = {
        "brick":     ("C", "depth-dom"),
        "linear":    ("C", "depth-dom"),
        "heavy_hex": ("B", "balanced"),
        "star":      ("C", "width-dom"),
        "ring":      ("C", "width-dom"),
    }

    df_valid = df[df['error_final'].notna()]

    for a in TOPOS:
        sub = df_valid[df_valid['ansatz'] == a]
        errs = {}
        for cfg in ['wide', 'square', 'deep']:
            s = sub[sub['config'] == cfg]['error_final']
            errs[cfg] = float(s.mean()) if len(s) > 0 else float('nan')

        best_cfg = min(errs, key=errs.get)
        pred_cfg, regime = preds[a]
        match = "YES" if best_cfg[0].upper() == pred_cfg else "NO"

        print(f"  {a:>12}  {errs['wide']:>10.4f}  {errs['square']:>10.4f}  "
              f"{errs['deep']:>10.4f}  {regime}  {match}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
