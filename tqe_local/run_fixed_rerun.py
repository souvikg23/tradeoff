"""
run_fixed_rerun.py
==================
Rerun after the Nov 2026 audit fixes:
  * noise applied after EVERY layer (ansatz layer_hook)
  * noisy and noiseless cells use the SAME seeds and the SAME n_runs (paired)
  * gradient variance averaged over ALL parameters, for two costs:
        z0    : C = <Z_0>
        local : C = (1/n) sum_i <Z_i>
    (the old theta_0 / Z_0 diagnostic is kept as grad_var_theta0 for continuity)
  * expressivity: Haar KL against the REAL Haar ensemble + frame-potential ratio
  * kernel rank with n_data large enough not to cap the rank (default 64)
  * heavy_hex_true: real IBM Falcon-27 coupling-map subgraphs

Groups
------
  noisy     : n = 2..6, all noise presets incl. noiseless (paired), gradients only
  noiseless : n = 2..N_MAX, gradients + expressivity + kernel rank

Usage
-----
  python run_fixed_rerun.py --group noisy --out outputs_fixed
  python run_fixed_rerun.py --group noiseless --nmax 12 --out outputs_fixed
  python run_fixed_rerun.py --group noisy --quick          # small preview run
Re-running resumes: finished cells are skipped.
"""

import argparse
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TOPOS = ["linear", "ring", "brick", "all_to_all", "star", "heavy_hex", "heavy_hex_true"]
NOISES = ["noiseless", "depolarizing_realistic", "depolarizing_strong",
          "amplitude_realistic", "amplitude_strong"]


def grad_stats(ansatz, n, L, noise, n_samples, seed):
    """One run: per-parameter gradient variances for z0 and local costs,
    computed from a single Jacobian of all <Z_i>."""
    import pennylane as qml
    from pennylane import numpy as pnp
    from ansatze import ANSATZE
    from noise import make_device, is_noiseless, apply_layer_noise

    dev = make_device(n, noise)
    hook = None if is_noiseless(noise) else (lambda: apply_layer_noise(noise, n))

    @qml.qnode(dev, interface="autograd")
    def circ(p):
        ANSATZE[ansatz](p, n, L, layer_hook=hook)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    def vec(p):
        return pnp.stack(circ(p))

    jac = qml.jacobian(vec)
    rng = np.random.default_rng(seed)
    n_p = n * L
    J = np.empty((n_samples, n, n_p))
    for s in range(n_samples):
        p = pnp.array(rng.uniform(0, 2 * np.pi, n_p), requires_grad=True)
        J[s] = np.array(jac(p), dtype=float).reshape(n, n_p)
    g_z0 = J[:, 0, :]
    g_loc = J.mean(axis=1)
    v_z0, v_loc = g_z0.var(axis=0), g_loc.var(axis=0)
    return {
        "grad_var_theta0": float(v_z0[0]),
        "grad_var_z0_all": float(v_z0.mean()),
        "grad_var_local_all": float(v_loc.mean()),
        "grad_var_local_min": float(v_loc.min()),
    }


def run_cell(task):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings("ignore")
    from analysis import bootstrap_ci
    from metrics import frame_potential_ratio, haar_kl_divergence, effective_kernel_rank

    group, ansatz, noise, n, L, n_runs, n_samples, n_data = task
    per_run = {}
    t0 = time.time()
    for r in range(n_runs):
        out = grad_stats(ansatz, n, L, noise, n_samples, seed=r)
        if group == "noiseless":
            out["frame_potential_ratio"] = frame_potential_ratio(ansatz, n, L, noise, n_samples=60, seed=r)
            out["haar_kl_real"] = haar_kl_divergence(ansatz, n, L, noise, n_samples=60, seed=r, reference="real")
            out["kernel_rank"] = effective_kernel_rank(ansatz, n, L, noise, n_data=n_data, seed=r)
        for k, v in out.items():
            per_run.setdefault(k, []).append(v)
    rows = []
    for metric, vals in per_run.items():
        arr = np.array(vals, dtype=float)
        mean, lo, hi = bootstrap_ci(arr, n_boot=1000, seed=0)
        rows.append({"ansatz": ansatz, "noise": noise, "n_qubits": n, "n_layers": L,
                     "n_params": n * L, "metric": metric, "mean": mean,
                     "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                     "lo": lo, "hi": hi, "n_runs": n_runs, "n_samples": n_samples,
                     "seeds": f"0-{n_runs - 1}", "n_data": n_data if group == "noiseless" else "",
                     "seconds": round(time.time() - t0, 1)})
    return rows


def build_tasks(args):
    tasks = []
    if args.group == "noisy":
        ns = range(2, (5 if args.quick else 6) + 1)
        Ls = [1, 4, 8] if args.quick else range(1, 9)
        for a in TOPOS:
            for noise in NOISES:
                for n in ns:
                    for L in Ls:
                        tasks.append(("noisy", a, noise, n, L, args.runs, args.samples, 0))
    else:
        ns = range(2, args.nmax + 1)
        Ls = [1, 4, 8] if args.quick else range(1, 9)
        for a in TOPOS:
            for n in ns:
                for L in Ls:
                    tasks.append(("noiseless", a, "noiseless", n, L, args.runs, args.samples, args.ndata))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["noisy", "noiseless"], required=True)
    ap.add_argument("--out", default="outputs_fixed")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--ndata", type=int, default=64)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="small preview grid")
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()
    if args.quick and args.runs == 20:
        args.runs, args.samples = 5, 30

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.group}{'_quick' if args.quick else ''}"
    cells = os.path.join(args.out, f"cells_{tag}.csv")
    done = set()
    if os.path.exists(cells):
        d = pd.read_csv(cells)
        done = set(zip(d.ansatz, d.noise, d.n_qubits, d.n_layers))
    tasks = [t for t in build_tasks(args) if (t[1], t[2], t[3], t[4]) not in done]
    tasks.sort(key=lambda t: t[3] * t[4] * (3 if t[2] != "noiseless" else 1))
    print(f"{len(tasks)} cells to run ({len(done)} already done), {args.procs} processes", flush=True)

    t0 = time.time()
    with mp.Pool(args.procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_cell, tasks, chunksize=1), 1):
            pd.DataFrame(rows).to_csv(cells, mode="a", header=not os.path.exists(cells), index=False)
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"[{i}/{len(tasks)}] {el/60:.1f} min elapsed, eta {el/i*(len(tasks)-i)/60:.1f} min", flush=True)
    print("done:", cells)


if __name__ == "__main__":
    mp.freeze_support()
    main()
