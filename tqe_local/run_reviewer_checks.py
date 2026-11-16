"""
run_reviewer_checks.py
======================
Two extra noiseless experiments for the topology paper (Nov 2026), written to
pre-empt the two most likely reviewer objections.

Part "vqe"  (check 1: Hamiltonian matched to the heavy-hex hardware)
    Same VQE protocol as run_gap4_vqe.py / run_hh_true_extras.py:
      RY ansatz, p = n*L = 24, configs wide (8,3) / square (6,4) / deep (4,6),
      Adam lr 0.05, 300 steps, seeds 0..14, error = E_final - E_ground.
    Only the Hamiltonian changes: the ZZ couplings sit on the edges of the
    IBM Falcon-27 heavy-hex subgraph (the same edges the heavy_hex_true
    ansatz uses), instead of on the 1D chain (i, i+1):
        H = -J sum_{(i,j) in E_hh} Z_i Z_j - h sum_i X_i ,  J = 1, h = 0.5
    All six paper topologies are run on this Hamiltonian.

Part "grad" (check 2: heavy-hex measured on a degree-3 qubit)
    Same protocol as run_fixed_rerun.py (noiseless group): heavy_hex_true,
    n = 2..12, L = 1..8, 20 runs (seeds 0..19) x 50 random parameter vectors,
    bootstrap mean over runs (1000 resamples, seed 0).
    From ONE Jacobian of all <Z_i> per parameter vector we record
      G_q0      = Var[d<Z_0>/d theta_0]   (the paper's diagnostic; regression)
      G_q1      = Var[d<Z_1>/d theta_1]   (measured qubit 1: degree 3 for n >= 4)
      Gall_q0 / Gall_q1 = all-parameter versions (mean over parameters of
                          Var[d<Z_m>/d theta_k]).
    theta_m is the layer-0 RY on the measured qubit m (parameter index m).

Usage
-----
  python run_reviewer_checks.py --part all            # both parts, resumes
  python run_reviewer_checks.py --part vqe --procs 4
  python run_reviewer_checks.py --part grad --nmax 12
  python run_reviewer_checks.py --quick               # small preview grid
  python run_reviewer_checks.py --analyze             # print numbers for the paper
  python run_reviewer_checks.py --part all --resume-from /kaggle/input
Rerunning skips finished rows. Failed tasks are written to failures.csv and
printed; they are never silently dropped.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

J_COUPLING = 1.0
H_FIELD = 0.5
N_STEPS = 300
LR = 0.05
N_SEEDS = 15
CONFIGS = [("wide", 8, 3), ("square", 6, 4), ("deep", 4, 6)]
VQE_TOPOS = ["brick", "linear", "heavy_hex_true", "heavy_hex", "star", "ring"]
PAPER_NAME = {"heavy_hex": "chord-chain", "heavy_hex_true": "heavy-hex (IBM)"}

GRAD_TOPO = "heavy_hex_true"
GRAD_RUNS = 20
GRAD_SAMPLES = 50

VQE_FILE = "vqe_hh_hamiltonian.csv"
GRAD_FILE = "grad_measured_qubit.csv"
FAIL_FILE = "failures.csv"


# ---------------------------------------------------------------------------
# Hamiltonians
# ---------------------------------------------------------------------------
def coupling_edges(n, ham):
    """ZZ coupling pairs. ham='hh': Falcon-27 subgraph edges; 'chain': (i, i+1)."""
    from ansatze import _falcon_subgraph
    if ham == "hh":
        return [tuple(sorted(e)) for e in _falcon_subgraph(n)]
    if ham == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    raise ValueError(ham)


def exact_ground_energy(n, edges, J=J_COUPLING, h=H_FIELD):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    X = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    I = sp.identity(2, format="csr")

    def op(single, i):
        out = None
        for q in range(n):
            m = single if q == i else I
            out = m if out is None else sp.kron(out, m, format="csr")
        return out

    H = sp.csr_matrix((2 ** n, 2 ** n))
    for a, b in edges:
        H = H - J * op(Z, a).dot(op(Z, b))
    for i in range(n):
        H = H - h * op(X, i)
    return float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])


def pl_hamiltonian(n, edges, J=J_COUPLING, h=H_FIELD):
    import pennylane as qml
    coeffs, obs = [], []
    for a, b in edges:
        coeffs.append(-J)
        obs.append(qml.PauliZ(a) @ qml.PauliZ(b))
    for i in range(n):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, obs)


# ---------------------------------------------------------------------------
# Part 1: VQE
# ---------------------------------------------------------------------------
def run_vqe(ansatz, n, L, seed, ham, n_steps=N_STEPS, lr=LR):
    import pennylane as qml
    from pennylane import numpy as pnp
    from ansatze import ANSATZE

    edges = coupling_edges(n, ham)
    dev = qml.device("default.qubit", wires=n)
    H = pl_hamiltonian(n, edges)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        ANSATZE[ansatz](params, n, L)
        return qml.expval(H)

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, n * L), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    energies = []
    for _ in range(n_steps):
        params, e = opt.step_and_cost(circuit, params)
        energies.append(float(e))
    return energies, edges


def vqe_task(task):
    ansatz, cfg, n, L, seed, ham, n_steps = task
    t0 = time.time()
    try:
        energies, edges = run_vqe(ansatz, n, L, seed, ham, n_steps=n_steps)
        e0 = exact_ground_energy(n, edges)
        return task, {
            "ansatz": ansatz, "hamiltonian": ham, "config": cfg, "n_qubits": n,
            "n_layers": L, "n_params": n * L, "seed": seed, "n_steps": n_steps,
            "E_ground": e0, "E_final": energies[-1],
            "error_final": energies[-1] - e0, "error_best": min(energies) - e0,
            "n_edges": len(edges), "seconds": round(time.time() - t0, 1),
        }, None
    except Exception as exc:  # reported, never dropped silently
        return task, None, repr(exc)


# ---------------------------------------------------------------------------
# Part 2: gradient variance on a chosen measured qubit
# ---------------------------------------------------------------------------
def grad_run(n, L, n_samples, seed, topo=GRAD_TOPO):
    import pennylane as qml
    from pennylane import numpy as pnp
    from ansatze import ANSATZE

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def circ(p):
        ANSATZE[topo](p, n, L)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    jac = qml.jacobian(lambda p: pnp.stack(circ(p)))
    rng = np.random.default_rng(seed)
    n_p = n * L
    Jm = np.empty((n_samples, n, n_p))
    for s in range(n_samples):
        p = pnp.array(rng.uniform(0, 2 * np.pi, n_p), requires_grad=True)
        Jm[s] = np.array(jac(p), dtype=float).reshape(n, n_p)
    out = {}
    for m in (0, 1):
        if m >= n:
            continue
        v = Jm[:, m, :].var(axis=0)
        out[f"G_q{m}"] = float(v[m])
        out[f"Gall_q{m}"] = float(v.mean())
    return out


def grad_task(task):
    n, L, n_runs, n_samples = task
    from analysis import bootstrap_ci
    from ansatze import _falcon_subgraph
    t0 = time.time()
    try:
        per = {}
        for r in range(n_runs):
            for k, v in grad_run(n, L, n_samples, seed=r).items():
                per.setdefault(k, []).append(v)
        deg = {}
        for a, b in _falcon_subgraph(n):
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        rows = []
        for metric, vals in per.items():
            arr = np.asarray(vals, float)
            mean, lo, hi = bootstrap_ci(arr, n_boot=1000, seed=0)
            m = int(metric[-1])
            rows.append({"ansatz": GRAD_TOPO, "n_qubits": n, "n_layers": L, "metric": metric,
                         "measured_qubit": m, "degree": deg.get(m, 0),
                         "mean": mean, "lo": lo, "hi": hi,
                         "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                         "n_runs": n_runs, "n_samples": n_samples,
                         "seconds": round(time.time() - t0, 1)})
        return task, rows, None
    except Exception as exc:
        return task, None, repr(exc)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def _read_all(out, name, extra):
    paths = [os.path.join(out, name)]
    for src in extra or []:
        paths += glob.glob(os.path.join(src, "**", name), recursive=True) if os.path.isdir(src) else [src]
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p) and os.path.basename(p) == name]
    return pd.concat(frames, ignore_index=True) if frames else None


def _merge_previous(out, extra):
    """Copy rows from --resume-from into the working files (dedup on keys)."""
    for name, keys in ((VQE_FILE, ["ansatz", "hamiltonian", "config", "seed", "n_steps"]),
                       (GRAD_FILE, ["n_qubits", "n_layers", "metric", "n_runs", "n_samples"])):
        d = _read_all(out, name, extra)
        if d is not None and len(d):
            d.drop_duplicates(subset=keys, keep="first").to_csv(os.path.join(out, name), index=False)


def build_tasks(args):
    vqe, grad = [], []
    steps = 60 if args.quick else N_STEPS
    seeds = range(2) if args.quick else range(N_SEEDS)
    runs, samples = (3, 10) if args.quick else (GRAD_RUNS, GRAD_SAMPLES)
    if args.part in ("vqe", "all"):
        vqe = [(a, c, n, L, s, "hh", steps) for a in VQE_TOPOS for c, n, L in CONFIGS for s in seeds]
    if args.part in ("grad", "all"):
        ns = range(2, (6 if args.quick else args.nmax) + 1)
        Ls = [1, 4, 8] if args.quick else range(1, 9)
        grad = [(n, L, runs, samples) for n in ns for L in Ls]
    return vqe, grad


def run(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    _merge_previous(out, args.resume_from)
    vqe, grad = build_tasks(args)
    vf, gf = os.path.join(out, VQE_FILE), os.path.join(out, GRAD_FILE)
    if os.path.exists(vf):
        d = pd.read_csv(vf)
        done = set(zip(d.ansatz, d.config, d.seed.astype(int), d.n_steps.astype(int)))
        vqe = [t for t in vqe if (t[0], t[1], t[4], t[6]) not in done]
    if os.path.exists(gf):
        d = pd.read_csv(gf)
        done = set(zip(d.n_qubits.astype(int), d.n_layers.astype(int), d.n_runs.astype(int)))
        grad = [t for t in grad if (t[0], t[1], t[2]) not in done]
    grad.sort(key=lambda t: t[0] * t[1])
    tasks = [("vqe", t) for t in vqe] + [("grad", t) for t in grad]
    print(f"VQE runs to do: {len(vqe)}, gradient cells to do: {len(grad)}, "
          f"{args.procs} processes", flush=True)
    if not tasks:
        print("Nothing to do.")
        return
    fails, t0 = [], time.time()
    with mp.Pool(args.procs) as pool:
        for i, (kind, task, res, err) in enumerate(pool.imap_unordered(_dispatch, tasks, chunksize=1), 1):
            if err is not None:
                fails.append({"part": kind, "task": str(task), "error": err})
                pd.DataFrame([fails[-1]]).to_csv(os.path.join(out, FAIL_FILE), mode="a",
                                                 header=not os.path.exists(os.path.join(out, FAIL_FILE)),
                                                 index=False)
            else:
                rows = [res] if kind == "vqe" else res
                path = vf if kind == "vqe" else gf
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            el = time.time() - t0
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {el/60:.1f} min elapsed, eta {el/i*(len(tasks)-i)/60:.1f} min",
                      flush=True)
            if args.max_hours and el > args.max_hours * 3600:
                print("Time budget reached; results so far are saved. Rerun to continue.", flush=True)
                pool.terminate()
                break
    if fails:
        print(f"\n{len(fails)} FAILED tasks (see {FAIL_FILE}); rerun to retry:")
        for f in fails:
            print("  ", f)
    else:
        print("\nAll tasks in this batch finished.", flush=True)


def _dispatch(item):
    kind, task = item
    t, res, err = (vqe_task if kind == "vqe" else grad_task)(task)
    return kind, t, res, err


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def analyze(out, quick=False):
    from scipy import stats
    vf, gf = os.path.join(out, VQE_FILE), os.path.join(out, GRAD_FILE)
    if os.path.exists(vf):
        v = pd.read_csv(vf)
        if not quick:
            v = v[v.n_steps == N_STEPS]
        print("=== Check 1: VQE with Ising couplings on the heavy-hex edges ===")
        print(f"{'topology':>16} {'wide':>8} {'square':>8} {'deep':>8} {'gap(w-d)':>9} {'Welch p':>9} {'runs':>5}")
        for a in VQE_TOPOS:
            s = v[v.ansatz == a]
            if s.empty:
                continue
            w, sq, d = (s[s.config == c].error_final for c in ("wide", "square", "deep"))
            p = stats.ttest_ind(w, d, equal_var=False).pvalue if len(w) > 1 and len(d) > 1 else float("nan")
            print(f"{PAPER_NAME.get(a, a):>16} {w.mean():8.3f} {sq.mean():8.3f} {d.mean():8.4f} "
                  f"{w.mean()-d.mean():9.3f} {p:9.2g} {len(s):5d}")
        print("Paper (chain Hamiltonian) gaps: heavy-hex 0.79, star 0.73, chord-chain 0.25, "
              "ring 0.23, brick 0.16, linear 0.10\n")
    if os.path.exists(gf):
        g = pd.read_csv(gf)
        if not quick:
            g = g[g.n_runs == GRAD_RUNS]
        print("=== Check 2: heavy-hex (IBM), measured qubit 0 (degree 1) vs qubit 1 (degree 3) ===")
        for metric in ("G_q0", "G_q1", "Gall_q0", "Gall_q1"):
            m = {(int(a), int(b)): x for a, b, x in
                 zip(g[g.metric == metric].n_qubits, g[g.metric == metric].n_layers, g[g.metric == metric]["mean"])}
            try:
                al = [_slope(range(3, 11), [m[(n, L)] for n in range(3, 11)]) for L in range(1, 9)]
                dadl = stats.linregress(range(2, 9), al[1:]).slope
                b8 = _slope(range(2, 9), [m[(8, L)] for L in range(2, 9)])
                extra = ""
                if all((n, 8) in m for n in range(3, 13)):
                    extra = f", alpha(L=8, n<=12) = {_slope(range(3, 13), [m[(n, 8)] for n in range(3, 13)]):.2f}"
                print(f"{metric:>8}: d(alpha)/dL = {dadl:.3f}, alpha(L=8) = {al[-1]:.2f}, "
                      f"beta(n=8) = {b8:.2f}, |alpha/beta| = {abs(al[-1]/b8):.2f}{extra}")
            except KeyError as e:
                print(f"{metric:>8}: sweep incomplete, missing cell {e}")
        print("Paper: heavy-hex (IBM, qubit 0) d(alpha)/dL -0.300, ratio 0.78; "
              "global group -0.443 to -0.449, ratio 0.99-1.07; linear -0.136, 0.48")
        print("Rule prediction for qubit 1 (>= 2 partners): global group.")


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["vqe", "grad", "all"], default="all")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs_reviewer_checks"))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--resume-from", nargs="*", default=[])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.quick and a.out.endswith("outputs_reviewer_checks"):
        a.out += "_quick"
    if a.analyze:
        analyze(a.out, a.quick)
    else:
        run(a)
