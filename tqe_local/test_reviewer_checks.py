"""
test_reviewer_checks.py — checks for run_reviewer_checks.py.
Run: python test_reviewer_checks.py   (about 2-4 minutes on 2-4 CPUs)

Regression values come from the data already used in the paper:
  cells_noiseless.csv, heavy_hex_true n=4 L=2: grad_var_theta0 = 0.377146,
                                                grad_var_z0_all = 0.108644
  outputs_hh_true/vqe_results.csv, heavy_hex_true deep seed 0: error 0.045394
  outputs_gap4/vqe_results.csv, brick deep seed 0: error 0.001515
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import run_reviewer_checks as R
from ansatze import _falcon_subgraph

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# 1. Structure: Hamiltonian couplings are exactly the heavy-hex ansatz edges
for n in (4, 6, 8):
    ham = set(R.coupling_edges(n, "hh"))
    ans = {tuple(sorted(e)) for e in _falcon_subgraph(n)}
    check(f"hh couplings == heavy_hex_true CNOT edges (n={n})", ham == ans, f"{sorted(ham)}")
check("chain couplings are (i,i+1)", R.coupling_edges(4, "chain") == [(0, 1), (1, 2), (2, 3)])

# 2. Measured qubit 1 has degree 3 for n >= 4, qubit 0 has degree 1
for n in range(4, 13):
    deg = {}
    for a, b in _falcon_subgraph(n):
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    check(f"degree(q0)=1, degree(q1)=3 at n={n}", deg[0] == 1 and deg[1] == 3)

# 3. Exact ground energy: chain version equals the original script's function
import run_gap4_vqe as V
for n in (4, 6):
    a, b = R.exact_ground_energy(n, R.coupling_edges(n, "chain")), V.exact_ground_energy(n)
    check(f"chain ground energy matches run_gap4_vqe (n={n})", abs(a - b) < 1e-9, f"{a:.6f} vs {b:.6f}")
# hh ground energy vs dense diagonalization (independent route)
import pennylane as qml
for n in (4, 6):
    e = R.coupling_edges(n, "hh")
    dense = np.linalg.eigvalsh(qml.matrix(R.pl_hamiltonian(n, e), wire_order=range(n))).min()
    check(f"hh ground energy matches dense eig (n={n})", abs(dense - R.exact_ground_energy(n, e)) < 1e-8)

# 4. Regression: the paper's diagnostic is reproduced exactly (same seeds)
_, rows, err = R.grad_task((4, 2, 20, 50))
check("grad_task ran without error", err is None, str(err))
m = {r["metric"]: r["mean"] for r in rows}
check("G_q0 reproduces grad_var_theta0 (hh n=4 L=2)", abs(m["G_q0"] - 0.377146) < 1e-5, f"{m['G_q0']:.6f}")
check("Gall_q0 reproduces grad_var_z0_all (hh n=4 L=2)", abs(m["Gall_q0"] - 0.108644) < 1e-5, f"{m['Gall_q0']:.6f}")
check("G_q1 present and positive", m.get("G_q1", 0) > 0, f"{m.get('G_q1')}")

# 5. Regression: VQE on the chain Hamiltonian reproduces the paper's runs
en, _ = R.run_vqe("heavy_hex_true", 4, 6, 0, "chain")
e0 = R.exact_ground_energy(4, R.coupling_edges(4, "chain"))
check("VQE chain heavy_hex_true deep seed0 reproduces 0.045394", abs(en[-1] - e0 - 0.045394) < 1e-5,
      f"{en[-1]-e0:.6f}")
en, _ = R.run_vqe("brick", 4, 6, 0, "chain")
check("VQE chain brick deep seed0 reproduces 0.001515", abs(en[-1] - e0 - 0.001515) < 1e-5, f"{en[-1]-e0:.6f}")

# 6. Variational bound on the new Hamiltonian
_, row, err = R.vqe_task(("heavy_hex_true", "deep", 4, 6, 0, "hh", 300))
check("hh VQE task ran", err is None, str(err))
check("hh VQE energy >= exact ground energy", row["error_final"] > -1e-8, f"{row['error_final']:.6f}")

print(f"\n{'ALL TESTS PASSED' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
