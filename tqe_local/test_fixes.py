"""
test_fixes.py
=============
Checks for the Nov 2026 fixes. Run:  python -m pytest -q test_fixes.py
(or just: python test_fixes.py)
"""
import numpy as np
import pennylane as qml

from ansatze import ANSATZE, ANSATZE_WITH_DATA, TOPOLOGIES, FALCON27_EDGES, _falcon_subgraph
from metrics import (_make_expval_qnode, _make_state_qnode, _haar_fidelities_real,
                     haar_frame_potential, frame_potential_ratio, gradient_variance_all,
                     effective_kernel_rank, haar_kl_divergence)
from noise import apply_layer_noise


def _ops(fn, *args, **kw):
    return qml.tape.make_qscript(fn)(*args, **kw).operations


def test_noise_applied_every_layer():
    n, L = 3, 4
    hook = lambda: apply_layer_noise("depolarizing_strong", n)
    for name in TOPOLOGIES:
        ops = _ops(ANSATZE[name], np.zeros(n * L), n, L, layer_hook=hook)
        k = sum(isinstance(o, qml.DepolarizingChannel) for o in ops)
        assert k == n * L, (name, k)


def test_noisy_qnode_differs_from_end_only_noise():
    """Per-layer noise must damp <Z0> more than a single channel at the end."""
    n, L = 2, 4
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 2 * np.pi, n * L)
    per_layer = float(_make_expval_qnode("linear", n, L, "depolarizing_strong")(p))
    dev = qml.device("default.mixed", wires=n)

    @qml.qnode(dev)
    def end_only(params):
        ANSATZE["linear"](params, n, L)
        apply_layer_noise("depolarizing_strong", n)
        return qml.expval(qml.PauliZ(0))
    assert abs(per_layer - float(end_only(p))) > 1e-6


def test_noiseless_equals_old_circuit():
    """With no hook the circuits are unchanged from the original code."""
    n, L = 3, 2
    p = np.linspace(0.1, 1.2, n * L)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def old_linear(params):
        idx = 0
        for _ in range(L):
            for q in range(n):
                qml.RY(params[idx], wires=q); idx += 1
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))
    assert np.isclose(float(_make_expval_qnode("linear", n, L, "noiseless")(p)), float(old_linear(p)))


def test_states_are_real():
    """RY + CNOT/CZ circuits give real amplitudes, so the real-Haar reference is right."""
    n, L = 4, 3
    rng = np.random.default_rng(2)
    for name in TOPOLOGIES:
        qnode, mode = _make_state_qnode(name, n, L, "noiseless")
        st = np.array(qnode(rng.uniform(0, 2 * np.pi, n * L)))
        assert np.max(np.abs(st.imag)) < 1e-10, name


def test_real_haar_sampler():
    rng = np.random.default_rng(3)
    for n in (2, 4, 6):
        d = 2 ** n
        f = _haar_fidelities_real(n, 200000, rng)
        assert abs(f.mean() * d - 1) < 0.03
        assert abs(np.mean(f ** 2) / haar_frame_potential(n) - 1) < 0.05


def test_falcon_subgraph():
    assert len(FALCON27_EDGES) == 28
    nodes = {v for e in FALCON27_EDGES for v in e}
    assert nodes == set(range(27))
    deg = {}
    for a, b in FALCON27_EDGES:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    assert max(deg.values()) <= 3
    for n in range(2, 11):
        E = _falcon_subgraph(n)
        adj = {i: set() for i in range(n)}
        for a, b in E:
            adj[a].add(b); adj[b].add(a)
        seen, stack = {0}, [0]
        while stack:
            v = stack.pop()
            for w in adj[v] - seen:
                seen.add(w); stack.append(w)
        assert len(seen) == n, n
        assert max(len(s) for s in adj.values()) <= 3


def test_new_metrics_run():
    for name in ("linear", "star", "heavy_hex_true"):
        r = gradient_variance_all(name, 3, 2, "noiseless", n_samples=5, cost="local")
        assert r["min"] <= r["mean"] <= r["max"]
        assert frame_potential_ratio(name, 3, 2, n_samples=10) > 0
        assert effective_kernel_rank(name, 3, 2, n_data=8) > 0
        assert haar_kl_divergence(name, 3, 2, n_samples=10) >= 0
    r = gradient_variance_all("ring", 2, 2, "amplitude_strong", n_samples=3, cost="z0")
    assert np.isfinite(r["mean"])


if __name__ == "__main__":
    import sys, inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
