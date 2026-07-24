"""
metrics.py
==========
Five metrics. Use apply_layer_noise() to inject noise at the end of every layer,
so any noise preset registered in noise.py works automatically.

Metrics:
    haar_kl_divergence       : KL from Haar fidelity distribution.
    effective_kernel_rank    : effective rank of empirical kernel matrix.
    gradient_variance        : variance of d<Z_0>/dtheta_0 over random params.
    function_output_variance : variance of circuit output over random inputs.
    qfim_metrics             : QFIM trace and Abbas effective dimension.
"""

import itertools
import numpy as np
import pennylane as qml
from scipy.stats import entropy

from ansatze import ANSATZE, ANSATZE_WITH_DATA, n_params
from noise import make_device, NOISE_PRESETS, apply_layer_noise, is_noiseless


# ---------------------------------------------------------------------------
# QNode builders
# ---------------------------------------------------------------------------
def _make_state_qnode(ansatz_name, n_qubits, n_layers, noise_preset):
    ansatz = ANSATZE[ansatz_name]
    dev = make_device(n_qubits, noise_preset)

    if is_noiseless(noise_preset):
        @qml.qnode(dev)
        def circuit(params):
            ansatz(params, n_qubits, n_layers)
            return qml.state()
        return circuit, "pure"

    @qml.qnode(dev)
    def circuit(params):
        ansatz(params, n_qubits, n_layers)
        apply_layer_noise(noise_preset, n_qubits)
        return qml.density_matrix(wires=range(n_qubits))
    return circuit, "mixed"


def _make_expval_qnode(ansatz_name, n_qubits, n_layers, noise_preset, observable=None):
    ansatz = ANSATZE[ansatz_name]
    dev = make_device(n_qubits, noise_preset)
    if observable is None:
        observable = qml.PauliZ(0)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        ansatz(params, n_qubits, n_layers)
        apply_layer_noise(noise_preset, n_qubits)
        return qml.expval(observable)
    return circuit


def _make_data_qnode(ansatz_name, n_qubits, n_layers, noise_preset):
    if ansatz_name not in ANSATZE_WITH_DATA:
        raise NotImplementedError(
            f"Data-encoding variant not implemented for {ansatz_name}. "
            f"Add it to ansatze.py if you need function-space metrics for this topology."
        )
    ansatz = ANSATZE_WITH_DATA[ansatz_name]
    dev = make_device(n_qubits, noise_preset)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        ansatz(x, params, n_qubits, n_layers)
        apply_layer_noise(noise_preset, n_qubits)
        return qml.expval(qml.PauliZ(0))
    return circuit


def _make_data_state_qnode(ansatz_name, n_qubits, n_layers, noise_preset):
    if ansatz_name not in ANSATZE_WITH_DATA:
        raise NotImplementedError(
            f"Data-encoding variant not implemented for {ansatz_name}."
        )
    ansatz = ANSATZE_WITH_DATA[ansatz_name]
    dev = make_device(n_qubits, noise_preset)

    if is_noiseless(noise_preset):
        @qml.qnode(dev)
        def circuit(x, params):
            ansatz(x, params, n_qubits, n_layers)
            return qml.state()
        return circuit

    @qml.qnode(dev)
    def circuit(x, params):
        ansatz(x, params, n_qubits, n_layers)
        apply_layer_noise(noise_preset, n_qubits)
        return qml.density_matrix(wires=range(n_qubits))
    return circuit


# ---------------------------------------------------------------------------
# Haar reference
# ---------------------------------------------------------------------------
def _haar_fidelities_pure(n_qubits, n_pairs, rng):
    dim = 2 ** n_qubits
    fids = np.empty(n_pairs)
    for k in range(n_pairs):
        a = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        a /= np.linalg.norm(a)
        b = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        b /= np.linalg.norm(b)
        fids[k] = np.abs(np.vdot(a, b)) ** 2
    return fids


def _fidelity(state_a, state_b, mode):
    if mode == "pure":
        return float(np.abs(np.vdot(state_a, state_b)) ** 2)
    return float(np.real(np.trace(state_a @ state_b)))


# ---------------------------------------------------------------------------
# Metric 1: Haar KL
# ---------------------------------------------------------------------------
def haar_kl_divergence(ansatz_name, n_qubits, n_layers, noise_preset="noiseless",
                       n_samples=60, n_bins=50, seed=0):
    rng = np.random.default_rng(seed)
    qnode, mode = _make_state_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
    n_p = n_params(n_qubits, n_layers)

    states = []
    for _ in range(n_samples):
        params = rng.uniform(0, 2 * np.pi, size=n_p)
        states.append(qnode(params))

    pairs = list(itertools.combinations(range(n_samples), 2))
    vqc_fids = np.array([_fidelity(states[i], states[j], mode) for i, j in pairs])
    haar_fids = _haar_fidelities_pure(n_qubits, len(pairs), rng)

    edges = np.linspace(0, 1, n_bins + 1)
    h_vqc, _ = np.histogram(vqc_fids, bins=edges, density=False)
    h_haar, _ = np.histogram(haar_fids, bins=edges, density=False)
    h_vqc = h_vqc.astype(float) + 1e-10
    h_haar = h_haar.astype(float) + 1e-10
    h_vqc /= h_vqc.sum()
    h_haar /= h_haar.sum()
    return float(entropy(h_vqc, h_haar))


# ---------------------------------------------------------------------------
# Metric 2: Effective kernel rank
# ---------------------------------------------------------------------------
def effective_kernel_rank(ansatz_name, n_qubits, n_layers, noise_preset="noiseless",
                          n_data=20, seed=0):
    rng = np.random.default_rng(seed)
    qnode = _make_data_state_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
    n_p = n_params(n_qubits, n_layers)
    params = rng.uniform(0, 2 * np.pi, size=n_p)
    X = rng.uniform(0, np.pi, size=(n_data, n_qubits))

    states = [qnode(x, params) for x in X]
    mode = "pure" if is_noiseless(noise_preset) else "mixed"

    K = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(n_data):
            K[i, j] = _fidelity(states[i], states[j], mode)
    K = (K + K.T) / 2

    tr = np.trace(K)
    tr_sq = np.trace(K @ K)
    if tr_sq <= 0:
        return 0.0
    return float(tr ** 2 / tr_sq)


# ---------------------------------------------------------------------------
# Metric 3: Gradient variance
# ---------------------------------------------------------------------------
def gradient_variance(ansatz_name, n_qubits, n_layers, noise_preset="noiseless",
                      n_samples=50, seed=0, use_data=False):
    """
    Variance of d<Z_0>/dtheta_0 over random parameter samples.
    Standard barren-plateau diagnostic (McClean 2018, Cerezo 2021).
    """
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    n_p = n_params(n_qubits, n_layers)

    if use_data:
        qnode = _make_data_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
        x_fixed = rng.uniform(0, np.pi, size=n_qubits)
        def cost(p):
            return qnode(x_fixed, p)
    else:
        qnode = _make_expval_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
        def cost(p):
            return qnode(p)

    grad_fn = qml.grad(cost, argnums=0)

    g0_samples = []
    for _ in range(n_samples):
        params = pnp.array(rng.uniform(0, 2 * np.pi, size=n_p), requires_grad=True)
        g = grad_fn(params)
        g0_samples.append(float(g[0]))

    return float(np.var(g0_samples))


# ---------------------------------------------------------------------------
# Metric 4: Function output variance
# ---------------------------------------------------------------------------
def function_output_variance(ansatz_name, n_qubits, n_layers, noise_preset="noiseless",
                             n_data=30, seed=0):
    rng = np.random.default_rng(seed)
    qnode = _make_data_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
    n_p = n_params(n_qubits, n_layers)
    params = rng.uniform(0, 2 * np.pi, size=n_p)
    X = rng.uniform(0, np.pi, size=(n_data, n_qubits))
    outs = np.array([float(qnode(x, params)) for x in X])
    return float(np.var(outs))


# ---------------------------------------------------------------------------
# Metric 5: QFIM
# ---------------------------------------------------------------------------
def qfim_metrics(ansatz_name, n_qubits, n_layers, noise_preset="noiseless",
                 n_samples=15, n_data_for_eff_dim=1000, seed=0,
                 gamma=1.0, approx="block-diag"):
    """
    Returns dict with 'trace_F' (mean QFIM trace) and 'eff_dim' (Abbas et al. 2021).
    Uses block-diagonal Fubini-Study metric (factor of 4 for QFIM).
    """
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    qnode = _make_expval_qnode(ansatz_name, n_qubits, n_layers, noise_preset)
    n_p = n_params(n_qubits, n_layers)

    traces = []
    fishers = []
    for _ in range(n_samples):
        params = pnp.array(rng.uniform(0, 2 * np.pi, size=n_p), requires_grad=True)
        try:
            F = qml.metric_tensor(qnode, approx=approx)(params)
            F = np.array(F)
            F_qfim = 4.0 * F
            traces.append(float(np.trace(F_qfim)))
            fishers.append(F_qfim)
        except Exception:
            continue

    if len(fishers) == 0:
        return {"trace_F": float("nan"), "eff_dim": float("nan")}

    trace_F = float(np.mean(traces))

    M = n_data_for_eff_dim
    kappa = gamma * M / (2 * np.pi * np.log(max(M, 2)))
    if kappa <= 1.0:
        return {"trace_F": trace_F, "eff_dim": float("nan")}

    sqrt_dets = []
    for F_qfim in fishers:
        norm = n_p / max(np.trace(F_qfim), 1e-12)
        F_hat = norm * F_qfim
        F_hat_sym = (F_hat + F_hat.T) / 2
        eigvals = np.linalg.eigvalsh(F_hat_sym)
        eigvals = np.clip(eigvals, 0.0, None)
        log_det = float(np.sum(np.log1p(kappa * eigvals)))
        sqrt_dets.append(0.5 * log_det)

    sd = np.array(sqrt_dets)
    smax = sd.max()
    log_mean_sqrt_det = smax + np.log(np.mean(np.exp(sd - smax)))
    eff_dim = float(2.0 * log_mean_sqrt_det / np.log(kappa))
    eff_dim = max(0.0, eff_dim)
    return {"trace_F": trace_F, "eff_dim": eff_dim}
