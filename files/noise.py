"""
noise.py
========
Noise model and device construction.

Two noise families:
    - Depolarizing: uniform mixing toward the maximally mixed state.
    - Amplitude damping: directional decay toward |0>. Physically dominant
      on superconducting hardware (T1 process).

Applied per-qubit at the end of every layer. The rate corresponds to total
error per qubit per layer.

Presets:
    noiseless              : no noise (statevector simulator).
    depolarizing_realistic : p = 1e-3 per qubit per layer (IBM Heron class).
    depolarizing_strong    : p = 1e-2 per qubit per layer (older NISQ).
    amplitude_realistic    : gamma = 1e-3 per qubit per layer.
    amplitude_strong       : gamma = 1e-2 per qubit per layer.

The channel `kind` and rate `rate` are queried by metrics.py to apply the
correct channel inline in the qnode.
"""

import pennylane as qml


NOISE_PRESETS = {
    "noiseless": {
        "kind": "none",
        "rate": 0.0,
    },
    "depolarizing_realistic": {
        "kind": "depolarizing",
        "rate": 1e-3,
    },
    "depolarizing_strong": {
        "kind": "depolarizing",
        "rate": 1e-2,
    },
    "amplitude_realistic": {
        "kind": "amplitude_damping",
        "rate": 1e-3,
    },
    "amplitude_strong": {
        "kind": "amplitude_damping",
        "rate": 1e-2,
    },
}


def make_device(n_qubits, noise_preset="noiseless"):
    cfg = NOISE_PRESETS[noise_preset]
    if cfg["kind"] == "none":
        return qml.device("default.qubit", wires=n_qubits)
    return qml.device("default.mixed", wires=n_qubits)


def is_noiseless(noise_preset):
    return NOISE_PRESETS[noise_preset]["kind"] == "none"


def apply_layer_noise(noise_preset, n_qubits):
    """
    Apply per-qubit noise channel for the configured preset.
    Call this at the end of every ansatz layer.
    """
    cfg = NOISE_PRESETS[noise_preset]
    kind = cfg["kind"]
    rate = cfg["rate"]

    if kind == "none":
        return
    if kind == "depolarizing":
        for q in range(n_qubits):
            qml.DepolarizingChannel(rate, wires=q)
    elif kind == "amplitude_damping":
        for q in range(n_qubits):
            qml.AmplitudeDamping(rate, wires=q)
    else:
        raise ValueError(f"Unknown noise kind: {kind}")
