"""
ansatze.py
==========
Three hardware-efficient ansatz topologies. Identical rotation layers
(single RY per qubit per layer) so depth and width are the only varying
factors. Topology only differs in the entangling block.

Topologies
----------
linear : nearest-neighbor CNOT chain, no wrap-around.
ring   : nearest-neighbor CNOT chain + wrap-around (your original HEA).
brick  : alternating even-odd / odd-even CZ pattern (your "alt" ansatz).

Parameter count: n_qubits * n_layers for all three.
"""

import pennylane as qml


def _rotation_layer(params, n_qubits, idx):
    """Single RY rotation on every qubit. Returns updated idx."""
    for q in range(n_qubits):
        qml.RY(params[idx], wires=q)
        idx += 1
    return idx


def ansatz_linear(params, n_qubits, n_layers):
    idx = 0
    for _ in range(n_layers):
        idx = _rotation_layer(params, n_qubits, idx)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


def ansatz_ring(params, n_qubits, n_layers):
    idx = 0
    for _ in range(n_layers):
        idx = _rotation_layer(params, n_qubits, idx)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])


def ansatz_brick(params, n_qubits, n_layers):
    idx = 0
    for layer in range(n_layers):
        idx = _rotation_layer(params, n_qubits, idx)
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])
        else:
            for q in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])


def ansatz_all_to_all(params, n_qubits, n_layers):
    """All-to-all: CNOT between every pair (i, j) with i < j in each layer."""
    idx = 0
    for _ in range(n_layers):
        idx = _rotation_layer(params, n_qubits, idx)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])


def ansatz_star(params, n_qubits, n_layers):
    """Star: qubit 0 is the hub, CNOT(0, j) for all j>0 in each layer."""
    idx = 0
    for _ in range(n_layers):
        idx = _rotation_layer(params, n_qubits, idx)
        for j in range(1, n_qubits):
            qml.CNOT(wires=[0, j])


def ansatz_linear_with_data(x, params, n_qubits, n_layers):
    """Data-encoding variant: RY(x[q]) before each trainable layer. Linear chain."""
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qml.RY(x[q], wires=q)
        idx = _rotation_layer(params, n_qubits, idx)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


def ansatz_ring_with_data(x, params, n_qubits, n_layers):
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qml.RY(x[q], wires=q)
        idx = _rotation_layer(params, n_qubits, idx)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])


def ansatz_brick_with_data(x, params, n_qubits, n_layers):
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RY(x[q], wires=q)
        idx = _rotation_layer(params, n_qubits, idx)
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])
        else:
            for q in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])


ANSATZE = {
    "linear": ansatz_linear,
    "ring": ansatz_ring,
    "brick": ansatz_brick,
    "all_to_all": ansatz_all_to_all,
    "star": ansatz_star,
}

ANSATZE_WITH_DATA = {
    "linear": ansatz_linear_with_data,
    "ring": ansatz_ring_with_data,
    "brick": ansatz_brick_with_data,
}


def n_params(n_qubits, n_layers, ansatz_name=None):
    """Total trainable parameters. Identical for all three topologies."""
    return n_qubits * n_layers
