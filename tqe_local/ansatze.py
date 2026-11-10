"""
ansatze.py
==========
Hardware-efficient ansatz topologies. Identical rotation layers
(single RY per qubit per layer) so depth and width are the only varying
factors. Topology only differs in the entangling block.

Topologies
----------
linear         : nearest-neighbour CNOT chain, no wrap-around.
ring           : nearest-neighbour CNOT chain + wrap-around.
brick          : alternating even-odd / odd-even CZ pattern.
all_to_all     : CNOT between every pair (i, j), i < j.
star           : qubit 0 is the hub, CNOT(0, j) for all j > 0.
heavy_hex      : LEGACY approximation (chain + (i, i+2) chords). Kept only so
                 old results can be reproduced. Not a real heavy-hex graph.
heavy_hex_true : connected n-qubit subgraph of the real IBM Falcon 27-qubit
                 heavy-hex coupling map (BFS order from qubit 0).

Every ansatz accepts an optional ``layer_hook``. It is called once at the end
of every layer (after the entangling block). metrics.py passes the noise
channel through it, so noise is applied per layer, as the paper states.

Parameter count: n_qubits * n_layers for all topologies.
"""

from collections import deque

import pennylane as qml


# ---------------------------------------------------------------------------
# Real IBM Falcon r4 27-qubit heavy-hex coupling map (undirected edges).
# Source: IBM 27-qubit Falcon backends (e.g. ibmq_montreal / ibmq_kolkata).
# ---------------------------------------------------------------------------
FALCON27_EDGES = [
    (0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8), (6, 7),
    (7, 10), (8, 9), (8, 11), (10, 12), (11, 14), (12, 13), (12, 15),
    (13, 14), (14, 16), (15, 18), (16, 19), (17, 18), (18, 21), (19, 20),
    (19, 22), (21, 23), (22, 25), (23, 24), (24, 25), (25, 26),
]


def _falcon_subgraph(n_qubits, root=0):
    """Edges of the connected n-node subgraph of Falcon-27 reached by BFS from
    `root`, relabelled to wires 0..n-1. A BFS prefix of a connected graph is
    always connected."""
    adj = {}
    for a, b in FALCON27_EDGES:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    order, seen, queue = [], {root}, deque([root])
    while queue and len(order) < n_qubits:
        v = queue.popleft()
        order.append(v)
        for w in sorted(adj[v]):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    if len(order) < n_qubits:
        raise ValueError(f"Falcon-27 has only {len(order)} qubits reachable; asked for {n_qubits}")
    label = {v: i for i, v in enumerate(order)}
    keep = set(order)
    return [(label[a], label[b]) for a, b in FALCON27_EDGES if a in keep and b in keep]


def _heavy_hex_gates(n_qubits):
    """LEGACY approximation used in the original sweeps (not real heavy-hex)."""
    gates = []
    for i in range(n_qubits - 1):
        gates.append((i, i + 1))
    for i in range(0, n_qubits - 2, 3):
        if i + 2 < n_qubits:
            gates.append((i, i + 2))
    seen, unique = set(), []
    for g in gates:
        key = (min(g), max(g))
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


# ---------------------------------------------------------------------------
# Entangling blocks
# ---------------------------------------------------------------------------
def _entangle(name, n_qubits, layer):
    if name == "linear":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
    elif name == "ring":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
    elif name == "brick":
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
    elif name == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    elif name == "star":
        for j in range(1, n_qubits):
            qml.CNOT(wires=[0, j])
    elif name == "heavy_hex":
        for c, t in _heavy_hex_gates(n_qubits):
            qml.CNOT(wires=[c, t])
    elif name == "heavy_hex_true":
        for c, t in _falcon_subgraph(n_qubits):
            qml.CNOT(wires=[c, t])
    else:
        raise ValueError(f"Unknown topology: {name}")


TOPOLOGIES = ["linear", "ring", "brick", "all_to_all", "star", "heavy_hex", "heavy_hex_true"]


def _rotation_layer(params, n_qubits, idx):
    """Single RY rotation on every qubit. Returns updated idx."""
    for q in range(n_qubits):
        qml.RY(params[idx], wires=q)
        idx += 1
    return idx


def _build(name, params, n_qubits, n_layers, x=None, layer_hook=None):
    idx = 0
    for layer in range(n_layers):
        if x is not None:
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
        idx = _rotation_layer(params, n_qubits, idx)
        _entangle(name, n_qubits, layer)
        if layer_hook is not None:
            layer_hook()


def _make(name):
    def ansatz(params, n_qubits, n_layers, layer_hook=None):
        _build(name, params, n_qubits, n_layers, layer_hook=layer_hook)
    ansatz.__name__ = f"ansatz_{name}"
    return ansatz


def _make_data(name):
    def ansatz(x, params, n_qubits, n_layers, layer_hook=None):
        _build(name, params, n_qubits, n_layers, x=x, layer_hook=layer_hook)
    ansatz.__name__ = f"ansatz_{name}_with_data"
    return ansatz


ANSATZE = {name: _make(name) for name in TOPOLOGIES}
ANSATZE_WITH_DATA = {name: _make_data(name) for name in TOPOLOGIES}

# Backward-compatible names
ansatz_linear = ANSATZE["linear"]
ansatz_ring = ANSATZE["ring"]
ansatz_brick = ANSATZE["brick"]
ansatz_all_to_all = ANSATZE["all_to_all"]
ansatz_star = ANSATZE["star"]
ansatz_heavy_hex = ANSATZE["heavy_hex"]
ansatz_linear_with_data = ANSATZE_WITH_DATA["linear"]
ansatz_ring_with_data = ANSATZE_WITH_DATA["ring"]
ansatz_brick_with_data = ANSATZE_WITH_DATA["brick"]


def n_params(n_qubits, n_layers, ansatz_name=None):
    """Total trainable parameters. Identical for all topologies."""
    return n_qubits * n_layers
