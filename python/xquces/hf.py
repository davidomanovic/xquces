# hf.py
from __future__ import annotations
from typing import Tuple, Union
from qiskit.circuit import QuantumCircuit, QuantumRegister

def hartree_fock(
    norb: int,
    nelec: Union[int, Tuple[int, int]],
    *,
    name: str = "HF",
) -> QuantumCircuit:
    if norb <= 0:
        raise ValueError("norb must be positive.")
    qreg = QuantumRegister(2 * norb, "q")
    qc = QuantumCircuit(qreg, name=name)
    for q in occupied_qubits_for_hf(norb, nelec):
        qc.x(q)
    return qc

def _parse_nelec(norb: int, nelec: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(nelec, tuple):
        n_alpha, n_beta = nelec
    else:
        if nelec < 0 or nelec % 2 != 0:
            raise ValueError("Integer nelec must be non-negative and even for closed-shell.")
        n_alpha = n_beta = nelec // 2
    if not (0 <= n_alpha <= norb and 0 <= n_beta <= norb):
        raise ValueError(f"n_alpha ({n_alpha}) and n_beta ({n_beta}) must be ≤ norb={norb}.")
    return int(n_alpha), int(n_beta)

def _index_blocked(norb: int, p: int, spin: str) -> int:
    return (p if spin == "alpha" else norb + p)

def occupied_qubits_for_hf(
    norb: int,
    nelec: Union[int, Tuple[int, int]],
) -> Tuple[int, ...]:
    if norb <= 0:
        raise ValueError("norb must be positive.")
    n_alpha, n_beta = _parse_nelec(norb, nelec)

    idx = []

    # Fill α block, then β block
    for p in range(n_alpha):
        idx.append(_index_blocked(norb, p, "alpha"))
    for p in range(n_beta):
        idx.append(_index_blocked(norb, p, "beta"))

    return tuple(idx)
