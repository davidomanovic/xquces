from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CXGate, PhaseGate, RZZGate, XGate

from xquces.gcr.pair_gcr2 import (
    PairGCR2Ansatz,
    PairGCR2Parameterization,
    pair_givens_indices,
    pair_jastrow_indices,
)
from xquces.gcr.product_pair_uccd import _pair_uccd_ov_pairs
from xquces.qiskit.gates.product_pair_uccd import PairRegisterUCCDGivensJW


def _nonzero(value: float, atol: float = 1e-15) -> bool:
    return abs(float(value)) > atol


def _yield_rzz_layers(
    qubits: Sequence[Qubit],
    edges: Sequence[tuple[float, int, int]],
) -> Iterator[CircuitInstruction]:
    remaining = list(edges)
    while remaining:
        used: set[int] = set()
        deferred: list[tuple[float, int, int]] = []
        for theta, p, q in remaining:
            if p in used or q in used:
                deferred.append((theta, p, q))
                continue
            yield CircuitInstruction(RZZGate(theta), (qubits[p], qubits[q]))
            used.add(p)
            used.add(q)
        remaining = deferred


def _number_edges_to_rzz_terms(
    edges: Sequence[tuple[float, int, int]],
    nqubits: int,
) -> tuple[list[tuple[float, int, int]], np.ndarray, float]:
    rzz_edges: list[tuple[float, int, int]] = []
    one_body_phases = np.zeros(nqubits, dtype=np.float64)
    global_phase = 0.0
    for theta, p, q in edges:
        rzz_edges.append((-0.5 * theta, p, q))
        one_body_phases[p] += 0.5 * theta
        one_body_phases[q] += 0.5 * theta
        global_phase -= 0.25 * theta
    return rzz_edges, one_body_phases, global_phase


class PairGCR2JW(Gate):
    """Pair-register GCR-2 block on logical pair qubits.

    This gate implements ``G_L exp(i J_pair) G_R`` on the pair register.  The
    product-pUCCD reference is applied by the state-preparation helpers below.
    """

    def __init__(
        self,
        ansatz: PairGCR2Ansatz,
        *,
        time: float = 1.0,
        label: str | None = None,
    ):
        if not isinstance(ansatz, PairGCR2Ansatz):
            raise TypeError("ansatz must be a PairGCR2Ansatz")
        self.ansatz = ansatz
        self.time = float(time)
        super().__init__("pair_gcr2_jw", ansatz.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        instructions, global_phase = _pair_gcr2_jw(
            qubits,
            self.ansatz,
            time=self.time,
        )
        circuit.global_phase += global_phase
        for instruction in instructions:
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "PairGCR2JW":
        ansatz = PairGCR2Ansatz(
            self.ansatz.norb,
            self.ansatz.nocc,
            self.ansatz.reference_params,
            -self.ansatz.right_params,
            -self.ansatz.diagonal.jastrow_params,
            -self.ansatz.left_params,
        )
        return PairGCR2JW(ansatz, time=self.time, label=self.label)


def pair_gcr2_stateprep_jw_circuit(
    ansatz: PairGCR2Ansatz,
    *,
    time: float = 1.0,
    embed_spin_orbital: bool = False,
) -> QuantumCircuit:
    """Prepare product-pUCCD + pair-GCR2 from ``|0...0>``.

    By default this returns a circuit on ``norb`` logical pair qubits.  With
    ``embed_spin_orbital=True`` it returns a ``2 * norb`` qubit circuit and
    copies the final pair register into the beta register with CNOTs.
    """

    return product_pair_uccd_pair_gcr2_stateprep_jw_circuit(
        ansatz,
        time=time,
        embed_spin_orbital=embed_spin_orbital,
    )


def product_pair_uccd_pair_gcr2_stateprep_jw_circuit(
    ansatz_or_parameterization,
    params: np.ndarray | None = None,
    *,
    time: float = 1.0,
    embed_spin_orbital: bool = False,
) -> QuantumCircuit:
    """Prepare ``G_L exp(iJ) G_R |Phi_pUCCD>`` on a pair register."""

    if isinstance(ansatz_or_parameterization, PairGCR2Ansatz):
        if params is not None:
            raise ValueError("params must be None when an ansatz is supplied")
        ansatz = ansatz_or_parameterization
    elif isinstance(ansatz_or_parameterization, PairGCR2Parameterization):
        if params is None:
            raise ValueError("params are required with a parameterization")
        ansatz = ansatz_or_parameterization.ansatz_from_parameters(params)
    else:
        raise TypeError(
            "expected a PairGCR2Ansatz or PairGCR2Parameterization"
        )

    nqubits = 2 * ansatz.norb if embed_spin_orbital else ansatz.norb
    circuit = QuantumCircuit(nqubits)
    pair_qubits = circuit.qubits[: ansatz.norb]

    for instruction in _product_pair_uccd_pair_gcr2_stateprep_jw(
        pair_qubits,
        ansatz,
        time=time,
    ):
        circuit.append(instruction)

    if embed_spin_orbital:
        for p in range(ansatz.norb):
            circuit.append(
                CXGate(),
                (circuit.qubits[p], circuit.qubits[ansatz.norb + p]),
            )

    return circuit


def _product_pair_uccd_pair_gcr2_stateprep_jw(
    qubits: Sequence[Qubit],
    ansatz: PairGCR2Ansatz,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != ansatz.norb:
        raise ValueError("Expected ansatz.norb pair-register qubits.")

    for p in range(ansatz.nocc):
        yield CircuitInstruction(XGate(), (qubits[p],))

    for theta, (i, a) in zip(
        ansatz.reference_params,
        _pair_uccd_ov_pairs(ansatz.norb, ansatz.nocc),
    ):
        if _nonzero(theta):
            yield CircuitInstruction(
                PairRegisterUCCDGivensJW(time * float(theta)),
                (qubits[i], qubits[a]),
            )

    yield CircuitInstruction(PairGCR2JW(ansatz, time=time), tuple(qubits))


def _pair_gcr2_jw(
    qubits: Sequence[Qubit],
    ansatz: PairGCR2Ansatz,
    *,
    time: float,
) -> tuple[list[CircuitInstruction], float]:
    if len(qubits) != ansatz.norb:
        raise ValueError("Expected ansatz.norb pair-register qubits.")

    instructions: list[CircuitInstruction] = []

    for theta, (p, q) in zip(ansatz.right_params, pair_givens_indices(ansatz.norb)):
        if _nonzero(theta):
            instructions.append(
                CircuitInstruction(
                    PairRegisterUCCDGivensJW(time * float(theta)),
                    (qubits[p], qubits[q]),
                )
            )

    edges = [
        (time * float(theta), p, q)
        for theta, (p, q) in zip(
            ansatz.diagonal.jastrow_params,
            pair_jastrow_indices(ansatz.norb),
        )
        if _nonzero(theta)
    ]
    rzz_edges, one_body_phases, global_phase = _number_edges_to_rzz_terms(
        edges,
        ansatz.norb,
    )
    instructions.extend(_yield_rzz_layers(qubits, rzz_edges))
    for p, theta in enumerate(one_body_phases):
        if _nonzero(theta):
            instructions.append(CircuitInstruction(PhaseGate(theta), (qubits[p],)))

    for theta, (p, q) in zip(ansatz.left_params, pair_givens_indices(ansatz.norb)):
        if _nonzero(theta):
            instructions.append(
                CircuitInstruction(
                    PairRegisterUCCDGivensJW(time * float(theta)),
                    (qubits[p], qubits[q]),
                )
            )

    return instructions, global_phase


__all__ = [
    "PairGCR2JW",
    "pair_gcr2_stateprep_jw_circuit",
    "product_pair_uccd_pair_gcr2_stateprep_jw_circuit",
]
