from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PhaseGate, RZZGate


def _iter_pairs(norb: int) -> Iterator[tuple[int, int]]:
    for distance in range(1, norb):
        for offset in range(distance + 1):
            for p in range(offset, norb - distance, distance + 1):
                yield p, p + distance


def _as_square_real_matrix(mat: np.ndarray, norb: int, name: str) -> np.ndarray:
    out = np.asarray(mat, dtype=np.float64)
    if out.shape != (norb, norb):
        raise ValueError(f"{name} must have shape {(norb, norb)}")
    return out


def _as_real_vector(vec: np.ndarray, norb: int, name: str) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64)
    if out.shape != (norb,):
        raise ValueError(f"{name} must have shape {(norb,)}")
    return out


def _nonzero(value: float, atol: float = 1e-15) -> bool:
    return abs(float(value)) > atol


def _yield_rzz_layers(
    qubits: Sequence[Qubit],
    edges: Sequence[tuple[float, int, int]],
) -> Iterator[CircuitInstruction]:
    """Emit commuting RZZ interactions in greedy disjoint-qubit layers."""
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
        # exp(i theta n_p n_q) = exp(-i theta / 4)
        #   * P_p(theta / 2) P_q(theta / 2) RZZ_pq(-theta / 2).
        rzz_edges.append((-0.5 * theta, p, q))
        one_body_phases[p] += 0.5 * theta
        one_body_phases[q] += 0.5 * theta
        global_phase -= 0.25 * theta
    return rzz_edges, one_body_phases, global_phase


def spin_balanced_number_terms_to_rzz(
    norb: int,
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
    *,
    time: float = 1.0,
) -> tuple[list[tuple[float, int, int]], np.ndarray, float]:
    same = _as_square_real_matrix(same_spin_params, norb, "same_spin_params")
    mixed = _as_square_real_matrix(mixed_spin_params, norb, "mixed_spin_params")
    edges: list[tuple[float, int, int]] = []
    one_body_phases = np.zeros(2 * norb, dtype=np.float64)

    for spin_offset in (0, norb):
        for p in range(norb):
            theta = 0.5 * time * same[p, p]
            if _nonzero(theta):
                one_body_phases[spin_offset + p] += theta

    for p in range(norb):
        theta = time * mixed[p, p]
        if _nonzero(theta):
            edges.append((theta, p, norb + p))

    for p, q in _iter_pairs(norb):
        theta_same = time * same[p, q]
        if _nonzero(theta_same):
            edges.append((theta_same, p, q))
            edges.append((theta_same, norb + p, norb + q))

        theta_pq = time * mixed[p, q]
        if _nonzero(theta_pq):
            edges.append((theta_pq, p, norb + q))
        theta_qp = time * mixed[q, p]
        if _nonzero(theta_qp):
            edges.append((theta_qp, q, norb + p))

    rzz_edges, pair_one_body_phases, global_phase = _number_edges_to_rzz_terms(
        edges,
        2 * norb,
    )
    one_body_phases += pair_one_body_phases
    return rzz_edges, one_body_phases, global_phase


def spin_restricted_number_terms_to_rzz(
    norb: int,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    *,
    time: float = 1.0,
) -> tuple[list[tuple[float, int, int]], np.ndarray, float]:
    double = _as_real_vector(double_params, norb, "double_params")
    pair = _as_square_real_matrix(pair_params, norb, "pair_params")
    edges: list[tuple[float, int, int]] = []

    for p in range(norb):
        theta = time * double[p]
        if _nonzero(theta):
            edges.append((theta, p, norb + p))

    for p, q in itertools.combinations(range(norb), 2):
        theta = time * pair[p, q]
        if not _nonzero(theta):
            continue
        edges.append((theta, p, q))
        edges.append((theta, norb + p, norb + q))
        edges.append((theta, p, norb + q))
        edges.append((theta, q, norb + p))

    return _number_edges_to_rzz_terms(edges, 2 * norb)


class Diag2SpinBalancedJW(Gate):
    """Spin-balanced iGCR-2 diagonal operator in Jordan-Wigner form.

    Qubits are ordered as all alpha spin-orbitals followed by all beta
    spin-orbitals.  The implemented unitary is the same number-representation
    diagonal used by :func:`xquces.gates.apply_ucj_spin_balanced`.
    """

    def __init__(
        self,
        norb: int,
        same_spin_params: np.ndarray,
        mixed_spin_params: np.ndarray,
        *,
        time: float = 1.0,
        emit_one_body_phases: bool = True,
        label: str | None = None,
    ):
        self.norb = int(norb)
        self.same_spin_params = _as_square_real_matrix(
            same_spin_params, self.norb, "same_spin_params"
        )
        self.mixed_spin_params = _as_square_real_matrix(
            mixed_spin_params, self.norb, "mixed_spin_params"
        )
        self.time = float(time)
        self.emit_one_body_phases = bool(emit_one_body_phases)
        super().__init__("igcr2_diag2_balanced_jw", 2 * self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        instructions, global_phase = _diag2_spin_balanced_jw(
            qubits,
            self.same_spin_params,
            self.mixed_spin_params,
            self.time,
            self.norb,
            emit_one_body_phases=self.emit_one_body_phases,
        )
        circuit.global_phase += global_phase
        for instruction in instructions:
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "Diag2SpinBalancedJW":
        return Diag2SpinBalancedJW(
            self.norb,
            self.same_spin_params,
            self.mixed_spin_params,
            time=-self.time,
            emit_one_body_phases=self.emit_one_body_phases,
            label=self.label,
        )


class Diag2SpinRestrictedJW(Gate):
    """Spin-restricted iGCR-2 diagonal operator in Jordan-Wigner form."""

    def __init__(
        self,
        norb: int,
        double_params: np.ndarray,
        pair_params: np.ndarray,
        *,
        time: float = 1.0,
        emit_one_body_phases: bool = True,
        label: str | None = None,
    ):
        self.norb = int(norb)
        self.double_params = _as_real_vector(double_params, self.norb, "double_params")
        self.pair_params = _as_square_real_matrix(pair_params, self.norb, "pair_params")
        self.time = float(time)
        self.emit_one_body_phases = bool(emit_one_body_phases)
        super().__init__("igcr2_diag2_restricted_jw", 2 * self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        instructions, global_phase = _diag2_spin_restricted_jw(
            qubits,
            self.double_params,
            self.pair_params,
            self.time,
            self.norb,
            emit_one_body_phases=self.emit_one_body_phases,
        )
        circuit.global_phase += global_phase
        for instruction in instructions:
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "Diag2SpinRestrictedJW":
        return Diag2SpinRestrictedJW(
            self.norb,
            self.double_params,
            self.pair_params,
            time=-self.time,
            emit_one_body_phases=self.emit_one_body_phases,
            label=self.label,
        )


def _diag2_spin_balanced_jw(
    qubits: Sequence[Qubit],
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
    time: float,
    norb: int,
    *,
    emit_one_body_phases: bool,
) -> tuple[list[CircuitInstruction], float]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")

    rzz_edges, one_body_phases, global_phase = spin_balanced_number_terms_to_rzz(
        norb,
        same_spin_params,
        mixed_spin_params,
        time=time,
    )
    instructions = list(_yield_rzz_layers(qubits, rzz_edges))
    if emit_one_body_phases:
        for i, theta in enumerate(one_body_phases):
            if _nonzero(theta):
                instructions.append(CircuitInstruction(PhaseGate(theta), (qubits[i],)))
    return instructions, global_phase


def _diag2_spin_restricted_jw(
    qubits: Sequence[Qubit],
    double_params: np.ndarray,
    pair_params: np.ndarray,
    time: float,
    norb: int,
    *,
    emit_one_body_phases: bool,
) -> tuple[list[CircuitInstruction], float]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")

    rzz_edges, one_body_phases, global_phase = spin_restricted_number_terms_to_rzz(
        norb,
        double_params,
        pair_params,
        time=time,
    )
    instructions = list(_yield_rzz_layers(qubits, rzz_edges))
    if emit_one_body_phases:
        for i, theta in enumerate(one_body_phases):
            if _nonzero(theta):
                instructions.append(CircuitInstruction(PhaseGate(theta), (qubits[i],)))
    return instructions, global_phase
