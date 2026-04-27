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
from qiskit.circuit.library import CPhaseGate, MCPhaseGate, PhaseGate

from xquces.qiskit.gates.diag_2 import (
    Diag2SpinRestrictedJW,
    _as_real_vector,
    _as_square_real_matrix,
    _nonzero,
)


def _as_omega_vector(omega_values: np.ndarray, norb: int) -> np.ndarray:
    omega = np.asarray(omega_values, dtype=np.float64)
    expected = (len(list(itertools.combinations(range(norb), 3))),)
    if omega.shape != expected:
        raise ValueError(f"omega_values must have shape {expected}")
    return omega


def _number_product_phase_gate(theta: float, nqubits: int) -> Gate:
    if nqubits == 1:
        return PhaseGate(theta)
    if nqubits == 2:
        return CPhaseGate(theta)
    return MCPhaseGate(theta, nqubits - 1)


def _yield_number_product_phase(
    qubits: Sequence[Qubit],
    theta: float,
    indices: Sequence[int],
) -> Iterator[CircuitInstruction]:
    if not _nonzero(theta):
        return
    if len(set(indices)) != len(indices):
        raise ValueError("number-product phase indices must be distinct")
    gate = _number_product_phase_gate(theta, len(indices))
    yield CircuitInstruction(gate, tuple(qubits[i] for i in indices))


class Diag3SpinRestrictedJW(Gate):
    """Spin-restricted iGCR-3 diagonal operator in Jordan-Wigner form.

    The quadratic ``D_p`` and ``N_p N_q`` sectors are emitted by the existing
    iGCR-2 restricted diagonal gate.  The cubic sectors are exact number-product
    phase gadgets:

    - ``D_p N_q`` -> two three-qubit phases
      ``n_pa n_pb n_qa`` and ``n_pa n_pb n_qb``.
    - ``N_p N_q N_r`` -> eight three-qubit phases, one for each spin choice.
    """

    def __init__(
        self,
        norb: int,
        double_params: np.ndarray,
        pair_params: np.ndarray,
        tau_params: np.ndarray,
        omega_values: np.ndarray,
        *,
        time: float = 1.0,
        label: str | None = None,
    ):
        self.norb = int(norb)
        self.double_params = _as_real_vector(double_params, self.norb, "double_params")
        self.pair_params = _as_square_real_matrix(pair_params, self.norb, "pair_params")
        self.tau_params = _as_square_real_matrix(tau_params, self.norb, "tau_params")
        self.omega_values = _as_omega_vector(omega_values, self.norb)
        self.time = float(time)
        super().__init__("igcr3_diag3_restricted_jw", 2 * self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _diag3_spin_restricted_jw(
            qubits,
            self.double_params,
            self.pair_params,
            self.tau_params,
            self.omega_values,
            self.time,
            self.norb,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "Diag3SpinRestrictedJW":
        return Diag3SpinRestrictedJW(
            self.norb,
            self.double_params,
            self.pair_params,
            self.tau_params,
            self.omega_values,
            time=-self.time,
            label=self.label,
        )


def _diag3_spin_restricted_jw(
    qubits: Sequence[Qubit],
    double_params: np.ndarray,
    pair_params: np.ndarray,
    tau_params: np.ndarray,
    omega_values: np.ndarray,
    time: float,
    norb: int,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")

    yield CircuitInstruction(
        Diag2SpinRestrictedJW(
            norb,
            double_params,
            pair_params,
            time=time,
        ),
        tuple(qubits),
    )

    tau = np.asarray(tau_params, dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            theta = time * tau[p, q]
            yield from _yield_number_product_phase(
                qubits,
                theta,
                (p, norb + p, q),
            )
            yield from _yield_number_product_phase(
                qubits,
                theta,
                (p, norb + p, norb + q),
            )

    for theta0, (p, q, r) in zip(omega_values, itertools.combinations(range(norb), 3)):
        theta = time * float(theta0)
        for p_spin in (p, norb + p):
            for q_spin in (q, norb + q):
                for r_spin in (r, norb + r):
                    yield from _yield_number_product_phase(
                        qubits,
                        theta,
                        (p_spin, q_spin, r_spin),
                    )
