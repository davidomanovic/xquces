from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import CircuitInstruction, Gate, QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library import PhaseGate, XXPlusYYGate

from xquces.orbitals import givens_decomposition


def _validate_unitary(mat: np.ndarray, name: str, rtol: float, atol: float) -> np.ndarray:
    out = np.asarray(mat, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(out.conj().T @ out, np.eye(out.shape[0]), rtol=rtol, atol=atol):
        raise ValueError(f"{name} must be unitary")
    return out


def _normalize_spinful_orbital_rotation(
    norb: int,
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    validate: bool,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        mat = _validate_unitary(orbital_rotation, "orbital_rotation", rtol, atol) if validate else np.asarray(orbital_rotation, dtype=np.complex128)
        if mat.shape != (norb, norb):
            raise ValueError("orbital_rotation has wrong shape")
        return mat, mat

    mat_a, mat_b = orbital_rotation
    identity = np.eye(norb, dtype=np.complex128)
    if mat_a is None:
        out_a = identity
    else:
        out_a = _validate_unitary(mat_a, "alpha orbital_rotation", rtol, atol) if validate else np.asarray(mat_a, dtype=np.complex128)
        if out_a.shape != (norb, norb):
            raise ValueError("alpha orbital_rotation has wrong shape")
    if mat_b is None:
        out_b = identity
    else:
        out_b = _validate_unitary(mat_b, "beta orbital_rotation", rtol, atol) if validate else np.asarray(mat_b, dtype=np.complex128)
        if out_b.shape != (norb, norb):
            raise ValueError("beta orbital_rotation has wrong shape")
    return out_a, out_b


class OrbitalRotationJW(Gate):
    """Spinful orbital rotation under the Jordan-Wigner transform.

    Qubits are ordered as all alpha spin-orbitals followed by all beta
    spin-orbitals.  The input matrix maps creation operators as
    ``a_i^dag -> sum_j U[j, i] a_j^dag``.
    """

    def __init__(
        self,
        norb: int,
        orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
        *,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        self.norb = int(norb)
        self.orbital_rotation_a, self.orbital_rotation_b = _normalize_spinful_orbital_rotation(
            self.norb,
            orbital_rotation,
            validate,
            rtol,
            atol,
        )
        super().__init__("orbital_rotation_jw", 2 * self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _orbital_rotation_jw(qubits[: self.norb], self.orbital_rotation_a):
            circuit.append(instruction)
        for instruction in _orbital_rotation_jw(qubits[self.norb :], self.orbital_rotation_b):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "OrbitalRotationJW":
        return OrbitalRotationJW(
            self.norb,
            (self.orbital_rotation_a.conj().T, self.orbital_rotation_b.conj().T),
            label=self.label,
            validate=False,
        )


class OrbitalRotationSpinlessJW(Gate):
    """Spinless orbital rotation under the Jordan-Wigner transform."""

    def __init__(
        self,
        norb: int,
        orbital_rotation: np.ndarray,
        *,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        self.norb = int(norb)
        self.orbital_rotation = (
            _validate_unitary(orbital_rotation, "orbital_rotation", rtol, atol)
            if validate
            else np.asarray(orbital_rotation, dtype=np.complex128)
        )
        if self.orbital_rotation.shape != (self.norb, self.norb):
            raise ValueError("orbital_rotation has wrong shape")
        super().__init__("orbital_rotation_spinless_jw", self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _orbital_rotation_jw(qubits, self.orbital_rotation):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "OrbitalRotationSpinlessJW":
        return OrbitalRotationSpinlessJW(
            self.norb,
            self.orbital_rotation.conj().T,
            label=self.label,
            validate=False,
        )


def _orbital_rotation_jw(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
) -> Iterator[CircuitInstruction]:
    givens_rotations, phase_shifts = givens_decomposition(orbital_rotation)
    for c, s, i, j in givens_rotations:
        theta = 2.0 * math.acos(max(-1.0, min(1.0, c)))
        beta = cmath.phase(s) - 0.5 * math.pi
        yield CircuitInstruction(XXPlusYYGate(theta, beta), (qubits[i], qubits[j]))
    for i, phase_shift in enumerate(phase_shifts):
        yield CircuitInstruction(PhaseGate(cmath.phase(phase_shift)), (qubits[i],))
