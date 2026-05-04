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

from xquces.gcr.utils import (
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
)
from xquces.qiskit.gates.diag_2 import (
    _as_real_vector,
    _as_square_real_matrix,
    _nonzero,
)
from xquces.qiskit.gates.diag_3 import (
    Diag3SpinRestrictedJW,
    _as_omega_vector,
    _yield_number_product_phase,
)


def _as_eta_vector(eta_values: np.ndarray, norb: int) -> np.ndarray:
    eta = np.asarray(eta_values, dtype=np.float64)
    expected = (len(_default_eta_indices(norb)),)
    if eta.shape != expected:
        raise ValueError(f"eta_values must have shape {expected}")
    return eta


def _as_rho_vector(rho_values: np.ndarray, norb: int) -> np.ndarray:
    rho = np.asarray(rho_values, dtype=np.float64)
    expected = (len(_default_rho_indices(norb)),)
    if rho.shape != expected:
        raise ValueError(f"rho_values must have shape {expected}")
    return rho


def _as_sigma_vector(sigma_values: np.ndarray, norb: int) -> np.ndarray:
    sigma = np.asarray(sigma_values, dtype=np.float64)
    expected = (len(_default_sigma_indices(norb)),)
    if sigma.shape != expected:
        raise ValueError(f"sigma_values must have shape {expected}")
    return sigma


class Diag4SpinRestrictedJW(Gate):
    """Spin-restricted iGCR-4 diagonal operator in Jordan-Wigner form.

    The lower-order sectors are emitted by :class:`Diag3SpinRestrictedJW`.
    The quartic sectors are exact number-product phase gadgets:

    - ``D_p D_q`` -> one four-qubit phase.
    - ``D_p N_q N_r`` -> four four-qubit phases, one for each spin choice
      on ``q`` and ``r``.
    - ``N_p N_q N_r N_s`` -> sixteen four-qubit phases.
    """

    def __init__(
        self,
        norb: int,
        double_params: np.ndarray,
        pair_params: np.ndarray,
        tau_params: np.ndarray,
        omega_values: np.ndarray,
        eta_values: np.ndarray,
        rho_values: np.ndarray,
        sigma_values: np.ndarray,
        *,
        time: float = 1.0,
        label: str | None = None,
    ):
        self.norb = int(norb)
        self.double_params = _as_real_vector(double_params, self.norb, "double_params")
        self.pair_params = _as_square_real_matrix(pair_params, self.norb, "pair_params")
        self.tau_params = _as_square_real_matrix(tau_params, self.norb, "tau_params")
        self.omega_values = _as_omega_vector(omega_values, self.norb)
        self.eta_values = _as_eta_vector(eta_values, self.norb)
        self.rho_values = _as_rho_vector(rho_values, self.norb)
        self.sigma_values = _as_sigma_vector(sigma_values, self.norb)
        self.time = float(time)
        super().__init__("igcr4_diag4_restricted_jw", 2 * self.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _diag4_spin_restricted_jw(
            qubits,
            self.double_params,
            self.pair_params,
            self.tau_params,
            self.omega_values,
            self.eta_values,
            self.rho_values,
            self.sigma_values,
            self.time,
            self.norb,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "Diag4SpinRestrictedJW":
        return Diag4SpinRestrictedJW(
            self.norb,
            self.double_params,
            self.pair_params,
            self.tau_params,
            self.omega_values,
            self.eta_values,
            self.rho_values,
            self.sigma_values,
            time=-self.time,
            label=self.label,
        )


def _diag4_spin_restricted_jw(
    qubits: Sequence[Qubit],
    double_params: np.ndarray,
    pair_params: np.ndarray,
    tau_params: np.ndarray,
    omega_values: np.ndarray,
    eta_values: np.ndarray,
    rho_values: np.ndarray,
    sigma_values: np.ndarray,
    time: float,
    norb: int,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")

    yield CircuitInstruction(
        Diag3SpinRestrictedJW(
            norb,
            double_params,
            pair_params,
            tau_params,
            omega_values,
            time=time,
        ),
        tuple(qubits),
    )

    for theta0, (p, q) in zip(eta_values, _default_eta_indices(norb)):
        theta = time * float(theta0)
        yield from _yield_number_product_phase(
            qubits,
            theta,
            (p, norb + p, q, norb + q),
        )

    for theta0, (p, q, r) in zip(rho_values, _default_rho_indices(norb)):
        theta = time * float(theta0)
        if not _nonzero(theta):
            continue
        for q_spin in (q, norb + q):
            for r_spin in (r, norb + r):
                yield from _yield_number_product_phase(
                    qubits,
                    theta,
                    (p, norb + p, q_spin, r_spin),
                )

    for theta0, (p, q, r, s) in zip(sigma_values, _default_sigma_indices(norb)):
        theta = time * float(theta0)
        if not _nonzero(theta):
            continue
        for p_spin, q_spin, r_spin, s_spin in itertools.product(
            (p, norb + p),
            (q, norb + q),
            (r, norb + r),
            (s, norb + s),
        ):
            yield from _yield_number_product_phase(
                qubits,
                theta,
                (p_spin, q_spin, r_spin, s_spin),
            )
