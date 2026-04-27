from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from ffsim.qiskit.gates import PrepareSlaterDeterminantJW
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)

from xquces.gcr.igcr2 import IGCR2Ansatz, IGCR2SpinBalancedSpec, IGCR2SpinRestrictedSpec
from xquces.qiskit.gates.diag_2 import (
    Diag2SpinBalancedJW,
    Diag2SpinRestrictedJW,
    spin_balanced_number_terms_to_rzz,
)
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW


def _diag_unitary(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * np.asarray(phases, dtype=np.float64)))


def _iter_upper_pairs(norb: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(norb) for q in range(p + 1, norb)]


def _zero_small_values(values: np.ndarray, atol: float) -> np.ndarray:
    out = np.array(values, copy=True)
    out[np.abs(out) <= atol] = 0.0
    return out


def _balanced_irreducible_pair_matrices(
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    same = np.asarray(same_spin_params, dtype=np.float64)
    mixed = np.asarray(mixed_spin_params, dtype=np.float64)
    mixed_diag = np.diag(mixed)
    shift = 0.5 * (mixed_diag[:, None] + mixed_diag[None, :])
    same_red = np.array(same, copy=True)
    mixed_red = np.array(mixed, copy=True)
    mask = ~np.eye(same.shape[0], dtype=bool)
    same_red[mask] -= shift[mask]
    mixed_red[mask] -= shift[mask]
    np.fill_diagonal(same_red, 0.0)
    np.fill_diagonal(mixed_red, 0.0)
    return same_red, mixed_red


def _balanced_left_phase_vector(
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
    nocc: int,
) -> np.ndarray:
    same_diag = np.diag(np.asarray(same_spin_params, dtype=np.float64))
    mixed_diag = np.diag(np.asarray(mixed_spin_params, dtype=np.float64))
    return 0.5 * same_diag + 0.5 * (2 * int(nocc) - 1) * mixed_diag


def _additive_pair_sparsifying_shift(
    mat: np.ndarray,
    *,
    diagonal_weight: int = 0,
    atol: float = 1e-12,
) -> np.ndarray:
    """Find a vertex shift ``v`` making ``mat[p, q] + v[p] + v[q]`` sparse."""
    mat = np.asarray(mat, dtype=np.float64)
    norb = mat.shape[0]
    pairs = _iter_upper_pairs(norb)
    if not pairs:
        return np.zeros(norb, dtype=np.float64)

    candidates: list[np.ndarray] = [np.zeros(norb, dtype=np.float64)]

    rows = []
    rhs = []
    for p, q in pairs:
        row = np.zeros(norb, dtype=np.float64)
        row[p] = 1.0
        row[q] = 1.0
        rows.append(row)
        rhs.append(-mat[p, q])
    candidates.append(np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0])

    # A single anchor plus one non-anchor edge fixes a full-rank additive
    # representative for complete graphs while keeping the search cheap.
    for anchor in range(norb):
        others = [p for p in range(norb) if p != anchor]
        for i, p in enumerate(others):
            for q in others[i + 1 :]:
                anchor_shift = 0.5 * (mat[p, q] - mat[anchor, p] - mat[anchor, q])
                shift = np.zeros(norb, dtype=np.float64)
                shift[anchor] = anchor_shift
                for r in others:
                    shift[r] = -mat[anchor, r] - anchor_shift
                candidates.append(shift)

    best = candidates[0]
    best_score: tuple[float, ...] | None = None
    for shift in candidates:
        shifted = np.array([mat[p, q] + shift[p] + shift[q] for p, q in pairs])
        offdiag_count = int(np.count_nonzero(np.abs(shifted) > atol))
        diagonal_count = int(np.count_nonzero(np.abs(shift) > atol))
        score = (
            2 * offdiag_count + diagonal_weight * diagonal_count,
            offdiag_count,
            diagonal_count,
            float(np.sum(np.abs(shifted))),
            float(np.linalg.norm(shift)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = shift
    return best


@dataclass(frozen=True)
class IGCR2SpinBalancedCircuitGauge:
    right: np.ndarray
    left: np.ndarray
    same_spin_params: np.ndarray
    mixed_spin_params: np.ndarray


@dataclass(frozen=True)
class IGCR2SpinBalancedRZZCircuitGauge:
    right: np.ndarray | tuple[np.ndarray, np.ndarray]
    left: np.ndarray | tuple[np.ndarray, np.ndarray]
    same_spin_params: np.ndarray
    mixed_spin_params: np.ndarray


def spin_balanced_circuit_gauge(
    ansatz: IGCR2Ansatz,
    *,
    atol: float = 1e-12,
) -> IGCR2SpinBalancedCircuitGauge:
    """Choose a fixed-sector circuit gauge for a spin-balanced iGCR-2 ansatz.

    The optimization parameterization stores an irreducible diagonal basis.  That
    basis is good for tangent conditioning, but it can be dense as a JW phase
    network.  On the fixed ``(nocc, nocc)`` sector, additive orbital-wise
    same-spin and mixed-spin pair shifts are one-body phases; this helper picks
    a sparse representative and absorbs the compensating phases into ``U_L``.
    """
    if not isinstance(ansatz.diagonal, IGCR2SpinBalancedSpec):
        raise TypeError("expected a spin-balanced iGCR-2 ansatz")

    diagonal = ansatz.diagonal.to_standard()
    same_base, mixed_base = _balanced_irreducible_pair_matrices(
        diagonal.same_spin_params,
        diagonal.mixed_spin_params,
    )
    initial_left_phase = _balanced_left_phase_vector(
        diagonal.same_spin_params,
        diagonal.mixed_spin_params,
        ansatz.nocc,
    )

    same_shift = _additive_pair_sparsifying_shift(
        same_base,
        diagonal_weight=0,
        atol=atol,
    )
    mixed_shift = _additive_pair_sparsifying_shift(
        mixed_base,
        diagonal_weight=1,
        atol=atol,
    )

    norb = ansatz.norb
    same_sparse = np.array(same_base, copy=True)
    mixed_sparse = np.array(mixed_base, copy=True)
    for p, q in _iter_upper_pairs(norb):
        same_sparse[p, q] = same_sparse[q, p] = (
            same_base[p, q] + same_shift[p] + same_shift[q]
        )
        mixed_sparse[p, q] = mixed_sparse[q, p] = (
            mixed_base[p, q] + mixed_shift[p] + mixed_shift[q]
        )
    np.fill_diagonal(same_sparse, 0.0)
    np.fill_diagonal(mixed_sparse, 2.0 * mixed_shift)

    same_sparse = _zero_small_values(same_sparse, atol)
    mixed_sparse = _zero_small_values(mixed_sparse, atol)

    added_one_body_phase = (ansatz.nocc - 1) * same_shift + ansatz.nocc * mixed_shift
    left_phase = initial_left_phase - added_one_body_phase
    left = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(left_phase)

    return IGCR2SpinBalancedCircuitGauge(
        right=np.asarray(ansatz.right, dtype=np.complex128),
        left=left,
        same_spin_params=same_sparse,
        mixed_spin_params=mixed_sparse,
    )


def spin_balanced_rzz_circuit_gauge(
    ansatz: IGCR2Ansatz,
    *,
    atol: float = 1e-12,
) -> IGCR2SpinBalancedRZZCircuitGauge:
    """Choose a sparse spin-balanced gauge suited for native RZZ lowering.

    The number-representation diagonal is first sparsified.  Its one-body
    phases from ``n_p n_q -> RZZ`` conversion are then split across the
    neighboring orbital rotations, so the diagonal layer itself emits only the
    entangling RZZ network.
    """
    gauge = spin_balanced_circuit_gauge(ansatz, atol=atol)
    _, one_body_phases, _ = spin_balanced_number_terms_to_rzz(
        ansatz.norb,
        gauge.same_spin_params,
        gauge.mixed_spin_params,
    )
    phase_a = one_body_phases[: ansatz.norb]
    phase_b = one_body_phases[ansatz.norb :]
    half_phase_a = _diag_unitary(0.5 * phase_a)
    half_phase_b = _diag_unitary(0.5 * phase_b)
    right = (
        half_phase_a @ gauge.right,
        half_phase_b @ gauge.right,
    )
    left = (
        gauge.left @ half_phase_a,
        gauge.left @ half_phase_b,
    )
    return IGCR2SpinBalancedRZZCircuitGauge(
        right=right,
        left=left,
        same_spin_params=gauge.same_spin_params,
        mixed_spin_params=gauge.mixed_spin_params,
    )


class IGCR2JW(Gate):
    """Full iGCR-2 ansatz gate under the Jordan-Wigner transform.

    The implemented unitary is ``U_L exp(i J) U_R`` with the same alpha-first,
    beta-second qubit ordering as the rest of the Qiskit gates in this package.
    """

    def __init__(
        self,
        ansatz: IGCR2Ansatz,
        *,
        label: str | None = None,
        validate_orbital_rotations: bool = True,
        sparsify_diagonal: bool = True,
        sparsify_atol: float = 1e-12,
    ):
        self.ansatz = ansatz
        self.validate_orbital_rotations = bool(validate_orbital_rotations)
        self.sparsify_diagonal = bool(sparsify_diagonal)
        self.sparsify_atol = float(sparsify_atol)
        super().__init__("igcr2_jw", 2 * ansatz.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _igcr2_jw(
                qubits,
                self.ansatz,
                validate_orbital_rotations=self.validate_orbital_rotations,
                sparsify_diagonal=self.sparsify_diagonal,
                sparsify_atol=self.sparsify_atol,
            ),
            qubits=qubits,
            name=self.name,
        )


def igcr2_jw_circuit(
    ansatz: IGCR2Ansatz,
    *,
    validate_orbital_rotations: bool = True,
    sparsify_diagonal: bool = True,
    sparsify_atol: float = 1e-12,
) -> QuantumCircuit:
    """Build a circuit for an iGCR-2 ansatz under Jordan-Wigner."""
    circuit = QuantumCircuit(2 * ansatz.norb)
    circuit.append(
        IGCR2JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            sparsify_diagonal=sparsify_diagonal,
            sparsify_atol=sparsify_atol,
        ),
        circuit.qubits,
    )
    return circuit


def igcr2_stateprep_jw_circuit(
    ansatz: IGCR2Ansatz,
    *,
    validate_orbital_rotations: bool = True,
    sparsify_diagonal: bool = True,
    sparsify_atol: float = 1e-12,
) -> QuantumCircuit:
    """Build an iGCR-2 state-preparation circuit under Jordan-Wigner.

    Unlike :class:`IGCR2JW`, this assumes the input is ``|0...0>`` and prepares
    ``U_L exp(iJ) U_R |Phi_0>`` directly.  Therefore the right factor is lowered
    with :class:`ffsim.qiskit.gates.PrepareSlaterDeterminantJW`, which only
    depends on the occupied columns of ``U_R`` instead of implementing ``U_R`` as
    a generic orbital rotation on an arbitrary input state.
    """
    circuit = QuantumCircuit(2 * ansatz.norb)
    for instruction in _igcr2_stateprep_jw(
        circuit.qubits,
        ansatz,
        validate_orbital_rotations=validate_orbital_rotations,
        sparsify_diagonal=sparsify_diagonal,
        sparsify_atol=sparsify_atol,
    ):
        circuit.append(instruction)
    return circuit


def _igcr2_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR2Ansatz,
    *,
    validate_orbital_rotations: bool,
    sparsify_diagonal: bool,
    sparsify_atol: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    right, left, diagonal, emit_one_body_phases = _igcr2_circuit_factors(
        ansatz,
        sparsify_diagonal=sparsify_diagonal,
        sparsify_atol=sparsify_atol,
    )

    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            right,
            validate=validate_orbital_rotations,
        ),
        qubits,
    )

    if isinstance(ansatz.diagonal, IGCR2SpinRestrictedSpec):
        yield CircuitInstruction(
            Diag2SpinRestrictedJW(
                ansatz.norb,
                diagonal.double_params,
                diagonal.pair_params,
            ),
            qubits,
        )
    elif isinstance(ansatz.diagonal, IGCR2SpinBalancedSpec):
        yield CircuitInstruction(
            Diag2SpinBalancedJW(
                ansatz.norb,
                diagonal.same_spin_params,
                diagonal.mixed_spin_params,
                emit_one_body_phases=emit_one_body_phases,
            ),
            qubits,
        )
    else:
        raise TypeError("Unsupported iGCR-2 diagonal specification.")

    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            left,
            validate=validate_orbital_rotations,
        ),
        qubits,
    )


def _igcr2_stateprep_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR2Ansatz,
    *,
    validate_orbital_rotations: bool,
    sparsify_diagonal: bool,
    sparsify_atol: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    right, left, diagonal, emit_one_body_phases = _igcr2_circuit_factors(
        ansatz,
        sparsify_diagonal=sparsify_diagonal,
        sparsify_atol=sparsify_atol,
    )

    occupied = (range(ansatz.nocc), range(ansatz.nocc))
    yield CircuitInstruction(
        PrepareSlaterDeterminantJW(
            ansatz.norb,
            occupied,
            orbital_rotation=right,
            validate=validate_orbital_rotations,
        ),
        qubits,
    )

    if isinstance(ansatz.diagonal, IGCR2SpinRestrictedSpec):
        yield CircuitInstruction(
            Diag2SpinRestrictedJW(
                ansatz.norb,
                diagonal.double_params,
                diagonal.pair_params,
            ),
            qubits,
        )
    elif isinstance(ansatz.diagonal, IGCR2SpinBalancedSpec):
        yield CircuitInstruction(
            Diag2SpinBalancedJW(
                ansatz.norb,
                diagonal.same_spin_params,
                diagonal.mixed_spin_params,
                emit_one_body_phases=emit_one_body_phases,
            ),
            qubits,
        )
    else:
        raise TypeError("Unsupported iGCR-2 diagonal specification.")

    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            left,
            validate=validate_orbital_rotations,
        ),
        qubits,
    )


def _igcr2_circuit_factors(
    ansatz: IGCR2Ansatz,
    *,
    sparsify_diagonal: bool,
    sparsify_atol: float,
):
    right = ansatz.right
    left = ansatz.left
    diagonal = ansatz.diagonal.to_standard()
    emit_one_body_phases = True
    if isinstance(ansatz.diagonal, IGCR2SpinBalancedSpec) and sparsify_diagonal:
        gauge = spin_balanced_rzz_circuit_gauge(ansatz, atol=sparsify_atol)
        right = gauge.right
        left = gauge.left
        diagonal = type(diagonal)(
            same_spin_params=gauge.same_spin_params,
            mixed_spin_params=gauge.mixed_spin_params,
        )
        emit_one_body_phases = False
    return right, left, diagonal, emit_one_body_phases
