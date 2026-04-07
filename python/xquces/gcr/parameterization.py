from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.orbitals import canonicalize_unitary
from xquces.ucj._unitary import GaugeFixedInternalUnitaryChart, OccupiedVirtualUnitaryChart
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz
from xquces.ucj.parameterization import (
    _GaugeReducedSpinBalancedMap,
    _symmetric_matrix_from_values,
    _validate_pairs,
)


def _gauge_fix_left_and_transfer_right(
    left: np.ndarray,
    right: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    left = canonicalize_unitary(np.asarray(left, dtype=np.complex128), tol=tol)
    right = np.asarray(right, dtype=np.complex128)
    norb = left.shape[0]
    phases = np.ones(norb, dtype=np.complex128)
    for j in range(norb):
        col = left[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > tol:
            phases[j] = np.exp(-1j * np.angle(val))
    p = np.diag(phases)
    left_gf = left @ p
    right_eff = p.conj().T @ right
    return left_gf, right_eff

def _stable_ov_params_from_unitary_subspace(unitary: np.ndarray, nocc: int) -> np.ndarray:
    unitary = np.asarray(unitary, dtype=np.complex128)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("unitary must be square")
    norb = unitary.shape[0]
    if not np.allclose(unitary.conj().T @ unitary, np.eye(norb), atol=1e-10):
        raise ValueError("unitary must be unitary")
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return np.zeros(0, dtype=np.float64)

    c_occ = unitary[:, :nocc]
    a = np.asarray(c_occ[:nocc, :], dtype=np.complex128)
    b = np.asarray(c_occ[nocc:, :], dtype=np.complex128)

    ua, _, vha = np.linalg.svd(a, full_matrices=False)
    q_occ = ua @ vha
    b_gf = b @ q_occ.conj().T

    ub, s, vh = np.linalg.svd(b_gf, full_matrices=False)
    theta = np.arcsin(np.clip(s, -1.0, 1.0))
    z = ub @ np.diag(theta) @ vh

    return np.concatenate([np.real(z).reshape(-1), np.imag(z).reshape(-1)])

@dataclass(frozen=True)
class GCRSpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: GaugeFixedInternalUnitaryChart = field(default_factory=GaugeFixedInternalUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self) -> OccupiedVirtualUnitaryChart:
        return OccupiedVirtualUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_diagonal_params(self) -> int:
        return self.norb

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diagonal_params
            + self.n_pair_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs = self.pair_indices
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n
        n = self.n_diagonal_params
        d = np.array(params[idx : idx + n], copy=True)
        idx += n
        n = self.n_pair_params
        p = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs)
        idx += n
        n = self.n_right_orbital_rotation_params
        right = self.right_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        return GCRAnsatz(
            diagonal=SpinRestrictedSpec(double_params=d, pair_params=p),
            left_orbital_rotation=left,
            right_orbital_rotation=right,
        )

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        pairs = self.pair_indices
        left_gf, right_eff = _gauge_fix_left_and_transfer_right(
            ansatz.left_orbital_rotation,
            ansatz.right_orbital_rotation,
        )
        out = np.zeros(self.n_params, dtype=np.float64)
        d = ansatz.diagonal
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = self.left_orbital_chart.parameters_from_unitary(left_gf)
        idx += n
        n = self.n_diagonal_params
        out[idx : idx + n] = np.asarray(d.double_params, dtype=np.float64)
        idx += n
        n = self.n_pair_params
        if n:
            out[idx : idx + n] = np.asarray([d.pair_params[p, q] for p, q in pairs], dtype=np.float64)
            idx += n
        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(right_eff)
        return out

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(gcr_from_ucj_ansatz(ansatz))

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func

@dataclass(frozen=True)
class GCRSpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: GaugeFixedInternalUnitaryChart = field(default_factory=GaugeFixedInternalUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)
        object.__setattr__(
            self,
            "_jastrow_gauge_map",
            _GaugeReducedSpinBalancedMap(
                self.norb,
                1,
                self.same_spin_indices,
                self.mixed_spin_indices,
            ),
        )

    @property
    def right_orbital_chart(self) -> OccupiedVirtualUnitaryChart:
        return OccupiedVirtualUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def jastrow_gauge_map(self) -> _GaugeReducedSpinBalancedMap:
        return self._jastrow_gauge_map

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.jastrow_gauge_map.n_reduced
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs_aa = self.same_spin_indices
        pairs_ab = self.mixed_spin_indices
        j_map = self.jastrow_gauge_map
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n
        n = j_map.n_reduced
        x_j_full = j_map.reduced_to_full(params[idx : idx + n])
        idx += n
        n_same = len(pairs_aa)
        n_mixed = len(pairs_ab)
        j0 = _symmetric_matrix_from_values(x_j_full[:n_same], self.norb, pairs_aa)
        j1 = _symmetric_matrix_from_values(x_j_full[n_same : n_same + n_mixed], self.norb, pairs_ab)
        n = self.n_right_orbital_rotation_params
        right = self.right_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        return GCRAnsatz(
            diagonal=SpinBalancedSpec(same_spin_params=j0, mixed_spin_params=j1),
            left_orbital_rotation=left,
            right_orbital_rotation=right,
        )

    @staticmethod
    def _reduced_coordinates(basis: np.ndarray, vec: np.ndarray) -> np.ndarray:
        basis = np.asarray(basis, dtype=np.float64)
        vec = np.asarray(vec, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("basis must be a matrix")
        if basis.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        return np.linalg.lstsq(basis, vec, rcond=None)[0]

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")

        same_pairs = self.same_spin_indices
        mixed_pairs = self.mixed_spin_indices
        j_map = self.jastrow_gauge_map

        left_gf, right_eff = _gauge_fix_left_and_transfer_right(
            ansatz.left_orbital_rotation,
            ansatz.right_orbital_rotation,
        )

        d = ansatz.diagonal

        same_diag = np.diag(np.asarray(d.same_spin_params, dtype=np.float64)).copy()

        same_full = np.asarray(
            [d.same_spin_params[p, q] for p, q in same_pairs],
            dtype=np.float64,
        )
        mixed_full = np.asarray(
            [d.mixed_spin_params[p, q] for p, q in mixed_pairs],
            dtype=np.float64,
        )

        x_same_red = self._reduced_coordinates(j_map.v_same, same_full)
        x_mixed_red = self._reduced_coordinates(j_map.v_mixed, mixed_full)

        same_red_full = (
            j_map.v_same @ x_same_red
            if j_map.n_same_reduced > 0
            else np.zeros_like(same_full)
        )
        mixed_red_full = (
            j_map.v_mixed @ x_mixed_red
            if j_map.n_mixed_reduced > 0
            else np.zeros_like(mixed_full)
        )

        same_gauge = same_full - same_red_full
        mixed_gauge = mixed_full - mixed_red_full

        a = np.zeros(self.norb, dtype=np.float64)
        if len(same_pairs) > 0:
            A_same = np.zeros((len(same_pairs), self.norb), dtype=np.float64)
            for k, (p, q) in enumerate(same_pairs):
                A_same[k, p] = 1.0
                A_same[k, q] = 1.0
            a = np.linalg.lstsq(A_same, same_gauge, rcond=None)[0]

        b = np.zeros(self.norb, dtype=np.float64)
        if len(mixed_pairs) > 0:
            A_mixed = np.zeros((len(mixed_pairs), self.norb), dtype=np.float64)
            for k, (p, q) in enumerate(mixed_pairs):
                if p == q:
                    A_mixed[k, p] = 2.0
                else:
                    A_mixed[k, p] = 1.0
                    A_mixed[k, q] = 1.0
            b = np.linalg.lstsq(A_mixed, mixed_gauge, rcond=None)[0]

        phi = 0.5 * same_diag + (self.nocc - 1) * a + self.nocc * b
        right_eff = np.diag(np.exp(1j * phi)) @ right_eff

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = self.left_orbital_chart.parameters_from_unitary(left_gf)
        idx += n

        n_same_red = j_map.n_same_reduced
        if n_same_red:
            out[idx : idx + n_same_red] = x_same_red
            idx += n_same_red

        n_mixed_red = j_map.n_mixed_reduced
        if n_mixed_red:
            out[idx : idx + n_mixed_red] = x_mixed_red
            idx += n_mixed_red

        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(right_eff)

        return out

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != 1:
            raise ValueError("only a single-layer UCJ ansatz can be mapped exactly to GCR")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        return self.parameters_from_ansatz(gcr_from_ucj_ansatz(ansatz))

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func