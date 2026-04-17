from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np

from xquces.gcr.fixed_sector_basis import (
    FixedSectorDiagonalBasis,
    constant_feature_matrix,
    cubic_feature_matrix,
    one_body_feature_matrix,
    pair_feature_matrix,
    quartic_feature_matrix,
)
from xquces.gcr import igcr2 as _igcr2
from xquces.gcr import igcr3 as _igcr3
from xquces.gcr import igcr4 as _igcr4


@cache
def _cubic_reduction_basis(norb: int, nocc: int) -> FixedSectorDiagonalBasis:
    lower = np.concatenate(
        [
            constant_feature_matrix(norb, nocc),
            one_body_feature_matrix(norb, nocc),
            pair_feature_matrix(norb, nocc),
        ],
        axis=1,
    )
    return FixedSectorDiagonalBasis(cubic_feature_matrix(norb, nocc), lower)


@cache
def _quartic_reduction_basis(norb: int, nocc: int) -> FixedSectorDiagonalBasis:
    lower = np.concatenate(
        [
            constant_feature_matrix(norb, nocc),
            cubic_feature_matrix(norb, nocc),
        ],
        axis=1,
    )
    return FixedSectorDiagonalBasis(quartic_feature_matrix(norb, nocc), lower)


@dataclass(frozen=True)
class IGCR3CubicReduction:
    norb: int
    nocc: int

    @property
    def pair_indices(self):
        return _igcr3._default_pair_indices(self.norb)

    @property
    def tau_indices(self):
        return _igcr3._default_tau_indices(self.norb)

    @property
    def omega_indices(self):
        return _igcr3._default_triple_indices(self.norb)

    @property
    def n_pair_full(self):
        return len(self.pair_indices)

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def physical_cubic_basis(self):
        return _cubic_reduction_basis(self.norb, self.nocc).physical_basis

    @property
    def n_params(self):
        return _cubic_reduction_basis(self.norb, self.nocc).n_params

    def full_from_reduced(self, params: np.ndarray) -> np.ndarray:
        return _cubic_reduction_basis(self.norb, self.nocc).full_from_reduced(params)

    def reduce_full(
        self,
        pair_values: np.ndarray,
        cubic_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_values = np.asarray(pair_values, dtype=np.float64)
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        if pair_values.shape != (self.n_pair_full,):
            raise ValueError(f"Expected pair shape {(self.n_pair_full,)}, got {pair_values.shape}.")
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}.")
        lower_values, reduced_values = _cubic_reduction_basis(self.norb, self.nocc).reduce_full(cubic_values)
        offset = 0
        offset += 1
        onebody_phase = lower_values[offset : offset + self.norb]
        offset += self.norb
        pair_shift = lower_values[offset : offset + self.n_pair_full]
        pair_reduced = pair_values + pair_shift
        return pair_reduced, reduced_values, onebody_phase


@dataclass(frozen=True)
class IGCR4QuarticReduction:
    norb: int
    nocc: int

    @property
    def tau_indices(self):
        return _igcr4._default_tau_indices(self.norb)

    @property
    def omega_indices(self):
        return _igcr4._default_triple_indices(self.norb)

    @property
    def eta_indices(self):
        return _igcr4._default_eta_indices(self.norb)

    @property
    def rho_indices(self):
        return _igcr4._default_rho_indices(self.norb)

    @property
    def sigma_indices(self):
        return _igcr4._default_sigma_indices(self.norb)

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def n_quartic_full(self):
        return len(self.eta_indices) + len(self.rho_indices) + len(self.sigma_indices)

    @property
    def physical_quartic_basis(self):
        return _quartic_reduction_basis(self.norb, self.nocc).physical_basis

    @property
    def n_params(self):
        return _quartic_reduction_basis(self.norb, self.nocc).n_params

    def full_from_reduced(self, params: np.ndarray) -> np.ndarray:
        return _quartic_reduction_basis(self.norb, self.nocc).full_from_reduced(params)

    def reduce_full(
        self,
        cubic_values: np.ndarray,
        quartic_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        quartic_values = np.asarray(quartic_values, dtype=np.float64)
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}.")
        if quartic_values.shape != (self.n_quartic_full,):
            raise ValueError(f"Expected quartic shape {(self.n_quartic_full,)}, got {quartic_values.shape}.")
        lower_values, reduced_values = _quartic_reduction_basis(self.norb, self.nocc).reduce_full(quartic_values)
        cubic_reduced = cubic_values + lower_values[1:]
        return cubic_reduced, reduced_values


@dataclass(frozen=True)
class IGCR2SpinBalancedSpec:
    same_diag: np.ndarray
    same: np.ndarray
    mixed: np.ndarray
    double: np.ndarray

    @property
    def norb(self):
        return int(np.asarray(self.double, dtype=np.float64).shape[0])

    def full_double(self):
        double = np.asarray(self.double, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double has inconsistent shape")
        return double

    def to_standard(self):
        same = np.array(self.same, copy=True, dtype=np.float64)
        mixed = np.array(self.mixed, copy=True, dtype=np.float64)
        np.fill_diagonal(same, np.asarray(self.same_diag, dtype=np.float64))
        np.fill_diagonal(mixed, self.full_double())
        return _igcr2.SpinBalancedSpec(
            same_spin_params=same,
            mixed_spin_params=mixed,
        )


def reduce_spin_balanced(diag: _igcr2.SpinBalancedSpec):
    same = np.asarray(diag.same_spin_params, dtype=np.float64).copy()
    mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64).copy()
    same_diag = np.diag(same).copy()
    double = np.diag(mixed).copy()
    np.fill_diagonal(same, 0.0)
    np.fill_diagonal(mixed, 0.0)
    return IGCR2SpinBalancedSpec(
        same_diag=same_diag,
        same=same,
        mixed=mixed,
        double=double,
    )


@dataclass(frozen=True)
class IGCR2SpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=_igcr2.IGCR2LeftUnitaryChart)
    left_right_ov_relative_scale: float | None = 3.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _igcr2._validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _igcr2._validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)
        if (
            self.left_right_ov_relative_scale is not None
            and (
                not np.isfinite(float(self.left_right_ov_relative_scale))
                or self.left_right_ov_relative_scale <= 0
            )
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def same_spin_indices(self):
        return _igcr2._validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self):
        return _igcr2._validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self):
        return _igcr2.IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_same_diag_params(self):
        return self.norb

    @property
    def n_double_params(self):
        return self.norb

    @property
    def n_same_spin_params(self):
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self):
        return len(self.mixed_spin_indices)

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_double_params
            + self.n_same_spin_params
            + self.n_mixed_spin_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _igcr2._left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _igcr2._native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_double_params
            + self.n_same_spin_params
            + self.n_mixed_spin_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n
        same_diag = np.asarray(params[idx:idx + self.n_same_diag_params], dtype=np.float64)
        idx += self.n_same_diag_params
        double = np.asarray(params[idx:idx + self.n_double_params], dtype=np.float64)
        idx += self.n_double_params
        same = _igcr2._symmetric_matrix_from_values(
            np.asarray(params[idx:idx + self.n_same_spin_params], dtype=np.float64),
            self.norb,
            self.same_spin_indices,
        )
        idx += self.n_same_spin_params
        mixed = _igcr2._symmetric_matrix_from_values(
            np.asarray(params[idx:idx + self.n_mixed_spin_params], dtype=np.float64),
            self.norb,
            self.mixed_spin_indices,
        )
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        right = _igcr2._right_unitary_from_left_and_final(left, final, self.nocc)
        return _igcr2.IGCR2Ansatz(
            diagonal=IGCR2SpinBalancedSpec(
                same_diag=same_diag,
                same=same,
                mixed=mixed,
                double=double,
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: _igcr2.IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        d = ansatz.diagonal.to_standard()
        same_mat = np.asarray(d.same_spin_params, dtype=np.float64).copy()
        mixed_mat = np.asarray(d.mixed_spin_params, dtype=np.float64).copy()
        same_diag = np.diag(same_mat).copy()
        mixed_double = np.diag(mixed_mat).copy()
        np.fill_diagonal(same_mat, 0.0)
        np.fill_diagonal(mixed_mat, 0.0)
        same_full = np.asarray([same_mat[p, q] for p, q in self.same_spin_indices], dtype=np.float64)
        mixed_full = np.asarray([mixed_mat[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(
                np.asarray(ansatz.left, dtype=np.complex128)
            )
        else:
            left_params = left_chart.parameters_from_unitary(np.asarray(ansatz.left, dtype=np.complex128))
            right_phase = np.zeros(self.norb, dtype=np.float64)
        right_eff = _igcr2._diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        out[idx:idx + self.n_left_orbital_rotation_params] = left_params
        idx += self.n_left_orbital_rotation_params
        out[idx:idx + self.n_same_diag_params] = same_diag
        idx += self.n_same_diag_params
        out[idx:idx + self.n_double_params] = mixed_double
        idx += self.n_double_params
        out[idx:idx + self.n_same_spin_params] = same_full
        idx += self.n_same_spin_params
        out[idx:idx + self.n_mixed_spin_params] = mixed_full
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        left_param_unitary = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _igcr2._final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        out[idx:idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)
        return self._public_parameters_from_native(out)

    def parameters_from_ucj_ansatz(self, ansatz: _igcr2.UCJAnsatz):
        return self.parameters_from_ansatz(_igcr2.IGCR2Ansatz.from_ucj_ansatz(ansatz, self.nocc))

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR2SpinBalancedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError("previous ansatz nocc does not match this parameterization")
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
            old_for_new, phases = _igcr2.orbital_relabeling_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
        if old_for_new is not None:
            ansatz = _igcr2.relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func


_igcr3.IGCR3CubicReduction = IGCR3CubicReduction
_igcr4.IGCR4QuarticReduction = IGCR4QuarticReduction
_igcr2.IGCR2SpinBalancedSpec = IGCR2SpinBalancedSpec
_igcr2.reduce_spin_balanced = reduce_spin_balanced
_igcr2.IGCR2SpinBalancedParameterization = IGCR2SpinBalancedParameterization
