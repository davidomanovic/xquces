from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj._unitary import ExactUnitaryChart
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz


def _default_triu_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(norb), 2))


def _default_upper_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _validate_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
    *,
    allow_diagonal: bool,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_triu_indices(norb) if allow_diagonal else _default_upper_indices(norb)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p, q in pairs:
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("interaction pair index out of bounds")
        if p > q:
            raise ValueError("interaction pairs must be upper triangular")
        if not allow_diagonal and p == q:
            raise ValueError("diagonal interaction pairs are not allowed here")
        if (p, q) in seen:
            raise ValueError("interaction pairs must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return out


def _symmetric_matrix_from_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((norb, norb), dtype=np.float64)
    if not pairs:
        return out
    rows, cols = zip(*pairs)
    vals = np.asarray(values, dtype=np.float64)
    out[rows, cols] = vals
    out[cols, rows] = vals
    return out


@dataclass(frozen=True)
class GCRSpinRestrictedParameterization:
    norb: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: ExactUnitaryChart = field(default_factory=ExactUnitaryChart)
    right_orbital_chart: ExactUnitaryChart = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_diagonal_params(self) -> int:
        return self.norb

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

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
        out = np.zeros(self.n_params, dtype=np.float64)
        d = ansatz.diagonal
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = self.left_orbital_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n
        n = self.n_diagonal_params
        out[idx : idx + n] = np.asarray(d.double_params, dtype=np.float64)
        idx += n
        n = self.n_pair_params
        if n:
            out[idx : idx + n] = np.asarray([d.pair_params[p, q] for p, q in pairs], dtype=np.float64)
            idx += n
        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(ansatz.right_orbital_rotation)
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
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: ExactUnitaryChart = field(default_factory=ExactUnitaryChart)
    right_orbital_chart: ExactUnitaryChart = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=True)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_same_spin_params(self) -> int:
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self) -> int:
        return len(self.mixed_spin_indices)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_spin_params
            + self.n_mixed_spin_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs_aa = self.same_spin_indices
        pairs_ab = self.mixed_spin_indices
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n
        n = self.n_same_spin_params
        j0 = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs_aa)
        idx += n
        n = self.n_mixed_spin_params
        j1 = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs_ab)
        idx += n
        n = self.n_right_orbital_rotation_params
        right = self.right_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        return GCRAnsatz(
            diagonal=SpinBalancedSpec(same_spin_params=j0, mixed_spin_params=j1),
            left_orbital_rotation=left,
            right_orbital_rotation=right,
        )

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        pairs_aa = self.same_spin_indices
        pairs_ab = self.mixed_spin_indices
        out = np.zeros(self.n_params, dtype=np.float64)
        d = ansatz.diagonal
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = self.left_orbital_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n
        n = self.n_same_spin_params
        if n:
            out[idx : idx + n] = np.asarray([d.same_spin_params[p, q] for p, q in pairs_aa], dtype=np.float64)
            idx += n
        n = self.n_mixed_spin_params
        if n:
            out[idx : idx + n] = np.asarray([d.mixed_spin_params[p, q] for p, q in pairs_ab], dtype=np.float64)
            idx += n
        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(ansatz.right_orbital_rotation)
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
