from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj._unitary import ExactUnitaryChart
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz
from xquces.ucj.parameterization import ov_final_unitary, ov_params_from_unitary


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
class _OVRightChart:
    nocc: int
    norb: int

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    def n_params(self, norb: int) -> int:
        if norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return 2 * self.nocc * self.nvirt

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        if norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return ov_final_unitary(np.asarray(params, dtype=np.float64), norb, self.nocc)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.norb, self.norb):
            raise ValueError("u has wrong shape")
        return ov_params_from_unitary(u, self.nocc)


@dataclass(frozen=True)
class GCRSpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=ExactUnitaryChart)
    right_orbital_chart: object = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def _effective_left_chart(self):
        return self.left_orbital_chart

    @property
    def _effective_right_chart(self):
        if isinstance(self.right_orbital_chart, ExactUnitaryChart):
            return _OVRightChart(self.nocc, self.norb)
        return self.right_orbital_chart

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._effective_left_chart.n_params(self.norb)

    @property
    def n_diagonal_params(self) -> int:
        return self.norb

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._effective_right_chart.n_params(self.norb)

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
        left = self._effective_left_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n = self.n_diagonal_params
        d = np.array(params[idx:idx + n], copy=True)
        idx += n

        n = self.n_pair_params
        p = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, pairs)
        idx += n

        n = self.n_right_orbital_rotation_params
        right = self._effective_right_chart.unitary_from_parameters(params[idx:idx + n], self.norb)

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
        d = ansatz.diagonal
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self._effective_left_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n

        n = self.n_diagonal_params
        out[idx:idx + n] = np.asarray(d.double_params, dtype=np.float64)
        idx += n

        n = self.n_pair_params
        if n:
            out[idx:idx + n] = np.asarray([d.pair_params[p, q] for p, q in pairs], dtype=np.float64)
            idx += n

        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = self._effective_right_chart.parameters_from_unitary(ansatz.right_orbital_rotation)

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
    left_orbital_chart: object = field(default_factory=ExactUnitaryChart)
    right_orbital_chart: object = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def _effective_left_chart(self):
        return self.left_orbital_chart

    @property
    def _effective_right_chart(self):
        if isinstance(self.right_orbital_chart, ExactUnitaryChart):
            return _OVRightChart(self.nocc, self.norb)
        return self.right_orbital_chart

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._effective_left_chart.n_params(self.norb)

    @property
    def n_same_diag_params(self) -> int:
        return self.norb

    @property
    def n_same_pair_params(self) -> int:
        return len(self.same_spin_indices)

    @property
    def n_mixed_params(self) -> int:
        return len(self.mixed_spin_indices)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._effective_right_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_same_pair_params
            + self.n_mixed_params
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
        left = self._effective_left_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n = self.n_same_diag_params
        same_diag = np.array(params[idx:idx + n], copy=True)
        idx += n

        n = self.n_same_pair_params
        same = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, pairs_aa)
        idx += n
        same[np.diag_indices(self.norb)] = same_diag

        n = self.n_mixed_params
        mixed = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, pairs_ab)
        idx += n

        n = self.n_right_orbital_rotation_params
        right = self._effective_right_chart.unitary_from_parameters(params[idx:idx + n], self.norb)

        return GCRAnsatz(
            diagonal=SpinBalancedSpec(
                same_spin_params=same,
                mixed_spin_params=mixed,
            ),
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
        d = ansatz.diagonal

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self._effective_left_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n

        n = self.n_same_diag_params
        out[idx:idx + n] = np.diag(np.asarray(d.same_spin_params, dtype=np.float64))
        idx += n

        n = self.n_same_pair_params
        if n:
            out[idx:idx + n] = np.asarray(
                [d.same_spin_params[p, q] for p, q in pairs_aa],
                dtype=np.float64,
            )
            idx += n

        n = self.n_mixed_params
        if n:
            out[idx:idx + n] = np.asarray(
                [d.mixed_spin_params[p, q] for p, q in pairs_ab],
                dtype=np.float64,
            )
            idx += n

        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = self._effective_right_chart.parameters_from_unitary(ansatz.right_orbital_rotation)

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