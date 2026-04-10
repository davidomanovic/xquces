from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.gates import apply_gcr_spin_balanced, apply_gcr_spin_restricted
from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj._unitary import ExactUnitaryChart, OccupiedVirtualUnitaryChart
from xquces.ucj.init import UCJBalancedDFSeed, UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz
from xquces.ucj.parameterization import ov_final_unitary


def _symmetric_matrix_from_values(values, norb, pairs):
    out = np.zeros((norb, norb), dtype=np.float64)
    if pairs:
        rows, cols = zip(*pairs)
        vals = np.asarray(values, dtype=np.float64)
        out[rows, cols] = vals
        out[cols, rows] = vals
    return out


def _validate_pairs(pairs, norb, allow_diagonal=False):
    if pairs is None:
        if allow_diagonal:
            return list(itertools.combinations_with_replacement(range(norb), 2))
        return list(itertools.combinations(range(norb), 2))
    out = []
    seen = set()
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


def exact_reference_ov_params_from_unitary(u, nocc):
    u = np.asarray(u, dtype=np.complex128)
    norb = u.shape[0]
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return np.zeros(0, dtype=np.float64)
    f = u[:, :nocc]
    a = f[:nocc, :]
    x, _ = scipy.linalg.polar(a, side="right")
    fp = f @ x.conj().T
    c = fp[nocc:, :]
    u_left, s, vh = np.linalg.svd(c, full_matrices=False)
    angles = np.arcsin(np.clip(s, -1.0, 1.0))
    z = u_left @ np.diag(angles) @ vh
    return np.concatenate([z.real.ravel(), z.imag.ravel()])


def exact_reference_ov_unitary(u, nocc):
    params = exact_reference_ov_params_from_unitary(u, nocc)
    return ov_final_unitary(params, u.shape[0], nocc)


@dataclass(frozen=True)
class IGCR2SpinRestrictedSpec:
    b_tail: np.ndarray
    pair: np.ndarray

    @property
    def norb(self):
        return self.b_tail.shape[0] + 1

    def full_double(self):
        return np.concatenate([np.zeros(1, dtype=np.float64), np.asarray(self.b_tail, dtype=np.float64)])

    def to_standard(self):
        pair = np.array(self.pair, copy=True, dtype=np.float64)
        np.fill_diagonal(pair, 0.0)
        return SpinRestrictedSpec(
            double_params=self.full_double(),
            pair_params=pair,
        )


@dataclass(frozen=True)
class IGCR2SpinBalancedSpec:
    same_diag: np.ndarray
    b_tail: np.ndarray
    same: np.ndarray
    mixed: np.ndarray

    @property
    def norb(self):
        return self.b_tail.shape[0] + 1

    def full_double(self):
        return np.concatenate([np.zeros(1, dtype=np.float64), np.asarray(self.b_tail, dtype=np.float64)])

    def to_standard(self):
        same = np.array(self.same, copy=True, dtype=np.float64)
        mixed = np.array(self.mixed, copy=True, dtype=np.float64)
        np.fill_diagonal(same, np.asarray(self.same_diag, dtype=np.float64))
        np.fill_diagonal(mixed, self.full_double())
        return SpinBalancedSpec(
            same_spin_params=same,
            mixed_spin_params=mixed,
        )


def reduce_spin_restricted(diag: SpinRestrictedSpec):
    b = np.asarray(diag.double_params, dtype=np.float64).copy()
    pair = np.asarray(diag.pair_params, dtype=np.float64).copy()
    delta = b[0]
    b_red = b.copy()
    b_red[1:] = b_red[1:] - delta
    pair_red = pair.copy()
    mask = ~np.eye(pair.shape[0], dtype=bool)
    pair_red[mask] -= delta
    np.fill_diagonal(pair_red, 0.0)
    return IGCR2SpinRestrictedSpec(
        b_tail=b_red[1:],
        pair=pair_red,
    )


def reduce_spin_balanced(diag: SpinBalancedSpec):
    same = np.asarray(diag.same_spin_params, dtype=np.float64).copy()
    mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64).copy()
    same_diag = np.diag(same).copy()
    np.fill_diagonal(same, 0.0)
    b = np.diag(mixed).copy()
    np.fill_diagonal(mixed, 0.0)
    delta = b[0]
    b_red = b.copy()
    b_red[1:] = b_red[1:] - delta
    mask = ~np.eye(same.shape[0], dtype=bool)
    same[mask] -= delta
    mixed[mask] -= delta
    return IGCR2SpinBalancedSpec(
        same_diag=same_diag,
        b_tail=b_red[1:],
        same=same,
        mixed=mixed,
    )


@dataclass(frozen=True)
class IGCR2Ansatz:
    diagonal: IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec
    left: np.ndarray
    right: np.ndarray
    nocc: int

    @property
    def norb(self):
        return self.diagonal.norb

    @property
    def is_spin_restricted(self):
        return isinstance(self.diagonal, IGCR2SpinRestrictedSpec)

    @property
    def is_spin_balanced(self):
        return isinstance(self.diagonal, IGCR2SpinBalancedSpec)

    def apply(self, vec, nelec, copy=True):
        if self.is_spin_restricted:
            d = self.diagonal.to_standard()
            return apply_gcr_spin_restricted(
                vec,
                d.double_params,
                d.pair_params,
                self.norb,
                nelec,
                left_orbital_rotation=self.left,
                right_orbital_rotation=self.right,
                copy=copy,
            )
        d = self.diagonal.to_standard()
        return apply_gcr_spin_balanced(
            vec,
            d.same_spin_params,
            d.mixed_spin_params,
            self.norb,
            nelec,
            left_orbital_rotation=self.left,
            right_orbital_rotation=self.right,
            copy=copy,
        )

    @classmethod
    def from_gcr_ansatz(cls, ansatz: GCRAnsatz, nocc: int):
        right_ov = exact_reference_ov_unitary(
            ansatz.right_orbital_rotation,
            nocc,
        )
        if ansatz.is_spin_restricted:
            diag = reduce_spin_restricted(ansatz.diagonal)
        else:
            diag = reduce_spin_balanced(ansatz.diagonal)
        return cls(
            diagonal=diag,
            left=np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128),
            right=np.asarray(right_ov, dtype=np.complex128),
            nocc=nocc,
        )

    @classmethod
    def from_ucj(cls, ucj: UCJAnsatz, nocc: int):
        gcr = gcr_from_ucj_ansatz(ucj)
        return cls.from_gcr_ansatz(gcr, nocc=nocc)

    @classmethod
    def from_ucj_ansatz(cls, ansatz: UCJAnsatz, nocc: int):
        return cls.from_ucj(ansatz, nocc=nocc)

    @classmethod
    def from_t_balanced(cls, t2, **kwargs):
        ucj = UCJBalancedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj(ucj, nocc=t2.shape[0])

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj(ucj, nocc=t2.shape[0])


@dataclass(frozen=True)
class IGCR2SpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self):
        return OccupiedVirtualUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def n_left_orbital_rotation_params(self):
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_double_params(self):
        return self.norb - 1

    @property
    def n_pair_params(self):
        return len(self.pair_indices)

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_double_params
            + self.n_pair_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n
        n = self.n_double_params
        b_tail = np.array(params[idx:idx + n], copy=True)
        idx += n
        n = self.n_pair_params
        pair = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.pair_indices)
        idx += n
        n = self.n_right_orbital_rotation_params
        right = ov_final_unitary(params[idx:idx + n], self.norb, self.nocc)
        return IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(
                b_tail=b_tail,
                pair=pair,
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        d = ansatz.diagonal
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self.left_orbital_chart.parameters_from_unitary(ansatz.left)
        idx += n
        n = self.n_double_params
        out[idx:idx + n] = d.b_tail
        idx += n
        n = self.n_pair_params
        out[idx:idx + n] = np.asarray([d.pair[p, q] for p, q in self.pair_indices], dtype=np.float64)
        idx += n
        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = exact_reference_ov_params_from_unitary(ansatz.right, self.nocc)
        return out

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        return self.parameters_from_ansatz(IGCR2Ansatz.from_ucj_ansatz(ansatz, self.nocc))

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func


@dataclass(frozen=True)
class IGCR2SpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def same_spin_indices(self):
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self):
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self):
        return OccupiedVirtualUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def n_left_orbital_rotation_params(self):
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_same_diag_params(self):
        return self.norb

    @property
    def n_double_params(self):
        return self.norb - 1

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
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n
        n = self.n_same_diag_params
        same_diag = np.array(params[idx:idx + n], copy=True)
        idx += n
        n = self.n_double_params
        b_tail = np.array(params[idx:idx + n], copy=True)
        idx += n
        n = self.n_same_spin_params
        same = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.same_spin_indices)
        idx += n
        n = self.n_mixed_spin_params
        mixed = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.mixed_spin_indices)
        idx += n
        n = self.n_right_orbital_rotation_params
        right = ov_final_unitary(params[idx:idx + n], self.norb, self.nocc)
        return IGCR2Ansatz(
            diagonal=IGCR2SpinBalancedSpec(
                same_diag=same_diag,
                b_tail=b_tail,
                same=same,
                mixed=mixed,
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        d = ansatz.diagonal
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self.left_orbital_chart.parameters_from_unitary(ansatz.left)
        idx += n
        n = self.n_same_diag_params
        out[idx:idx + n] = d.same_diag
        idx += n
        n = self.n_double_params
        out[idx:idx + n] = d.b_tail
        idx += n
        n = self.n_same_spin_params
        out[idx:idx + n] = np.asarray([d.same[p, q] for p, q in self.same_spin_indices], dtype=np.float64)
        idx += n
        n = self.n_mixed_spin_params
        out[idx:idx + n] = np.asarray([d.mixed[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
        idx += n
        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = exact_reference_ov_params_from_unitary(ansatz.right, self.nocc)
        return out

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        return self.parameters_from_ansatz(IGCR2Ansatz.from_ucj_ansatz(ansatz, self.nocc))

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func