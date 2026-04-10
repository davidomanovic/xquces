from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

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


def _diag_unitary(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * np.asarray(phases, dtype=np.float64)))


@dataclass(frozen=True)
class _AntiHermitianLeftChart:
    def n_params(self, norb: int) -> int:
        return norb * norb

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (norb * norb,):
            raise ValueError(f"Expected {(norb * norb,)}, got {params.shape}.")
        n_strict = norb * (norb - 1) // 2
        re = params[:n_strict]
        im = params[n_strict : 2 * n_strict]
        diag = params[2 * n_strict :]
        k = np.zeros((norb, norb), dtype=np.complex128)
        rows, cols = np.triu_indices(norb, k=1)
        z = re + 1j * im
        k[rows, cols] = z
        k[cols, rows] = -np.conj(z)
        k[np.diag_indices(norb)] = 1j * diag
        return scipy.linalg.expm(k)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        norb = u.shape[0]
        if u.shape != (norb, norb):
            raise ValueError("u has wrong shape")
        k = scipy.linalg.logm(u)
        k = 0.5 * (k - k.conj().T)
        rows, cols = np.triu_indices(norb, k=1)
        z = k[rows, cols]
        diag = np.imag(np.diag(k))
        return np.concatenate([np.real(z), np.imag(z), diag]).astype(np.float64, copy=False)


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
class _CoupledSpinBalancedDiagonalGaugeMap:
    norb: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        same_pairs = list(self.same_spin_pairs)
        mixed_pairs = list(self.mixed_spin_pairs)

        n_same = len(same_pairs)
        n_mixed = len(mixed_pairs)
        n_full = n_same + n_mixed

        a = np.zeros((n_full, self.norb), dtype=np.float64)

        for k, (p, q) in enumerate(same_pairs):
            a[k, p] = 1.0
            a[k, q] = 1.0

        for j, (p, q) in enumerate(mixed_pairs):
            row = n_same + j
            if p != q:
                a[row, p] = 1.0
                a[row, q] = 1.0

        u, s, _ = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > 1e-10))
        v = np.array(u[:, rank:], copy=True)

        for j in range(v.shape[1]):
            col = v[:, j]
            idx = int(np.argmax(np.abs(col)))
            if abs(col[idx]) > 1e-14 and col[idx] < 0:
                v[:, j] *= -1.0

        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_v", v)
        object.__setattr__(self, "_n_same", n_same)
        object.__setattr__(self, "_n_mixed", n_mixed)

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def n_same(self) -> int:
        return self._n_same

    @property
    def n_mixed(self) -> int:
        return self._n_mixed

    @property
    def n_full(self) -> int:
        return self.n_same + self.n_mixed

    @property
    def n_reduced(self) -> int:
        return self.v.shape[1]

    def reduced_to_full(self, x_reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        x_full = self.v @ x_reduced
        return x_full[: self.n_same], x_full[self.n_same :]

    def full_to_reduced(self, same_full: np.ndarray, mixed_full: np.ndarray) -> np.ndarray:
        x_full = np.concatenate(
            [
                np.asarray(same_full, dtype=np.float64),
                np.asarray(mixed_full, dtype=np.float64),
            ]
        )
        return self.v.T @ x_full

    def gauge_lambda(self, same_full: np.ndarray, mixed_full: np.ndarray) -> np.ndarray:
        x_full = np.concatenate(
            [
                np.asarray(same_full, dtype=np.float64),
                np.asarray(mixed_full, dtype=np.float64),
            ]
        )
        x_phys = self.v @ (self.v.T @ x_full)
        x_gauge = x_full - x_phys
        lam, *_ = np.linalg.lstsq(self.a, x_gauge, rcond=None)
        return lam


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
        if isinstance(self.left_orbital_chart, ExactUnitaryChart):
            return _AntiHermitianLeftChart()
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
        left = self._effective_left_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n

        n = self.n_diagonal_params
        d = np.array(params[idx : idx + n], copy=True)
        idx += n

        n = self.n_pair_params
        p = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs)
        idx += n

        n = self.n_right_orbital_rotation_params
        right = self._effective_right_chart.unitary_from_parameters(params[idx : idx + n], self.norb)

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
        out[idx : idx + n] = self._effective_left_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n

        n = self.n_diagonal_params
        out[idx : idx + n] = np.asarray(d.double_params, dtype=np.float64)
        idx += n

        n = self.n_pair_params
        if n:
            out[idx : idx + n] = np.asarray([d.pair_params[p, q] for p, q in pairs], dtype=np.float64)
            idx += n

        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self._effective_right_chart.parameters_from_unitary(ansatz.right_orbital_rotation)

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
        if isinstance(self.left_orbital_chart, ExactUnitaryChart):
            return _AntiHermitianLeftChart()
        return self.left_orbital_chart

    @property
    def _effective_right_chart(self):
        if isinstance(self.right_orbital_chart, ExactUnitaryChart):
            return _OVRightChart(self.nocc, self.norb)
        return self.right_orbital_chart

    @property
    def _diag_gauge_map(self) -> _CoupledSpinBalancedDiagonalGaugeMap:
        return _CoupledSpinBalancedDiagonalGaugeMap(
            norb=self.norb,
            same_spin_pairs=self.same_spin_indices,
            mixed_spin_pairs=self.mixed_spin_indices,
        )

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._effective_left_chart.n_params(self.norb)

    @property
    def n_jastrow_params(self) -> int:
        return self._diag_gauge_map.n_reduced

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._effective_right_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_jastrow_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self._effective_left_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n

        n = self.n_jastrow_params
        same_pair_full, mixed_full = self._diag_gauge_map.reduced_to_full(params[idx : idx + n])
        idx += n

        same = _symmetric_matrix_from_values(same_pair_full, self.norb, self.same_spin_indices)
        mixed = _symmetric_matrix_from_values(mixed_full, self.norb, self.mixed_spin_indices)

        n = self.n_right_orbital_rotation_params
        right = self._effective_right_chart.unitary_from_parameters(params[idx : idx + n], self.norb)

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

        d = ansatz.diagonal
        left = np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128)
        right = np.asarray(ansatz.right_orbital_rotation, dtype=np.complex128)

        same_diag = np.diag(np.asarray(d.same_spin_params, dtype=np.float64))

        same_pair_full = np.asarray(
            [d.same_spin_params[p, q] for p, q in self.same_spin_indices],
            dtype=np.float64,
        )

        mixed_full = np.asarray(
            [d.mixed_spin_params[p, q] for p, q in self.mixed_spin_indices],
            dtype=np.float64,
        )

        lam = self._diag_gauge_map.gauge_lambda(same_pair_full, mixed_full)

        phase_vec = 0.5 * same_diag + (2 * self.nocc - 1) * lam
        left_eff = left @ _diag_unitary(phase_vec)

        mixed_full_eff = np.array(mixed_full, copy=True)
        for k, (p, q) in enumerate(self.mixed_spin_indices):
            if p == q:
                mixed_full_eff[k] -= 2.0 * lam[p]

        out = np.zeros(self.n_params, dtype=np.float64)

        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = self._effective_left_chart.parameters_from_unitary(left_eff)
        idx += n

        n = self.n_jastrow_params
        out[idx : idx + n] = self._diag_gauge_map.full_to_reduced(
            same_pair_full,
            mixed_full_eff,
        )
        idx += n

        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self._effective_right_chart.parameters_from_unitary(right)

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