from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.gates import apply_gcr_spin_balanced, apply_gcr_spin_restricted
from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
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


def _assert_square_matrix(a: np.ndarray, name: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def _diag_unitary(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * np.asarray(phases, dtype=np.float64)))


def _antihermitian_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    expected = norb * norb
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    n_strict = norb * (norb - 1) // 2
    re = params[:n_strict]
    im = params[n_strict : 2 * n_strict]
    diag = params[2 * n_strict :]
    k = np.zeros((norb, norb), dtype=np.complex128)
    rows, cols = np.triu_indices(norb, k=1)
    z = re + 1j * im
    k[rows, cols] = z
    k[cols, rows] = -np.conjugate(z)
    k[np.diag_indices(norb)] = 1j * diag
    return k


def _parameters_from_antihermitian(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.complex128)
    _assert_square_matrix(k, "k")
    if not np.allclose(k.conj().T, -k, atol=1e-10):
        raise ValueError("k must be antihermitian")
    norb = k.shape[0]
    rows, cols = np.triu_indices(norb, k=1)
    z = k[rows, cols]
    diag = np.imag(np.diag(k))
    return np.concatenate([np.real(z), np.imag(z), diag]).astype(np.float64, copy=False)


def _left_unitary_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    return np.asarray(scipy.linalg.expm(_antihermitian_from_parameters(params, norb)), dtype=np.complex128)


def _left_parameters_from_unitary(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    _assert_square_matrix(u, "u")
    norb = u.shape[0]
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")
    k = scipy.linalg.logm(u)
    k = 0.5 * (k - k.conj().T)
    k[np.diag_indices(norb)] = 1j * np.imag(np.diag(k))
    return _parameters_from_antihermitian(k)


@dataclass(frozen=True)
class IGCR2LeftUnitaryChart:
    def n_params(self, norb: int) -> int:
        return norb * norb

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return _left_unitary_from_parameters(params, norb)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return _left_parameters_from_unitary(u)


@dataclass(frozen=True)
class IGCR2ReferenceOVUnitaryChart:
    nocc: int
    nvirt: int

    def __post_init__(self):
        if self.nocc < 0 or self.nvirt < 0:
            raise ValueError("nocc and nvirt must be nonnegative")

    @property
    def norb(self) -> int:
        return self.nocc + self.nvirt

    def n_params(self, norb: int | None = None) -> int:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return 2 * self.nocc * self.nvirt

    def unitary_from_parameters(self, params: np.ndarray, norb: int | None = None) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return ov_final_unitary(np.asarray(params, dtype=np.float64), self.norb, self.nocc)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.norb, self.norb):
            raise ValueError("u has wrong shape")
        return exact_reference_ov_params_from_unitary(u, self.nocc)


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


def _n_total_from_nocc(nocc: int) -> int:
    return 2 * int(nocc)


def _restricted_irreducible_pair_matrix(double_params: np.ndarray, pair_params: np.ndarray) -> np.ndarray:
    b = np.asarray(double_params, dtype=np.float64)
    pair = np.asarray(pair_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    out = np.array(pair, copy=True, dtype=np.float64)
    mask = ~np.eye(pair.shape[0], dtype=bool)
    out[mask] -= shift[mask]
    np.fill_diagonal(out, 0.0)
    return out


def _restricted_left_phase_vector(double_params: np.ndarray, nocc: int) -> np.ndarray:
    return 0.5 * (_n_total_from_nocc(nocc) - 1) * np.asarray(double_params, dtype=np.float64)


def _balanced_irreducible_pair_matrices(
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    same = np.asarray(same_spin_params, dtype=np.float64)
    mixed = np.asarray(mixed_spin_params, dtype=np.float64)
    b = np.diag(mixed)
    shift = 0.5 * (b[:, None] + b[None, :])
    same_red = np.array(same, copy=True, dtype=np.float64)
    mixed_red = np.array(mixed, copy=True, dtype=np.float64)
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
    return 0.5 * same_diag + 0.5 * (_n_total_from_nocc(nocc) - 1) * mixed_diag


@dataclass(frozen=True)
class _SameSpinPairGaugeMap:
    norb: int
    same_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        pairs = list(self.same_spin_pairs)
        n_pairs = len(pairs)
        a = np.zeros((n_pairs, self.norb), dtype=np.float64)
        for k, (p, q) in enumerate(pairs):
            a[k, p] = 1.0
            a[k, q] = 1.0
        if n_pairs == 0:
            u = np.zeros((0, 0), dtype=np.float64)
            v = np.zeros((0, 0), dtype=np.float64)
        else:
            u_full, s, _ = np.linalg.svd(a, full_matrices=True)
            rank = int(np.sum(s > 1e-10))
            u = np.array(u_full[:, :rank], copy=True)
            v = np.array(u_full[:, rank:], copy=True)
            for j in range(v.shape[1]):
                col = v[:, j]
                idx = int(np.argmax(np.abs(col)))
                if abs(col[idx]) > 1e-14 and col[idx] < 0:
                    v[:, j] *= -1.0
        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_u", u)
        object.__setattr__(self, "_v", v)

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def u(self) -> np.ndarray:
        return self._u

    @property
    def n_full(self) -> int:
        return self.a.shape[0]

    @property
    def n_reduced(self) -> int:
        return self.v.shape[1]

    @property
    def n_gauge(self) -> int:
        return self.u.shape[1]

    def reduced_to_full(self, x_reduced: np.ndarray) -> np.ndarray:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        if self.n_full == 0:
            return np.zeros(0, dtype=np.float64)
        return self.v @ x_reduced

    def gauge_to_full(self, x_gauge: np.ndarray) -> np.ndarray:
        x_gauge = np.asarray(x_gauge, dtype=np.float64)
        if self.n_full == 0:
            return np.zeros(0, dtype=np.float64)
        return self.u @ x_gauge

    def full_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        if self.n_full == 0:
            return np.zeros(0, dtype=np.float64)
        return self.v.T @ x_full

    def full_to_gauge(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        if self.n_full == 0:
            return np.zeros(0, dtype=np.float64)
        return self.u.T @ x_full

    def gauge_lambda(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        if self.n_full == 0:
            return np.zeros(self.norb, dtype=np.float64)
        x_phys = self.v @ (self.v.T @ x_full)
        x_gauge = x_full - x_phys
        lam, *_ = np.linalg.lstsq(self.a, x_gauge, rcond=None)
        return lam


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
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self):
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def n_left_orbital_rotation_params(self):
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_double_params(self):
        return 0

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
        n = self.n_pair_params
        pair = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.pair_indices)
        idx += n
        n = self.n_right_orbital_rotation_params
        right = ov_final_unitary(params[idx:idx + n], self.norb, self.nocc)
        return IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(
                b_tail=np.zeros(max(self.norb - 1, 0), dtype=np.float64),
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
        d_std = d.to_standard()
        phase_vec = _restricted_left_phase_vector(d_std.double_params, self.nocc)
        left_eff = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(phase_vec)
        pair_eff = _restricted_irreducible_pair_matrix(d_std.double_params, d_std.pair_params)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self.left_orbital_chart.parameters_from_unitary(left_eff)
        idx += n
        n = self.n_pair_params
        out[idx:idx + n] = np.asarray([pair_eff[p, q] for p, q in self.pair_indices], dtype=np.float64)
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
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)

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
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _same_spin_gauge_map(self):
        return _SameSpinPairGaugeMap(
            norb=self.norb,
            same_spin_pairs=self.same_spin_indices,
        )

    @property
    def _same_mixed_identical(self):
        return self.same_spin_indices == self.mixed_spin_indices

    @property
    def n_left_orbital_rotation_params(self):
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_same_diag_params(self):
        return 0

    @property
    def n_double_params(self):
        return 0

    @property
    def n_same_spin_params(self):
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self):
        return len(self.mixed_spin_indices)

    @property
    def _n_same_spin_reduced_params(self):
        return self._same_spin_gauge_map.n_reduced

    @property
    def _n_same_spin_gauge_params(self):
        return self._same_spin_gauge_map.n_gauge

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        if self._same_mixed_identical:
            return (
                self.n_left_orbital_rotation_params
                + self._n_same_spin_reduced_params
                + self._n_same_spin_reduced_params
                + self._n_same_spin_gauge_params
                + self.n_right_orbital_rotation_params
            )
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_double_params
            + self._n_same_spin_reduced_params
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
        if self._same_mixed_identical:
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            n = self._n_same_spin_reduced_params
            charge_phys = np.asarray(params[idx:idx + n], dtype=np.float64)
            idx += n
            spin_phys = np.asarray(params[idx:idx + n], dtype=np.float64)
            idx += n
            n = self._n_same_spin_gauge_params
            mixed_gauge = np.asarray(params[idx:idx + n], dtype=np.float64)
            idx += n
            same_phys = inv_sqrt2 * (charge_phys + spin_phys)
            mixed_phys = inv_sqrt2 * (charge_phys - spin_phys)
            same_full = self._same_spin_gauge_map.reduced_to_full(same_phys)
            mixed_full = (
                self._same_spin_gauge_map.reduced_to_full(mixed_phys)
                + self._same_spin_gauge_map.gauge_to_full(mixed_gauge)
            )
            same = _symmetric_matrix_from_values(same_full, self.norb, self.same_spin_indices)
            mixed = _symmetric_matrix_from_values(mixed_full, self.norb, self.mixed_spin_indices)
        else:
            n = self._n_same_spin_reduced_params
            same_full = self._same_spin_gauge_map.reduced_to_full(params[idx:idx + n])
            same = _symmetric_matrix_from_values(same_full, self.norb, self.same_spin_indices)
            idx += n
            n = self.n_mixed_spin_params
            mixed = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.mixed_spin_indices)
            idx += n

        n = self.n_right_orbital_rotation_params
        right = ov_final_unitary(params[idx:idx + n], self.norb, self.nocc)
        return IGCR2Ansatz(
            diagonal=IGCR2SpinBalancedSpec(
                same_diag=np.zeros(self.norb, dtype=np.float64),
                b_tail=np.zeros(max(self.norb - 1, 0), dtype=np.float64),
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
        d_std = d.to_standard()
        phase_vec = _balanced_left_phase_vector(
            d_std.same_spin_params,
            d_std.mixed_spin_params,
            self.nocc,
        )
        left_eff = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(phase_vec)
        same_eff, mixed_eff = _balanced_irreducible_pair_matrices(
            d_std.same_spin_params,
            d_std.mixed_spin_params,
        )
        same_full = np.asarray([same_eff[p, q] for p, q in self.same_spin_indices], dtype=np.float64)
        lam = self._same_spin_gauge_map.gauge_lambda(same_full)
        left_eff = left_eff @ _diag_unitary((self.nocc - 1) * lam)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self.left_orbital_chart.parameters_from_unitary(left_eff)
        idx += n
        if self._same_mixed_identical:
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            mixed_full = np.asarray([mixed_eff[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
            same_phys = self._same_spin_gauge_map.full_to_reduced(same_full)
            mixed_phys = self._same_spin_gauge_map.full_to_reduced(mixed_full)
            mixed_gauge = self._same_spin_gauge_map.full_to_gauge(mixed_full)
            n = self._n_same_spin_reduced_params
            out[idx:idx + n] = inv_sqrt2 * (same_phys + mixed_phys)
            idx += n
            out[idx:idx + n] = inv_sqrt2 * (same_phys - mixed_phys)
            idx += n
            n = self._n_same_spin_gauge_params
            out[idx:idx + n] = mixed_gauge
            idx += n
        else:
            n = self._n_same_spin_reduced_params
            out[idx:idx + n] = self._same_spin_gauge_map.full_to_reduced(same_full)
            idx += n
            n = self.n_mixed_spin_params
            out[idx:idx + n] = np.asarray([mixed_eff[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
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
