from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np

from xquces._lib import apply_igcr4_spin_combo6_in_place_num_rep
from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gcr.igcr2 import (
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    _right_unitary_from_left_and_final,
)
from xquces.orbitals import apply_orbital_rotation


SPIN_COMBO6_LABELS = ("aabb", "abab", "abba", "baab", "baba", "bbaa")


def _default_q4_indices(norb: int) -> list[tuple[int, int, int, int]]:
    return [
        (p, q, r, s)
        for p in range(norb)
        for q in range(p + 1)
        for r in range(q + 1)
        for s in range(r + 1)
    ]


def _validate_q4_indices(indices, norb: int) -> list[tuple[int, int, int, int]]:
    if indices is None:
        return _default_q4_indices(norb)
    out = []
    seen = set()
    for p, q, r, s in indices:
        key = (int(p), int(q), int(r), int(s))
        if not (0 <= key[3] <= key[2] <= key[1] <= key[0] < norb):
            raise ValueError("q4 indices must satisfy 0 <= s <= r <= q <= p < norb")
        if key in seen:
            raise ValueError("q4 indices must not contain duplicates")
        seen.add(key)
        out.append(key)
    return out


def _q4_features(norb: int, nelec: tuple[int, int], q4_indices) -> np.ndarray:
    occ_a = occ_indicator_rows(norb, nelec[0])
    occ_b = occ_indicator_rows(norb, nelec[1])
    out = np.zeros((len(occ_a) * len(occ_b), 6 * len(q4_indices)), dtype=np.float64)
    row = 0
    for a in occ_a:
        for b in occ_b:
            col = 0
            for p, q, r, s in q4_indices:
                out[row, col] = a[p] * a[q] * b[r] * b[s]
                out[row, col + 1] = a[p] * b[q] * a[r] * b[s]
                out[row, col + 2] = a[p] * b[q] * b[r] * a[s]
                out[row, col + 3] = b[p] * a[q] * a[r] * b[s]
                out[row, col + 4] = b[p] * a[q] * b[r] * a[s]
                out[row, col + 5] = b[p] * b[q] * a[r] * a[s]
                col += 6
            row += 1
    return out


@dataclass(frozen=True)
class IGCR4IndependentQ4Basis:
    norb: int
    nelec: tuple[int, int]
    q4_indices: list[tuple[int, int, int, int]]
    raw_basis: np.ndarray
    singular_values: np.ndarray
    raw_n_params: int

    @property
    def n_params(self) -> int:
        return int(self.raw_basis.shape[1])

    def raw_from_params(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return (self.raw_basis @ params).reshape((len(self.q4_indices), 6))

    def params_from_raw(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (len(self.q4_indices), 6):
            raise ValueError(f"Expected {(len(self.q4_indices), 6)}, got {raw.shape}.")
        return self.raw_basis.T @ raw.reshape(-1)


def make_independent_q4_basis(norb: int, nelec: tuple[int, int], q4_indices=None, *, rtol: float = 1e-10):
    nelec = tuple(int(x) for x in nelec)
    q4_indices = _validate_q4_indices(q4_indices, norb)
    raw = _q4_features(norb, nelec, q4_indices)
    centered = raw - raw.mean(axis=0, keepdims=True)
    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    rank = 0 if svals.size == 0 else int(np.sum(svals > float(rtol) * svals[0]))
    return IGCR4IndependentQ4Basis(
        norb=norb,
        nelec=nelec,
        q4_indices=q4_indices,
        raw_basis=np.asarray(vh[:rank].T, dtype=np.float64),
        singular_values=np.asarray(svals, dtype=np.float64),
        raw_n_params=raw.shape[1],
    )


@dataclass(frozen=True)
class IGCR4SpinCombo6Spec:
    q4_values: np.ndarray
    norb_: int
    q4_indices_: list[tuple[int, int, int, int]] | None = None

    @property
    def norb(self) -> int:
        return int(self.norb_)

    @property
    def q4_indices(self):
        return _validate_q4_indices(self.q4_indices_, self.norb)

    def q4_matrix(self) -> np.ndarray:
        values = np.asarray(self.q4_values, dtype=np.float64)
        if values.shape != (len(self.q4_indices), 6):
            raise ValueError("q4_values must have shape (n_q4, 6)")
        default = _default_q4_indices(self.norb)
        if self.q4_indices == default:
            return values
        out = np.zeros((len(default), 6), dtype=np.float64)
        pos = {idx: k for k, idx in enumerate(default)}
        for value, idx in zip(values, self.q4_indices):
            out[pos[idx]] = value
        return out


def apply_igcr4_spin_combo6_diagonal(vec, diagonal, norb: int, nelec: tuple[int, int], *, time: float = 1.0, copy: bool = True):
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    state2 = reshape_state(arr, norb, nelec)
    apply_igcr4_spin_combo6_in_place_num_rep(
        state2,
        np.asarray(diagonal.q4_matrix(), dtype=np.float64) * time,
        norb,
        occ_indicator_rows(norb, nelec[0]),
        occ_indicator_rows(norb, nelec[1]),
    )
    return flatten_state(state2)


@dataclass(frozen=True)
class IGCR4SpinCombo6Ansatz:
    diagonal: IGCR4SpinCombo6Spec
    left: np.ndarray
    right: np.ndarray
    nocc: int

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(arr, self.right, norb=self.norb, nelec=nelec, copy=False)
        arr = apply_igcr4_spin_combo6_diagonal(arr, self.diagonal, self.norb, nelec, copy=False)
        return apply_orbital_rotation(arr, self.left, norb=self.norb, nelec=nelec, copy=False)


@dataclass(frozen=True)
class IGCR4SpinCombo6Parameterization:
    norb: int
    nelec: tuple[int, int]
    q4_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_q4: bool = True
    q4_rtol: float = 1e-10
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False

    def __post_init__(self):
        nelec = tuple(int(x) for x in self.nelec)
        if len(nelec) != 2:
            raise ValueError("nelec must be a two-tuple")
        if nelec[0] != nelec[1]:
            raise ValueError("spin-combo6 iGCR4 currently requires N_alpha = N_beta")
        if not (0 <= nelec[0] <= self.norb and 0 <= nelec[1] <= self.norb):
            raise ValueError("nelec must be compatible with norb")
        if not np.isfinite(float(self.q4_rtol)) or self.q4_rtol < 0:
            raise ValueError("q4_rtol must be nonnegative")
        object.__setattr__(self, "nelec", nelec)
        _validate_q4_indices(self.q4_indices_, self.norb)

    @property
    def nocc(self) -> int:
        return self.nelec[0]

    @property
    def q4_indices(self):
        return _validate_q4_indices(self.q4_indices_, self.norb)

    @cached_property
    def q4_basis(self):
        return make_independent_q4_basis(self.norb, self.nelec, self.q4_indices, rtol=self.q4_rtol)

    @property
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def n_left_orbital_rotation_params(self):
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_q4_params(self):
        return self.q4_basis.n_params if self.reduce_q4 else 6 * len(self.q4_indices)

    @property
    def n_raw_q4_params(self):
        return 6 * len(self.q4_indices)

    @property
    def n_diagonal_params(self):
        return self.n_q4_params

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        return self.n_left_orbital_rotation_params + self.n_q4_params + self.n_right_orbital_rotation_params

    def sector_sizes(self):
        return {
            "left": self.n_left_orbital_rotation_params,
            "q4": self.n_q4_params,
            "q4_raw": self.n_raw_q4_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _split_params(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        n_left = self.n_left_orbital_rotation_params
        n_q4 = self.n_q4_params
        return params[:n_left], params[n_left : n_left + n_q4], params[n_left + n_q4 :]

    def q4_values_from_parameters(self, params):
        _, q4_params, _ = self._split_params(params)
        if self.reduce_q4:
            return self.q4_basis.raw_from_params(q4_params)
        return q4_params.reshape((len(self.q4_indices), 6))

    def diagonal_from_parameters(self, params):
        return IGCR4SpinCombo6Spec(self.q4_values_from_parameters(params), self.norb, self.q4_indices)

    def ansatz_from_parameters(self, params):
        params = self._native_parameters_from_public(params)
        left_params, _, right_params = self._split_params(params)
        left = self.left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final = self.right_orbital_chart.unitary_from_parameters(right_params, self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)
        return IGCR4SpinCombo6Ansatz(self.diagonal_from_parameters(params), left, right, self.nocc)

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError("spin-combo6 iGCR4 parameterization got wrong nelec")
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=self.nelec, copy=True)

        return func


IGCR4SpinBalancedFixedSectorAnsatz = IGCR4SpinCombo6Ansatz
IGCR4SpinBalancedFixedSectorParameterization = IGCR4SpinCombo6Parameterization
