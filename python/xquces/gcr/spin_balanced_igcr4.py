from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces._lib import apply_igcr4_spin_combo6_in_place_num_rep
from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gcr.igcr2 import (
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _right_unitary_from_left_and_final,
)
from xquces.orbitals import apply_orbital_rotation


SPIN_COMBO6_LABELS = (
    "aabb",
    "abab",
    "abba",
    "baab",
    "bbaa",
)


def _default_q4_indices(norb: int) -> list[tuple[int, int, int, int]]:
    return [(p, q, r, s) for p in range(norb) for q in range(p + 1) for r in range(q + 1) for s in range(r + 1)]


def _validate_q4_indices(
    indices: list[tuple[int, int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int, int]]:
    if indices is None:
        return _default_q4_indices(norb)
    out = []
    seen = set()
    for p, q, r, s in indices:
        if not (0 <= s <= r <= q <= p < norb):
            raise ValueError("q4 indices must satisfy 0 <= s <= r <= q <= p < norb")
        key = (p, q, r, s)
        if key in seen:
            raise ValueError("q4 indices must not contain duplicates")
        seen.add(key)
        out.append(key)
    return out


@dataclass(frozen=True)
class IGCR4SpinCombo6Spec:
    q4_values: np.ndarray
    norb_: int
    q4_indices_: list[tuple[int, int, int, int]] | None = None

    @property
    def norb(self) -> int:
        return int(self.norb_)

    @property
    def q4_indices(self) -> list[tuple[int, int, int, int]]:
        return _validate_q4_indices(self.q4_indices_, self.norb)

    def q4_matrix(self) -> np.ndarray:
        values = np.asarray(self.q4_values, dtype=np.float64)
        if values.shape != (len(self.q4_indices), 6):
            raise ValueError("q4_values must have shape (n_q4, 6)")
        if self.q4_indices == _default_q4_indices(self.norb):
            return values
        out = np.zeros((len(_default_q4_indices(self.norb)), 6), dtype=np.float64)
        pos = {idx: k for k, idx in enumerate(_default_q4_indices(self.norb))}
        for value, idx in zip(values, self.q4_indices):
            out[pos[idx]] = value
        return out

    def phase_from_occupations(
        self,
        occ_alpha: np.ndarray,
        occ_beta: np.ndarray,
    ) -> float:
        a = np.zeros(self.norb, dtype=np.float64)
        b = np.zeros(self.norb, dtype=np.float64)
        a[np.asarray(occ_alpha, dtype=np.int64)] = 1.0
        b[np.asarray(occ_beta, dtype=np.int64)] = 1.0
        return self.phase_from_spin_arrays(a, b)

    def phase_from_spin_arrays(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != (self.norb,) or b.shape != (self.norb,):
            raise ValueError("a and b must have shape (norb,)")
        phase = 0.0
        for values, (p, q, r, s) in zip(self.q4_matrix(), _default_q4_indices(self.norb)):
            phase += values[0] * a[p] * a[q] * b[r] * b[s]
            phase += values[1] * a[p] * b[q] * a[r] * b[s]
            phase += values[2] * a[p] * b[q] * b[r] * a[s]
            phase += values[3] * b[p] * a[q] * a[r] * b[s]
            phase += values[4] * b[p] * a[q] * b[r] * a[s]
            phase += values[5] * b[p] * b[q] * a[r] * a[s]
        return float(phase)


def apply_igcr4_spin_combo6_diagonal(
    vec: np.ndarray,
    diagonal: IGCR4SpinCombo6Spec,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    state2 = reshape_state(arr, norb, nelec)
    occ_alpha = occ_indicator_rows(norb, nelec[0])
    occ_beta = occ_indicator_rows(norb, nelec[1])
    apply_igcr4_spin_combo6_in_place_num_rep(
        state2,
        np.asarray(diagonal.q4_matrix(), dtype=np.float64) * time,
        norb,
        occ_alpha,
        occ_beta,
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
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        arr = apply_igcr4_spin_combo6_diagonal(
            arr,
            self.diagonal,
            self.norb,
            nelec,
            copy=False,
        )
        arr = apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        return arr


@dataclass(frozen=True)
class IGCR4SpinCombo6Parameterization:
    norb: int
    nelec: tuple[int, int]
    q4_indices_: list[tuple[int, int, int, int]] | None = None
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
        object.__setattr__(self, "nelec", nelec)
        _validate_q4_indices(self.q4_indices_, self.norb)

    @property
    def nocc(self) -> int:
        return self.nelec[0]

    @property
    def q4_indices(self) -> list[tuple[int, int, int, int]]:
        return _validate_q4_indices(self.q4_indices_, self.norb)

    @property
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_q4_params(self):
        return 6 * len(self.q4_indices)

    @property
    def n_diagonal_params(self):
        return self.n_q4_params

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return self.n_left_orbital_rotation_params + self.n_diagonal_params

    @property
    def _left_right_ov_transform_scale(self):
        return None

    @property
    def n_params(self):
        return self.n_left_orbital_rotation_params + self.n_q4_params + self.n_right_orbital_rotation_params

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "q4": self.n_q4_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _split_params(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        i0 = 0
        i1 = self.n_left_orbital_rotation_params
        i2 = i1 + self.n_q4_params
        return (
            params[i0:i1],
            params[i1:i2].reshape((len(self.q4_indices), 6)),
            params[i2:],
        )

    def diagonal_from_parameters(self, params: np.ndarray) -> IGCR4SpinCombo6Spec:
        _, q4_values, _ = self._split_params(params)
        return IGCR4SpinCombo6Spec(
            q4_values=np.asarray(q4_values, dtype=np.float64),
            norb_=self.norb,
            q4_indices_=self.q4_indices,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4SpinCombo6Ansatz:
        params = self._native_parameters_from_public(params)
        left_params, _, right_params = self._split_params(params)
        left = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final = self.right_orbital_chart.unitary_from_parameters(right_params, self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)
        return IGCR4SpinCombo6Ansatz(
            diagonal=self.diagonal_from_parameters(params),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_restricted_igcr4_ansatz(self, ansatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(ansatz.left)
        else:
            left_params = left_chart.parameters_from_unitary(ansatz.left)
            right_phase = np.zeros(self.norb, dtype=np.float64)
        right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
        left_param_unitary = left_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        right_params = self.right_orbital_chart.parameters_from_unitary(final_eff)
        sigma_map = dict(zip(ansatz.diagonal.sigma_indices, ansatz.diagonal.sigma_vector()))
        q4_values = np.zeros((len(self.q4_indices), 6), dtype=np.float64)
        for k, idx in enumerate(self.q4_indices):
            rev = tuple(sorted(idx))
            if len(set(idx)) == 4 and rev in sigma_map:
                q4_values[k] = sigma_map[rev]
        return self._public_parameters_from_native(np.concatenate([left_params, q4_values.reshape(-1), right_params]))

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError("spin-combo6 iGCR4 parameterization got wrong nelec")
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=self.nelec, copy=True)

        return func


IGCR4SpinBalancedFixedSectorAnsatz = IGCR4SpinCombo6Ansatz
IGCR4SpinBalancedFixedSectorParameterization = IGCR4SpinCombo6Parameterization
