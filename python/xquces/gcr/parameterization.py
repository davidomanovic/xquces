from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj._unitary import ExactUnitaryChart, GaugeFixedInternalUnitaryChart
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


def _canonicalize_internal_unitary_with_phase_matrix(
    u: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("unitary must be square")
    out = np.array(u, copy=True)
    norb = out.shape[0]
    phases = np.ones(norb, dtype=np.complex128)
    for j in range(norb):
        col = out[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > tol:
            phases[j] = np.exp(-1j * np.angle(val))
    d = np.diag(phases)
    return out @ d, d


def _diag_unitary_from_real_phases(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * np.asarray(phases, dtype=np.float64)))


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
class _GaugeReducedSpinBalancedMap:
    norb: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        a_same = np.zeros((len(self.same_spin_pairs), self.norb), dtype=np.float64)
        for k, (p, q) in enumerate(self.same_spin_pairs):
            a_same[k, p] = 1.0
            a_same[k, q] = 1.0

        a_mixed = np.zeros((len(self.mixed_spin_pairs), self.norb), dtype=np.float64)
        for k, (p, q) in enumerate(self.mixed_spin_pairs):
            if p == q:
                a_mixed[k, p] = 2.0
            else:
                a_mixed[k, p] = 1.0
                a_mixed[k, q] = 1.0

        u_same, s_same, _ = np.linalg.svd(a_same, full_matrices=True)
        rank_same = int(np.sum(s_same > 1e-10))
        v_same = np.array(u_same[:, rank_same:], copy=True)

        u_mixed, s_mixed, _ = np.linalg.svd(a_mixed, full_matrices=True)
        rank_mixed = int(np.sum(s_mixed > 1e-10))
        v_mixed = np.array(u_mixed[:, rank_mixed:], copy=True)

        for j in range(v_same.shape[1]):
            col = v_same[:, j]
            idx = int(np.argmax(np.abs(col)))
            if abs(col[idx]) > 1e-14 and col[idx] < 0:
                v_same[:, j] *= -1.0

        for j in range(v_mixed.shape[1]):
            col = v_mixed[:, j]
            idx = int(np.argmax(np.abs(col)))
            if abs(col[idx]) > 1e-14 and col[idx] < 0:
                v_mixed[:, j] *= -1.0

        object.__setattr__(self, "_a_same", a_same)
        object.__setattr__(self, "_a_mixed", a_mixed)
        object.__setattr__(self, "_v_same", v_same)
        object.__setattr__(self, "_v_mixed", v_mixed)

    @property
    def a_same(self) -> np.ndarray:
        return self._a_same

    @property
    def a_mixed(self) -> np.ndarray:
        return self._a_mixed

    @property
    def v_same(self) -> np.ndarray:
        return self._v_same

    @property
    def v_mixed(self) -> np.ndarray:
        return self._v_mixed

    @property
    def n_same_reduced(self) -> int:
        return self.v_same.shape[1]

    @property
    def n_mixed_reduced(self) -> int:
        return self.v_mixed.shape[1]

    @property
    def n_reduced(self) -> int:
        return self.n_same_reduced + self.n_mixed_reduced

    def reduced_to_full(self, x_reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        i = 0
        x_same = self.v_same @ x_reduced[i:i + self.n_same_reduced]
        i += self.n_same_reduced
        x_mixed = self.v_mixed @ x_reduced[i:i + self.n_mixed_reduced]
        return x_same, x_mixed

    def full_to_reduced(
        self,
        x_same_full: np.ndarray,
        x_mixed_full: np.ndarray,
    ) -> np.ndarray:
        x_same_full = np.asarray(x_same_full, dtype=np.float64)
        x_mixed_full = np.asarray(x_mixed_full, dtype=np.float64)
        return np.concatenate(
            [
                self.v_same.T @ x_same_full,
                self.v_mixed.T @ x_mixed_full,
            ]
        )

    def split_gauge(
        self,
        x_same_full: np.ndarray,
        x_mixed_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_same_full = np.asarray(x_same_full, dtype=np.float64)
        x_mixed_full = np.asarray(x_mixed_full, dtype=np.float64)

        x_same_phys = self.v_same @ (self.v_same.T @ x_same_full)
        x_mixed_phys = self.v_mixed @ (self.v_mixed.T @ x_mixed_full)

        g_same = x_same_full - x_same_phys
        g_mixed = x_mixed_full - x_mixed_phys

        lam_same, *_ = np.linalg.lstsq(self.a_same, g_same, rcond=None)
        lam_mixed, *_ = np.linalg.lstsq(self.a_mixed, g_mixed, rcond=None)

        return x_same_phys, x_mixed_phys, lam_same, lam_mixed


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
        if isinstance(self.left_orbital_chart, ExactUnitaryChart):
            return GaugeFixedInternalUnitaryChart()
        return self.left_orbital_chart

    @property
    def _effective_right_chart(self):
        if isinstance(self.right_orbital_chart, ExactUnitaryChart):
            return _OVRightChart(self.nocc, self.norb)
        return self.right_orbital_chart

    @property
    def _jastrow_gauge_map(self) -> _GaugeReducedSpinBalancedMap:
        return _GaugeReducedSpinBalancedMap(
            self.norb,
            self.same_spin_indices,
            self.mixed_spin_indices,
        )

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._effective_left_chart.n_params(self.norb)

    @property
    def n_jastrow_params(self) -> int:
        return self._jastrow_gauge_map.n_reduced

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
        left = self._effective_left_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n = self.n_jastrow_params
        same_full, mixed_full = self._jastrow_gauge_map.reduced_to_full(params[idx:idx + n])
        idx += n

        same = _symmetric_matrix_from_values(same_full, self.norb, self.same_spin_indices)
        mixed = _symmetric_matrix_from_values(mixed_full, self.norb, self.mixed_spin_indices)

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

        d = ansatz.diagonal
        left = np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128)
        right = np.asarray(ansatz.right_orbital_rotation, dtype=np.complex128)

        if isinstance(self.left_orbital_chart, ExactUnitaryChart):
            left_eff, d_left = _canonicalize_internal_unitary_with_phase_matrix(left)
        else:
            left_eff = left
            d_left = np.eye(self.norb, dtype=np.complex128)

        same_diag = np.diag(np.asarray(d.same_spin_params, dtype=np.float64))
        same_full = np.asarray(
            [d.same_spin_params[p, q] for p, q in self.same_spin_indices],
            dtype=np.float64,
        )
        mixed_full = np.asarray(
            [d.mixed_spin_params[p, q] for p, q in self.mixed_spin_indices],
            dtype=np.float64,
        )

        same_phys, mixed_phys, lam_same, lam_mixed = self._jastrow_gauge_map.split_gauge(
            same_full,
            mixed_full,
        )

        phase_vec = 0.5 * same_diag + (self.nocc - 1) * lam_same + self.nocc * lam_mixed
        d_phase = _diag_unitary_from_real_phases(phase_vec)
        right_eff = d_phase @ d_left.conj().T @ right

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self._effective_left_chart.parameters_from_unitary(left_eff)
        idx += n

        n = self.n_jastrow_params
        out[idx:idx + n] = self._jastrow_gauge_map.full_to_reduced(same_phys, mixed_phys)
        idx += n

        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = self._effective_right_chart.parameters_from_unitary(right_eff)

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