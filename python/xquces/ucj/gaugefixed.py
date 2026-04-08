from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.ucj._unitary import AntiHermitianUnitaryChart
from xquces.ucj.init import UCJBalancedDFSeed
from xquces.ucj.model import SpinBalancedSpec, UCJAnsatz, UCJLayer
from xquces.ucj.parameterization import ov_final_param_dim, ov_final_unitary, ov_params_from_unitary


def _default_triu_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(norb), 2))


def _validate_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
    *,
    allow_diagonal: bool,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_triu_indices(norb)
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


def _symmetric_matrix_from_values(values: np.ndarray, norb: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((norb, norb), dtype=np.float64)
    if not pairs:
        return out
    rows, cols = zip(*pairs)
    vals = np.asarray(values, dtype=np.float64)
    out[rows, cols] = vals
    out[cols, rows] = vals
    return out


def _canonicalize_internal_unitary(u: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    u = np.array(u, dtype=np.complex128, copy=True)
    norb = u.shape[0]
    phases = np.ones(norb, dtype=np.complex128)
    for j in range(norb):
        col = u[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > tol:
            phases[j] = np.exp(-1j * np.angle(val))
    return u @ np.diag(phases)


@dataclass(frozen=True)
class _InternalOrbitalGaugeMap:
    norb: int
    n_layers: int

    @property
    def n_orb_rot_full(self) -> int:
        return self.norb**2

    @property
    def phase_indices(self) -> np.ndarray:
        n_triu_strict = self.norb * (self.norb - 1) // 2
        rows_full, cols_full = np.triu_indices(self.norb, k=0)
        diag_positions_in_imag = [idx for idx, (r, c) in enumerate(zip(rows_full, cols_full)) if r == c]
        return np.array([n_triu_strict + p for p in diag_positions_in_imag], dtype=int)

    @property
    def kept_indices(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n_orb_rot_full), self.phase_indices)

    @property
    def n_orb_rot_reduced(self) -> int:
        return len(self.kept_indices)

    @property
    def n_full(self) -> int:
        return self.n_layers * self.n_orb_rot_full

    @property
    def n_reduced(self) -> int:
        return self.n_layers * self.n_orb_rot_reduced

    def reduced_to_full(self, x_reduced: np.ndarray) -> np.ndarray:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        out = np.zeros(self.n_full, dtype=np.float64)
        ir = 0
        iff = 0
        for _ in range(self.n_layers):
            block = np.zeros(self.n_orb_rot_full, dtype=np.float64)
            block[self.kept_indices] = x_reduced[ir : ir + self.n_orb_rot_reduced]
            out[iff : iff + self.n_orb_rot_full] = block
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full
        return out

    def full_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        out = np.empty(self.n_reduced, dtype=np.float64)
        ir = 0
        iff = 0
        for _ in range(self.n_layers):
            out[ir : ir + self.n_orb_rot_reduced] = x_full[iff : iff + self.n_orb_rot_full][self.kept_indices]
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full
        return out


@dataclass(frozen=True)
class _GaugeReducedUCJMap:
    norb: int
    n_layers: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        v_same, n_same = self._build_gauge_basis(self.same_spin_pairs, diag_factor=False)
        v_mixed, n_mixed = self._build_gauge_basis(self.mixed_spin_pairs, diag_factor=True)
        object.__setattr__(self, "_v_same", v_same)
        object.__setattr__(self, "_v_mixed", v_mixed)
        object.__setattr__(self, "_n_same_reduced", n_same)
        object.__setattr__(self, "_n_mixed_reduced", n_mixed)

    def _build_gauge_basis(self, pairs: list[tuple[int, int]], diag_factor: bool) -> tuple[np.ndarray, int]:
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros((0, 0), dtype=np.float64), 0
        a = np.zeros((n_pairs, self.norb), dtype=np.float64)
        for k, (p, q) in enumerate(pairs):
            if p == q and diag_factor:
                a[k, p] = 2.0
            else:
                a[k, p] = 1.0
                a[k, q] = 1.0
        u, s, _ = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > 1e-10))
        v_indep = np.array(u[:, rank:], copy=True)
        for j in range(v_indep.shape[1]):
            col = v_indep[:, j]
            idx = int(np.argmax(np.abs(col)))
            if abs(col[idx]) > 1e-14 and col[idx] < 0:
                v_indep[:, j] *= -1.0
        return v_indep, v_indep.shape[1]

    @property
    def v_same(self) -> np.ndarray:
        return self._v_same

    @property
    def v_mixed(self) -> np.ndarray:
        return self._v_mixed

    @property
    def n_same_reduced(self) -> int:
        return self._n_same_reduced

    @property
    def n_mixed_reduced(self) -> int:
        return self._n_mixed_reduced

    @property
    def n_full_per_layer(self) -> int:
        return len(self.same_spin_pairs) + len(self.mixed_spin_pairs)

    @property
    def n_reduced_per_layer(self) -> int:
        return self.n_same_reduced + self.n_mixed_reduced

    @property
    def n_full(self) -> int:
        return self.n_layers * self.n_full_per_layer

    @property
    def n_reduced(self) -> int:
        return self.n_layers * self.n_reduced_per_layer

    def reduced_to_full(self, x_reduced: np.ndarray) -> np.ndarray:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        out = np.zeros(self.n_full, dtype=np.float64)
        ir = 0
        iff = 0
        n_same_full = len(self.same_spin_pairs)
        n_mixed_full = len(self.mixed_spin_pairs)
        for _ in range(self.n_layers):
            if self.n_same_reduced > 0:
                out[iff : iff + n_same_full] = self.v_same @ x_reduced[ir : ir + self.n_same_reduced]
            ir += self.n_same_reduced
            iff += n_same_full
            if self.n_mixed_reduced > 0:
                out[iff : iff + n_mixed_full] = self.v_mixed @ x_reduced[ir : ir + self.n_mixed_reduced]
            ir += self.n_mixed_reduced
            iff += n_mixed_full
        return out

    def full_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        out = np.empty(self.n_reduced, dtype=np.float64)
        ir = 0
        iff = 0
        n_same_full = len(self.same_spin_pairs)
        n_mixed_full = len(self.mixed_spin_pairs)
        for _ in range(self.n_layers):
            if self.n_same_reduced > 0:
                out[ir : ir + self.n_same_reduced] = self.v_same.T @ x_full[iff : iff + n_same_full]
            ir += self.n_same_reduced
            iff += n_same_full
            if self.n_mixed_reduced > 0:
                out[ir : ir + self.n_mixed_reduced] = self.v_mixed.T @ x_full[iff : iff + n_mixed_full]
            ir += self.n_mixed_reduced
            iff += n_mixed_full
        return out


@dataclass(frozen=True)
class GaugeFixedUCJSpinBalancedParameterizationV2:
    norb: int
    nocc: int
    n_layers: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    with_final_orbital_rotation: bool = False

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=True)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def orbital_chart(self) -> AntiHermitianUnitaryChart:
        return AntiHermitianUnitaryChart()

    @property
    def internal_orbital_map(self) -> _InternalOrbitalGaugeMap:
        return _InternalOrbitalGaugeMap(self.norb, self.n_layers)

    @property
    def diagonal_gauge_map(self) -> _GaugeReducedUCJMap:
        return _GaugeReducedUCJMap(
            self.norb,
            self.n_layers,
            self.same_spin_indices,
            self.mixed_spin_indices,
        )

    @property
    def n_internal_orbital_rotation_params(self) -> int:
        return self.internal_orbital_map.n_reduced // self.n_layers

    @property
    def n_diagonal_params(self) -> int:
        return self.diagonal_gauge_map.n_reduced // self.n_layers

    @property
    def n_layer_params(self) -> int:
        return self.n_internal_orbital_rotation_params + self.n_diagonal_params

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return ov_final_param_dim(self.norb, self.nocc) if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_layer_params + self.n_final_orbital_rotation_params

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        layers: list[UCJLayer] = []
        orb_map = self.internal_orbital_map
        diag_map = self.diagonal_gauge_map
        n_orb = self.n_internal_orbital_rotation_params
        n_diag = self.n_diagonal_params
        n_same = len(self.same_spin_indices)
        n_mixed = len(self.mixed_spin_indices)
        for _ in range(self.n_layers):
            x_orb_reduced = params[idx : idx + n_orb]
            idx += n_orb
            x_orb_full = np.zeros(self.norb**2, dtype=np.float64)
            x_orb_full[orb_map.kept_indices] = x_orb_reduced
            u = _canonicalize_internal_unitary(self.orbital_chart.unitary_from_parameters(x_orb_full, self.norb))
            x_diag_reduced = params[idx : idx + n_diag]
            idx += n_diag
            x_diag_full = np.zeros(n_same + n_mixed, dtype=np.float64)
            if diag_map.n_reduced_per_layer > 0:
                x_diag_full = diag_map.reduced_to_full(x_diag_reduced)
            j0 = _symmetric_matrix_from_values(x_diag_full[:n_same], self.norb, self.same_spin_indices)
            j1 = _symmetric_matrix_from_values(x_diag_full[n_same : n_same + n_mixed], self.norb, self.mixed_spin_indices)
            layers.append(
                UCJLayer(
                    diagonal=SpinBalancedSpec(same_spin_params=j0, mixed_spin_params=j1),
                    orbital_rotation=u,
                )
            )
        final_orbital_rotation = None
        if self.with_final_orbital_rotation:
            x_final = params[idx : idx + self.n_final_orbital_rotation_params]
            final_orbital_rotation = ov_final_unitary(x_final, self.norb, self.nocc)
        return UCJAnsatz(layers=tuple(layers), final_orbital_rotation=final_orbital_rotation)

    def parameters_from_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != self.n_layers:
            raise ValueError("ansatz n_layers does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        if self.with_final_orbital_rotation != (ansatz.final_orbital_rotation is not None):
            raise ValueError("final orbital rotation presence does not match parameterization")
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        orb_map = self.internal_orbital_map
        diag_map = self.diagonal_gauge_map
        n_orb = self.n_internal_orbital_rotation_params
        n_diag = self.n_diagonal_params
        n_same = len(self.same_spin_indices)
        n_mixed = len(self.mixed_spin_indices)
        for layer in ansatz.layers:
            d = layer.diagonal
            x_orb_full = self.orbital_chart.parameters_from_unitary(_canonicalize_internal_unitary(layer.orbital_rotation))
            out[idx : idx + n_orb] = x_orb_full[orb_map.kept_indices]
            idx += n_orb
            x_diag_full = np.zeros(n_same + n_mixed, dtype=np.float64)
            if n_same:
                x_diag_full[:n_same] = np.asarray([d.same_spin_params[p, q] for p, q in self.same_spin_indices], dtype=np.float64)
            if n_mixed:
                x_diag_full[n_same : n_same + n_mixed] = np.asarray([d.mixed_spin_params[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
            out[idx : idx + n_diag] = diag_map.full_to_reduced(x_diag_full)
            idx += n_diag
        if self.with_final_orbital_rotation:
            out[idx : idx + self.n_final_orbital_rotation_params] = ov_params_from_unitary(ansatz.final_orbital_rotation, self.nocc)
        return out

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(ansatz)

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)
        return func


@dataclass(frozen=True)
class GaugeFixedUCJBalancedDFSeedV2:
    t2: np.ndarray
    t1: np.ndarray | None = None
    n_reps: int | None = None
    tol: float = 1e-8
    optimize: bool = False
    method: str = "L-BFGS-B"
    callback: object = None
    options: dict | None = None
    regularization: float = 0.0
    multi_stage_start: int | None = None
    multi_stage_step: int | None = None

    def build_parameters(self) -> tuple[UCJAnsatz, GaugeFixedUCJSpinBalancedParameterizationV2, np.ndarray]:
        ucj_ansatz = UCJBalancedDFSeed(
            t2=self.t2,
            t1=self.t1,
            n_reps=self.n_reps,
            tol=self.tol,
            optimize=self.optimize,
            method=self.method,
            callback=self.callback,
            options=self.options,
            regularization=self.regularization,
            multi_stage_start=self.multi_stage_start,
            multi_stage_step=self.multi_stage_step,
        ).build_ansatz()
        param = GaugeFixedUCJSpinBalancedParameterizationV2(
            norb=ucj_ansatz.norb,
            nocc=np.asarray(self.t2).shape[0],
            n_layers=ucj_ansatz.n_layers,
            with_final_orbital_rotation=ucj_ansatz.final_orbital_rotation is not None,
        )
        x0 = param.parameters_from_ansatz(ucj_ansatz)
        ansatz = param.ansatz_from_parameters(x0)
        return ansatz, param, x0

    def build_ansatz(self) -> UCJAnsatz:
        ansatz, _, _ = self.build_parameters()
        return ansatz
