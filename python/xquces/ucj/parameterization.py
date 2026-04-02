from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.orbitals import canonicalize_unitary
from xquces.ucj._unitary import (
    AntiHermitianUnitaryChart,
    OccupiedVirtualUnitaryChart,
)
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz, UCJLayer


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
class _GaugeReducedUCJMap:
    norb: int
    n_layers: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    @staticmethod
    def _build_gauge_basis(
        norb: int,
        pairs: list[tuple[int, int]],
        *,
        diag_factor: bool,
    ) -> tuple[np.ndarray, int]:
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros((0, 0), dtype=np.float64), 0
        a = np.zeros((n_pairs, norb), dtype=np.float64)
        for k, (p, q) in enumerate(pairs):
            if p == q and diag_factor:
                a[k, p] = 2.0
            else:
                a[k, p] += 1.0
                a[k, q] += 1.0
        u, s, _ = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > 1e-10))
        v_indep = u[:, rank:]
        return v_indep, v_indep.shape[1]

    @property
    def n_orb_rot_full(self) -> int:
        return self.norb**2

    @property
    def phase_indices(self) -> np.ndarray:
        return np.arange(self.norb, dtype=int)

    @property
    def kept_indices(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n_orb_rot_full), self.phase_indices)

    @property
    def n_orb_rot_reduced(self) -> int:
        return len(self.kept_indices)

    @property
    def v_aa(self) -> np.ndarray:
        return self._build_gauge_basis(
            self.norb,
            self.same_spin_pairs,
            diag_factor=False,
        )[0]

    @property
    def n_indep_aa(self) -> int:
        return self._build_gauge_basis(
            self.norb,
            self.same_spin_pairs,
            diag_factor=False,
        )[1]

    @property
    def v_ab(self) -> np.ndarray:
        return self._build_gauge_basis(
            self.norb,
            self.mixed_spin_pairs,
            diag_factor=True,
        )[0]

    @property
    def n_indep_ab(self) -> int:
        return self._build_gauge_basis(
            self.norb,
            self.mixed_spin_pairs,
            diag_factor=True,
        )[1]

    @property
    def n_full_per_layer(self) -> int:
        return self.n_orb_rot_full + len(self.same_spin_pairs) + len(self.mixed_spin_pairs)

    @property
    def n_reduced_per_layer(self) -> int:
        return self.n_orb_rot_reduced + self.n_indep_aa + self.n_indep_ab

    @property
    def n_full(self) -> int:
        return self.n_layers * self.n_full_per_layer

    @property
    def n_reduced(self) -> int:
        return self.n_layers * self.n_reduced_per_layer

    def reduced_to_full(self, x_reduced: np.ndarray) -> np.ndarray:
        x_reduced = np.asarray(x_reduced, dtype=np.float64)
        x_full = np.zeros(self.n_full, dtype=np.float64)
        ir = 0
        iff = 0
        for _ in range(self.n_layers):
            x_full_orb = np.zeros(self.n_orb_rot_full, dtype=np.float64)
            x_full_orb[self.kept_indices] = x_reduced[ir : ir + self.n_orb_rot_reduced]
            x_full[iff : iff + self.n_orb_rot_full] = x_full_orb
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full

            n = len(self.same_spin_pairs)
            if self.n_indep_aa > 0:
                x_full[iff : iff + n] = self.v_aa @ x_reduced[ir : ir + self.n_indep_aa]
            ir += self.n_indep_aa
            iff += n

            n = len(self.mixed_spin_pairs)
            if self.n_indep_ab > 0:
                x_full[iff : iff + n] = self.v_ab @ x_reduced[ir : ir + self.n_indep_ab]
            ir += self.n_indep_ab
            iff += n

        return x_full

    def full_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        x_full = np.asarray(x_full, dtype=np.float64)
        x_reduced = np.empty(self.n_reduced, dtype=np.float64)
        ir = 0
        iff = 0
        for _ in range(self.n_layers):
            x_reduced[ir : ir + self.n_orb_rot_reduced] = x_full[
                iff : iff + self.n_orb_rot_full
            ][self.kept_indices]
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full

            n = len(self.same_spin_pairs)
            if self.n_indep_aa > 0:
                x_reduced[ir : ir + self.n_indep_aa] = self.v_aa.T @ x_full[iff : iff + n]
            ir += self.n_indep_aa
            iff += n

            n = len(self.mixed_spin_pairs)
            if self.n_indep_ab > 0:
                x_reduced[ir : ir + self.n_indep_ab] = self.v_ab.T @ x_full[iff : iff + n]
            ir += self.n_indep_ab
            iff += n

        return x_reduced


@dataclass(frozen=True)
class UCJSpinRestrictedParameterization:
    norb: int
    n_layers: int
    interaction_pairs: list[tuple[int, int]] | None = None
    with_final_orbital_rotation: bool = False
    orbital_chart: AntiHermitianUnitaryChart = field(default_factory=AntiHermitianUnitaryChart)

    def __post_init__(self):
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def n_orbital_rotation_params(self) -> int:
        return self.orbital_chart.n_params(self.norb)

    @property
    def n_diagonal_params(self) -> int:
        return self.norb

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_layer_params(self) -> int:
        return self.n_orbital_rotation_params + self.n_diagonal_params + self.n_pair_params

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return self.n_orbital_rotation_params if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_layer_params + self.n_final_orbital_rotation_params

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs = self.pair_indices
        idx = 0
        layers: list[UCJLayer] = []
        for _ in range(self.n_layers):
            n = self.n_orbital_rotation_params
            u = self.orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
            idx += n

            n = self.n_diagonal_params
            d = np.array(params[idx : idx + n], copy=True)
            idx += n

            n = self.n_pair_params
            p = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs)
            idx += n

            layers.append(
                UCJLayer(
                    diagonal=SpinRestrictedSpec(double_params=d, pair_params=p),
                    orbital_rotation=u,
                )
            )

        final_orbital_rotation = None
        if self.with_final_orbital_rotation:
            n = self.n_orbital_rotation_params
            final_orbital_rotation = self.orbital_chart.unitary_from_parameters(
                params[idx : idx + n],
                self.norb,
            )

        return UCJAnsatz(
            layers=tuple(layers),
            final_orbital_rotation=final_orbital_rotation,
        )

    def parameters_from_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != self.n_layers:
            raise ValueError("ansatz n_layers does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        if self.with_final_orbital_rotation != (ansatz.final_orbital_rotation is not None):
            raise ValueError("final orbital rotation presence does not match parameterization")

        pairs = self.pair_indices
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        for layer in ansatz.layers:
            d = layer.diagonal

            n = self.n_orbital_rotation_params
            out[idx : idx + n] = self.orbital_chart.parameters_from_unitary(layer.orbital_rotation)
            idx += n

            n = self.n_diagonal_params
            out[idx : idx + n] = np.asarray(d.double_params, dtype=np.float64)
            idx += n

            n = self.n_pair_params
            if n:
                out[idx : idx + n] = np.asarray(
                    [d.pair_params[p, q] for p, q in pairs],
                    dtype=np.float64,
                )
                idx += n

        if self.with_final_orbital_rotation:
            n = self.n_orbital_rotation_params
            out[idx : idx + n] = self.orbital_chart.parameters_from_unitary(ansatz.final_orbital_rotation)

        return out

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
class UCJSpinBalancedParameterization:
    norb: int
    n_layers: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    with_final_orbital_rotation: bool = False
    orbital_chart: AntiHermitianUnitaryChart = field(default_factory=AntiHermitianUnitaryChart)

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
    def n_orbital_rotation_params(self) -> int:
        return self.orbital_chart.n_params(self.norb)

    @property
    def n_same_spin_params(self) -> int:
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self) -> int:
        return len(self.mixed_spin_indices)

    @property
    def n_layer_params(self) -> int:
        return self.n_orbital_rotation_params + self.n_same_spin_params + self.n_mixed_spin_params

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return self.n_orbital_rotation_params if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_layer_params + self.n_final_orbital_rotation_params

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs_aa = self.same_spin_indices
        pairs_ab = self.mixed_spin_indices
        idx = 0
        layers: list[UCJLayer] = []
        for _ in range(self.n_layers):
            n = self.n_orbital_rotation_params
            u = self.orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
            idx += n

            n = self.n_same_spin_params
            j0 = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs_aa)
            idx += n

            n = self.n_mixed_spin_params
            j1 = _symmetric_matrix_from_values(params[idx : idx + n], self.norb, pairs_ab)
            idx += n

            layers.append(
                UCJLayer(
                    diagonal=SpinBalancedSpec(
                        same_spin_params=j0,
                        mixed_spin_params=j1,
                    ),
                    orbital_rotation=u,
                )
            )

        final_orbital_rotation = None
        if self.with_final_orbital_rotation:
            n = self.n_orbital_rotation_params
            final_orbital_rotation = self.orbital_chart.unitary_from_parameters(
                params[idx : idx + n],
                self.norb,
            )

        return UCJAnsatz(
            layers=tuple(layers),
            final_orbital_rotation=final_orbital_rotation,
        )

    def parameters_from_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != self.n_layers:
            raise ValueError("ansatz n_layers does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        if self.with_final_orbital_rotation != (ansatz.final_orbital_rotation is not None):
            raise ValueError("final orbital rotation presence does not match parameterization")

        pairs_aa = self.same_spin_indices
        pairs_ab = self.mixed_spin_indices
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        for layer in ansatz.layers:
            d = layer.diagonal

            n = self.n_orbital_rotation_params
            out[idx : idx + n] = self.orbital_chart.parameters_from_unitary(layer.orbital_rotation)
            idx += n

            n = self.n_same_spin_params
            if n:
                out[idx : idx + n] = np.asarray(
                    [d.same_spin_params[p, q] for p, q in pairs_aa],
                    dtype=np.float64,
                )
                idx += n

            n = self.n_mixed_spin_params
            if n:
                out[idx : idx + n] = np.asarray(
                    [d.mixed_spin_params[p, q] for p, q in pairs_ab],
                    dtype=np.float64,
                )
                idx += n

        if self.with_final_orbital_rotation:
            n = self.n_orbital_rotation_params
            out[idx : idx + n] = self.orbital_chart.parameters_from_unitary(ansatz.final_orbital_rotation)

        return out

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
class GaugeFixedUCJSpinRestrictedParameterization:
    norb: int
    nocc: int
    n_layers: int
    interaction_pairs: list[tuple[int, int]] | None = None
    with_final_orbital_rotation: bool = False
    orbital_chart: AntiHermitianUnitaryChart = field(default_factory=AntiHermitianUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def gauge_map(self) -> _GaugeReducedUCJMap:
        return _GaugeReducedUCJMap(
            norb=self.norb,
            n_layers=self.n_layers,
            same_spin_pairs=self.pair_indices,
            mixed_spin_pairs=[],
        )

    @property
    def final_orbital_chart(self) -> OccupiedVirtualUnitaryChart:
        return OccupiedVirtualUnitaryChart(self.nocc, self.nvirt)

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return self.final_orbital_chart.n_params() if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return self.gauge_map.n_reduced + self.n_layers * self.norb + self.n_final_orbital_rotation_params

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")

        n_ucj = self.gauge_map.n_reduced
        x_ucj = params[:n_ucj]
        x_rest = params[n_ucj:]

        x_full = self.gauge_map.reduced_to_full(x_ucj)

        idx_full = 0
        idx_d = 0
        layers: list[UCJLayer] = []

        for _ in range(self.n_layers):
            n = self.gauge_map.n_orb_rot_full
            u = self.orbital_chart.unitary_from_parameters(
                x_full[idx_full : idx_full + n],
                self.norb,
            )
            idx_full += n

            n = len(self.pair_indices)
            p = _symmetric_matrix_from_values(
                x_full[idx_full : idx_full + n],
                self.norb,
                self.pair_indices,
            )
            idx_full += n

            d = np.array(x_rest[idx_d : idx_d + self.norb], copy=True)
            idx_d += self.norb

            layers.append(
                UCJLayer(
                    diagonal=SpinRestrictedSpec(double_params=d, pair_params=p),
                    orbital_rotation=u,
                )
            )

        final_orbital_rotation = None
        if self.with_final_orbital_rotation:
            final_orbital_rotation = self.final_orbital_chart.unitary_from_parameters(
                params[-self.n_final_orbital_rotation_params :]
            )

        return UCJAnsatz(
            layers=tuple(layers),
            final_orbital_rotation=final_orbital_rotation,
        )

    def parameters_from_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != self.n_layers:
            raise ValueError("ansatz n_layers does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        if self.with_final_orbital_rotation != (ansatz.final_orbital_rotation is not None):
            raise ValueError("final orbital rotation presence does not match parameterization")

        x_full = np.zeros(self.gauge_map.n_full, dtype=np.float64)
        x_d = np.zeros(self.n_layers * self.norb, dtype=np.float64)

        idx_full = 0
        idx_d = 0

        for layer in ansatz.layers:
            d = layer.diagonal

            n = self.gauge_map.n_orb_rot_full
            x_full[idx_full : idx_full + n] = self.orbital_chart.parameters_from_unitary(
                _canonicalize_internal_unitary(layer.orbital_rotation)
            )
            idx_full += n

            n = len(self.pair_indices)
            x_full[idx_full : idx_full + n] = np.asarray(
                [d.pair_params[p, q] for p, q in self.pair_indices],
                dtype=np.float64,
            )
            idx_full += n

            x_d[idx_d : idx_d + self.norb] = np.asarray(d.double_params, dtype=np.float64)
            idx_d += self.norb

        x_ucj = self.gauge_map.full_to_reduced(x_full)

        if not self.with_final_orbital_rotation:
            return np.concatenate([x_ucj, x_d])

        x_final = self.final_orbital_chart.parameters_from_unitary(ansatz.final_orbital_rotation)
        return np.concatenate([x_ucj, x_d, x_final])

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
class GaugeFixedUCJSpinBalancedParameterization:
    norb: int
    nocc: int
    n_layers: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    with_final_orbital_rotation: bool = False
    orbital_chart: AntiHermitianUnitaryChart = field(default_factory=AntiHermitianUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        if self.same_spin_interaction_pairs is not None:
            _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        if self.mixed_spin_interaction_pairs is not None:
            _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        if self.same_spin_interaction_pairs is None:
            return _default_upper_indices(self.norb)
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        if self.mixed_spin_interaction_pairs is None:
            return _default_triu_indices(self.norb)
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=True)

    @property
    def gauge_map(self) -> _GaugeReducedUCJMap:
        return _GaugeReducedUCJMap(
            norb=self.norb,
            n_layers=self.n_layers,
            same_spin_pairs=self.same_spin_indices,
            mixed_spin_pairs=self.mixed_spin_indices,
        )

    @property
    def final_orbital_chart(self) -> OccupiedVirtualUnitaryChart:
        return OccupiedVirtualUnitaryChart(self.nocc, self.nvirt)

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return self.final_orbital_chart.n_params() if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return self.gauge_map.n_reduced + self.n_final_orbital_rotation_params

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")

        n_ucj = self.gauge_map.n_reduced
        x_ucj = params[:n_ucj]
        x_full = self.gauge_map.reduced_to_full(x_ucj)

        idx = 0
        layers: list[UCJLayer] = []

        for _ in range(self.n_layers):
            n = self.gauge_map.n_orb_rot_full
            u = self.orbital_chart.unitary_from_parameters(
                x_full[idx : idx + n],
                self.norb,
            )
            idx += n

            n = len(self.same_spin_indices)
            j0 = _symmetric_matrix_from_values(
                x_full[idx : idx + n],
                self.norb,
                self.same_spin_indices,
            )
            idx += n

            n = len(self.mixed_spin_indices)
            j1 = _symmetric_matrix_from_values(
                x_full[idx : idx + n],
                self.norb,
                self.mixed_spin_indices,
            )
            idx += n

            layers.append(
                UCJLayer(
                    diagonal=SpinBalancedSpec(
                        same_spin_params=j0,
                        mixed_spin_params=j1,
                    ),
                    orbital_rotation=u,
                )
            )

        final_orbital_rotation = None
        if self.with_final_orbital_rotation:
            final_orbital_rotation = self.final_orbital_chart.unitary_from_parameters(
                params[-self.n_final_orbital_rotation_params :]
            )

        return UCJAnsatz(
            layers=tuple(layers),
            final_orbital_rotation=final_orbital_rotation,
        )

    def parameters_from_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != self.n_layers:
            raise ValueError("ansatz n_layers does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        if self.with_final_orbital_rotation != (ansatz.final_orbital_rotation is not None):
            raise ValueError("final orbital rotation presence does not match parameterization")

        x_full = np.zeros(self.gauge_map.n_full, dtype=np.float64)
        idx = 0

        for layer in ansatz.layers:
            d = layer.diagonal

            n = self.gauge_map.n_orb_rot_full
            x_full[idx : idx + n] = self.orbital_chart.parameters_from_unitary(
                _canonicalize_internal_unitary(layer.orbital_rotation)
            )
            idx += n

            n = len(self.same_spin_indices)
            x_full[idx : idx + n] = np.asarray(
                [d.same_spin_params[p, q] for p, q in self.same_spin_indices],
                dtype=np.float64,
            )
            idx += n

            n = len(self.mixed_spin_indices)
            x_full[idx : idx + n] = np.asarray(
                [d.mixed_spin_params[p, q] for p, q in self.mixed_spin_indices],
                dtype=np.float64,
            )
            idx += n

        x_ucj = self.gauge_map.full_to_reduced(x_full)

        if not self.with_final_orbital_rotation:
            return x_ucj

        x_final = self.final_orbital_chart.parameters_from_unitary(ansatz.final_orbital_rotation)
        return np.concatenate([x_ucj, x_final])

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func