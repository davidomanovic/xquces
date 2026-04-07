from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np

from ffsim.linalg.util import (
    unitary_from_parameters as ffsim_unitary_from_parameters,
)
from ffsim.linalg.util import (
    unitary_to_parameters as ffsim_unitary_to_parameters,
)

from xquces.ucj._unitary import AntiHermitianUnitaryChart, OccupiedVirtualUnitaryChart
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
        diag_positions_in_imag = [
            idx for idx, (r, c) in enumerate(zip(rows_full, cols_full)) if r == c
        ]
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
            out[ir : ir + self.n_orb_rot_reduced] = x_full[
                iff : iff + self.n_orb_rot_full
            ][self.kept_indices]
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full
        return out


@dataclass(frozen=True)
class _GaugeReducedSpinBalancedMap:
    norb: int
    n_layers: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        v_same, n_same = self._build_gauge_basis(self.same_spin_pairs, False)
        v_mixed, n_mixed = self._build_gauge_basis(self.mixed_spin_pairs, True)
        object.__setattr__(self, "_v_same", v_same)
        object.__setattr__(self, "_v_mixed", v_mixed)
        object.__setattr__(self, "_n_same_reduced", n_same)
        object.__setattr__(self, "_n_mixed_reduced", n_mixed)

    def _build_gauge_basis(
        self,
        pairs: list[tuple[int, int]],
        diag_factor: bool,
    ) -> tuple[np.ndarray, int]:
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
                out[iff : iff + n_same_full] = self.v_same @ x_reduced[
                    ir : ir + self.n_same_reduced
                ]
            ir += self.n_same_reduced
            iff += n_same_full
            if self.n_mixed_reduced > 0:
                out[iff : iff + n_mixed_full] = self.v_mixed @ x_reduced[
                    ir : ir + self.n_mixed_reduced
                ]
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
                out[ir : ir + self.n_same_reduced] = self.v_same.T @ x_full[
                    iff : iff + n_same_full
                ]
            ir += self.n_same_reduced
            iff += n_same_full
            if self.n_mixed_reduced > 0:
                out[ir : ir + self.n_mixed_reduced] = self.v_mixed.T @ x_full[
                    iff : iff + n_mixed_full
                ]
            ir += self.n_mixed_reduced
            iff += n_mixed_full
        return out


def ov_final_param_dim(norb: int, nocc: int) -> int:
    return 2 * nocc * (norb - nocc)


def _ov_z_from_params(params: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    nvirt = norb - nocc
    ncomplex = nocc * nvirt
    xr = np.asarray(params[:ncomplex], dtype=float).reshape(nvirt, nocc)
    xi = np.asarray(params[ncomplex:], dtype=float).reshape(nvirt, nocc)
    return xr + 1j * xi


def ov_final_unitary(params: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    nvirt = norb - nocc
    eye = np.eye(norb, dtype=complex)
    if nocc == 0 or nvirt == 0:
        return eye
    z = _ov_z_from_params(params, norb, nocc)
    u_left, s, vh = np.linalg.svd(z, full_matrices=False)
    v_right = vh.conj().T
    cos_s = np.cos(s)
    sin_s = np.sin(s)
    occ_eye = np.eye(nocc, dtype=complex)
    virt_eye = np.eye(nvirt, dtype=complex)
    a = occ_eye + v_right @ (np.diag(cos_s - 1.0)) @ v_right.conj().T
    d = virt_eye + u_left @ (np.diag(cos_s - 1.0)) @ u_left.conj().T
    b = -(v_right @ np.diag(sin_s) @ u_left.conj().T)
    c = u_left @ np.diag(sin_s) @ v_right.conj().T
    out = np.zeros((norb, norb), dtype=complex)
    out[:nocc, :nocc] = a
    out[:nocc, nocc:] = b
    out[nocc:, :nocc] = c
    out[nocc:, nocc:] = d
    return out


def ov_params_from_unitary(unitary: np.ndarray, nocc: int) -> np.ndarray:
    norb = unitary.shape[0]
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return np.zeros(0, dtype=float)
    a = np.asarray(unitary[:nocc, :nocc], dtype=complex)
    c = np.asarray(unitary[nocc:, :nocc], dtype=complex)
    z = None
    try:
        m = np.linalg.solve(a.T, c.T).T
        u_left, s_tan, vh = np.linalg.svd(m, full_matrices=False)
        s = np.arctan(s_tan)
        z = u_left @ np.diag(s) @ vh
    except np.linalg.LinAlgError:
        z = None
    if z is None:
        u_left, s_sin, vh = np.linalg.svd(c, full_matrices=False)
        s = np.arcsin(np.clip(s_sin, -1.0, 1.0))
        z = u_left @ np.diag(s) @ vh
    return np.concatenate([np.real(z).reshape(-1), np.imag(z).reshape(-1)])


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
class GaugeFixedUCJSpinBalancedParameterization:
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
    def internal_orbital_gauge_map(self) -> _InternalOrbitalGaugeMap:
        return _InternalOrbitalGaugeMap(self.norb, self.n_layers)

    @property
    def n_internal_orbital_rotation_params(self) -> int:
        return self.internal_orbital_gauge_map.n_reduced

    @property
    def n_same_spin_params(self) -> int:
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self) -> int:
        return len(self.mixed_spin_indices)

    @property
    def n_final_orbital_rotation_params(self) -> int:
        return ov_final_param_dim(self.norb, self.nocc) if self.with_final_orbital_rotation else 0

    @property
    def n_params(self) -> int:
        return (
            self.n_internal_orbital_rotation_params
            + self.n_layers * (self.n_same_spin_params + self.n_mixed_spin_params)
            + self.n_final_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")

        orb_map = self.internal_orbital_gauge_map
        n_orb_red = orb_map.n_reduced
        x_orb_red = params[:n_orb_red]
        x_j = params[n_orb_red : n_orb_red + self.n_layers * (self.n_same_spin_params + self.n_mixed_spin_params)]
        x_final = params[n_orb_red + self.n_layers * (self.n_same_spin_params + self.n_mixed_spin_params) :]

        x_orb_full = orb_map.reduced_to_full(x_orb_red)

        i_orb = 0
        i_j = 0
        n_same = self.n_same_spin_params
        n_mixed = self.n_mixed_spin_params
        layers: list[UCJLayer] = []

        for _ in range(self.n_layers):
            u = _canonicalize_internal_unitary(
                ffsim_unitary_from_parameters(x_orb_full[i_orb : i_orb + self.norb**2], self.norb)
            )
            i_orb += self.norb**2

            j0 = _symmetric_matrix_from_values(
                x_j[i_j : i_j + n_same],
                self.norb,
                self.same_spin_indices,
            )
            i_j += n_same

            j1 = _symmetric_matrix_from_values(
                x_j[i_j : i_j + n_mixed],
                self.norb,
                self.mixed_spin_indices,
            )
            i_j += n_mixed

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
            final_orbital_rotation = ov_final_unitary(x_final, self.norb, self.nocc)

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

        orb_map = self.internal_orbital_gauge_map
        x_orb_full = np.zeros(orb_map.n_full, dtype=np.float64)
        x_j = np.zeros(self.n_layers * (self.n_same_spin_params + self.n_mixed_spin_params), dtype=np.float64)

        i_orb = 0
        i_j = 0
        n_same = self.n_same_spin_params
        n_mixed = self.n_mixed_spin_params

        for layer in ansatz.layers:
            d = cast(SpinBalancedSpec, layer.diagonal)

            x_orb_full[i_orb : i_orb + self.norb**2] = ffsim_unitary_to_parameters(
                _canonicalize_internal_unitary(layer.orbital_rotation)
            )
            i_orb += self.norb**2

            if n_same:
                x_j[i_j : i_j + n_same] = np.asarray(
                    [d.same_spin_params[p, q] for p, q in self.same_spin_indices],
                    dtype=np.float64,
                )
                i_j += n_same

            if n_mixed:
                x_j[i_j : i_j + n_mixed] = np.asarray(
                    [d.mixed_spin_params[p, q] for p, q in self.mixed_spin_indices],
                    dtype=np.float64,
                )
                i_j += n_mixed

        x_orb_red = orb_map.full_to_reduced(x_orb_full)

        if not self.with_final_orbital_rotation:
            return np.concatenate([x_orb_red, x_j])

        x_final = ov_params_from_unitary(ansatz.final_orbital_rotation, self.nocc)
        return np.concatenate([x_orb_red, x_j, x_final])

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func