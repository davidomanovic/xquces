from __future__ import annotations

import itertools
from dataclasses import InitVar, dataclass, field
from typing import cast

import numpy as np

from xquces.ucj.native import (
    UCJOpSpinBalanced,
    is_real_symmetric,
    is_unitary,
    validate_interaction_pairs,
    orbital_rotation_from_t1_amplitudes,
)


def _canonicalize_internal_unitary(u: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    u = np.array(u, dtype=complex, copy=True)
    norb = u.shape[0]
    phases = np.ones(norb, dtype=complex)
    for j in range(norb):
        col = u[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > tol:
            phases[j] = np.exp(-1j * np.angle(val))
    return u @ np.diag(phases)


class _GaugeReducedUCJMap:
    def __init__(self, norb: int, n_reps: int, interaction_pairs=None):
        self.norb = norb
        self.n_reps = n_reps
        if interaction_pairs is not None:
            self.pairs_aa, self.pairs_ab = interaction_pairs
        else:
            self.pairs_aa = [(p, q) for p in range(norb) for q in range(p + 1, norb)]
            self.pairs_ab = [(p, q) for p in range(norb) for q in range(p, norb)]
        self.n_aa = len(self.pairs_aa)
        self.n_ab = len(self.pairs_ab)
        self.n_orb_rot_full = norb * norb
        self.v_aa, self.n_indep_aa = self._build_gauge_basis(self.pairs_aa, diag_factor=False)
        self.v_ab, self.n_indep_ab = self._build_gauge_basis(self.pairs_ab, diag_factor=True)
        self.phase_indices = np.zeros(0, dtype=int)
        self.kept_indices = np.arange(self.n_orb_rot_full, dtype=int)
        self.n_orb_rot_reduced = self.n_orb_rot_full
        self.n_full_per_layer = self.n_orb_rot_full + self.n_aa + self.n_ab
        self.n_reduced_per_layer = self.n_orb_rot_reduced + self.n_indep_aa + self.n_indep_ab
        self.n_full = n_reps * self.n_full_per_layer
        self.n_reduced = n_reps * self.n_reduced_per_layer

    def _build_gauge_basis(self, pairs, diag_factor: bool):
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros((0, 0)), 0
        a = np.zeros((n_pairs, self.norb))
        for k, (p, q) in enumerate(pairs):
            if p == q and diag_factor:
                a[k, p] = 2.0
            else:
                a[k, p] = 1.0
                a[k, q] = 1.0
        u, s, _ = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > 1e-10))
        v_indep = u[:, rank:]
        return v_indep, v_indep.shape[1]

    def reduced_to_full(self, x_reduced: np.ndarray) -> np.ndarray:
        x_full = np.zeros(self.n_full, dtype=float)
        ir = 0
        iff = 0
        for _ in range(self.n_reps):
            x_full_orb = np.zeros(self.n_orb_rot_full)
            x_full_orb[self.kept_indices] = x_reduced[ir : ir + self.n_orb_rot_reduced]
            x_full[iff : iff + self.n_orb_rot_full] = x_full_orb
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full
            if self.n_indep_aa > 0:
                x_full[iff : iff + self.n_aa] = self.v_aa @ x_reduced[ir : ir + self.n_indep_aa]
            ir += self.n_indep_aa
            iff += self.n_aa
            if self.n_indep_ab > 0:
                x_full[iff : iff + self.n_ab] = self.v_ab @ x_reduced[ir : ir + self.n_indep_ab]
            ir += self.n_indep_ab
            iff += self.n_ab
        return x_full

    def full_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        x_reduced = np.empty(self.n_reduced, dtype=float)
        ir = 0
        iff = 0
        for _ in range(self.n_reps):
            x_reduced[ir : ir + self.n_orb_rot_reduced] = x_full[iff : iff + self.n_orb_rot_full][self.kept_indices]
            ir += self.n_orb_rot_reduced
            iff += self.n_orb_rot_full
            if self.n_indep_aa > 0:
                x_reduced[ir : ir + self.n_indep_aa] = self.v_aa.T @ x_full[iff : iff + self.n_aa]
            ir += self.n_indep_aa
            iff += self.n_aa
            if self.n_indep_ab > 0:
                x_reduced[ir : ir + self.n_indep_ab] = self.v_ab.T @ x_full[iff : iff + self.n_ab]
            ir += self.n_indep_ab
            iff += self.n_ab
        return x_reduced


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
class UCJOpGaugeFixed:
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    final_orbital_rotation: np.ndarray | None = None
    nocc: int | None = None
    _final_ov_params: np.ndarray | None = field(default=None, repr=False, compare=False)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if not validate:
            return
        if self.diag_coulomb_mats.ndim != 4 or self.diag_coulomb_mats.shape[1] != 2:
            raise ValueError(f"diag_coulomb_mats should have shape (n_reps, 2, norb, norb). Got shape {self.diag_coulomb_mats.shape}.")
        if self.orbital_rotations.ndim != 3:
            raise ValueError(f"orbital_rotations should have shape (n_reps, norb, norb). Got shape {self.orbital_rotations.shape}.")
        if self.final_orbital_rotation is not None and self.final_orbital_rotation.ndim != 2:
            raise ValueError("final_orbital_rotation should have shape (norb, norb).")
        norb = self.diag_coulomb_mats.shape[-1]
        if self.orbital_rotations.shape != (self.diag_coulomb_mats.shape[0], norb, norb):
            raise ValueError("orbital_rotations shape was inconsistent with diag_coulomb_mats.")
        if self.final_orbital_rotation is not None and self.final_orbital_rotation.shape != (norb, norb):
            raise ValueError("final_orbital_rotation shape was inconsistent with diag_coulomb_mats.")
        if not all(is_real_symmetric(mats[0], rtol=rtol, atol=atol) and is_real_symmetric(mats[1], rtol=rtol, atol=atol) for mats in self.diag_coulomb_mats):
            raise ValueError("Diagonal Coulomb matrices were not all real symmetric.")
        if not all(is_unitary(orbital_rotation, rtol=rtol, atol=atol) for orbital_rotation in self.orbital_rotations):
            raise ValueError("Orbital rotations were not all unitary.")
        if self.final_orbital_rotation is not None and not is_unitary(self.final_orbital_rotation, rtol=rtol, atol=atol):
            raise ValueError("Final orbital rotation was not unitary.")
        if self.final_orbital_rotation is not None and self.nocc is None:
            raise ValueError("nocc must be provided when final_orbital_rotation is present.")

    @property
    def norb(self) -> int:
        return self.diag_coulomb_mats.shape[-1]

    @property
    def n_reps(self) -> int:
        return self.diag_coulomb_mats.shape[0]

    @staticmethod
    def n_params(
        norb: int,
        n_reps: int,
        *,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        with_final_orbital_rotation: bool = False,
        nocc: int | None = None,
    ) -> int:
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False)
        validate_interaction_pairs(pairs_ab, ordered=False)
        triu_indices = cast(list[tuple[int, int]], list(itertools.combinations_with_replacement(range(norb), 2)))
        if pairs_aa is None:
            pairs_aa = [(p, q) for p, q in triu_indices if p < q]
        if pairs_ab is None:
            pairs_ab = triu_indices
        gf = _GaugeReducedUCJMap(norb=norb, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab))
        n = gf.n_reduced
        if with_final_orbital_rotation:
            if nocc is None:
                raise ValueError("nocc must be provided when with_final_orbital_rotation is True.")
            n += ov_final_param_dim(norb, nocc)
        return n

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        with_final_orbital_rotation: bool = False,
        nocc: int | None = None,
    ) -> "UCJOpGaugeFixed":
        params = np.asarray(params, dtype=float)
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False)
        validate_interaction_pairs(pairs_ab, ordered=False)
        triu_indices = cast(list[tuple[int, int]], list(itertools.combinations_with_replacement(range(norb), 2)))
        if pairs_aa is None:
            pairs_aa = [(p, q) for p, q in triu_indices if p < q]
        if pairs_ab is None:
            pairs_ab = triu_indices
        gf = _GaugeReducedUCJMap(norb=norb, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab))
        n_ucj = gf.n_reduced
        n_final = 0
        if with_final_orbital_rotation:
            if nocc is None:
                raise ValueError("nocc must be provided when with_final_orbital_rotation is True.")
            n_final = ov_final_param_dim(norb, nocc)
        n_expected = n_ucj + n_final
        if len(params) != n_expected:
            raise ValueError("The number of parameters passed did not match the number expected based on the function inputs. " f"Expected {n_expected} but got {len(params)}.")
        x_ucj = params[:n_ucj]
        x_final = params[n_ucj:]
        x_full = gf.reduced_to_full(x_ucj)
        stock = UCJOpSpinBalanced.from_parameters(
            x_full,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=False,
        )
        orbital_rotations = np.array(stock.orbital_rotations, copy=True)
        final_orbital_rotation = None
        final_params = None
        if with_final_orbital_rotation:
            final_params = np.array(x_final, dtype=float, copy=True)
            final_orbital_rotation = ov_final_unitary(final_params, norb, cast(int, nocc))
        return UCJOpGaugeFixed(
            diag_coulomb_mats=stock.diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
            nocc=nocc,
            _final_ov_params=final_params,
        )

    def to_parameters(
        self,
        *,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
    ) -> np.ndarray:
        norb = self.norb
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False)
        validate_interaction_pairs(pairs_ab, ordered=False)
        triu_indices = cast(list[tuple[int, int]], list(itertools.combinations_with_replacement(range(norb), 2)))
        if pairs_aa is None:
            pairs_aa = [(p, q) for p, q in triu_indices if p < q]
        if pairs_ab is None:
            pairs_ab = triu_indices
        gf = _GaugeReducedUCJMap(norb=norb, n_reps=self.n_reps, interaction_pairs=(pairs_aa, pairs_ab))
        stock = UCJOpSpinBalanced(
            diag_coulomb_mats=np.array(self.diag_coulomb_mats, copy=True),
            orbital_rotations=np.array(self.orbital_rotations, copy=True),
            final_orbital_rotation=None,
        )
        x_full = stock.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        x_ucj = gf.full_to_reduced(x_full)
        if self.final_orbital_rotation is None:
            return x_ucj
        if self.nocc is None:
            raise ValueError("nocc must be present when final_orbital_rotation is present.")
        if self._final_ov_params is not None:
            x_final = np.array(self._final_ov_params, dtype=float, copy=True)
        else:
            x_final = ov_params_from_unitary(self.final_orbital_rotation, self.nocc)
        return np.concatenate([x_ucj, x_final])

    @staticmethod
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback=None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> "UCJOpGaugeFixed":
        stock = UCJOpSpinBalanced.from_t_amplitudes(
            np.asarray(t2, dtype=float),
            t1=None,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            tol=tol,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )
        orbital_rotations = np.array(stock.orbital_rotations, copy=True)
        final_orbital_rotation = None
        nocc = None
        final_params = None
        if t1 is not None:
            t1 = np.asarray(t1)
            nocc = t1.shape[0]
            final_orbital_rotation = orbital_rotation_from_t1_amplitudes(t1)
            final_params = ov_params_from_unitary(final_orbital_rotation, nocc)
        return UCJOpGaugeFixed(
            diag_coulomb_mats=stock.diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
            nocc=nocc,
            _final_ov_params=final_params,
        )

    def as_stock_ucj(self) -> UCJOpSpinBalanced:
        return UCJOpSpinBalanced(
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
            final_orbital_rotation=self.final_orbital_rotation,
        )

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        return self.as_stock_ucj().apply(vec, nelec=nelec, copy=copy)

    def _apply_unitary_(self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool) -> np.ndarray:
        return self.as_stock_ucj()._apply_unitary_(vec, norb=norb, nelec=nelec, copy=copy)

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if not isinstance(other, UCJOpGaugeFixed):
            return NotImplemented
        if not np.allclose(self.diag_coulomb_mats, other.diag_coulomb_mats, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.orbital_rotations, other.orbital_rotations, rtol=rtol, atol=atol):
            return False
        if self.final_orbital_rotation is None and other.final_orbital_rotation is None:
            return True
        if (self.final_orbital_rotation is None) != (other.final_orbital_rotation is None):
            return False
        return np.allclose(cast(np.ndarray, self.final_orbital_rotation), cast(np.ndarray, other.final_orbital_rotation), rtol=rtol, atol=atol)


@dataclass(frozen=True)
class GaugeFixedUCJSpinBalancedParameterizationExact:
    norb: int
    nocc: int
    n_reps: int
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None
    with_final_orbital_rotation: bool = True

    @property
    def n_params(self) -> int:
        return UCJOpGaugeFixed.n_params(
            self.norb,
            self.n_reps,
            interaction_pairs=self.interaction_pairs,
            with_final_orbital_rotation=self.with_final_orbital_rotation,
            nocc=self.nocc,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> UCJOpGaugeFixed:
        return UCJOpGaugeFixed.from_parameters(
            params,
            norb=self.norb,
            n_reps=self.n_reps,
            interaction_pairs=self.interaction_pairs,
            with_final_orbital_rotation=self.with_final_orbital_rotation,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: UCJOpGaugeFixed) -> np.ndarray:
        return ansatz.to_parameters(interaction_pairs=self.interaction_pairs)


@dataclass(frozen=True)
class GaugeFixedUCJBalancedDFSeedExact:
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

    def build_ansatz(self) -> UCJOpGaugeFixed:
        ansatz, _, _ = self.build_parameters()
        return ansatz

    def build_parameters(self) -> tuple[UCJOpGaugeFixed, GaugeFixedUCJSpinBalancedParameterizationExact, np.ndarray]:
        stock = UCJOpSpinBalanced.from_t_amplitudes(
            np.asarray(self.t2, dtype=float),
            t1=None if self.t1 is None else np.asarray(self.t1),
            n_reps=self.n_reps,
            interaction_pairs=None,
            tol=self.tol,
            optimize=self.optimize,
            method=self.method,
            callback=self.callback,
            options=self.options,
            regularization=self.regularization,
            multi_stage_start=self.multi_stage_start,
            multi_stage_step=self.multi_stage_step,
        )
        nocc = np.asarray(self.t2).shape[0]
        param = GaugeFixedUCJSpinBalancedParameterizationExact(
            norb=stock.norb,
            nocc=nocc,
            n_reps=stock.n_reps,
            interaction_pairs=None,
            with_final_orbital_rotation=stock.final_orbital_rotation is not None,
        )
        seed_ansatz = UCJOpGaugeFixed(
            diag_coulomb_mats=np.array(stock.diag_coulomb_mats, copy=True),
            orbital_rotations=np.array(stock.orbital_rotations, copy=True),
            final_orbital_rotation=None if stock.final_orbital_rotation is None else np.array(stock.final_orbital_rotation, copy=True),
            nocc=nocc if stock.final_orbital_rotation is not None else None,
            _final_ov_params=None if stock.final_orbital_rotation is None else ov_params_from_unitary(stock.final_orbital_rotation, nocc),
        )
        x0 = param.parameters_from_ansatz(seed_ansatz)
        ansatz_roundtrip = param.ansatz_from_parameters(x0)
        return ansatz_roundtrip, param, x0