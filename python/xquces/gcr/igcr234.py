from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.basis import flatten_state, occ_indicator_rows, occ_rows, reshape_state, sector_shape
from xquces.gates import apply_igcr2_spin_restricted
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    _symmetric_matrix_from_values,
    _validate_pairs,
    _zero_diag_antihermitian_from_parameters,
    reduce_spin_restricted,
)
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3CubicReduction,
    IGCR3SpinRestrictedSpec,
    _default_pair_indices,
    _default_tau_indices,
    _default_triple_indices,
    _ordered_matrix_from_values,
    _validate_ordered_pairs,
    _validate_triples,
    apply_igcr3_spin_restricted_diagonal,
    spin_restricted_triples_seed_from_pair_params,
)
from xquces.gcr.igcr4 import (
    IGCR4Ansatz,
    IGCR4QuarticReduction,
    IGCR4SpinRestrictedSpec,
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _validate_rho_indices,
    _validate_sigma_indices,
    apply_igcr4_spin_restricted_diagonal,
    spin_restricted_quartic_seed_from_pair_params,
)
from xquces.gcr.model import GCRAnsatz
from xquces.orbitals import apply_orbital_rotation, ov_generator_from_params
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz


def _infer_norb_from_unique_pair_count(count: int) -> int:
    n = 0
    while n * (n - 1) // 2 < count:
        n += 1
    if n * (n - 1) // 2 != count:
        raise ValueError("pair_values has inconsistent length")
    return n


def _pair_values_from_matrix(pair: np.ndarray) -> np.ndarray:
    pair = np.asarray(pair, dtype=np.float64)
    norb = pair.shape[0]
    return np.asarray([pair[p, q] for p, q in _default_pair_indices(norb)], dtype=np.float64)


def _tau_values_from_matrix(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    norb = tau.shape[0]
    return np.asarray([tau[p, q] for p, q in _default_tau_indices(norb)], dtype=np.float64)


def _ordered_matrix_from_sparse_values(values: np.ndarray, norb: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((norb, norb), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),):
        raise ValueError(f"Expected {(len(pairs),)}, got {values.shape}.")
    for value, (p, q) in zip(values, pairs):
        out[p, q] = value
    np.fill_diagonal(out, 0.0)
    return out


def _full_sparse_dict(values: np.ndarray, indices) -> dict:
    return {idx: float(val) for idx, val in zip(indices, np.asarray(values, dtype=np.float64))}


def _vector_from_sparse_dict(default_indices, sparse_dict: dict) -> np.ndarray:
    return np.asarray([sparse_dict.get(idx, 0.0) for idx in default_indices], dtype=np.float64)


def _irreducible_pair_from_diagonal(diagonal: SpinRestrictedSpec) -> np.ndarray:
    if not isinstance(diagonal, SpinRestrictedSpec):
        raise TypeError("expected a spin-restricted diagonal")
    return reduce_spin_restricted(diagonal).pair


@cache
def _number_arrays(norb: int, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    occ_a = occ_indicator_rows(norb, nelec[0]).astype(np.float64)
    occ_b = occ_indicator_rows(norb, nelec[1]).astype(np.float64)
    n = occ_a[:, None, :] + occ_b[None, :, :]
    d = occ_a[:, None, :] * occ_b[None, :, :]
    return n.reshape(-1, norb), d.reshape(-1, norb)


@cache
def _pair_feature_matrix(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    n, _ = _number_arrays(norb, nelec)
    cols = [n[:, p] * n[:, q] for p, q in _default_pair_indices(norb)]
    return np.column_stack(cols) if cols else np.zeros((n.shape[0], 0), dtype=np.float64)


@cache
def _cubic_feature_matrix(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    n, d = _number_arrays(norb, nelec)
    cols = [d[:, p] * n[:, q] for p, q in _default_tau_indices(norb)]
    cols.extend(n[:, p] * n[:, q] * n[:, r] for p, q, r in _default_triple_indices(norb))
    return np.column_stack(cols) if cols else np.zeros((n.shape[0], 0), dtype=np.float64)


@cache
def _quartic_feature_matrix(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    n, d = _number_arrays(norb, nelec)
    cols = [d[:, p] * d[:, q] for p, q in _default_eta_indices(norb)]
    cols.extend(d[:, p] * n[:, q] * n[:, r] for p, q, r in _default_rho_indices(norb))
    cols.extend(n[:, p] * n[:, q] * n[:, r] * n[:, s] for p, q, r, s in _default_sigma_indices(norb))
    return np.column_stack(cols) if cols else np.zeros((n.shape[0], 0), dtype=np.float64)


@cache
def _spinless_one_body_transitions(norb: int, nocc: int):
    occs = [tuple(int(x) for x in row) for row in occ_rows(norb, nocc)]
    index = {occ: i for i, occ in enumerate(occs)}
    trans = {}
    for p in range(norb):
        for q in range(norb):
            rows = []
            cols = []
            signs = []
            for j, occ in enumerate(occs):
                occ_set = set(occ)
                if q not in occ_set:
                    continue
                if p == q:
                    rows.append(j)
                    cols.append(j)
                    signs.append(1.0)
                    continue
                if p in occ_set:
                    continue
                between = 0
                lo = min(p, q)
                hi = max(p, q)
                for orb in occ:
                    if lo < orb < hi:
                        between += 1
                sign = -1.0 if between % 2 else 1.0
                new_occ = tuple(sorted((p if orb == q else orb) for orb in occ))
                rows.append(index[new_occ])
                cols.append(j)
                signs.append(sign)
            trans[(p, q)] = (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
                np.asarray(signs, dtype=np.complex128),
            )
    return trans


def _apply_spinless_one_body_generator(state: np.ndarray, x: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    dim = len(occ_rows(norb, nocc))
    vec = np.asarray(state, dtype=np.complex128).reshape(dim)
    out = np.zeros_like(vec)
    trans = _spinless_one_body_transitions(norb, nocc)
    for p in range(norb):
        for q in range(norb):
            coeff = x[p, q]
            if abs(coeff) < 1e-15:
                continue
            rows, cols, signs = trans[(p, q)]
            if len(rows):
                out[rows] += coeff * signs * vec[cols]
    return out


def _apply_one_body_generator_to_state(vec: np.ndarray, x: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    mat = np.asarray(vec, dtype=np.complex128).reshape(dim_a, dim_b)
    out = np.zeros_like(mat)
    xa = _apply_spinless_one_body_generator
    for j in range(dim_b):
        out[:, j] += xa(mat[:, j], x, norb, nelec[0])
    for i in range(dim_a):
        out[i, :] += xa(mat[i, :], x, norb, nelec[1])
    return flatten_state(out)


@cache
def _left_chart_basis_matrices(norb: int) -> tuple[np.ndarray, ...]:
    mats = []
    for p, q in _default_pair_indices(norb):
        m = np.zeros((norb, norb), dtype=np.complex128)
        m[p, q] = 1.0
        m[q, p] = -1.0
        mats.append(m)
        m = np.zeros((norb, norb), dtype=np.complex128)
        m[p, q] = 1.0j
        m[q, p] = 1.0j
        mats.append(m)
    return tuple(mats)


@cache
def _ov_chart_basis_matrices(norb: int, nocc: int, real_only: bool) -> tuple[np.ndarray, ...]:
    nvirt = norb - nocc
    mats = []
    for a in range(nvirt):
        p = nocc + a
        for i in range(nocc):
            m = np.zeros((norb, norb), dtype=np.complex128)
            m[p, i] = 1.0
            m[i, p] = -1.0
            mats.append(m)
    if not real_only:
        for a in range(nvirt):
            p = nocc + a
            for i in range(nocc):
                m = np.zeros((norb, norb), dtype=np.complex128)
                m[p, i] = 1.0j
                m[i, p] = 1.0j
                mats.append(m)
    return tuple(mats)


def _chart_unitary_and_tangents(chart, params: np.ndarray, norb: int, nocc: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    params = np.asarray(params, dtype=np.float64)
    if isinstance(chart, IGCR2LeftUnitaryChart):
        kappa = _zero_diag_antihermitian_from_parameters(params, norb)
        u = chart.unitary_from_parameters(params, norb)
        basis = _left_chart_basis_matrices(norb)
    elif isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        if nocc is None:
            raise ValueError("nocc is required for OV chart tangents")
        kappa = ov_generator_from_params(params, norb, nocc)
        u = chart.unitary_from_parameters(params, norb)
        basis = _ov_chart_basis_matrices(norb, nocc, False)
    elif isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        if nocc is None:
            raise ValueError("nocc is required for OV chart tangents")
        full = np.concatenate([params, np.zeros_like(params)])
        kappa = ov_generator_from_params(full, norb, nocc)
        u = chart.unitary_from_parameters(params, norb)
        basis = _ov_chart_basis_matrices(norb, nocc, True)
    else:
        raise NotImplementedError("analytic Jacobian is implemented only for IGCR2 left and OV charts")
    u_dag = u.conj().T
    tangents = []
    for b in basis:
        du = scipy.linalg.expm_frechet(kappa, b, compute_expm=False)
        tangents.append(du @ u_dag)
    return u, tangents


@dataclass(frozen=True)
class IGCR234SpinRestrictedSpec:
    pair_values: np.ndarray
    tau_values: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma_values: np.ndarray

    @property
    def norb(self) -> int:
        return _infer_norb_from_unique_pair_count(len(np.asarray(self.pair_values, dtype=np.float64)))

    @property
    def pair_indices(self):
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self):
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self):
        return _default_triple_indices(self.norb)

    @property
    def eta_indices(self):
        return _default_eta_indices(self.norb)

    @property
    def rho_indices(self):
        return _default_rho_indices(self.norb)

    @property
    def sigma_indices(self):
        return _default_sigma_indices(self.norb)

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(np.asarray(self.pair_values, dtype=np.float64), self.norb, self.pair_indices)

    def tau_vector(self) -> np.ndarray:
        tau = np.asarray(self.tau_values, dtype=np.float64)
        if tau.shape != (len(self.tau_indices),):
            raise ValueError("tau_values has inconsistent shape")
        return tau

    def tau_matrix(self) -> np.ndarray:
        return _ordered_matrix_from_values(self.tau_vector(), self.norb, self.tau_indices)

    def omega_vector(self) -> np.ndarray:
        omega = np.asarray(self.omega_values, dtype=np.float64)
        if omega.shape != (len(self.omega_indices),):
            raise ValueError("omega_values has inconsistent shape")
        return omega

    def eta_vector(self) -> np.ndarray:
        eta = np.asarray(self.eta_values, dtype=np.float64)
        if eta.shape != (len(self.eta_indices),):
            raise ValueError("eta_values has inconsistent shape")
        return eta

    def rho_vector(self) -> np.ndarray:
        rho = np.asarray(self.rho_values, dtype=np.float64)
        if rho.shape != (len(self.rho_indices),):
            raise ValueError("rho_values has inconsistent shape")
        return rho

    def sigma_vector(self) -> np.ndarray:
        sigma = np.asarray(self.sigma_values, dtype=np.float64)
        if sigma.shape != (len(self.sigma_indices),):
            raise ValueError("sigma_values has inconsistent shape")
        return sigma

    def t_diagonal(self) -> IGCR3SpinRestrictedSpec:
        return IGCR3SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=np.zeros(len(self.pair_indices), dtype=np.float64),
            tau=self.tau_matrix(),
            omega_values=self.omega_vector(),
        )

    def q_diagonal(self) -> IGCR4SpinRestrictedSpec:
        return IGCR4SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=np.zeros(len(self.pair_indices), dtype=np.float64),
            tau=np.zeros((self.norb, self.norb), dtype=np.float64),
            omega_values=np.zeros(len(self.omega_indices), dtype=np.float64),
            eta_values=self.eta_vector(),
            rho_values=self.rho_vector(),
            sigma_values=self.sigma_vector(),
        )


@dataclass(frozen=True)
class IGCR234Ansatz:
    diagonal: IGCR234SpinRestrictedSpec
    u1: np.ndarray
    u2: np.ndarray
    u3: np.ndarray
    u4: np.ndarray
    nocc: int

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(arr, self.u4, norb=self.norb, nelec=nelec, copy=False)
        arr = apply_igcr4_spin_restricted_diagonal(arr, self.diagonal.q_diagonal(), self.norb, nelec, copy=False)
        arr = apply_orbital_rotation(arr, self.u3, norb=self.norb, nelec=nelec, copy=False)
        arr = apply_igcr3_spin_restricted_diagonal(arr, self.diagonal.t_diagonal(), self.norb, nelec, copy=False)
        arr = apply_igcr2_spin_restricted(arr, self.diagonal.pair_matrix(), self.norb, nelec, left_orbital_rotation=self.u1, right_orbital_rotation=self.u2, copy=False)
        return arr

    @classmethod
    def from_igcr2_ansatz(cls, ansatz: IGCR2Ansatz, *, tau_scale: float = 0.0, omega_scale: float = 0.0, eta_scale: float = 0.0, rho_scale: float = 0.0, sigma_scale: float = 0.0) -> "IGCR234Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR234 is currently implemented only for spin-restricted seeds")
        pair = ansatz.diagonal.to_standard().pair_params
        tau, omega = spin_restricted_triples_seed_from_pair_params(pair, ansatz.nocc, tau_scale=tau_scale, omega_scale=omega_scale)
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(pair, ansatz.nocc, eta_scale=eta_scale, rho_scale=rho_scale, sigma_scale=sigma_scale)
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair),
            tau_values=_tau_values_from_matrix(tau),
            omega_values=np.asarray(omega, dtype=np.float64),
            eta_values=np.asarray(eta, dtype=np.float64),
            rho_values=np.asarray(rho, dtype=np.float64),
            sigma_values=np.asarray(sigma, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(diagonal=diagonal, u1=np.asarray(ansatz.left, dtype=np.complex128), u2=identity, u3=identity, u4=np.asarray(ansatz.right, dtype=np.complex128), nocc=ansatz.nocc)

    @classmethod
    def from_ucj_ansatz(cls, ansatz: UCJAnsatz, nocc: int, *, tau_scale: float = 1.0, omega_scale: float = 1.0, eta_scale: float = 1.0, rho_scale: float = 1.0, sigma_scale: float = 1.0) -> "IGCR234Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR234 is currently implemented only for spin-restricted seeds")
        if ansatz.n_layers == 1:
            return cls.from_igcr2_ansatz(IGCR2Ansatz.from_ucj_ansatz(ansatz, nocc=nocc), tau_scale=tau_scale, omega_scale=omega_scale, eta_scale=eta_scale, rho_scale=rho_scale, sigma_scale=sigma_scale)
        if ansatz.n_layers != 3:
            raise ValueError("iGCR234 UCJ seeding currently expects either 1 or 3 layers")
        layer_q = ansatz.layers[0]
        layer_t = ansatz.layers[1]
        layer_j = ansatz.layers[2]
        final = ansatz.final_orbital_rotation
        if final is None:
            final = np.eye(ansatz.norb, dtype=np.complex128)
        pair_j = _irreducible_pair_from_diagonal(layer_j.diagonal)
        pair_t = _irreducible_pair_from_diagonal(layer_t.diagonal)
        pair_q = _irreducible_pair_from_diagonal(layer_q.diagonal)
        tau, omega = spin_restricted_triples_seed_from_pair_params(pair_t, nocc, tau_scale=tau_scale, omega_scale=omega_scale)
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(pair_q, nocc, eta_scale=eta_scale, rho_scale=rho_scale, sigma_scale=sigma_scale)
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair_j),
            tau_values=_tau_values_from_matrix(tau),
            omega_values=np.asarray(omega, dtype=np.float64),
            eta_values=np.asarray(eta, dtype=np.float64),
            rho_values=np.asarray(rho, dtype=np.float64),
            sigma_values=np.asarray(sigma, dtype=np.float64),
        )
        return cls(
            diagonal=diagonal,
            u1=np.asarray(final @ layer_j.orbital_rotation, dtype=np.complex128),
            u2=np.asarray(layer_j.orbital_rotation.conj().T @ layer_t.orbital_rotation, dtype=np.complex128),
            u3=np.asarray(layer_t.orbital_rotation.conj().T @ layer_q.orbital_rotation, dtype=np.complex128),
            u4=np.asarray(layer_q.orbital_rotation.conj().T, dtype=np.complex128),
            nocc=nocc,
        )

    @classmethod
    def from_ucj(cls, ansatz: UCJAnsatz, nocc: int, **kwargs) -> "IGCR234Ansatz":
        return cls.from_ucj_ansatz(ansatz, nocc, **kwargs)

    @classmethod
    def from_igcr3_ansatz(cls, ansatz: IGCR3Ansatz, *, eta_scale: float = 0.0, rho_scale: float = 0.0, sigma_scale: float = 0.0) -> "IGCR234Ansatz":
        pair = ansatz.diagonal.pair_matrix()
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(pair, ansatz.nocc, eta_scale=eta_scale, rho_scale=rho_scale, sigma_scale=sigma_scale)
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair),
            tau_values=_tau_values_from_matrix(ansatz.diagonal.tau_matrix()),
            omega_values=np.asarray(ansatz.diagonal.omega_vector(), dtype=np.float64),
            eta_values=np.asarray(eta, dtype=np.float64),
            rho_values=np.asarray(rho, dtype=np.float64),
            sigma_values=np.asarray(sigma, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(diagonal=diagonal, u1=np.asarray(ansatz.left, dtype=np.complex128), u2=identity, u3=identity, u4=np.asarray(ansatz.right, dtype=np.complex128), nocc=ansatz.nocc)

    @classmethod
    def from_igcr4_ansatz(cls, ansatz: IGCR4Ansatz) -> "IGCR234Ansatz":
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=np.asarray(ansatz.diagonal.pair_values, dtype=np.float64),
            tau_values=_tau_values_from_matrix(ansatz.diagonal.tau_matrix()),
            omega_values=np.asarray(ansatz.diagonal.omega_values, dtype=np.float64),
            eta_values=np.asarray(ansatz.diagonal.eta_values, dtype=np.float64),
            rho_values=np.asarray(ansatz.diagonal.rho_values, dtype=np.float64),
            sigma_values=np.asarray(ansatz.diagonal.sigma_values, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(diagonal=diagonal, u1=np.asarray(ansatz.left, dtype=np.complex128), u2=identity, u3=identity, u4=np.asarray(ansatz.right, dtype=np.complex128), nocc=ansatz.nocc)

    @classmethod
    def from_gcr_ansatz(cls, ansatz: GCRAnsatz, nocc: int, **kwargs) -> "IGCR234Ansatz":
        return cls.from_igcr2_ansatz(IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc), **kwargs)

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        n_reps = kwargs.pop("n_reps", 3)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, n_reps=n_reps, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(ucj, nocc=t2.shape[0])


@dataclass(frozen=True)
class IGCR234SpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    reduce_quartic_gauge: bool = True
    general_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)
        _validate_rho_indices(self.rho_indices_, self.norb)
        _validate_sigma_indices(self.sigma_indices_, self.norb)

    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def tau_indices(self):
        return _validate_ordered_pairs(self.tau_indices_, self.norb)

    @property
    def omega_indices(self):
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def eta_indices(self):
        return _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)

    @property
    def rho_indices(self):
        return _validate_rho_indices(self.rho_indices_, self.norb)

    @property
    def sigma_indices(self):
        return _validate_sigma_indices(self.sigma_indices_, self.norb)

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return self.reduce_cubic_gauge and self.tau_indices == _default_tau_indices(self.norb) and self.omega_indices == _default_triple_indices(self.norb)

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return self.reduce_quartic_gauge and self.eta_indices == _default_eta_indices(self.norb) and self.rho_indices == _default_rho_indices(self.norb) and self.sigma_indices == _default_sigma_indices(self.norb)

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return IGCR3CubicReduction(self.norb, self.nocc)

    @property
    def quartic_reduction(self) -> IGCR4QuarticReduction:
        return IGCR4QuarticReduction(self.norb, self.nocc)

    @property
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def n_u1_params(self):
        return self.general_orbital_chart.n_params(self.norb)

    @property
    def n_pair_params(self):
        return len(self.pair_indices)

    @property
    def n_u2_params(self):
        return self.general_orbital_chart.n_params(self.norb)

    @property
    def n_tau_params(self):
        return self.cubic_reduction.n_params if self.uses_reduced_cubic_chart else len(self.tau_indices)

    @property
    def n_omega_params(self):
        return 0 if self.uses_reduced_cubic_chart else len(self.omega_indices)

    @property
    def n_u3_params(self):
        return self.general_orbital_chart.n_params(self.norb)

    @property
    def n_eta_params(self):
        return self.quartic_reduction.n_params if self.uses_reduced_quartic_chart else len(self.eta_indices)

    @property
    def n_rho_params(self):
        return 0 if self.uses_reduced_quartic_chart else len(self.rho_indices)

    @property
    def n_sigma_params(self):
        return 0 if self.uses_reduced_quartic_chart else len(self.sigma_indices)

    @property
    def n_u4_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        return self.n_u1_params + self.n_pair_params + self.n_u2_params + self.n_tau_params + self.n_omega_params + self.n_u3_params + self.n_eta_params + self.n_rho_params + self.n_sigma_params + self.n_u4_params

    def sector_sizes(self) -> dict[str, int]:
        return {
            "u1": self.n_u1_params,
            "j": self.n_pair_params,
            "u2": self.n_u2_params,
            "cubic": self.n_tau_params if self.uses_reduced_cubic_chart else self.n_tau_params + self.n_omega_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "u3": self.n_u3_params,
            "quartic": self.n_eta_params if self.uses_reduced_quartic_chart else self.n_eta_params + self.n_rho_params + self.n_sigma_params,
            "eta": 0 if self.uses_reduced_quartic_chart else self.n_eta_params,
            "rho": self.n_rho_params,
            "sigma": self.n_sigma_params,
            "u4": self.n_u4_params,
            "total": self.n_params,
        }

    def _split_public_params(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        p = {}
        n = self.n_u1_params
        p["u1"] = params[idx: idx + n]
        idx += n
        n = self.n_pair_params
        p["j"] = params[idx: idx + n]
        idx += n
        n = self.n_u2_params
        p["u2"] = params[idx: idx + n]
        idx += n
        n = self.n_tau_params
        p["tau"] = params[idx: idx + n]
        idx += n
        n = self.n_omega_params
        p["omega"] = params[idx: idx + n]
        idx += n
        n = self.n_u3_params
        p["u3"] = params[idx: idx + n]
        idx += n
        n = self.n_eta_params
        p["eta"] = params[idx: idx + n]
        idx += n
        n = self.n_rho_params
        p["rho"] = params[idx: idx + n]
        idx += n
        n = self.n_sigma_params
        p["sigma"] = params[idx: idx + n]
        idx += n
        n = self.n_u4_params
        p["u4"] = params[idx: idx + n]
        return p

    def _full_diagonal_vectors_from_public(self, params: np.ndarray):
        pieces = self._split_public_params(params)
        pair_sparse = _symmetric_matrix_from_values(np.asarray(pieces["j"], dtype=np.float64), self.norb, self.pair_indices)
        pair_values = _pair_values_from_matrix(pair_sparse)
        if self.uses_reduced_cubic_chart:
            cubic = self.cubic_reduction.full_from_reduced(pieces["tau"])
            n_tau_full = len(_default_tau_indices(self.norb))
            tau_values = np.asarray(cubic[:n_tau_full], dtype=np.float64)
            omega_values = np.asarray(cubic[n_tau_full:], dtype=np.float64)
        else:
            tau_values = _vector_from_sparse_dict(_default_tau_indices(self.norb), _full_sparse_dict(pieces["tau"], self.tau_indices))
            omega_values = _vector_from_sparse_dict(_default_triple_indices(self.norb), _full_sparse_dict(pieces["omega"], self.omega_indices))
        if self.uses_reduced_quartic_chart:
            quartic = self.quartic_reduction.full_from_reduced(pieces["eta"])
            n_eta = len(_default_eta_indices(self.norb))
            n_rho = len(_default_rho_indices(self.norb))
            eta_values = np.asarray(quartic[:n_eta], dtype=np.float64)
            rho_values = np.asarray(quartic[n_eta:n_eta + n_rho], dtype=np.float64)
            sigma_values = np.asarray(quartic[n_eta + n_rho:], dtype=np.float64)
        else:
            eta_values = _vector_from_sparse_dict(_default_eta_indices(self.norb), _full_sparse_dict(pieces["eta"], self.eta_indices))
            rho_values = _vector_from_sparse_dict(_default_rho_indices(self.norb), _full_sparse_dict(pieces["rho"], self.rho_indices))
            sigma_values = _vector_from_sparse_dict(_default_sigma_indices(self.norb), _full_sparse_dict(pieces["sigma"], self.sigma_indices))
        return pieces, pair_values, tau_values, omega_values, eta_values, rho_values, sigma_values

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR234Ansatz:
        pieces, pair_values, tau_values, omega_values, eta_values, rho_values, sigma_values = self._full_diagonal_vectors_from_public(params)
        u1 = self.general_orbital_chart.unitary_from_parameters(pieces["u1"], self.norb)
        u2 = self.general_orbital_chart.unitary_from_parameters(pieces["u2"], self.norb)
        u3 = self.general_orbital_chart.unitary_from_parameters(pieces["u3"], self.norb)
        u4 = self.right_orbital_chart.unitary_from_parameters(pieces["u4"], self.norb)
        return IGCR234Ansatz(
            diagonal=IGCR234SpinRestrictedSpec(pair_values=pair_values, tau_values=tau_values, omega_values=omega_values, eta_values=eta_values, rho_values=rho_values, sigma_values=sigma_values),
            u1=u1,
            u2=u2,
            u3=u3,
            u4=u4,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR234Ansatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        d = ansatz.diagonal
        pair = d.pair_matrix()
        tau = d.tau_vector()
        omega = d.omega_vector()
        eta = d.eta_vector()
        rho = d.rho_vector()
        sigma = d.sigma_vector()
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_u1_params
        out[idx: idx + n] = self.general_orbital_chart.parameters_from_unitary(ansatz.u1)
        idx += n
        n = self.n_pair_params
        out[idx: idx + n] = np.asarray([pair[p, q] for p, q in self.pair_indices], dtype=np.float64)
        idx += n
        n = self.n_u2_params
        out[idx: idx + n] = self.general_orbital_chart.parameters_from_unitary(ansatz.u2)
        idx += n
        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            reduced = self.cubic_reduction.reduce_full(np.zeros(len(_default_pair_indices(self.norb)), dtype=np.float64), np.concatenate([tau, omega]))[1]
            out[idx: idx + n] = reduced
            idx += n
        else:
            n = self.n_tau_params
            tau_mat = d.tau_matrix()
            out[idx: idx + n] = np.asarray([tau_mat[p, q] for p, q in self.tau_indices], dtype=np.float64)
            idx += n
            n = self.n_omega_params
            omega_dict = {ix: val for ix, val in zip(d.omega_indices, omega)}
            out[idx: idx + n] = np.asarray([omega_dict.get(ix, 0.0) for ix in self.omega_indices], dtype=np.float64)
            idx += n
        n = self.n_u3_params
        out[idx: idx + n] = self.general_orbital_chart.parameters_from_unitary(ansatz.u3)
        idx += n
        if self.uses_reduced_quartic_chart:
            n = self.n_eta_params
            reduced = self.quartic_reduction.reduce_full(np.concatenate([tau, omega]), np.concatenate([eta, rho, sigma]))[1]
            out[idx: idx + n] = reduced
            idx += n
        else:
            n = self.n_eta_params
            eta_dict = {ix: val for ix, val in zip(d.eta_indices, eta)}
            out[idx: idx + n] = np.asarray([eta_dict.get(ix, 0.0) for ix in self.eta_indices], dtype=np.float64)
            idx += n
            n = self.n_rho_params
            rho_dict = {ix: val for ix, val in zip(d.rho_indices, rho)}
            out[idx: idx + n] = np.asarray([rho_dict.get(ix, 0.0) for ix in self.rho_indices], dtype=np.float64)
            idx += n
            n = self.n_sigma_params
            sigma_dict = {ix: val for ix, val in zip(d.sigma_indices, sigma)}
            out[idx: idx + n] = np.asarray([sigma_dict.get(ix, 0.0) for ix in self.sigma_indices], dtype=np.float64)
            idx += n
        n = self.n_u4_params
        out[idx: idx + n] = self.right_orbital_chart.parameters_from_unitary(ansatz.u4)
        return out

    def parameters_from_igcr2_ansatz(self, ansatz: IGCR2Ansatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR234Ansatz.from_igcr2_ansatz(ansatz, **kwargs))

    def parameters_from_igcr3_ansatz(self, ansatz: IGCR3Ansatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR234Ansatz.from_igcr3_ansatz(ansatz, **kwargs))

    def parameters_from_igcr4_ansatz(self, ansatz: IGCR4Ansatz) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR234Ansatz.from_igcr4_ansatz(ansatz))

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR234Ansatz.from_ucj_ansatz(ansatz, self.nocc, **kwargs))

    def parameters_from_gcr_ansatz(self, ansatz: GCRAnsatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR234Ansatz.from_gcr_ansatz(ansatz, self.nocc, **kwargs))

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)
        return func

    def params_to_vec_jacobian(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        nelec = tuple(int(x) for x in nelec)
        pair_features_full = _pair_feature_matrix(self.norb, nelec)
        cubic_features_full = _cubic_feature_matrix(self.norb, nelec)
        quartic_features_full = _quartic_feature_matrix(self.norb, nelec)
        pair_cols = [_default_pair_indices(self.norb).index(ix) for ix in self.pair_indices]
        if self.uses_reduced_cubic_chart:
            cubic_features = cubic_features_full @ self.cubic_reduction.physical_cubic_basis
        else:
            tau_all = _default_tau_indices(self.norb)
            omega_all = _default_triple_indices(self.norb)
            tau_cols = [tau_all.index(ix) for ix in self.tau_indices]
            omega_cols = [omega_all.index(ix) for ix in self.omega_indices]
            cubic_features = np.concatenate([cubic_features_full[:, tau_cols], cubic_features_full[:, len(tau_all) + np.asarray(omega_cols, dtype=int)]], axis=1)
        if self.uses_reduced_quartic_chart:
            quartic_features = quartic_features_full @ self.quartic_reduction.physical_quartic_basis
        else:
            eta_all = _default_eta_indices(self.norb)
            rho_all = _default_rho_indices(self.norb)
            sigma_all = _default_sigma_indices(self.norb)
            eta_cols = [eta_all.index(ix) for ix in self.eta_indices]
            rho_cols = [rho_all.index(ix) for ix in self.rho_indices]
            sigma_cols = [sigma_all.index(ix) for ix in self.sigma_indices]
            quartic_features = np.concatenate([
                quartic_features_full[:, eta_cols],
                quartic_features_full[:, len(eta_all) + np.asarray(rho_cols, dtype=int)],
                quartic_features_full[:, len(eta_all) + len(rho_all) + np.asarray(sigma_cols, dtype=int)],
            ], axis=1)
        def jac(params: np.ndarray) -> np.ndarray:
            pieces, pair_values, tau_values, omega_values, eta_values, rho_values, sigma_values = self._full_diagonal_vectors_from_public(params)
            u1, x1s = _chart_unitary_and_tangents(self.general_orbital_chart, pieces["u1"], self.norb, self.nocc)
            u2, x2s = _chart_unitary_and_tangents(self.general_orbital_chart, pieces["u2"], self.norb, self.nocc)
            u3, x3s = _chart_unitary_and_tangents(self.general_orbital_chart, pieces["u3"], self.norb, self.nocc)
            u4, x4s = _chart_unitary_and_tangents(self.right_orbital_chart, pieces["u4"], self.norb, self.nocc)
            diagonal = IGCR234SpinRestrictedSpec(pair_values=pair_values, tau_values=tau_values, omega_values=omega_values, eta_values=eta_values, rho_values=rho_values, sigma_values=sigma_values)
            tdiag = diagonal.t_diagonal()
            qdiag = diagonal.q_diagonal()
            psi4 = apply_orbital_rotation(reference_vec, u4, norb=self.norb, nelec=nelec, copy=True)
            psiq = apply_igcr4_spin_restricted_diagonal(psi4, qdiag, self.norb, nelec, copy=True)
            psi3 = apply_orbital_rotation(psiq, u3, norb=self.norb, nelec=nelec, copy=True)
            psit = apply_igcr3_spin_restricted_diagonal(psi3, tdiag, self.norb, nelec, copy=True)
            psi2 = apply_orbital_rotation(psit, u2, norb=self.norb, nelec=nelec, copy=True)
            psij = apply_igcr2_spin_restricted(psi2, diagonal.pair_matrix(), self.norb, nelec, copy=True)
            psif = apply_orbital_rotation(psij, u1, norb=self.norb, nelec=nelec, copy=True)
            def after_u1(v):
                return apply_orbital_rotation(v, u1, norb=self.norb, nelec=nelec, copy=True)
            def after_j(v):
                return after_u1(apply_igcr2_spin_restricted(v, diagonal.pair_matrix(), self.norb, nelec, copy=True))
            def after_u2(v):
                return after_j(apply_orbital_rotation(v, u2, norb=self.norb, nelec=nelec, copy=True))
            def after_t(v):
                return after_u2(apply_igcr3_spin_restricted_diagonal(v, tdiag, self.norb, nelec, copy=True))
            def after_u3(v):
                return after_t(apply_orbital_rotation(v, u3, norb=self.norb, nelec=nelec, copy=True))
            def after_q(v):
                return after_u3(apply_igcr4_spin_restricted_diagonal(v, qdiag, self.norb, nelec, copy=True))
            cols = []
            for x in x1s:
                cols.append(_apply_one_body_generator_to_state(psif, x, self.norb, nelec))
            for k in pair_cols:
                cols.append(after_u1(1j * pair_features_full[:, k] * psij))
            for x in x2s:
                cols.append(after_j(_apply_one_body_generator_to_state(psi2, x, self.norb, nelec)))
            for k in range(cubic_features.shape[1]):
                cols.append(after_u2(1j * cubic_features[:, k] * psit))
            for x in x3s:
                cols.append(after_t(_apply_one_body_generator_to_state(psi3, x, self.norb, nelec)))
            for k in range(quartic_features.shape[1]):
                cols.append(after_u3(1j * quartic_features[:, k] * psiq))
            for x in x4s:
                cols.append(after_q(_apply_one_body_generator_to_state(psi4, x, self.norb, nelec)))
            return np.column_stack(cols)
        return jac
