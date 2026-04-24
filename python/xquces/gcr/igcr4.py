from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gcr.model import GCRAnsatz
from xquces._lib import apply_igcr4_spin_restricted_in_place_num_rep
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    IGCR2SpinRestrictedSpec,
    _assert_square_matrix,
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _left_right_ov_adapted_to_native,
    _native_to_left_right_ov_adapted,
    _orbital_relabeling_unitary,
    _restricted_irreducible_pair_matrix,
    _restricted_left_phase_vector,
    _right_unitary_from_left_and_final,
    _symmetric_matrix_from_values,
    _validate_pairs,
    orbital_relabeling_from_overlap,
    relabel_igcr2_ansatz_orbitals,
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
    _values_from_ordered_matrix,
    _validate_ordered_pairs,
    _validate_triples,
    relabel_igcr3_ansatz_orbitals,
    spin_restricted_triples_seed_from_pair_params,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz


def _default_eta_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _default_rho_indices(norb: int) -> list[tuple[int, int, int]]:
    out = []
    for p in range(norb):
        for q in range(norb):
            if q == p:
                continue
            for r in range(q + 1, norb):
                if r == p:
                    continue
                out.append((p, q, r))
    return out


def _default_sigma_indices(norb: int) -> list[tuple[int, int, int, int]]:
    return list(itertools.combinations(range(norb), 4))


def _validate_rho_indices(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int]]:
    if triples is None:
        return _default_rho_indices(norb)
    out = []
    seen = set()
    for p, q, r in triples:
        if not (0 <= p < norb and 0 <= q < norb and 0 <= r < norb):
            raise ValueError("rho indices out of bounds")
        if p == q or p == r or q == r:
            raise ValueError("rho indices must be distinct")
        if q >= r:
            raise ValueError("rho indices must satisfy q < r")
        if (p, q, r) in seen:
            raise ValueError("rho indices must not contain duplicates")
        seen.add((p, q, r))
        out.append((p, q, r))
    return out


def _validate_sigma_indices(
    quads: list[tuple[int, int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int, int]]:
    if quads is None:
        return _default_sigma_indices(norb)
    out = []
    seen = set()
    for p, q, r, s in quads:
        if not (0 <= p < q < r < s < norb):
            raise ValueError("sigma indices must satisfy 0 <= p < q < r < s < norb")
        if (p, q, r, s) in seen:
            raise ValueError("sigma indices must not contain duplicates")
        seen.add((p, q, r, s))
        out.append((p, q, r, s))
    return out


@dataclass(frozen=True)
class IGCR4QuarticReduction:
    norb: int
    nocc: int

    def __post_init__(self):
        if self.norb < 0:
            raise ValueError("norb must be nonnegative")
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")

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

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def n_quartic_full(self):
        return len(self.eta_indices) + len(self.rho_indices) + len(self.sigma_indices)

    @property
    def gauge_cubic_matrix(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[0]

    @property
    def gauge_quartic_matrix(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[1]

    @property
    def physical_quartic_basis(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[2]

    @property
    def n_params(self):
        return self.physical_quartic_basis.shape[1]

    def full_from_reduced(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.physical_quartic_basis @ params

    def reduce_full(
        self,
        cubic_values: np.ndarray,
        quartic_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        quartic_values = np.asarray(quartic_values, dtype=np.float64)
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}.")
        if quartic_values.shape != (self.n_quartic_full,):
            raise ValueError(f"Expected quartic shape {(self.n_quartic_full,)}, got {quartic_values.shape}.")

        basis = self.physical_quartic_basis
        reduced = basis.T @ quartic_values
        residual = quartic_values - basis @ reduced
        gauge_coeff, *_ = np.linalg.lstsq(
            self.gauge_quartic_matrix,
            residual,
            rcond=None,
        )
        cubic_reduced = cubic_values - self.gauge_cubic_matrix @ gauge_coeff
        return cubic_reduced, reduced


@cache
def _igcr4_quartic_reduction_matrices(norb: int, nocc: int):
    tau_indices = _default_tau_indices(norb)
    omega_indices = _default_triple_indices(norb)
    eta_indices = _default_eta_indices(norb)
    rho_indices = _default_rho_indices(norb)
    sigma_indices = _default_sigma_indices(norb)

    n_tau = len(tau_indices)
    n_omega = len(omega_indices)
    n_eta = len(eta_indices)
    n_rho = len(rho_indices)
    n_sigma = len(sigma_indices)
    nelec_total = 2 * int(nocc)

    n_id_tau = n_tau
    n_id_omega = n_omega
    n_id = n_id_tau + n_id_omega

    tau_index = {pair: i for i, pair in enumerate(tau_indices)}
    omega_index = {triple: i for i, triple in enumerate(omega_indices)}
    eta_index = {pair: i for i, pair in enumerate(eta_indices)}
    rho_index = {triple: i for i, triple in enumerate(rho_indices)}
    sigma_index = {quad: i for i, quad in enumerate(sigma_indices)}

    gauge_cubic = np.zeros((n_tau + n_omega, n_id), dtype=np.float64)
    gauge_quartic = np.zeros((n_eta + n_rho + n_sigma, n_id), dtype=np.float64)

    for col, (p, q) in enumerate(tau_indices):
        gauge_cubic[tau_index[(p, q)], col] -= nelec_total - 3
        gauge_quartic[eta_index[(p, q) if p < q else (q, p)], col] += 2.0
        for r in range(norb):
            if r == p or r == q:
                continue
            a, b = (q, r) if q < r else (r, q)
            gauge_quartic[n_eta + rho_index[(p, a, b)], col] += 1.0

    for local_col, (p, q, r) in enumerate(omega_indices):
        col = n_id_tau + local_col
        gauge_cubic[n_tau + omega_index[(p, q, r)], col] -= nelec_total - 3
        gauge_quartic[n_eta + rho_index[(p, q, r)], col] += 2.0
        gauge_quartic[n_eta + rho_index[(q, p, r) if p < r else (q, r, p)], col] += 2.0
        gauge_quartic[n_eta + rho_index[(r, p, q) if p < q else (r, q, p)], col] += 2.0
        for s in range(norb):
            if s == p or s == q or s == r:
                continue
            quad = tuple(sorted((p, q, r, s)))
            gauge_quartic[n_eta + n_rho + sigma_index[quad], col] += 1.0

    if gauge_quartic.size == 0:
        physical = np.zeros((0, 0), dtype=np.float64)
        return gauge_cubic, gauge_quartic, physical

    u, s, _ = np.linalg.svd(gauge_quartic, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > max(gauge_quartic.shape) * np.finfo(float).eps * s[0]))
    physical = np.array(u[:, rank:], copy=True, dtype=np.float64)
    for j in range(physical.shape[1]):
        col = physical[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            physical[:, j] *= -1.0
    return gauge_cubic, gauge_quartic, physical


def spin_restricted_quartic_seed_from_pair_params(
    pair_params: np.ndarray,
    nocc: int,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    _assert_square_matrix(pair, "pair_params")
    norb = pair.shape[0]
    nelec_total = 2 * int(nocc)
    denom = max(nelec_total - 3, 1)

    eta = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
    if eta_scale != 0.0:
        for k, (p, q) in enumerate(_default_eta_indices(norb)):
            eta[k] = float(eta_scale) * 0.5 * (pair[p, q]) / denom

    rho = np.zeros(len(_default_rho_indices(norb)), dtype=np.float64)
    if rho_scale != 0.0:
        for k, (p, q, r) in enumerate(_default_rho_indices(norb)):
            rho[k] = float(rho_scale) * (pair[p, q] + pair[p, r] + pair[q, r]) / (3.0 * denom)

    sigma = np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64)
    if sigma_scale != 0.0:
        for k, (p, q, r, s) in enumerate(_default_sigma_indices(norb)):
            avg = (
                pair[p, q] + pair[p, r] + pair[p, s] +
                pair[q, r] + pair[q, s] + pair[r, s]
            ) / 6.0
            sigma[k] = float(sigma_scale) * avg / denom

    return eta, rho, sigma


@dataclass(frozen=True)
class IGCR4SpinRestrictedSpec:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma_values: np.ndarray

    @property
    def norb(self) -> int:
        return int(np.asarray(self.double_params, dtype=np.float64).shape[0])

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

    def full_double(self) -> np.ndarray:
        double = np.asarray(self.double_params, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        return double

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(
            np.asarray(self.pair_values, dtype=np.float64),
            self.norb,
            self.pair_indices,
        )

    def tau_matrix(self) -> np.ndarray:
        tau = np.asarray(self.tau, dtype=np.float64)
        if tau.shape != (self.norb, self.norb):
            raise ValueError("tau must have shape (norb, norb)")
        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)
        return tau

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

    def phase_from_occupations(
        self,
        occ_alpha: np.ndarray,
        occ_beta: np.ndarray,
    ) -> float:
        n = np.zeros(self.norb, dtype=np.float64)
        n[np.asarray(occ_alpha, dtype=np.int64)] += 1.0
        n[np.asarray(occ_beta, dtype=np.int64)] += 1.0
        d = np.zeros(self.norb, dtype=np.float64)
        d[np.intersect1d(occ_alpha, occ_beta, assume_unique=True)] = 1.0
        return self.phase_from_number_arrays(n, d)

    def phase_from_number_arrays(self, n: np.ndarray, d: np.ndarray) -> float:
        n = np.asarray(n, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        if n.shape != (self.norb,) or d.shape != (self.norb,):
            raise ValueError("n and d must have shape (norb,)")

        phase = float(np.dot(self.full_double(), d))
        pair = self.pair_matrix()
        for p, q in self.pair_indices:
            phase += pair[p, q] * n[p] * n[q]
        tau = self.tau_matrix()
        for p, q in self.tau_indices:
            phase += tau[p, q] * d[p] * n[q]
        omega = self.omega_vector()
        for value, (p, q, r) in zip(omega, self.omega_indices):
            phase += value * n[p] * n[q] * n[r]
        eta = self.eta_vector()
        for value, (p, q) in zip(eta, self.eta_indices):
            phase += value * d[p] * d[q]
        rho = self.rho_vector()
        for value, (p, q, r) in zip(rho, self.rho_indices):
            phase += value * d[p] * n[q] * n[r]
        sigma = self.sigma_vector()
        for value, (p, q, r, s) in zip(sigma, self.sigma_indices):
            phase += value * n[p] * n[q] * n[r] * n[s]
        return float(phase)

    def to_igcr3_diagonal(self) -> IGCR3SpinRestrictedSpec:
        return IGCR3SpinRestrictedSpec(
            double_params=self.full_double(),
            pair_values=self.pair_values,
            tau=self.tau,
            omega_values=self.omega_values,
        )

    def to_igcr2_diagonal(self) -> IGCR2SpinRestrictedSpec:
        return reduce_spin_restricted(
            SpinRestrictedSpec(
                double_params=self.full_double(),
                pair_params=self.pair_matrix(),
            )
        )

    @classmethod
    def from_igcr3_diagonal(
        cls,
        diagonal: IGCR3SpinRestrictedSpec,
        *,
        eta_values: np.ndarray | None = None,
        rho_values: np.ndarray | None = None,
        sigma_values: np.ndarray | None = None,
    ) -> "IGCR4SpinRestrictedSpec":
        norb = diagonal.norb
        if eta_values is None:
            eta_values = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
        if rho_values is None:
            rho_values = np.zeros(len(_default_rho_indices(norb)), dtype=np.float64)
        if sigma_values is None:
            sigma_values = np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64)
        return cls(
            double_params=np.asarray(diagonal.full_double(), dtype=np.float64),
            pair_values=np.asarray(diagonal.pair_values, dtype=np.float64),
            tau=np.asarray(diagonal.tau, dtype=np.float64),
            omega_values=np.asarray(diagonal.omega_values, dtype=np.float64),
            eta_values=np.asarray(eta_values, dtype=np.float64),
            rho_values=np.asarray(rho_values, dtype=np.float64),
            sigma_values=np.asarray(sigma_values, dtype=np.float64),
        )


def apply_igcr4_spin_restricted_diagonal(
    vec: np.ndarray,
    diagonal: IGCR4SpinRestrictedSpec,
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
    apply_igcr4_spin_restricted_in_place_num_rep(
        state2,
        np.asarray(diagonal.full_double(), dtype=np.float64) * time,
        np.asarray(diagonal.pair_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.tau_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.omega_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.eta_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.rho_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.sigma_vector(), dtype=np.float64) * time,
        norb,
        occ_alpha,
        occ_beta,
    )
    return flatten_state(state2)


@dataclass(frozen=True)
class IGCR4Ansatz:
    diagonal: IGCR4SpinRestrictedSpec
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
        arr = apply_igcr4_spin_restricted_diagonal(
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

    def to_igcr3_ansatz(self) -> IGCR3Ansatz:
        if np.linalg.norm(self.diagonal.eta_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero eta sector to iGCR-3")
        if np.linalg.norm(self.diagonal.rho_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero rho sector to iGCR-3")
        if np.linalg.norm(self.diagonal.sigma_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero sigma sector to iGCR-3")
        return IGCR3Ansatz(
            diagonal=self.diagonal.to_igcr3_diagonal(),
            left=np.asarray(self.left, dtype=np.complex128),
            right=np.asarray(self.right, dtype=np.complex128),
            nocc=self.nocc,
        )

    @classmethod
    def from_igcr3_ansatz(
        cls,
        ansatz: IGCR3Ansatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        d3 = ansatz.diagonal
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
            d3.pair_matrix(),
            ansatz.nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )
        diagonal = IGCR4SpinRestrictedSpec.from_igcr3_diagonal(
            d3,
            eta_values=eta,
            rho_values=rho,
            sigma_values=sigma,
        )
        return cls(
            diagonal=diagonal,
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_igcr2_ansatz(
        cls,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_igcr3_ansatz(
            IGCR3Ansatz.from_igcr2_ansatz(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            ),
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_ucj_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_ucj(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_ucj_ansatz(
            ansatz,
            nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_gcr_ansatz(
        cls,
        ansatz: GCRAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        tau_scale = kwargs.pop("tau_scale", 0.0)
        omega_scale = kwargs.pop("omega_scale", 0.0)
        eta_scale = kwargs.pop("eta_scale", 0.0)
        rho_scale = kwargs.pop("rho_scale", 0.0)
        sigma_scale = kwargs.pop("sigma_scale", 0.0)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(
            ucj,
            nocc=t2.shape[0],
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )


def relabel_igcr4_ansatz_orbitals(
    ansatz: IGCR4Ansatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR4Ansatz:
    if ansatz.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)

    d = ansatz.diagonal
    double = d.full_double()[old_for_new]
    pair = d.pair_matrix()[np.ix_(old_for_new, old_for_new)]
    tau = d.tau_matrix()[np.ix_(old_for_new, old_for_new)]

    pair_values = np.asarray(
        [pair[p, q] for p, q in _default_pair_indices(ansatz.norb)],
        dtype=np.float64,
    )

    omega_old = {idx: val for idx, val in zip(d.omega_indices, d.omega_vector())}
    omega_values = np.asarray(
        [
            omega_old[tuple(sorted((int(old_for_new[p]), int(old_for_new[q]), int(old_for_new[r]))))]
            for p, q, r in _default_triple_indices(ansatz.norb)
        ],
        dtype=np.float64,
    )

    eta_old = {idx: val for idx, val in zip(d.eta_indices, d.eta_vector())}
    eta_values = np.asarray(
        [
            eta_old[(int(old_for_new[p]), int(old_for_new[q])) if old_for_new[p] < old_for_new[q] else (int(old_for_new[q]), int(old_for_new[p]))]
            for p, q in _default_eta_indices(ansatz.norb)
        ],
        dtype=np.float64,
    )

    rho_old = {idx: val for idx, val in zip(d.rho_indices, d.rho_vector())}
    rho_values = np.asarray(
        [
            rho_old[
                (
                    int(old_for_new[p]),
                    min(int(old_for_new[q]), int(old_for_new[r])),
                    max(int(old_for_new[q]), int(old_for_new[r])),
                )
            ]
            for p, q, r in _default_rho_indices(ansatz.norb)
        ],
        dtype=np.float64,
    )

    sigma_old = {idx: val for idx, val in zip(d.sigma_indices, d.sigma_vector())}
    sigma_values = np.asarray(
        [
            sigma_old[tuple(sorted((int(old_for_new[p]), int(old_for_new[q]), int(old_for_new[r]), int(old_for_new[s]))))]
            for p, q, r, s in _default_sigma_indices(ansatz.norb)
        ],
        dtype=np.float64,
    )

    diagonal = IGCR4SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega_values,
        eta_values=eta_values,
        rho_values=rho_values,
        sigma_values=sigma_values,
    )

    return IGCR4Ansatz(
        diagonal=diagonal,
        left=relabel.conj().T @ ansatz.left @ relabel,
        right=relabel.conj().T @ ansatz.right @ relabel,
        nocc=ansatz.nocc,
    )


@dataclass(frozen=True)
class IGCR4SpinRestrictedParameterization:
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
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 3.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)
        _validate_rho_indices(self.rho_indices_, self.norb)
        _validate_sigma_indices(self.sigma_indices_, self.norb)
        if (
            self.left_right_ov_relative_scale is not None
            and (
                not np.isfinite(float(self.left_right_ov_relative_scale))
                or self.left_right_ov_relative_scale <= 0
            )
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

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
        return (
            self.reduce_cubic_gauge
            and self.pair_indices == _default_pair_indices(self.norb)
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
        )

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return (
            self.reduce_quartic_gauge
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
            and self.eta_indices == _default_eta_indices(self.norb)
            and self.rho_indices == _default_rho_indices(self.norb)
            and self.sigma_indices == _default_sigma_indices(self.norb)
        )

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
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_pair_params(self):
        return len(self.pair_indices)

    @property
    def n_tau_params(self):
        if self.uses_reduced_cubic_chart:
            return self.cubic_reduction.n_params
        return len(self.tau_indices)

    @property
    def n_omega_params(self):
        if self.uses_reduced_cubic_chart:
            return 0
        return len(self.omega_indices)

    @property
    def n_eta_params(self):
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.eta_indices)

    @property
    def n_rho_params(self):
        if self.uses_reduced_quartic_chart:
            return self.quartic_reduction.n_params
        return len(self.rho_indices)

    @property
    def n_sigma_params(self):
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.sigma_indices)

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
            + self.n_right_orbital_rotation_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "pair": self.n_pair_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "eta": 0 if self.uses_reduced_quartic_chart else self.n_eta_params,
            "rho": self.n_rho_params,
            "sigma": 0 if self.uses_reduced_quartic_chart else self.n_sigma_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4Ansatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n = self.n_pair_params
        pair_sparse_values = np.asarray(params[idx:idx + n], dtype=np.float64)
        pair_sparse = _symmetric_matrix_from_values(pair_sparse_values, self.norb, self.pair_indices)
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            cubic = self.cubic_reduction.full_from_reduced(params[idx:idx + n])
            n_tau_full = len(_default_tau_indices(self.norb))
            tau = _ordered_matrix_from_values(
                cubic[:n_tau_full],
                self.norb,
                _default_tau_indices(self.norb),
            )
            omega_values = np.asarray(cubic[n_tau_full:], dtype=np.float64)
            idx += n
        else:
            n = self.n_tau_params
            tau = _ordered_matrix_from_values(params[idx:idx + n], self.norb, self.tau_indices)
            idx += n

            n = self.n_omega_params
            omega_sparse_values = np.asarray(params[idx:idx + n], dtype=np.float64)
            omega_sparse = {triple: value for triple, value in zip(self.omega_indices, omega_sparse_values)}
            omega_values = np.asarray(
                [omega_sparse.get(triple, 0.0) for triple in _default_triple_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

        if self.uses_reduced_quartic_chart:
            n = self.n_rho_params
            quartic = self.quartic_reduction.full_from_reduced(params[idx:idx + n])
            n_eta_full = len(_default_eta_indices(self.norb))
            n_rho_full = len(_default_rho_indices(self.norb))
            eta_values = np.asarray(quartic[:n_eta_full], dtype=np.float64)
            rho_values = np.asarray(quartic[n_eta_full:n_eta_full + n_rho_full], dtype=np.float64)
            sigma_values = np.asarray(quartic[n_eta_full + n_rho_full:], dtype=np.float64)
            idx += n
        else:
            n = self.n_eta_params
            eta_sparse_values = np.asarray(params[idx:idx + n], dtype=np.float64)
            eta_sparse = {pair: value for pair, value in zip(self.eta_indices, eta_sparse_values)}
            eta_values = np.asarray(
                [eta_sparse.get(pair, 0.0) for pair in _default_eta_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

            n = self.n_rho_params
            rho_sparse_values = np.asarray(params[idx:idx + n], dtype=np.float64)
            rho_sparse = {triple: value for triple, value in zip(self.rho_indices, rho_sparse_values)}
            rho_values = np.asarray(
                [rho_sparse.get(triple, 0.0) for triple in _default_rho_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

            n = self.n_sigma_params
            sigma_sparse_values = np.asarray(params[idx:idx + n], dtype=np.float64)
            sigma_sparse = {quad: value for quad, value in zip(self.sigma_indices, sigma_sparse_values)}
            sigma_values = np.asarray(
                [sigma_sparse.get(quad, 0.0) for quad in _default_sigma_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)

        return IGCR4Ansatz(
            diagonal=IGCR4SpinRestrictedSpec(
                double_params=np.zeros(self.norb, dtype=np.float64),
                pair_values=pair_values,
                tau=tau,
                omega_values=omega_values,
                eta_values=eta_values,
                rho_values=rho_values,
                sigma_values=sigma_values,
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR4Ansatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")

        d = ansatz.diagonal
        pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
        tau = d.tau_matrix()
        omega = d.omega_vector()
        eta = d.eta_vector()
        rho = d.rho_vector()
        sigma = d.sigma_vector()

        full_pair_values = np.asarray(
            [pair_eff[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        full_cubic = np.concatenate(
            [
                _values_from_ordered_matrix(tau, _default_tau_indices(self.norb)),
                omega,
            ]
        )
        full_quartic = np.concatenate([eta, rho, sigma])

        if self.uses_reduced_quartic_chart:
            full_cubic, reduced_quartic_values = self.quartic_reduction.reduce_full(
                full_cubic,
                full_quartic,
            )
        else:
            reduced_quartic_values = None

        cubic_onebody_phase = np.zeros(self.norb, dtype=np.float64)
        if self.uses_reduced_cubic_chart:
            reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = self.cubic_reduction.reduce_full(
                full_pair_values,
                full_cubic,
            )
        else:
            reduced_pair_values = None
            reduced_cubic_values = None

        phase_vec = _restricted_left_phase_vector(d.full_double(), self.nocc) + cubic_onebody_phase
        left_eff = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(phase_vec)
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(left_eff)
        else:
            left_params = left_chart.parameters_from_unitary(left_eff)
            right_phase = np.zeros(self.norb, dtype=np.float64)

        right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = left_params
        idx += n

        n = self.n_pair_params
        if self.uses_reduced_cubic_chart:
            out[idx:idx + n] = reduced_pair_values
        else:
            out[idx:idx + n] = np.asarray([pair_eff[p, q] for p, q in self.pair_indices], dtype=np.float64)
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            out[idx:idx + n] = reduced_cubic_values
            idx += n
        else:
            n = self.n_tau_params
            out[idx:idx + n] = _values_from_ordered_matrix(tau, self.tau_indices)
            idx += n

            n = self.n_omega_params
            full_omega = {triple: value for value, triple in zip(omega, d.omega_indices)}
            out[idx:idx + n] = np.asarray([full_omega[t] for t in self.omega_indices], dtype=np.float64)
            idx += n

        if self.uses_reduced_quartic_chart:
            n = self.n_rho_params
            out[idx:idx + n] = reduced_quartic_values
            idx += n
        else:
            n = self.n_eta_params
            full_eta = {pair: value for value, pair in zip(eta, d.eta_indices)}
            out[idx:idx + n] = np.asarray([full_eta[t] for t in self.eta_indices], dtype=np.float64)
            idx += n

            n = self.n_rho_params
            full_rho = {triple: value for value, triple in zip(rho, d.rho_indices)}
            out[idx:idx + n] = np.asarray([full_rho[t] for t in self.rho_indices], dtype=np.float64)
            idx += n

            n = self.n_sigma_params
            full_sigma = {quad: value for value, quad in zip(sigma, d.sigma_indices)}
            out[idx:idx + n] = np.asarray([full_sigma[t] for t in self.sigma_indices], dtype=np.float64)
            idx += n

        n = self.n_right_orbital_rotation_params
        left_param_unitary = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        out[idx:idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)

        return self._public_parameters_from_native(out)

    def parameters_from_igcr3_ansatz(
        self,
        ansatz: IGCR3Ansatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_igcr3_ansatz(
                ansatz,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_igcr2_ansatz(
        self,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_igcr2_ansatz(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_ucj_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_gcr_ansatz(
        self,
        ansatz: GCRAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_gcr_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR4SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError("previous ansatz nocc does not match this parameterization")
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
            old_for_new, phases = orbital_relabeling_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
        if old_for_new is not None:
            if isinstance(ansatz, IGCR4Ansatz):
                ansatz = relabel_igcr4_ansatz_orbitals(ansatz, old_for_new, phases)
            elif isinstance(ansatz, IGCR3Ansatz):
                ansatz = relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases)
            elif isinstance(ansatz, IGCR2Ansatz):
                ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
            else:
                raise TypeError(f"Unsupported ansatz type for transfer: {type(ansatz)!r}")
        if isinstance(ansatz, IGCR4Ansatz):
            return self.parameters_from_ansatz(ansatz)
        if isinstance(ansatz, IGCR3Ansatz):
            return self.parameters_from_igcr3_ansatz(ansatz)
        if isinstance(ansatz, IGCR2Ansatz):
            return self.parameters_from_igcr2_ansatz(ansatz)
        raise TypeError(f"Unsupported ansatz type for transfer: {type(ansatz)!r}")

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func


def igcr4_from_igcr3_ansatz(
    ansatz: IGCR3Ansatz,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz:
    return IGCR4Ansatz.from_igcr3_ansatz(
        ansatz,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )


def igcr4_from_igcr2_ansatz(
    ansatz: IGCR2Ansatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz:
    return IGCR4Ansatz.from_igcr2_ansatz(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )
