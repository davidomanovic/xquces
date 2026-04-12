from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import flatten_state, occ_rows, reshape_state
from xquces.gcr.model import GCRAnsatz
from xquces._lib import apply_igcr3_spin_restricted_in_place_num_rep
from xquces.igcr2 import (
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
    reduce_spin_restricted,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz


def _default_tau_indices(norb: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(norb) for q in range(norb) if p != q]


def _default_triple_indices(norb: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(norb), 3))


def _default_pair_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _validate_ordered_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_tau_indices(norb)
    out = []
    seen = set()
    for p, q in pairs:
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("ordered-pair index out of bounds")
        if p == q:
            raise ValueError("ordered-pair diagonal entries are not allowed")
        if (p, q) in seen:
            raise ValueError("ordered pairs must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return out


def _validate_triples(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int]]:
    if triples is None:
        return _default_triple_indices(norb)
    out = []
    seen = set()
    for p, q, r in triples:
        if not (0 <= p < q < r < norb):
            raise ValueError("triple indices must satisfy 0 <= p < q < r < norb")
        if (p, q, r) in seen:
            raise ValueError("triple indices must not contain duplicates")
        seen.add((p, q, r))
        out.append((p, q, r))
    return out


def _ordered_matrix_from_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((norb, norb), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),):
        raise ValueError(f"Expected {(len(pairs),)}, got {values.shape}.")
    for value, (p, q) in zip(values, pairs):
        out[p, q] = value
    np.fill_diagonal(out, 0.0)
    return out


def _values_from_ordered_matrix(
    mat: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    return np.asarray([mat[p, q] for p, q in pairs], dtype=np.float64)


@dataclass(frozen=True)
class IGCR3CubicReduction:
    """Quotient the restricted cubic diagonal basis by fixed-N identities.

    The identities used here are, modulo constants and one-body N_p phases,

        sum_{q != p} D_p N_q
          + 1/2 (Ne - 2) sum_{q != p} N_p N_q == 0

    and, for p < q,

        sum_{r != p,q} N_p N_q N_r
          + 2 D_p N_q + 2 D_q N_p - (Ne - 2) N_p N_q == 0.

    They are exact on the fixed-(N_alpha, N_beta) sector and remove the
    persistent iGCR-3 cubic nullspace without changing the represented state.
    """

    norb: int
    nocc: int

    def __post_init__(self):
        if self.norb < 0:
            raise ValueError("norb must be nonnegative")
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")

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
    def n_pair_full(self):
        return len(self.pair_indices)

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def gauge_pair_matrix(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[0]

    @property
    def gauge_cubic_matrix(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[1]

    @property
    def physical_cubic_basis(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[2]

    @property
    def n_params(self):
        return self.physical_cubic_basis.shape[1]

    def full_from_reduced(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.physical_cubic_basis @ params

    def reduce_full(self, pair_values, cubic_values):
        pair_values = np.asarray(pair_values, dtype=np.float64)
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        if pair_values.shape != (self.n_pair_full,):
            raise ValueError(f"Expected pair shape {(self.n_pair_full,)}, got {pair_values.shape}.")
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}.")

        basis = self.physical_cubic_basis
        reduced = basis.T @ cubic_values
        residual = cubic_values - basis @ reduced
        gauge_coeff, *_ = np.linalg.lstsq(
            self.gauge_cubic_matrix,
            residual,
            rcond=None,
        )
        # old - new is an exact gauge vector, so the pair coefficients move
        # with the opposite sign of the pair part of that vector.
        pair_reduced = pair_values - self.gauge_pair_matrix @ gauge_coeff
        onebody_phase = np.zeros(self.norb, dtype=np.float64)
        if self.norb:
            nelec_total = 2 * int(self.nocc)
            onebody_phase[:] = (
                0.5
                * (nelec_total - 2)
                * (nelec_total - 1)
                * gauge_coeff[: self.norb]
            )
        return pair_reduced, reduced, onebody_phase


@cache
def _igcr3_cubic_reduction_matrices(norb: int, nocc: int):
    pair_indices = _default_pair_indices(norb)
    tau_indices = _default_tau_indices(norb)
    omega_indices = _default_triple_indices(norb)
    n_pair = len(pair_indices)
    n_tau = len(tau_indices)
    n_omega = len(omega_indices)
    n_identity = norb + n_pair
    nelec_total = 2 * int(nocc)

    pair_index = {pair: i for i, pair in enumerate(pair_indices)}
    tau_index = {pair: i for i, pair in enumerate(tau_indices)}
    omega_index = {triple: i for i, triple in enumerate(omega_indices)}

    gauge_pair = np.zeros((n_pair, n_identity), dtype=np.float64)
    gauge_cubic = np.zeros((n_tau + n_omega, n_identity), dtype=np.float64)

    for p in range(norb):
        col = p
        for q in range(norb):
            if p == q:
                continue
            pair = (p, q) if p < q else (q, p)
            gauge_pair[pair_index[pair], col] += 0.5 * (nelec_total - 2)
            gauge_cubic[tau_index[(p, q)], col] += 1.0

    for k, (p, q) in enumerate(pair_indices):
        col = norb + k
        gauge_pair[pair_index[(p, q)], col] -= nelec_total - 2
        gauge_cubic[tau_index[(p, q)], col] += 2.0
        gauge_cubic[tau_index[(q, p)], col] += 2.0
        for r in range(norb):
            if r == p or r == q:
                continue
            triple = tuple(sorted((p, q, r)))
            gauge_cubic[n_tau + omega_index[triple], col] += 1.0

    if gauge_cubic.size == 0:
        physical = np.zeros((0, 0), dtype=np.float64)
        return gauge_pair, gauge_cubic, physical

    u, s, _ = np.linalg.svd(gauge_cubic, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > max(gauge_cubic.shape) * np.finfo(float).eps * s[0]))
    physical = np.array(u[:, rank:], copy=True, dtype=np.float64)
    for j in range(physical.shape[1]):
        col = physical[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            physical[:, j] *= -1.0
    return gauge_pair, gauge_cubic, physical


def spin_restricted_triples_seed_from_pair_params(
    pair_params: np.ndarray,
    nocc: int,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    _assert_square_matrix(pair, "pair_params")
    norb = pair.shape[0]
    nelec_total = 2 * int(nocc)
    denom = max(nelec_total - 2, 1)

    tau = np.zeros((norb, norb), dtype=np.float64)
    if tau_scale != 0.0:
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    tau[p, q] = float(tau_scale) * pair[p, q] / denom

    omega = np.zeros(len(_default_triple_indices(norb)), dtype=np.float64)
    if omega_scale != 0.0:
        for k, (p, q, r) in enumerate(_default_triple_indices(norb)):
            avg_pair = (pair[p, q] + pair[p, r] + pair[q, r]) / 3.0
            omega[k] = float(omega_scale) * avg_pair / denom
    return tau, omega


@dataclass(frozen=True)
class IGCR3SpinRestrictedSpec:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray

    @property
    def norb(self) -> int:
        return int(np.asarray(self.double_params, dtype=np.float64).shape[0])

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _default_triple_indices(self.norb)

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
        return float(phase)

    def to_igcr2_diagonal(self) -> IGCR2SpinRestrictedSpec:
        return reduce_spin_restricted(
            SpinRestrictedSpec(
                double_params=self.full_double(),
                pair_params=self.pair_matrix(),
            )
        )

    @classmethod
    def from_igcr2_diagonal(
        cls,
        diagonal: IGCR2SpinRestrictedSpec,
        *,
        tau: np.ndarray | None = None,
        omega_values: np.ndarray | None = None,
    ) -> "IGCR3SpinRestrictedSpec":
        double = diagonal.full_double()
        norb = double.shape[0]
        pair = diagonal.to_standard().pair_params
        pair_values = np.asarray(
            [pair[p, q] for p, q in _default_pair_indices(norb)],
            dtype=np.float64,
        )
        if tau is None:
            tau = np.zeros((norb, norb), dtype=np.float64)
        if omega_values is None:
            omega_values = np.zeros(len(_default_triple_indices(norb)), dtype=np.float64)
        return cls(
            double_params=double,
            pair_values=pair_values,
            tau=np.asarray(tau, dtype=np.float64),
            omega_values=np.asarray(omega_values, dtype=np.float64),
        )


def apply_igcr3_spin_restricted_diagonal(
    vec: np.ndarray,
    diagonal: IGCR3SpinRestrictedSpec,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    state2 = reshape_state(arr, norb, nelec)
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    double = diagonal.full_double()
    pair = diagonal.pair_matrix()
    tau = diagonal.tau_matrix()
    omega = diagonal.omega_vector()
    apply_igcr3_spin_restricted_in_place_num_rep(
        state2,
        np.exp(1j * time * double),
        np.exp(1j * time * pair),
        np.exp(1j * time * tau),
        np.exp(1j * time * omega),
        norb,
        occ_alpha,
        occ_beta,
    )
    return flatten_state(state2)


@dataclass(frozen=True)
class IGCR3Ansatz:
    diagonal: IGCR3SpinRestrictedSpec
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
        arr = apply_igcr3_spin_restricted_diagonal(
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

    def to_igcr2_ansatz(self) -> IGCR2Ansatz:
        if np.linalg.norm(self.diagonal.tau_matrix()) > 1e-14:
            raise ValueError("cannot convert nonzero tau sector to iGCR-2")
        if np.linalg.norm(self.diagonal.omega_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero omega sector to iGCR-2")
        return IGCR2Ansatz(
            diagonal=self.diagonal.to_igcr2_diagonal(),
            left=np.asarray(self.left, dtype=np.complex128),
            right=np.asarray(self.right, dtype=np.complex128),
            nocc=self.nocc,
        )

    @classmethod
    def from_igcr2_ansatz(
        cls,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR-3 is currently implemented only for spin-restricted seeds")
        d = ansatz.diagonal.to_standard()
        tau, omega = spin_restricted_triples_seed_from_pair_params(
            d.pair_params,
            ansatz.nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        diagonal = IGCR3SpinRestrictedSpec.from_igcr2_diagonal(
            ansatz.diagonal,
            tau=tau,
            omega_values=omega,
        )
        return cls(
            diagonal=diagonal,
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_ucj_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )

    @classmethod
    def from_ucj(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        return cls.from_ucj_ansatz(
            ansatz,
            nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )

    @classmethod
    def from_gcr_ansatz(
        cls,
        ansatz: GCRAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        tau_scale = kwargs.pop("tau_scale", 0.0)
        omega_scale = kwargs.pop("omega_scale", 0.0)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(
            ucj,
            nocc=t2.shape[0],
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )


def relabel_igcr3_ansatz_orbitals(
    ansatz: IGCR3Ansatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR3Ansatz:
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
    omega_old = {
        (p, q, r): value
        for value, (p, q, r) in zip(d.omega_vector(), d.omega_indices)
    }
    omega_values = np.asarray(
        [
            omega_old[
                tuple(
                    sorted(
                        (
                            int(old_for_new[p]),
                            int(old_for_new[q]),
                            int(old_for_new[r]),
                        )
                    )
                )
            ]
            for p, q, r in _default_triple_indices(ansatz.norb)
        ],
        dtype=np.float64,
    )
    diagonal = IGCR3SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega_values,
    )
    return IGCR3Ansatz(
        diagonal=diagonal,
        left=relabel.conj().T @ ansatz.left @ relabel,
        right=relabel.conj().T @ ansatz.right @ relabel,
        nocc=ansatz.nocc,
    )


@dataclass(frozen=True)
class IGCR3SpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
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
        if (
            self.left_right_ov_relative_scale is not None
            and (
                not np.isfinite(float(self.left_right_ov_relative_scale))
                or self.left_right_ov_relative_scale <= 0
            )
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _validate_ordered_pairs(self.tau_indices_, self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return (
            self.reduce_cubic_gauge
            and self.pair_indices == _default_pair_indices(self.norb)
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
        )

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return IGCR3CubicReduction(self.norb, self.nocc)

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
    def n_double_params(self):
        return 0

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
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
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
            + self.n_right_orbital_rotation_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "double": self.n_double_params,
            "pair": self.n_pair_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "cubic": self.n_tau_params if self.uses_reduced_cubic_chart else (
                self.n_tau_params + self.n_omega_params
            ),
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR3Ansatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        idx += n

        n = self.n_pair_params
        pair_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
        pair_sparse = _symmetric_matrix_from_values(
            pair_sparse_values,
            self.norb,
            self.pair_indices,
        )
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            cubic = self.cubic_reduction.full_from_reduced(params[idx : idx + n])
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
            tau = _ordered_matrix_from_values(params[idx : idx + n], self.norb, self.tau_indices)
            idx += n

            n = self.n_omega_params
            omega_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            omega_sparse = {
                triple: value
                for triple, value in zip(self.omega_indices, omega_sparse_values)
            }
            omega_values = np.asarray(
                [
                    omega_sparse.get(triple, 0.0)
                    for triple in _default_triple_indices(self.norb)
                ],
                dtype=np.float64,
            )
            idx += n

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(params[idx : idx + n], self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)

        return IGCR3Ansatz(
            diagonal=IGCR3SpinRestrictedSpec(
                double_params=np.zeros(self.norb, dtype=np.float64),
                pair_values=pair_values,
                tau=tau,
                omega_values=omega_values,
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR3Ansatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        d = ansatz.diagonal
        pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
        tau = d.tau_matrix()
        omega = d.omega_vector()

        cubic_onebody_phase = np.zeros(self.norb, dtype=np.float64)
        reduced_pair_values = None
        reduced_cubic_values = None
        if self.uses_reduced_cubic_chart:
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
            reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = (
                self.cubic_reduction.reduce_full(full_pair_values, full_cubic)
            )

        phase_vec = (
            _restricted_left_phase_vector(d.full_double(), self.nocc)
            + cubic_onebody_phase
        )
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
        out[idx : idx + n] = left_params
        idx += n

        n = self.n_pair_params
        out[idx : idx + n] = np.asarray([pair_eff[p, q] for p, q in self.pair_indices], dtype=np.float64)
        idx += n

        if self.uses_reduced_cubic_chart:
            out[
                self.n_left_orbital_rotation_params :
                self.n_left_orbital_rotation_params + self.n_pair_params
            ] = reduced_pair_values
            n = self.n_tau_params
            out[idx : idx + n] = reduced_cubic_values
            idx += n
        else:
            n = self.n_tau_params
            out[idx : idx + n] = _values_from_ordered_matrix(tau, self.tau_indices)
            idx += n

            n = self.n_omega_params
            full_omega = {
                triple: value for value, triple in zip(omega, d.omega_indices)
            }
            out[idx : idx + n] = np.asarray([full_omega[t] for t in self.omega_indices], dtype=np.float64)
            idx += n

        n = self.n_right_orbital_rotation_params
        left_param_unitary = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)
        return self._public_parameters_from_native(out)

    def parameters_from_igcr2_ansatz(
        self,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR3Ansatz.from_igcr2_ansatz(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR3Ansatz.from_ucj_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def parameters_from_gcr_ansatz(
        self,
        ansatz: GCRAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR3Ansatz.from_gcr_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR3SpinRestrictedParameterization | None" = None,
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
            ansatz = relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func


def igcr3_from_igcr2_ansatz(
    ansatz: IGCR2Ansatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCR3Ansatz:
    return IGCR3Ansatz.from_igcr2_ansatz(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    )
