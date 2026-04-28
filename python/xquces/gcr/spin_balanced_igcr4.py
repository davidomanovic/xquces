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
    "baba",
    "bbaa",
)


def _default_pair_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _default_tau_indices(norb: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(norb) for q in range(norb) if p != q]


def _default_triple_indices(norb: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(norb), 3))


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


def _validate_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_pair_indices(norb)
    out = []
    seen = set()
    for p, q in pairs:
        if not (0 <= p < q < norb):
            raise ValueError("pair indices must satisfy 0 <= p < q < norb")
        if (p, q) in seen:
            raise ValueError("pair indices must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return out


def _validate_tau_indices(
    pairs: list[tuple[int, int]] | None,
    norb: int,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_tau_indices(norb)
    out = []
    seen = set()
    for p, q in pairs:
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("tau indices out of bounds")
        if p == q:
            raise ValueError("tau diagonal entries are not allowed")
        if (p, q) in seen:
            raise ValueError("tau indices must not contain duplicates")
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


def _validate_rho_indices(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int]]:
    if triples is None:
        return _default_rho_indices(norb)
    out = []
    seen = set()
    for p, q, r in triples:
        if not (0 <= p < norb and 0 <= q < r < norb):
            raise ValueError("rho indices must satisfy 0 <= p < norb and 0 <= q < r < norb")
        if p == q or p == r:
            raise ValueError("rho indices must be distinct")
        if (p, q, r) in seen:
            raise ValueError("rho indices must not contain duplicates")
        seen.add((p, q, r))
        out.append((p, q, r))
    return out


def _validate_sigmas(
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


def _symmetric_matrix_from_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),):
        raise ValueError(f"Expected {(len(pairs),)}, got {values.shape}.")
    out = np.zeros((norb, norb), dtype=np.float64)
    for value, (p, q) in zip(values, pairs):
        out[p, q] = value
        out[q, p] = value
    return out


def _ordered_matrix_from_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),):
        raise ValueError(f"Expected {(len(pairs),)}, got {values.shape}.")
    out = np.zeros((norb, norb), dtype=np.float64)
    for value, (p, q) in zip(values, pairs):
        out[p, q] = value
    np.fill_diagonal(out, 0.0)
    return out


@dataclass(frozen=True)
class IGCR4SpinCombo6Spec:
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma6_values: np.ndarray
    double_params: np.ndarray | None = None

    @property
    def norb(self) -> int:
        tau = np.asarray(self.tau, dtype=np.float64)
        if tau.ndim != 2 or tau.shape[0] != tau.shape[1]:
            raise ValueError("tau must have shape (norb, norb)")
        return int(tau.shape[0])

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _default_triple_indices(self.norb)

    @property
    def eta_indices(self) -> list[tuple[int, int]]:
        return _default_eta_indices(self.norb)

    @property
    def rho_indices(self) -> list[tuple[int, int, int]]:
        return _default_rho_indices(self.norb)

    @property
    def sigma_indices(self) -> list[tuple[int, int, int, int]]:
        return _default_sigma_indices(self.norb)

    def full_double(self) -> np.ndarray:
        if self.double_params is None:
            return np.zeros(self.norb, dtype=np.float64)
        out = np.asarray(self.double_params, dtype=np.float64)
        if out.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        return out

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
        out = np.asarray(self.omega_values, dtype=np.float64)
        if out.shape != (len(self.omega_indices),):
            raise ValueError("omega_values has inconsistent shape")
        return out

    def eta_vector(self) -> np.ndarray:
        out = np.asarray(self.eta_values, dtype=np.float64)
        if out.shape != (len(self.eta_indices),):
            raise ValueError("eta_values has inconsistent shape")
        return out

    def rho_vector(self) -> np.ndarray:
        out = np.asarray(self.rho_values, dtype=np.float64)
        if out.shape != (len(self.rho_indices),):
            raise ValueError("rho_values has inconsistent shape")
        return out

    def sigma6_matrix(self) -> np.ndarray:
        out = np.asarray(self.sigma6_values, dtype=np.float64)
        if out.shape != (len(self.sigma_indices), 6):
            raise ValueError("sigma6_values must have shape (n_sigma, 6)")
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
        n = a + b
        d = a * b
        phase = float(np.dot(self.full_double(), d))
        pair = self.pair_matrix()
        for p, q in self.pair_indices:
            phase += pair[p, q] * n[p] * n[q]
        tau = self.tau_matrix()
        for p, q in self.tau_indices:
            phase += tau[p, q] * d[p] * n[q]
        for value, (p, q, r) in zip(self.omega_vector(), self.omega_indices):
            phase += value * n[p] * n[q] * n[r]
        for value, (p, q) in zip(self.eta_vector(), self.eta_indices):
            phase += value * d[p] * d[q]
        for value, (p, q, r) in zip(self.rho_vector(), self.rho_indices):
            phase += value * d[p] * n[q] * n[r]
        sigma6 = self.sigma6_matrix()
        for values, (p, q, r, s) in zip(sigma6, self.sigma_indices):
            phase += values[0] * a[p] * a[q] * b[r] * b[s]
            phase += values[1] * a[p] * b[q] * a[r] * b[s]
            phase += values[2] * a[p] * b[q] * b[r] * a[s]
            phase += values[3] * b[p] * a[q] * a[r] * b[s]
            phase += values[4] * b[p] * a[q] * b[r] * a[s]
            phase += values[5] * b[p] * b[q] * a[r] * a[s]
        return float(phase)

    @classmethod
    def from_restricted_diagonal(cls, diagonal) -> "IGCR4SpinCombo6Spec":
        sigma = np.asarray(diagonal.sigma_vector(), dtype=np.float64)
        return cls(
            double_params=np.asarray(diagonal.full_double(), dtype=np.float64),
            pair_values=np.asarray(diagonal.pair_values, dtype=np.float64),
            tau=np.asarray(diagonal.tau_matrix(), dtype=np.float64),
            omega_values=np.asarray(diagonal.omega_vector(), dtype=np.float64),
            eta_values=np.asarray(diagonal.eta_vector(), dtype=np.float64),
            rho_values=np.asarray(diagonal.rho_vector(), dtype=np.float64),
            sigma6_values=np.repeat(sigma[:, None], 6, axis=1),
        )


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
        np.asarray(diagonal.full_double(), dtype=np.float64) * time,
        np.asarray(diagonal.pair_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.tau_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.omega_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.eta_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.rho_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.sigma6_matrix(), dtype=np.float64) * time,
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

    @classmethod
    def from_restricted_igcr4_ansatz(cls, ansatz) -> "IGCR4SpinCombo6Ansatz":
        return cls(
            diagonal=IGCR4SpinCombo6Spec.from_restricted_diagonal(ansatz.diagonal),
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )


@dataclass(frozen=True)
class IGCR4SpinCombo6Parameterization:
    norb: int
    nelec: tuple[int, int]
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    max_body: int = 4
    spin_balanced: bool = True
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
        if self.max_body != 4:
            raise ValueError("spin-combo6 iGCR4 requires max_body=4")
        object.__setattr__(self, "nelec", nelec)
        _validate_pairs(self.interaction_pairs, self.norb)
        _validate_tau_indices(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        _validate_pairs(self.eta_indices_, self.norb)
        _validate_rho_indices(self.rho_indices_, self.norb)
        _validate_sigmas(self.sigma_indices_, self.norb)

    @property
    def nocc(self) -> int:
        return self.nelec[0]

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _validate_tau_indices(self.tau_indices_, self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def eta_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.eta_indices_, self.norb)

    @property
    def rho_indices(self) -> list[tuple[int, int, int]]:
        return _validate_rho_indices(self.rho_indices_, self.norb)

    @property
    def sigma_indices(self) -> list[tuple[int, int, int, int]]:
        return _validate_sigmas(self.sigma_indices_, self.norb)

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
        return len(self.tau_indices)

    @property
    def n_omega_params(self):
        return len(self.omega_indices)

    @property
    def n_eta_params(self):
        return len(self.eta_indices)

    @property
    def n_rho_params(self):
        return len(self.rho_indices)

    @property
    def n_sigma6_params(self):
        return 6 * len(self.sigma_indices)

    @property
    def n_diagonal_params(self):
        return (
            self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma6_params
        )

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
        return (
            self.n_left_orbital_rotation_params
            + self.n_diagonal_params
            + self.n_right_orbital_rotation_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "pair": self.n_pair_params,
            "tau": self.n_tau_params,
            "omega": self.n_omega_params,
            "eta": self.n_eta_params,
            "rho": self.n_rho_params,
            "sigma6": self.n_sigma6_params,
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
        idx = 0
        n = self.n_left_orbital_rotation_params
        left_params = params[idx : idx + n]
        idx += n
        n = self.n_pair_params
        pair_values = params[idx : idx + n]
        idx += n
        n = self.n_tau_params
        tau_values = params[idx : idx + n]
        idx += n
        n = self.n_omega_params
        omega_values = params[idx : idx + n]
        idx += n
        n = self.n_eta_params
        eta_values = params[idx : idx + n]
        idx += n
        n = self.n_rho_params
        rho_values = params[idx : idx + n]
        idx += n
        n = self.n_sigma6_params
        sigma6_values = params[idx : idx + n].reshape((len(self.sigma_indices), 6))
        idx += n
        n = self.n_right_orbital_rotation_params
        right_params = params[idx : idx + n]
        return (
            left_params,
            pair_values,
            tau_values,
            omega_values,
            eta_values,
            rho_values,
            sigma6_values,
            right_params,
        )

    def diagonal_from_parameters(self, params: np.ndarray) -> IGCR4SpinCombo6Spec:
        (
            _,
            pair_sparse_values,
            tau_sparse_values,
            omega_sparse_values,
            eta_sparse_values,
            rho_sparse_values,
            sigma6_sparse_values,
            _,
        ) = self._split_params(params)
        pair_sparse = _symmetric_matrix_from_values(
            pair_sparse_values,
            self.norb,
            self.pair_indices,
        )
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        tau = _ordered_matrix_from_values(
            tau_sparse_values,
            self.norb,
            self.tau_indices,
        )
        omega_sparse = {
            triple: value for triple, value in zip(self.omega_indices, omega_sparse_values)
        }
        omega_values = np.asarray(
            [omega_sparse.get(triple, 0.0) for triple in _default_triple_indices(self.norb)],
            dtype=np.float64,
        )
        eta_sparse = {
            pair: value for pair, value in zip(self.eta_indices, eta_sparse_values)
        }
        eta_values = np.asarray(
            [eta_sparse.get(pair, 0.0) for pair in _default_eta_indices(self.norb)],
            dtype=np.float64,
        )
        rho_sparse = {
            triple: value for triple, value in zip(self.rho_indices, rho_sparse_values)
        }
        rho_values = np.asarray(
            [rho_sparse.get(triple, 0.0) for triple in _default_rho_indices(self.norb)],
            dtype=np.float64,
        )
        sigma_sparse = {
            quad: values for quad, values in zip(self.sigma_indices, sigma6_sparse_values)
        }
        sigma6_values = np.asarray(
            [
                sigma_sparse.get(quad, np.zeros(6, dtype=np.float64))
                for quad in _default_sigma_indices(self.norb)
            ],
            dtype=np.float64,
        )
        return IGCR4SpinCombo6Spec(
            pair_values=pair_values,
            tau=tau,
            omega_values=omega_values,
            eta_values=eta_values,
            rho_values=rho_values,
            sigma6_values=sigma6_values,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4SpinCombo6Ansatz:
        params = self._native_parameters_from_public(params)
        (
            left_params,
            _,
            _,
            _,
            _,
            _,
            _,
            right_params,
        ) = self._split_params(params)
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
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(
                ansatz.left
            )
        else:
            left_params = left_chart.parameters_from_unitary(ansatz.left)
            right_phase = np.zeros(self.norb, dtype=np.float64)
        right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
        left_param_unitary = left_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(
            left_param_unitary,
            right_eff,
            self.nocc,
        )
        right_params = self.right_orbital_chart.parameters_from_unitary(final_eff)
        d = ansatz.diagonal
        pair = d.pair_matrix()
        pair_values = np.asarray([pair[p, q] for p, q in self.pair_indices], dtype=np.float64)
        tau = d.tau_matrix()
        tau_values = np.asarray([tau[p, q] for p, q in self.tau_indices], dtype=np.float64)
        omega_map = dict(zip(d.omega_indices, d.omega_vector()))
        omega_values = np.asarray(
            [omega_map.get(triple, 0.0) for triple in self.omega_indices],
            dtype=np.float64,
        )
        eta_map = dict(zip(d.eta_indices, d.eta_vector()))
        eta_values = np.asarray(
            [eta_map.get(pair_idx, 0.0) for pair_idx in self.eta_indices],
            dtype=np.float64,
        )
        rho_map = dict(zip(d.rho_indices, d.rho_vector()))
        rho_values = np.asarray(
            [rho_map.get(triple, 0.0) for triple in self.rho_indices],
            dtype=np.float64,
        )
        sigma_map = dict(zip(d.sigma_indices, d.sigma_vector()))
        sigma6_values = np.asarray(
            [
                np.repeat(sigma_map.get(quad, 0.0), 6)
                for quad in self.sigma_indices
            ],
            dtype=np.float64,
        ).reshape(-1)
        return self._public_parameters_from_native(
            np.concatenate(
                [
                    left_params,
                    pair_values,
                    tau_values,
                    omega_values,
                    eta_values,
                    rho_values,
                    sigma6_values,
                    right_params,
                ]
            )
        )

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError("spin-combo6 iGCR4 parameterization got wrong nelec")
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=self.nelec,
                copy=True,
            )

        return func


IGCR4SpinBalancedFixedSectorAnsatz = IGCR4SpinCombo6Ansatz
IGCR4SpinBalancedFixedSectorParameterization = IGCR4SpinCombo6Parameterization
