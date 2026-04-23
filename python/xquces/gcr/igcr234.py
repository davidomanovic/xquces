from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gates import apply_igcr2_spin_restricted
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    _symmetric_matrix_from_values,
    _validate_pairs,
    reduce_spin_restricted,
)
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3SpinRestrictedSpec,
    _default_pair_indices,
    _default_tau_indices,
    _default_triple_indices,
    _validate_ordered_pairs,
    _validate_triples,
    apply_igcr3_spin_restricted_diagonal,
    spin_restricted_triples_seed_from_pair_params,
)
from xquces.gcr.igcr4 import (
    IGCR4Ansatz,
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
from xquces.orbitals import apply_orbital_rotation
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


def _ordered_matrix_from_sparse_values(
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


def _full_tau_values_from_sparse_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    tau = _ordered_matrix_from_sparse_values(values, norb, pairs)
    return _tau_values_from_matrix(tau)


def _full_sparse_dict(
    values: np.ndarray,
    indices,
):
    return {idx: float(val) for idx, val in zip(indices, np.asarray(values, dtype=np.float64))}


def _vector_from_sparse_dict(default_indices, sparse_dict) -> np.ndarray:
    return np.asarray([sparse_dict.get(idx, 0.0) for idx in default_indices], dtype=np.float64)


def _irreducible_pair_from_diagonal(diagonal: SpinRestrictedSpec) -> np.ndarray:
    if not isinstance(diagonal, SpinRestrictedSpec):
        raise TypeError("expected a spin-restricted diagonal")
    return reduce_spin_restricted(diagonal).pair


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
        return _symmetric_matrix_from_values(
            np.asarray(self.pair_values, dtype=np.float64),
            self.norb,
            self.pair_indices,
        )

    def tau_vector(self) -> np.ndarray:
        tau = np.asarray(self.tau_values, dtype=np.float64)
        if tau.shape != (len(self.tau_indices),):
            raise ValueError("tau_values has inconsistent shape")
        return tau

    def tau_matrix(self) -> np.ndarray:
        return _ordered_matrix_from_sparse_values(self.tau_vector(), self.norb, self.tau_indices)

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
        arr = apply_igcr2_spin_restricted(
            arr,
            self.diagonal.pair_matrix(),
            self.norb,
            nelec,
            left_orbital_rotation=self.u1,
            right_orbital_rotation=self.u2,
            copy=False,
        )
        return arr

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
    ) -> "IGCR234Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR234 is currently implemented only for spin-restricted seeds")
        pair = ansatz.diagonal.to_standard().pair_params
        tau, omega = spin_restricted_triples_seed_from_pair_params(
            pair,
            ansatz.nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
            pair,
            ansatz.nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair),
            tau_values=_tau_values_from_matrix(tau),
            omega_values=np.asarray(omega, dtype=np.float64),
            eta_values=np.asarray(eta, dtype=np.float64),
            rho_values=np.asarray(rho, dtype=np.float64),
            sigma_values=np.asarray(sigma, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(
            diagonal=diagonal,
            u1=np.asarray(ansatz.left, dtype=np.complex128),
            u2=identity,
            u3=identity,
            u4=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_igcr3_ansatz(
        cls,
        ansatz: IGCR3Ansatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR234Ansatz":
        pair = ansatz.diagonal.pair_matrix()
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
            pair,
            ansatz.nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )
        diagonal = IGCR234SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair),
            tau_values=_tau_values_from_matrix(ansatz.diagonal.tau_matrix()),
            omega_values=np.asarray(ansatz.diagonal.omega_vector(), dtype=np.float64),
            eta_values=np.asarray(eta, dtype=np.float64),
            rho_values=np.asarray(rho, dtype=np.float64),
            sigma_values=np.asarray(sigma, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(
            diagonal=diagonal,
            u1=np.asarray(ansatz.left, dtype=np.complex128),
            u2=identity,
            u3=identity,
            u4=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

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
        return cls(
            diagonal=diagonal,
            u1=np.asarray(ansatz.left, dtype=np.complex128),
            u2=identity,
            u3=identity,
            u4=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 1.0,
        omega_scale: float = 1.0,
        eta_scale: float = 1.0,
        rho_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> "IGCR234Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR234 is currently implemented only for spin-restricted seeds")
        if ansatz.n_layers == 1:
            return cls.from_igcr2_ansatz(
                IGCR2Ansatz.from_ucj_ansatz(ansatz, nocc=nocc),
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
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

        tau, omega = spin_restricted_triples_seed_from_pair_params(
            pair_t,
            nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
            pair_q,
            nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

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
    def from_ucj(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 1.0,
        omega_scale: float = 1.0,
        eta_scale: float = 1.0,
        rho_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> "IGCR234Ansatz":
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
    ) -> "IGCR234Ansatz":
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
        tau_scale = kwargs.pop("tau_scale", 1.0)
        omega_scale = kwargs.pop("omega_scale", 1.0)
        eta_scale = kwargs.pop("eta_scale", 1.0)
        rho_scale = kwargs.pop("rho_scale", 1.0)
        sigma_scale = kwargs.pop("sigma_scale", 1.0)
        n_reps = kwargs.pop("n_reps", 3)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, n_reps=n_reps, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(
            ucj,
            nocc=t2.shape[0],
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )


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
        return len(self.tau_indices)

    @property
    def n_omega_params(self):
        return len(self.omega_indices)

    @property
    def n_u3_params(self):
        return self.general_orbital_chart.n_params(self.norb)

    @property
    def n_eta_params(self):
        return len(self.eta_indices)

    @property
    def n_rho_params(self):
        return len(self.rho_indices)

    @property
    def n_sigma_params(self):
        return len(self.sigma_indices)

    @property
    def n_u4_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        return (
            self.n_u1_params
            + self.n_pair_params
            + self.n_u2_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_u3_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
            + self.n_u4_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "u1": self.n_u1_params,
            "j": self.n_pair_params,
            "u2": self.n_u2_params,
            "tau": self.n_tau_params,
            "omega": self.n_omega_params,
            "u3": self.n_u3_params,
            "eta": self.n_eta_params,
            "rho": self.n_rho_params,
            "sigma": self.n_sigma_params,
            "u4": self.n_u4_params,
            "total": self.n_params,
        }

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR234Ansatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0

        n = self.n_u1_params
        u1 = self.general_orbital_chart.unitary_from_parameters(params[idx: idx + n], self.norb)
        idx += n

        n = self.n_pair_params
        pair_sparse = _symmetric_matrix_from_values(np.asarray(params[idx: idx + n], dtype=np.float64), self.norb, self.pair_indices)
        pair_values = _pair_values_from_matrix(pair_sparse)
        idx += n

        n = self.n_u2_params
        u2 = self.general_orbital_chart.unitary_from_parameters(params[idx: idx + n], self.norb)
        idx += n

        n = self.n_tau_params
        tau_values = _full_tau_values_from_sparse_values(np.asarray(params[idx: idx + n], dtype=np.float64), self.norb, self.tau_indices)
        idx += n

        n = self.n_omega_params
        omega_values = _vector_from_sparse_dict(
            _default_triple_indices(self.norb),
            _full_sparse_dict(np.asarray(params[idx: idx + n], dtype=np.float64), self.omega_indices),
        )
        idx += n

        n = self.n_u3_params
        u3 = self.general_orbital_chart.unitary_from_parameters(params[idx: idx + n], self.norb)
        idx += n

        n = self.n_eta_params
        eta_values = _vector_from_sparse_dict(
            _default_eta_indices(self.norb),
            _full_sparse_dict(np.asarray(params[idx: idx + n], dtype=np.float64), self.eta_indices),
        )
        idx += n

        n = self.n_rho_params
        rho_values = _vector_from_sparse_dict(
            _default_rho_indices(self.norb),
            _full_sparse_dict(np.asarray(params[idx: idx + n], dtype=np.float64), self.rho_indices),
        )
        idx += n

        n = self.n_sigma_params
        sigma_values = _vector_from_sparse_dict(
            _default_sigma_indices(self.norb),
            _full_sparse_dict(np.asarray(params[idx: idx + n], dtype=np.float64), self.sigma_indices),
        )
        idx += n

        n = self.n_u4_params
        u4 = self.right_orbital_chart.unitary_from_parameters(params[idx: idx + n], self.norb)

        return IGCR234Ansatz(
            diagonal=IGCR234SpinRestrictedSpec(
                pair_values=pair_values,
                tau_values=tau_values,
                omega_values=omega_values,
                eta_values=eta_values,
                rho_values=rho_values,
                sigma_values=sigma_values,
            ),
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
        tau = d.tau_matrix()
        omega = {idx: val for idx, val in zip(d.omega_indices, d.omega_vector())}
        eta = {idx: val for idx, val in zip(d.eta_indices, d.eta_vector())}
        rho = {idx: val for idx, val in zip(d.rho_indices, d.rho_vector())}
        sigma = {idx: val for idx, val in zip(d.sigma_indices, d.sigma_vector())}

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

        n = self.n_tau_params
        out[idx: idx + n] = np.asarray([tau[p, q] for p, q in self.tau_indices], dtype=np.float64)
        idx += n

        n = self.n_omega_params
        out[idx: idx + n] = np.asarray([omega.get(idx_, 0.0) for idx_ in self.omega_indices], dtype=np.float64)
        idx += n

        n = self.n_u3_params
        out[idx: idx + n] = self.general_orbital_chart.parameters_from_unitary(ansatz.u3)
        idx += n

        n = self.n_eta_params
        out[idx: idx + n] = np.asarray([eta.get(idx_, 0.0) for idx_ in self.eta_indices], dtype=np.float64)
        idx += n

        n = self.n_rho_params
        out[idx: idx + n] = np.asarray([rho.get(idx_, 0.0) for idx_ in self.rho_indices], dtype=np.float64)
        idx += n

        n = self.n_sigma_params
        out[idx: idx + n] = np.asarray([sigma.get(idx_, 0.0) for idx_ in self.sigma_indices], dtype=np.float64)
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

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func
