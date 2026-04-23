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
)
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3CubicReduction,
    IGCR3SpinRestrictedSpec,
    _default_pair_indices,
    _default_tau_indices,
    _default_triple_indices,
    _validate_ordered_pairs,
    _validate_triples,
    apply_igcr3_spin_restricted_diagonal,
    spin_restricted_triples_seed_from_pair_params,
)
from xquces.gcr.igcr234 import (
    _apply_one_body_generator_to_state,
    _chart_unitary_and_tangents,
    _cubic_feature_matrix,
    _full_sparse_dict,
    _infer_norb_from_unique_pair_count,
    _irreducible_pair_from_diagonal,
    _ordered_matrix_from_sparse_values,
    _pair_feature_matrix,
    _pair_values_from_matrix,
    _tau_values_from_matrix,
    _vector_from_sparse_dict,
)
from xquces.gcr.model import GCRAnsatz
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.ucj.model import UCJAnsatz


@dataclass(frozen=True)
class IGCR23SpinRestrictedSpec:
    pair_values: np.ndarray
    tau_values: np.ndarray
    omega_values: np.ndarray

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

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(np.asarray(self.pair_values, dtype=np.float64), self.norb, self.pair_indices)

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

    def t_diagonal(self) -> IGCR3SpinRestrictedSpec:
        return IGCR3SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=np.zeros(len(self.pair_indices), dtype=np.float64),
            tau=self.tau_matrix(),
            omega_values=self.omega_vector(),
        )


@dataclass(frozen=True)
class IGCR23Ansatz:
    diagonal: IGCR23SpinRestrictedSpec
    u1: np.ndarray
    u2: np.ndarray
    u3: np.ndarray
    nocc: int

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        arr = np.array(vec, dtype=np.complex128, copy=copy)
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
    ) -> "IGCR23Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR23 is currently implemented only for spin-restricted seeds")
        pair = ansatz.diagonal.to_standard().pair_params
        tau, omega = spin_restricted_triples_seed_from_pair_params(
            pair,
            ansatz.nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        diagonal = IGCR23SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair),
            tau_values=_tau_values_from_matrix(tau),
            omega_values=np.asarray(omega, dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(
            diagonal=diagonal,
            u1=np.asarray(ansatz.left, dtype=np.complex128),
            u2=identity,
            u3=np.asarray(ansatz.right, dtype=np.complex128),
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
    ) -> "IGCR23Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError("iGCR23 is currently implemented only for spin-restricted seeds")
        if ansatz.n_layers == 1:
            return cls.from_igcr2_ansatz(
                IGCR2Ansatz.from_ucj_ansatz(ansatz, nocc=nocc),
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        if ansatz.n_layers != 2:
            raise ValueError("iGCR23 UCJ seeding currently expects either 1 or 2 layers")
        layer_t = ansatz.layers[0]
        layer_j = ansatz.layers[1]
        final = ansatz.final_orbital_rotation
        if final is None:
            final = np.eye(ansatz.norb, dtype=np.complex128)
        pair_j = _irreducible_pair_from_diagonal(layer_j.diagonal)
        pair_t = _irreducible_pair_from_diagonal(layer_t.diagonal)
        tau, omega = spin_restricted_triples_seed_from_pair_params(
            pair_t,
            nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        diagonal = IGCR23SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(pair_j),
            tau_values=_tau_values_from_matrix(tau),
            omega_values=np.asarray(omega, dtype=np.float64),
        )
        return cls(
            diagonal=diagonal,
            u1=np.asarray(final @ layer_j.orbital_rotation, dtype=np.complex128),
            u2=np.asarray(layer_j.orbital_rotation.conj().T @ layer_t.orbital_rotation, dtype=np.complex128),
            u3=np.asarray(layer_t.orbital_rotation.conj().T, dtype=np.complex128),
            nocc=nocc,
        )

    @classmethod
    def from_ucj(cls, ansatz: UCJAnsatz, nocc: int, **kwargs) -> "IGCR23Ansatz":
        return cls.from_ucj_ansatz(ansatz, nocc, **kwargs)

    @classmethod
    def from_igcr3_ansatz(cls, ansatz: IGCR3Ansatz) -> "IGCR23Ansatz":
        diagonal = IGCR23SpinRestrictedSpec(
            pair_values=_pair_values_from_matrix(ansatz.diagonal.pair_matrix()),
            tau_values=_tau_values_from_matrix(ansatz.diagonal.tau_matrix()),
            omega_values=np.asarray(ansatz.diagonal.omega_vector(), dtype=np.float64),
        )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        return cls(
            diagonal=diagonal,
            u1=np.asarray(ansatz.left, dtype=np.complex128),
            u2=identity,
            u3=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_gcr_ansatz(cls, ansatz: GCRAnsatz, nocc: int, **kwargs) -> "IGCR23Ansatz":
        return cls.from_igcr2_ansatz(IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc), **kwargs)

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        n_reps = kwargs.pop("n_reps", 2)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, n_reps=n_reps, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(ucj, nocc=t2.shape[0])


@dataclass(frozen=True)
class IGCR23SpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    general_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)

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
    def uses_reduced_cubic_chart(self) -> bool:
        return self.reduce_cubic_gauge and self.tau_indices == _default_tau_indices(self.norb) and self.omega_indices == _default_triple_indices(self.norb)

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
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self):
        return self.n_u1_params + self.n_pair_params + self.n_u2_params + self.n_tau_params + self.n_omega_params + self.n_u3_params

    def sector_sizes(self) -> dict[str, int]:
        return {
            "u1": self.n_u1_params,
            "j": self.n_pair_params,
            "u2": self.n_u2_params,
            "cubic": self.n_tau_params if self.uses_reduced_cubic_chart else self.n_tau_params + self.n_omega_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "u3": self.n_u3_params,
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
        return pieces, pair_values, tau_values, omega_values

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR23Ansatz:
        pieces, pair_values, tau_values, omega_values = self._full_diagonal_vectors_from_public(params)
        u1 = self.general_orbital_chart.unitary_from_parameters(pieces["u1"], self.norb)
        u2 = self.general_orbital_chart.unitary_from_parameters(pieces["u2"], self.norb)
        u3 = self.right_orbital_chart.unitary_from_parameters(pieces["u3"], self.norb)
        return IGCR23Ansatz(
            diagonal=IGCR23SpinRestrictedSpec(pair_values=pair_values, tau_values=tau_values, omega_values=omega_values),
            u1=u1,
            u2=u2,
            u3=u3,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR23Ansatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        d = ansatz.diagonal
        pair = d.pair_matrix()
        tau = d.tau_vector()
        omega = d.omega_vector()
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
        out[idx: idx + n] = self.right_orbital_chart.parameters_from_unitary(ansatz.u3)
        return out

    def parameters_from_igcr2_ansatz(self, ansatz: IGCR2Ansatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR23Ansatz.from_igcr2_ansatz(ansatz, **kwargs))

    def parameters_from_igcr3_ansatz(self, ansatz: IGCR3Ansatz) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR23Ansatz.from_igcr3_ansatz(ansatz))

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR23Ansatz.from_ucj_ansatz(ansatz, self.nocc, **kwargs))

    def parameters_from_gcr_ansatz(self, ansatz: GCRAnsatz, **kwargs) -> np.ndarray:
        return self.parameters_from_ansatz(IGCR23Ansatz.from_gcr_ansatz(ansatz, self.nocc, **kwargs))

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
        pair_cols = [_default_pair_indices(self.norb).index(ix) for ix in self.pair_indices]
        if self.uses_reduced_cubic_chart:
            cubic_features = cubic_features_full @ self.cubic_reduction.physical_cubic_basis
        else:
            tau_all = _default_tau_indices(self.norb)
            omega_all = _default_triple_indices(self.norb)
            tau_cols = [tau_all.index(ix) for ix in self.tau_indices]
            omega_cols = [omega_all.index(ix) for ix in self.omega_indices]
            cubic_features = np.concatenate([
                cubic_features_full[:, tau_cols],
                cubic_features_full[:, len(tau_all) + np.asarray(omega_cols, dtype=int)],
            ], axis=1)
        def jac(params: np.ndarray) -> np.ndarray:
            pieces, pair_values, tau_values, omega_values = self._full_diagonal_vectors_from_public(params)
            u1, x1s = _chart_unitary_and_tangents(self.general_orbital_chart, pieces["u1"], self.norb, self.nocc)
            u2, x2s = _chart_unitary_and_tangents(self.general_orbital_chart, pieces["u2"], self.norb, self.nocc)
            u3, x3s = _chart_unitary_and_tangents(self.right_orbital_chart, pieces["u3"], self.norb, self.nocc)
            diagonal = IGCR23SpinRestrictedSpec(pair_values=pair_values, tau_values=tau_values, omega_values=omega_values)
            tdiag = diagonal.t_diagonal()
            psi3 = apply_orbital_rotation(reference_vec, u3, norb=self.norb, nelec=nelec, copy=True)
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
                cols.append(after_u2(apply_igcr3_spin_restricted_diagonal(_apply_one_body_generator_to_state(psi3, x, self.norb, nelec), tdiag, self.norb, nelec, copy=True)))
            return np.column_stack(cols)
        return jac
