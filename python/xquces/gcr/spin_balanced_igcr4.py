from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np

from xquces.basis import occ_rows, sector_shape
from xquces.gcr.diagonal_rank import (
    spin_orbital_diagonal_features,
)
from xquces.gcr.igcr2 import (
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _right_unitary_from_left_and_final,
)
from xquces.gcr.igcr4 import IGCR4SpinRestrictedSpec
from xquces.orbitals import apply_orbital_rotation


@dataclass(frozen=True)
class FixedSectorDiagonalBasis:
    features: np.ndarray
    raw_n_features: int
    singular_values: np.ndarray
    max_body: int
    spin_balanced: bool

    @property
    def n_determinants(self) -> int:
        return self.features.shape[0]

    @property
    def n_params(self) -> int:
        return self.features.shape[1]

    def phase(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.features @ params

    def project_phase(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=np.float64).reshape(self.n_determinants)
        centered = phase - float(np.mean(phase))
        return self.features.T @ centered


def make_spin_orbital_diagonal_basis(
    norb: int,
    nelec: tuple[int, int],
    *,
    max_body: int = 4,
    spin_balanced: bool = True,
    rtol: float = 1e-10,
) -> FixedSectorDiagonalBasis:
    raw = spin_orbital_diagonal_features(
        norb,
        nelec,
        max_body=max_body,
        spin_balanced=spin_balanced,
    )
    centered = raw - raw.mean(axis=0, keepdims=True)
    u, svals, _ = np.linalg.svd(centered, full_matrices=False)
    rank = 0 if svals.size == 0 else int(np.sum(svals > float(rtol) * svals[0]))
    features = np.asarray(u[:, :rank], dtype=np.float64)
    return FixedSectorDiagonalBasis(
        features=features,
        raw_n_features=raw.shape[1],
        singular_values=svals,
        max_body=max_body,
        spin_balanced=spin_balanced,
    )


def restricted_igcr4_phase_vector(
    diagonal: IGCR4SpinRestrictedSpec,
    nelec: tuple[int, int],
) -> np.ndarray:
    norb = diagonal.norb
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    phases = np.empty(len(occ_a) * len(occ_b), dtype=np.float64)
    k = 0
    for alpha in occ_a:
        for beta in occ_b:
            phases[k] = diagonal.phase_from_occupations(alpha, beta)
            k += 1
    return phases


def orbital_rotation_operator(
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    dim = dim_a * dim_b
    eye = np.eye(dim, dtype=np.complex128)
    return np.column_stack(
        [
            apply_orbital_rotation(
                eye[:, k],
                orbital_rotation,
                norb=norb,
                nelec=nelec,
                copy=True,
            )
            for k in range(dim)
        ]
    )


@dataclass(frozen=True)
class FixedOrbitalDiagonalModel:
    basis: FixedSectorDiagonalBasis
    left_operator: np.ndarray
    right_state: np.ndarray

    @classmethod
    def from_orbitals(
        cls,
        basis: FixedSectorDiagonalBasis,
        reference_vec: np.ndarray,
        *,
        left: np.ndarray,
        right: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
    ) -> "FixedOrbitalDiagonalModel":
        right_state = apply_orbital_rotation(
            reference_vec,
            right,
            norb=norb,
            nelec=nelec,
            copy=True,
        )
        left_operator = orbital_rotation_operator(left, norb, nelec)
        return cls(
            basis=basis,
            left_operator=left_operator,
            right_state=np.asarray(right_state, dtype=np.complex128),
        )

    @property
    def n_params(self) -> int:
        return self.basis.n_params

    def state(self, params: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * self.basis.phase(params))
        return self.left_operator @ (phase * self.right_state)

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * self.basis.phase(params))
        diagonalized = phase * self.right_state
        local = 1j * diagonalized[:, None] * self.basis.features
        return self.left_operator @ local

    def phase_overlap_bound(
        self,
        target_state: np.ndarray,
    ) -> dict[str, float]:
        target_state = np.asarray(target_state, dtype=np.complex128).reshape(-1)
        target_frame = self.left_operator.conj().T @ target_state
        right_norm = float(np.linalg.norm(self.right_state))
        target_norm = float(np.linalg.norm(target_frame))
        if right_norm == 0.0 or target_norm == 0.0:
            return {
                "best_overlap": 0.0,
                "best_overlap_squared": 0.0,
                "amplitude_l2": float("inf"),
            }
        right_amp = np.abs(self.right_state) / right_norm
        target_amp = np.abs(target_frame) / target_norm
        best_overlap = float(np.dot(right_amp, target_amp))
        return {
            "best_overlap": best_overlap,
            "best_overlap_squared": best_overlap**2,
            "amplitude_l2": float(np.linalg.norm(right_amp - target_amp)),
        }

    def overlap_with(
        self,
        params: np.ndarray,
        target_state: np.ndarray,
    ) -> float:
        state = self.state(params)
        target = np.asarray(target_state, dtype=np.complex128).reshape(-1)
        denom = np.linalg.norm(state) * np.linalg.norm(target)
        if denom == 0.0:
            return 0.0
        return float(abs(np.vdot(target, state)) / denom)


@dataclass(frozen=True)
class IGCR4SpinBalancedFixedSectorAnsatz:
    basis: FixedSectorDiagonalBasis
    diagonal_params: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nelec: tuple[int, int]

    @property
    def nocc(self) -> int:
        return self.nelec[0]

    def apply(self, vec, nelec, copy=True):
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError("fixed-sector spin-balanced iGCR4 ansatz got wrong nelec")
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )
        arr *= np.exp(1j * self.basis.phase(self.diagonal_params))
        arr = apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )
        return arr


@dataclass(frozen=True)
class IGCR4SpinBalancedFixedSectorParameterization:
    norb: int
    nelec: tuple[int, int]
    max_body: int = 4
    spin_balanced: bool = True
    left_orbital_chart: object | None = None
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False

    def __post_init__(self):
        nelec = tuple(int(x) for x in self.nelec)
        if len(nelec) != 2:
            raise ValueError("nelec must be a two-tuple")
        if not (0 <= nelec[0] <= self.norb and 0 <= nelec[1] <= self.norb):
            raise ValueError("nelec must be compatible with norb")
        object.__setattr__(self, "nelec", nelec)
        if self.left_orbital_chart is None:
            object.__setattr__(self, "left_orbital_chart", IGCR2LeftUnitaryChart())

    @property
    def nocc(self) -> int:
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("nocc is only defined for spin-balanced electron counts")
        return self.nelec[0]

    @cached_property
    def diagonal_basis(self) -> FixedSectorDiagonalBasis:
        return make_spin_orbital_diagonal_basis(
            self.norb,
            self.nelec,
            max_body=self.max_body,
            spin_balanced=self.spin_balanced,
        )

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
    def n_diagonal_params(self):
        return self.diagonal_basis.n_params

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
            "spin_balanced_diag": self.n_diagonal_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def ansatz_from_parameters(
        self,
        params: np.ndarray,
    ) -> IGCR4SpinBalancedFixedSectorAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")

        idx = 0
        n = self.n_left_orbital_rotation_params
        left_params = params[idx : idx + n]
        left = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        idx += n

        n = self.n_diagonal_params
        diagonal_params = np.asarray(params[idx : idx + n], dtype=np.float64)
        idx += n

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n],
            self.norb,
        )
        right = _right_unitary_from_left_and_final(left, final, self.nocc)

        return IGCR4SpinBalancedFixedSectorAnsatz(
            basis=self.diagonal_basis,
            diagonal_params=diagonal_params,
            left=left,
            right=right,
            norb=self.norb,
            nelec=self.nelec,
        )

    def parameters_from_restricted_igcr4_ansatz(
        self,
        ansatz,
    ) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")

        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = (
                left_chart.parameters_and_right_phase_from_unitary(ansatz.left)
            )
        else:
            left_params = left_chart.parameters_from_unitary(ansatz.left)
            right_phase = np.zeros(self.norb, dtype=np.float64)

        right_eff = _diag_unitary(right_phase) @ np.asarray(
            ansatz.right,
            dtype=np.complex128,
        )
        left_param_unitary = left_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(
            left_param_unitary,
            right_eff,
            self.nocc,
        )
        right_params = self.right_orbital_chart.parameters_from_unitary(final_eff)

        phase = restricted_igcr4_phase_vector(ansatz.diagonal, self.nelec)
        diagonal_params = self.diagonal_basis.project_phase(phase)
        return np.concatenate([left_params, diagonal_params, right_params])

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError(
                "fixed-sector spin-balanced parameterization got wrong nelec"
            )
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=self.nelec,
                copy=True,
            )

        return func


@dataclass(frozen=True)
class IGCR4SpinSeparatedFixedSectorAnsatz:
    basis: FixedSectorDiagonalBasis
    diagonal_params: np.ndarray
    left: tuple[np.ndarray, np.ndarray]
    right: tuple[np.ndarray, np.ndarray]
    norb: int
    nelec: tuple[int, int]

    @property
    def nocc(self) -> int:
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("nocc is only defined for spin-balanced electron counts")
        return self.nelec[0]

    def apply(self, vec, nelec, copy=True):
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError("spin-separated fixed-sector iGCR4 ansatz got wrong nelec")
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )
        arr *= np.exp(1j * self.basis.phase(self.diagonal_params))
        arr = apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )
        return arr


@dataclass(frozen=True)
class IGCR4SpinSeparatedFixedSectorParameterization:
    norb: int
    nelec: tuple[int, int]
    max_body: int = 4
    spin_balanced: bool = False
    left_orbital_chart_alpha: object | None = None
    left_orbital_chart_beta: object | None = None
    right_orbital_chart_alpha_override: object | None = None
    right_orbital_chart_beta_override: object | None = None
    real_right_orbital_chart: bool = False

    def __post_init__(self):
        nelec = tuple(int(x) for x in self.nelec)
        if len(nelec) != 2:
            raise ValueError("nelec must be a two-tuple")
        if not (0 <= nelec[0] <= self.norb and 0 <= nelec[1] <= self.norb):
            raise ValueError("nelec must be compatible with norb")
        object.__setattr__(self, "nelec", nelec)
        if self.left_orbital_chart_alpha is None:
            object.__setattr__(
                self, "left_orbital_chart_alpha", IGCR2LeftUnitaryChart()
            )
        if self.left_orbital_chart_beta is None:
            object.__setattr__(
                self, "left_orbital_chart_beta", IGCR2LeftUnitaryChart()
            )

    @property
    def nocc(self) -> int:
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("nocc is only defined for spin-balanced electron counts")
        return self.nelec[0]

    @cached_property
    def diagonal_basis(self) -> FixedSectorDiagonalBasis:
        return make_spin_orbital_diagonal_basis(
            self.norb,
            self.nelec,
            max_body=self.max_body,
            spin_balanced=self.spin_balanced,
        )

    @property
    def right_orbital_chart_alpha(self):
        if self.right_orbital_chart_alpha_override is not None:
            return self.right_orbital_chart_alpha_override
        nocc = self.nelec[0]
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(nocc, self.norb - nocc)
        return IGCR2ReferenceOVUnitaryChart(nocc, self.norb - nocc)

    @property
    def right_orbital_chart_beta(self):
        if self.right_orbital_chart_beta_override is not None:
            return self.right_orbital_chart_beta_override
        nocc = self.nelec[1]
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(nocc, self.norb - nocc)
        return IGCR2ReferenceOVUnitaryChart(nocc, self.norb - nocc)

    @property
    def _left_orbital_chart(self):
        raise TypeError("spin-separated parameterization has alpha/beta left charts")

    @property
    def right_orbital_chart(self):
        raise TypeError("spin-separated parameterization has alpha/beta right charts")

    @property
    def n_left_alpha_orbital_rotation_params(self):
        return self.left_orbital_chart_alpha.n_params(self.norb)

    @property
    def n_left_beta_orbital_rotation_params(self):
        return self.left_orbital_chart_beta.n_params(self.norb)

    @property
    def n_left_orbital_rotation_params(self):
        return (
            self.n_left_alpha_orbital_rotation_params
            + self.n_left_beta_orbital_rotation_params
        )

    @property
    def n_diagonal_params(self):
        return self.diagonal_basis.n_params

    @property
    def n_right_alpha_orbital_rotation_params(self):
        return self.right_orbital_chart_alpha.n_params(self.norb)

    @property
    def n_right_beta_orbital_rotation_params(self):
        return self.right_orbital_chart_beta.n_params(self.norb)

    @property
    def n_right_orbital_rotation_params(self):
        return (
            self.n_right_alpha_orbital_rotation_params
            + self.n_right_beta_orbital_rotation_params
        )

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
            "left_alpha": self.n_left_alpha_orbital_rotation_params,
            "left_beta": self.n_left_beta_orbital_rotation_params,
            "diagonal": self.n_diagonal_params,
            "right_alpha": self.n_right_alpha_orbital_rotation_params,
            "right_beta": self.n_right_beta_orbital_rotation_params,
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
        n = self.n_left_alpha_orbital_rotation_params
        left_alpha = params[idx : idx + n]
        idx += n
        n = self.n_left_beta_orbital_rotation_params
        left_beta = params[idx : idx + n]
        idx += n
        n = self.n_diagonal_params
        diagonal = params[idx : idx + n]
        idx += n
        n = self.n_right_alpha_orbital_rotation_params
        right_alpha = params[idx : idx + n]
        idx += n
        n = self.n_right_beta_orbital_rotation_params
        right_beta = params[idx : idx + n]
        return left_alpha, left_beta, diagonal, right_alpha, right_beta

    def ansatz_from_parameters(
        self,
        params: np.ndarray,
    ) -> IGCR4SpinSeparatedFixedSectorAnsatz:
        left_alpha_params, left_beta_params, diagonal_params, right_alpha_params, (
            right_beta_params
        ) = self._split_params(params)

        left_alpha = self.left_orbital_chart_alpha.unitary_from_parameters(
            left_alpha_params,
            self.norb,
        )
        left_beta = self.left_orbital_chart_beta.unitary_from_parameters(
            left_beta_params,
            self.norb,
        )
        final_alpha = self.right_orbital_chart_alpha.unitary_from_parameters(
            right_alpha_params,
            self.norb,
        )
        final_beta = self.right_orbital_chart_beta.unitary_from_parameters(
            right_beta_params,
            self.norb,
        )
        right_alpha = left_alpha.conj().T @ final_alpha
        right_beta = left_beta.conj().T @ final_beta

        return IGCR4SpinSeparatedFixedSectorAnsatz(
            basis=self.diagonal_basis,
            diagonal_params=np.asarray(diagonal_params, dtype=np.float64),
            left=(left_alpha, left_beta),
            right=(right_alpha, right_beta),
            norb=self.norb,
            nelec=self.nelec,
        )

    def parameters_from_restricted_igcr4_ansatz(
        self,
        ansatz,
    ) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if self.nelec[0] != ansatz.nocc or self.nelec[1] != ansatz.nocc:
            raise ValueError("ansatz nocc does not match parameterization")

        left_params, right_phase = (
            self.left_orbital_chart_alpha.parameters_and_right_phase_from_unitary(
                ansatz.left
            )
        )
        left_alpha_params = left_params
        left_beta_params = self.left_orbital_chart_beta.parameters_from_unitary(
            ansatz.left
        )
        right_eff = _diag_unitary(right_phase) @ np.asarray(
            ansatz.right,
            dtype=np.complex128,
        )
        left_alpha_unitary = self.left_orbital_chart_alpha.unitary_from_parameters(
            left_alpha_params,
            self.norb,
        )
        left_beta_unitary = self.left_orbital_chart_beta.unitary_from_parameters(
            left_beta_params,
            self.norb,
        )
        final_alpha = _final_unitary_from_left_and_right(
            left_alpha_unitary,
            right_eff,
            self.nelec[0],
        )
        final_beta = _final_unitary_from_left_and_right(
            left_beta_unitary,
            right_eff,
            self.nelec[1],
        )
        right_alpha_params = self.right_orbital_chart_alpha.parameters_from_unitary(
            final_alpha
        )
        right_beta_params = self.right_orbital_chart_beta.parameters_from_unitary(
            final_beta
        )
        phase = restricted_igcr4_phase_vector(ansatz.diagonal, self.nelec)
        diagonal_params = self.diagonal_basis.project_phase(phase)
        return np.concatenate(
            [
                left_alpha_params,
                left_beta_params,
                diagonal_params,
                right_alpha_params,
                right_beta_params,
            ]
        )

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        if tuple(nelec) != tuple(self.nelec):
            raise ValueError(
                "fixed-sector spin-separated parameterization got wrong nelec"
            )
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=self.nelec,
                copy=True,
            )

        return func
