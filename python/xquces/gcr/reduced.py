from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj._unitary import ExactUnitaryChart, GaugeFixedInternalUnitaryChart
from xquces.ucj.init import GaugeFixedUCJBalancedDFSeed
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz


def _default_upper_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _validate_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
    *,
    allow_diagonal: bool,
) -> list[tuple[int, int]]:
    if pairs is None:
        if allow_diagonal:
            return list(itertools.combinations_with_replacement(range(norb), 2))
        return _default_upper_indices(norb)
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


@dataclass(frozen=True)
class _CoupledReducedSpinBalancedDiagonal:
    norb: int
    nocc: int
    same_spin_pairs: list[tuple[int, int]]
    mixed_spin_pairs: list[tuple[int, int]]

    def __post_init__(self):
        if self.nocc <= 1:
            raise ValueError("coupled reduced diagonal map requires nocc > 1")
        if self.same_spin_pairs != _default_upper_indices(self.norb):
            raise ValueError("same_spin_pairs must be all strict upper-triangular pairs")
        if self.mixed_spin_pairs != _default_upper_indices(self.norb):
            raise ValueError("mixed_spin_pairs must be all strict upper-triangular pairs")

    @property
    def n_alpha(self) -> int:
        return len(self.same_spin_pairs)

    @property
    def n_beta(self) -> int:
        return len(self.mixed_spin_pairs)

    @property
    def n_params(self) -> int:
        return self.n_alpha + self.n_beta

    def full_to_reduced(
        self,
        same_spin_params: np.ndarray,
        mixed_spin_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        same_spin_params = np.asarray(same_spin_params, dtype=np.float64)
        mixed_spin_params = np.asarray(mixed_spin_params, dtype=np.float64)

        if same_spin_params.shape != (self.norb, self.norb):
            raise ValueError("same_spin_params must have shape (norb, norb)")
        if mixed_spin_params.shape != (self.norb, self.norb):
            raise ValueError("mixed_spin_params must have shape (norb, norb)")
        if not np.allclose(same_spin_params, same_spin_params.T):
            raise ValueError("same_spin_params must be symmetric")
        if not np.allclose(mixed_spin_params, mixed_spin_params.T):
            raise ValueError("mixed_spin_params must be symmetric")

        mu = 0.5 * np.diag(same_spin_params)
        nu = np.diag(mixed_spin_params)

        alpha = np.asarray(
            [same_spin_params[p, q] for p, q in self.same_spin_pairs],
            dtype=np.float64,
        )
        beta = np.asarray(
            [mixed_spin_params[p, q] for p, q in self.mixed_spin_pairs],
            dtype=np.float64,
        )

        alpha_red = np.array(alpha, copy=True)
        beta_red = np.array(beta, copy=True)

        c_mu = 1.0 / (self.nocc - 1.0)
        c_nu_a = self.nocc / (2.0 * (self.nocc - 1.0))
        c_nu_b = -0.5

        for k, (p, q) in enumerate(self.same_spin_pairs):
            alpha_red[k] += c_mu * (mu[p] + mu[q]) + c_nu_a * (nu[p] + nu[q])

        for k, (p, q) in enumerate(self.mixed_spin_pairs):
            beta_red[k] += c_nu_b * (nu[p] + nu[q])

        return alpha_red, beta_red

    def reduced_to_full(
        self,
        alpha_red: np.ndarray,
        beta_red: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha_red = np.asarray(alpha_red, dtype=np.float64)
        beta_red = np.asarray(beta_red, dtype=np.float64)

        if alpha_red.shape != (self.n_alpha,):
            raise ValueError(f"Expected {(self.n_alpha,)}, got {alpha_red.shape}.")
        if beta_red.shape != (self.n_beta,):
            raise ValueError(f"Expected {(self.n_beta,)}, got {beta_red.shape}.")

        same = _symmetric_matrix_from_values(alpha_red, self.norb, self.same_spin_pairs)
        mixed = _symmetric_matrix_from_values(beta_red, self.norb, self.mixed_spin_pairs)

        np.fill_diagonal(same, 0.0)
        np.fill_diagonal(mixed, 0.0)

        return same, mixed


@dataclass(frozen=True)
class GCRSpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=GaugeFixedInternalUnitaryChart)
    right_orbital_chart: object = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_diagonal_params(self) -> int:
        return self.norb

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diagonal_params
            + self.n_pair_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        pairs = self.pair_indices
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n = self.n_diagonal_params
        d = np.array(params[idx:idx + n], copy=True)
        idx += n

        n = self.n_pair_params
        p = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, pairs)
        idx += n

        n = self.n_right_orbital_rotation_params
        right = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)

        return GCRAnsatz(
            diagonal=SpinRestrictedSpec(double_params=d, pair_params=p),
            left_orbital_rotation=left,
            right_orbital_rotation=right,
        )

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")

        pairs = self.pair_indices
        d = ansatz.diagonal
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = self.left_orbital_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        idx += n

        n = self.n_diagonal_params
        out[idx:idx + n] = np.asarray(d.double_params, dtype=np.float64)
        idx += n

        n = self.n_pair_params
        if n:
            out[idx:idx + n] = np.asarray([d.pair_params[p, q] for p, q in pairs], dtype=np.float64)
            idx += n

        n = self.n_right_orbital_rotation_params
        out[idx:idx + n] = self.right_orbital_chart.parameters_from_unitary(ansatz.right_orbital_rotation)

        return out

    def from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        return self.ansatz_from_parameters(params)

    def to_parameters(self, ansatz: GCRAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(ansatz)

    def apply(
        self,
        params: np.ndarray,
        vec: np.ndarray,
        *,
        nelec: tuple[int, int],
        copy: bool = True,
    ) -> np.ndarray:
        return self.ansatz_from_parameters(params).apply(vec, nelec=nelec, copy=copy)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(gcr_from_ucj_ansatz(ansatz))

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
class GCRSpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: GaugeFixedInternalUnitaryChart = field(default_factory=GaugeFixedInternalUnitaryChart)
    right_orbital_chart: ExactUnitaryChart = field(default_factory=ExactUnitaryChart)

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)
        if self.nocc <= 1:
            raise ValueError("this reduced spin-balanced parameterization requires nocc > 1")

    @property
    def same_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def mixed_spin_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def _diag_reduction(self) -> _CoupledReducedSpinBalancedDiagonal:
        return _CoupledReducedSpinBalancedDiagonal(
            norb=self.norb,
            nocc=self.nocc,
            same_spin_pairs=self.same_spin_indices,
            mixed_spin_pairs=self.mixed_spin_indices,
        )

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_jastrow_params(self) -> int:
        return self._diag_reduction.n_params

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_jastrow_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n

        n_alpha = self._diag_reduction.n_alpha
        alpha_red = np.array(params[idx:idx + n_alpha], copy=True)
        idx += n_alpha

        n_beta = self._diag_reduction.n_beta
        beta_red = np.array(params[idx:idx + n_beta], copy=True)
        idx += n_beta

        same, mixed = self._diag_reduction.reduced_to_full(alpha_red, beta_red)

        n = self.n_right_orbital_rotation_params
        right = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)

        return GCRAnsatz(
            diagonal=SpinBalancedSpec(
                same_spin_params=same,
                mixed_spin_params=mixed,
            ),
            left_orbital_rotation=left,
            right_orbital_rotation=right,
        )

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")

        alpha_red, beta_red = self._diag_reduction.full_to_reduced(
            ansatz.diagonal.same_spin_params,
            ansatz.diagonal.mixed_spin_params,
        )

        left_params = self.left_orbital_chart.parameters_from_unitary(ansatz.left_orbital_rotation)
        right_params = self.right_orbital_chart.parameters_from_unitary(ansatz.right_orbital_rotation)

        return np.concatenate(
            [
                np.asarray(left_params, dtype=np.float64),
                np.asarray(alpha_red, dtype=np.float64),
                np.asarray(beta_red, dtype=np.float64),
                np.asarray(right_params, dtype=np.float64),
            ]
        )

    def from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        return self.ansatz_from_parameters(params)

    def to_parameters(self, ansatz: GCRAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(ansatz)

    def apply(
        self,
        params: np.ndarray,
        vec: np.ndarray,
        *,
        nelec: tuple[int, int],
        copy: bool = True,
    ) -> np.ndarray:
        return self.ansatz_from_parameters(params).apply(vec, nelec=nelec, copy=copy)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.n_layers != 1:
            raise ValueError("only a single-layer UCJ ansatz can be mapped exactly to GCR")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        return self.parameters_from_ansatz(gcr_from_ucj_ansatz(ansatz))

    def ansatz_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> GCRAnsatz:
        return self.ansatz_from_parameters(self.parameters_from_ucj_ansatz(ansatz))

    def parameters_from_ccsd(
        self,
        t1: np.ndarray | None,
        t2: np.ndarray,
        *,
        n_reps: int = 1,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback: object = None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> np.ndarray:
        ucj_seed = GaugeFixedUCJBalancedDFSeed(
            t2=t2,
            t1=t1,
            n_reps=n_reps,
            tol=tol,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        ).build_ansatz()
        return self.parameters_from_ucj_ansatz(ucj_seed)

    def ansatz_from_ccsd(
        self,
        t1: np.ndarray | None,
        t2: np.ndarray,
        *,
        n_reps: int = 1,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback: object = None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> GCRAnsatz:
        return self.ansatz_from_parameters(
            self.parameters_from_ccsd(
                t1=t1,
                t2=t2,
                n_reps=n_reps,
                tol=tol,
                optimize=optimize,
                method=method,
                callback=callback,
                options=options,
                regularization=regularization,
                multi_stage_start=multi_stage_start,
                multi_stage_step=multi_stage_step,
            )
        )

    @classmethod
    def build_from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        *,
        nocc: int,
        same_spin_interaction_pairs: list[tuple[int, int]] | None = None,
        mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple["GCRSpinBalancedParameterization", GCRAnsatz, np.ndarray]:
        param = cls(
            norb=ansatz.norb,
            nocc=nocc,
            same_spin_interaction_pairs=same_spin_interaction_pairs,
            mixed_spin_interaction_pairs=mixed_spin_interaction_pairs,
        )
        x0 = param.parameters_from_ucj_ansatz(ansatz)
        gcr = param.ansatz_from_parameters(x0)
        return param, gcr, x0

    @classmethod
    def build_from_ccsd(
        cls,
        *,
        norb: int,
        nocc: int,
        t1: np.ndarray | None,
        t2: np.ndarray,
        n_reps: int = 1,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback: object = None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
        same_spin_interaction_pairs: list[tuple[int, int]] | None = None,
        mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple["GCRSpinBalancedParameterization", GCRAnsatz, np.ndarray]:
        param = cls(
            norb=norb,
            nocc=nocc,
            same_spin_interaction_pairs=same_spin_interaction_pairs,
            mixed_spin_interaction_pairs=mixed_spin_interaction_pairs,
        )
        x0 = param.parameters_from_ccsd(
            t1=t1,
            t2=t2,
            n_reps=n_reps,
            tol=tol,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )
        gcr = param.ansatz_from_parameters(x0)
        return param, gcr, x0

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)

        return func