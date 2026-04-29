from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gates import apply_igcr4_pair_spin
from xquces.gcr.igcr2 import (
    IGCR2LeftUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
)


def _validate_spin_pairs(spin_pairs, norb: int) -> tuple[tuple[int, int], ...]:
    pairs = tuple(tuple(int(x) for x in pair) for pair in spin_pairs)
    used = set()
    out = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each spin pair must contain exactly two orbitals")
        p, q = pair
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("spin pair index out of bounds")
        if p == q:
            raise ValueError("spin pair orbitals must be distinct")
        if p in used or q in used:
            raise ValueError("spin pairs must be disjoint")
        used.add(p)
        used.add(q)
        out.append((p, q))
    return tuple(out)


def default_adjacent_spin_pairs(norb: int) -> tuple[tuple[int, int], ...]:
    return tuple((p, p + 1) for p in range(0, norb - 1, 2))


def _upper_values_to_matrix(values: np.ndarray, n_pairs: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    expected = n_pairs * (n_pairs - 1) // 2
    if values.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {values.shape}.")
    out = np.zeros((n_pairs, n_pairs), dtype=np.float64)
    idx = 0
    for a in range(n_pairs):
        for b in range(a + 1, n_pairs):
            out[a, b] = values[idx]
            out[b, a] = values[idx]
            idx += 1
    return out


def _matrix_upper_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    out = []
    for a in range(matrix.shape[0]):
        for b in range(a + 1, matrix.shape[0]):
            out.append(matrix[a, b])
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class IGCR4PairSpinSpec:
    theta_singlet: np.ndarray
    theta_triplet: np.ndarray
    spin_pairs: tuple[tuple[int, int], ...]

    @property
    def n_spin_pairs(self) -> int:
        return len(self.spin_pairs)

    def validate(self, norb: int) -> "IGCR4PairSpinSpec":
        spin_pairs = _validate_spin_pairs(self.spin_pairs, norb)
        n_pairs = len(spin_pairs)
        theta_singlet = np.asarray(self.theta_singlet, dtype=np.float64)
        theta_triplet = np.asarray(self.theta_triplet, dtype=np.float64)
        if theta_singlet.shape != (n_pairs, n_pairs):
            raise ValueError("theta_singlet must have shape (n_spin_pairs, n_spin_pairs)")
        if theta_triplet.shape != (n_pairs, n_pairs):
            raise ValueError("theta_triplet must have shape (n_spin_pairs, n_spin_pairs)")
        theta_singlet = 0.5 * (theta_singlet + theta_singlet.T)
        theta_triplet = 0.5 * (theta_triplet + theta_triplet.T)
        np.fill_diagonal(theta_singlet, 0.0)
        np.fill_diagonal(theta_triplet, 0.0)
        return IGCR4PairSpinSpec(theta_singlet, theta_triplet, spin_pairs)


@dataclass(frozen=True)
class IGCR4PairSpinAnsatz:
    pair_spin: IGCR4PairSpinSpec
    left: np.ndarray
    right: np.ndarray
    nocc: int
    norb: int

    def __post_init__(self):
        self.pair_spin.validate(self.norb)

    def apply(self, vec, nelec, copy=True):
        spec = self.pair_spin.validate(self.norb)
        return apply_igcr4_pair_spin(
            vec,
            spec.theta_singlet,
            spec.theta_triplet,
            self.norb,
            nelec,
            np.asarray(spec.spin_pairs, dtype=np.uintp),
            left_orbital_rotation=self.left,
            right_orbital_rotation=self.right,
            copy=copy,
        )


@dataclass(frozen=True)
class IGCR4PairSpinParameterization:
    norb: int
    nocc: int
    spin_pairs: tuple[tuple[int, int], ...] | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart: object | None = None

    def __post_init__(self):
        if self.norb < 0 or self.nocc < 0 or self.nocc > self.norb:
            raise ValueError("invalid norb/nocc")
        spin_pairs = default_adjacent_spin_pairs(self.norb) if self.spin_pairs is None else self.spin_pairs
        _validate_spin_pairs(spin_pairs, self.norb)

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    @property
    def validated_spin_pairs(self) -> tuple[tuple[int, int], ...]:
        spin_pairs = default_adjacent_spin_pairs(self.norb) if self.spin_pairs is None else self.spin_pairs
        return _validate_spin_pairs(spin_pairs, self.norb)

    @property
    def n_spin_pairs(self) -> int:
        return len(self.validated_spin_pairs)

    @property
    def n_pair_interactions(self) -> int:
        return self.n_spin_pairs * (self.n_spin_pairs - 1) // 2

    @property
    def n_pair_spin_params(self) -> int:
        return 2 * self.n_pair_interactions

    @property
    def right_chart(self):
        if self.right_orbital_chart is not None:
            return self.right_orbital_chart
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.nvirt)

    @property
    def n_left_params(self) -> int:
        return self.left_orbital_chart.n_params(self.norb)

    @property
    def n_right_params(self) -> int:
        return self.right_chart.n_params(self.norb)

    @property
    def n_params(self) -> int:
        return self.n_left_params + self.n_pair_spin_params + self.n_right_params

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_params,
            "pair_spin_singlet": self.n_pair_interactions,
            "pair_spin_triplet": self.n_pair_interactions,
            "right": self.n_right_params,
            "total": self.n_params,
        }

    def _split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        n_l = self.n_left_params
        n_i = self.n_pair_interactions
        left = params[:n_l]
        singlet = params[n_l : n_l + n_i]
        triplet = params[n_l + n_i : n_l + 2 * n_i]
        right = params[n_l + 2 * n_i :]
        return left, singlet, triplet, right

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4PairSpinAnsatz:
        left_params, singlet_params, triplet_params, right_params = self._split_params(params)
        left = self.left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        right = self.right_chart.unitary_from_parameters(right_params, self.norb)
        spec = IGCR4PairSpinSpec(
            theta_singlet=_upper_values_to_matrix(singlet_params, self.n_spin_pairs),
            theta_triplet=_upper_values_to_matrix(triplet_params, self.n_spin_pairs),
            spin_pairs=self.validated_spin_pairs,
        )
        return IGCR4PairSpinAnsatz(
            pair_spin=spec,
            left=left,
            right=right,
            nocc=self.nocc,
            norb=self.norb,
        )

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def fun(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec, copy=True)
        return fun

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float64)

    def parameters_from_ansatz(self, ansatz: IGCR4PairSpinAnsatz) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz.norb does not match parameterization")
        spec = ansatz.pair_spin.validate(self.norb)
        if spec.spin_pairs != self.validated_spin_pairs:
            raise ValueError("ansatz spin_pairs do not match parameterization")
        left = self.left_orbital_chart.parameters_from_unitary(ansatz.left)
        singlet = _matrix_upper_values(spec.theta_singlet)
        triplet = _matrix_upper_values(spec.theta_triplet)
        right = self.right_chart.parameters_from_unitary(ansatz.right)
        return np.concatenate([left, singlet, triplet, right])

    @classmethod
    def default(
        cls,
        norb: int,
        nocc: int,
        *,
        spin_pairs: tuple[tuple[int, int], ...] | None = None,
        **kwargs,
    ) -> "IGCR4PairSpinParameterization":
        return cls(norb=norb, nocc=nocc, spin_pairs=spin_pairs, **kwargs)
