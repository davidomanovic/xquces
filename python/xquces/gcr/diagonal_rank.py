from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagonalRank:
    n_rows: int
    n_cols: int
    rank_mod_constant: int
    max_rank_mod_constant: int
    singular_values: np.ndarray

    @property
    def deficit(self) -> int:
        return self.max_rank_mod_constant - self.rank_mod_constant


def determinant_occupations(
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_alpha, n_beta = nelec
    spin_rows = []
    spatial_rows = []
    double_rows = []
    for alpha in itertools.combinations(range(norb), n_alpha):
        alpha_set = set(alpha)
        for beta in itertools.combinations(range(norb), n_beta):
            beta_set = set(beta)
            spin = np.zeros(2 * norb, dtype=np.float64)
            spatial = np.zeros(norb, dtype=np.float64)
            double = np.zeros(norb, dtype=np.float64)
            for p in alpha_set:
                spin[p] = 1.0
                spatial[p] += 1.0
            for p in beta_set:
                spin[norb + p] = 1.0
                spatial[p] += 1.0
            for p in alpha_set & beta_set:
                double[p] = 1.0
            spin_rows.append(spin)
            spatial_rows.append(spatial)
            double_rows.append(double)
    return (
        np.asarray(spin_rows, dtype=np.float64),
        np.asarray(spatial_rows, dtype=np.float64),
        np.asarray(double_rows, dtype=np.float64),
    )


def rank_mod_constant(
    features: np.ndarray,
    *,
    rtol: float = 1e-10,
) -> tuple[int, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if features.size == 0:
        return 0, np.zeros(0, dtype=np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(centered, compute_uv=False)
    if svals.size == 0 or svals[0] == 0.0:
        return 0, svals
    return int(np.sum(svals > float(rtol) * svals[0])), svals


def spin_flip_orbit_count(norb: int, nelec: tuple[int, int]) -> int:
    n_alpha, n_beta = nelec
    if n_alpha != n_beta:
        return 0
    n_fixed = 0
    n_total = 0
    for alpha in itertools.combinations(range(norb), n_alpha):
        alpha_set = set(alpha)
        for beta in itertools.combinations(range(norb), n_beta):
            beta_set = set(beta)
            n_total += 1
            if alpha_set == beta_set:
                n_fixed += 1
    return (n_total + n_fixed) // 2


def spin_restricted_igcr_diagonal_features(
    norb: int,
    nelec: tuple[int, int],
    *,
    max_body: int = 4,
) -> np.ndarray:
    if max_body < 1 or max_body > 4:
        raise ValueError("max_body must be in [1, 4]")
    _, n_rows, d_rows = determinant_occupations(norb, nelec)
    cols = []

    if max_body >= 1:
        for p in range(norb):
            cols.append(d_rows[:, p])

    if max_body >= 2:
        for p, q in itertools.combinations(range(norb), 2):
            cols.append(n_rows[:, p] * n_rows[:, q])

    if max_body >= 3:
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    cols.append(d_rows[:, p] * n_rows[:, q])
        for p, q, r in itertools.combinations(range(norb), 3):
            cols.append(n_rows[:, p] * n_rows[:, q] * n_rows[:, r])

    if max_body >= 4:
        for p, q in itertools.combinations(range(norb), 2):
            cols.append(d_rows[:, p] * d_rows[:, q])
        for p in range(norb):
            others = [q for q in range(norb) if q != p]
            for q, r in itertools.combinations(others, 2):
                cols.append(d_rows[:, p] * n_rows[:, q] * n_rows[:, r])
        for p, q, r, s in itertools.combinations(range(norb), 4):
            cols.append(n_rows[:, p] * n_rows[:, q] * n_rows[:, r] * n_rows[:, s])

    if not cols:
        return np.zeros((n_rows.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols)


def spin_orbital_diagonal_features(
    norb: int,
    nelec: tuple[int, int],
    *,
    max_body: int = 4,
    spin_balanced: bool = False,
) -> np.ndarray:
    if max_body < 1:
        raise ValueError("max_body must be positive")
    spin_rows, _, _ = determinant_occupations(norb, nelec)
    spin_orbs = tuple(range(2 * norb))
    flip = {i: (i + norb if i < norb else i - norb) for i in spin_orbs}
    seen: set[tuple[int, ...]] = set()
    cols = []

    for body in range(1, max_body + 1):
        for monomial in itertools.combinations(spin_orbs, body):
            monomial = tuple(monomial)
            if spin_balanced:
                flipped = tuple(sorted(flip[i] for i in monomial))
                key = min(monomial, flipped)
                if key in seen:
                    continue
                seen.add(key)
                orbit = tuple(sorted({monomial, flipped}))
                values = np.zeros(spin_rows.shape[0], dtype=np.float64)
                for term in orbit:
                    values += np.prod(spin_rows[:, term], axis=1)
                cols.append(values)
            else:
                cols.append(np.prod(spin_rows[:, monomial], axis=1))

    if not cols:
        return np.zeros((spin_rows.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols)


def diagonal_rank(features: np.ndarray, *, rtol: float = 1e-10) -> DiagonalRank:
    features = np.asarray(features, dtype=np.float64)
    rank, svals = rank_mod_constant(features, rtol=rtol)
    return DiagonalRank(
        n_rows=features.shape[0],
        n_cols=features.shape[1],
        rank_mod_constant=rank,
        max_rank_mod_constant=max(features.shape[0] - 1, 0),
        singular_values=svals,
    )
