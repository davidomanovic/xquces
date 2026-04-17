from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from xquces.basis import occ_indicator_rows


def _svd_rank(singular_values: np.ndarray, shape: tuple[int, int], tol: float) -> int:
    if singular_values.size == 0:
        return 0
    cutoff = tol * max(shape) * singular_values[0]
    return int(np.sum(singular_values > cutoff))


def _orthonormal_span(mat: np.ndarray, tol: float) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("mat must be two-dimensional")
    if mat.shape[1] == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    rank = _svd_rank(s, mat.shape, tol)
    return np.array(u[:, :rank], copy=True, dtype=np.float64)


@cache
def restricted_number_arrays(norb: int, nocc: int) -> tuple[np.ndarray, np.ndarray]:
    occ_a = occ_indicator_rows(norb, nocc).astype(np.float64, copy=False)
    occ_b = occ_indicator_rows(norb, nocc).astype(np.float64, copy=False)
    n = (occ_a[:, None, :] + occ_b[None, :, :]).reshape(-1, norb)
    d = (occ_a[:, None, :] * occ_b[None, :, :]).reshape(-1, norb)
    return np.array(n, copy=True), np.array(d, copy=True)


@cache
def constant_feature_matrix(norb: int, nocc: int) -> np.ndarray:
    n, _ = restricted_number_arrays(norb, nocc)
    return np.ones((n.shape[0], 1), dtype=np.float64)


@cache
def one_body_feature_matrix(norb: int, nocc: int) -> np.ndarray:
    n, _ = restricted_number_arrays(norb, nocc)
    return np.array(n, copy=True)


@cache
def pair_feature_matrix(norb: int, nocc: int) -> np.ndarray:
    n, _ = restricted_number_arrays(norb, nocc)
    cols = [n[:, p] * n[:, q] for p in range(norb) for q in range(p + 1, norb)]
    if not cols:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols)


@cache
def cubic_feature_matrix(norb: int, nocc: int) -> np.ndarray:
    n, d = restricted_number_arrays(norb, nocc)
    cols = []
    for p in range(norb):
        for q in range(norb):
            if p != q:
                cols.append(d[:, p] * n[:, q])
    for p in range(norb):
        for q in range(p + 1, norb):
            for r in range(q + 1, norb):
                cols.append(n[:, p] * n[:, q] * n[:, r])
    if not cols:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols)


@cache
def quartic_feature_matrix(norb: int, nocc: int) -> np.ndarray:
    n, d = restricted_number_arrays(norb, nocc)
    cols = []
    for p in range(norb):
        for q in range(p + 1, norb):
            cols.append(d[:, p] * d[:, q])
    for p in range(norb):
        for q in range(norb):
            if q == p:
                continue
            for r in range(q + 1, norb):
                if r == p:
                    continue
                cols.append(d[:, p] * n[:, q] * n[:, r])
    for p in range(norb):
        for q in range(p + 1, norb):
            for r in range(q + 1, norb):
                for s in range(r + 1, norb):
                    cols.append(n[:, p] * n[:, q] * n[:, r] * n[:, s])
    if not cols:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols)


@dataclass(frozen=True)
class FixedSectorDiagonalBasis:
    full_feature_matrix: np.ndarray
    lower_feature_matrix: np.ndarray | None = None
    tol: float = 1e-10

    def __post_init__(self):
        full = np.asarray(self.full_feature_matrix, dtype=np.float64)
        if full.ndim != 2:
            raise ValueError("full_feature_matrix must be two-dimensional")
        if self.lower_feature_matrix is None:
            lower = np.zeros((full.shape[0], 0), dtype=np.float64)
        else:
            lower = np.asarray(self.lower_feature_matrix, dtype=np.float64)
            if lower.ndim != 2:
                raise ValueError("lower_feature_matrix must be two-dimensional")
            if lower.shape[0] != full.shape[0]:
                raise ValueError("feature matrices must have the same number of rows")

        q_lower = _orthonormal_span(lower, self.tol)
        proj = q_lower @ (q_lower.T @ full) if q_lower.shape[1] else np.zeros_like(full)
        if lower.shape[1]:
            lower_transfer, *_ = np.linalg.lstsq(lower, proj, rcond=None)
        else:
            lower_transfer = np.zeros((0, full.shape[1]), dtype=np.float64)
        residual = full - proj

        if residual.shape[1] == 0:
            physical = np.zeros((full.shape[1], 0), dtype=np.float64)
        else:
            _, s, vh = np.linalg.svd(residual, full_matrices=False)
            rank = _svd_rank(s, residual.shape, self.tol)
            physical = np.array(vh[:rank].T, copy=True, dtype=np.float64)
            for j in range(physical.shape[1]):
                col = physical[:, j]
                pivot = int(np.argmax(np.abs(col)))
                if abs(col[pivot]) > 1e-14 and col[pivot] < 0:
                    physical[:, j] *= -1.0

        object.__setattr__(self, "_full_feature_matrix", full)
        object.__setattr__(self, "_lower_feature_matrix", lower)
        object.__setattr__(self, "_lower_transfer", np.array(lower_transfer, copy=True, dtype=np.float64))
        object.__setattr__(self, "_physical_basis", physical)

    @property
    def n_full(self) -> int:
        return self._full_feature_matrix.shape[1]

    @property
    def n_lower(self) -> int:
        return self._lower_feature_matrix.shape[1]

    @property
    def n_params(self) -> int:
        return self._physical_basis.shape[1]

    @property
    def lower_transfer(self) -> np.ndarray:
        return np.array(self._lower_transfer, copy=True)

    @property
    def physical_basis(self) -> np.ndarray:
        return np.array(self._physical_basis, copy=True)

    @property
    def reduced_feature_matrix(self) -> np.ndarray:
        return self._full_feature_matrix @ self._physical_basis

    def full_from_reduced(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self._physical_basis @ params

    def reduce_full(self, full_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        full_values = np.asarray(full_values, dtype=np.float64)
        if full_values.shape != (self.n_full,):
            raise ValueError(f"Expected {(self.n_full,)}, got {full_values.shape}.")
        lower_values = self._lower_transfer @ full_values
        reduced_values = self._physical_basis.T @ full_values
        return lower_values, reduced_values
