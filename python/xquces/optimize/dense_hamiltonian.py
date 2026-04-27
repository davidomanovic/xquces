from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse.linalg import LinearOperator


def build_dense_hamiltonian(
    hamiltonian: LinearOperator,
    *,
    n_workers: int | None = None,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"hamiltonian must be square. Got shape {hamiltonian.shape}.")

    dim = hamiltonian.shape[0]
    if dtype is None:
        ham_dtype = getattr(hamiltonian, "dtype", None)
        dtype = np.result_type(
            np.complex128 if ham_dtype is None else ham_dtype,
            np.complex128,
        )

    eye = np.eye(dim, dtype=dtype)

    def _column(i: int) -> np.ndarray:
        return np.asarray(hamiltonian @ eye[:, i], dtype=dtype).reshape(dim)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        cols = list(executor.map(_column, range(dim)))

    return np.column_stack(cols)


def make_dense_hamiltonian(
    hamiltonian: LinearOperator,
    *,
    n_workers: int | None = None,
    dtype: np.dtype | type | None = None,
) -> LinearOperator:
    dense = build_dense_hamiltonian(
        hamiltonian,
        n_workers=n_workers,
        dtype=dtype,
    )
    return LinearOperator(
        shape=dense.shape,
        matvec=lambda vec: dense @ vec,
        matmat=lambda mat: dense @ mat,
        dtype=dense.dtype,
    )
