from __future__ import annotations

import itertools
import math

import numpy as np


def occ_rows(norb: int, nocc: int) -> np.ndarray:
    if nocc < 0 or nocc > norb:
        raise ValueError("invalid occupation number")
    if nocc == 0:
        return np.zeros((1, 0), dtype=np.uintp)
    return np.asarray(list(itertools.combinations(range(norb), nocc)), dtype=np.uintp)


def sector_dim(norb: int, nocc: int) -> int:
    return math.comb(norb, nocc)


def sector_shape(norb: int, nelec: tuple[int, int]) -> tuple[int, int]:
    return sector_dim(norb, nelec[0]), sector_dim(norb, nelec[1])


def reshape_state(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    arr = np.asarray(vec, dtype=np.complex128)
    if arr.size != dim_a * dim_b:
        raise ValueError("state size does not match norb and nelec")
    return np.ascontiguousarray(arr.reshape(dim_a, dim_b))


def flatten_state(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat, dtype=np.complex128).reshape(-1)