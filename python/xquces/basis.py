from __future__ import annotations

import math
from functools import cache

import numpy as np
from pyscf.fci import cistring


def occ_rows(norb: int, nocc: int) -> np.ndarray:
    if nocc < 0 or nocc > norb:
        raise ValueError("invalid occupation number")
    if nocc == 0:
        return np.zeros((1, 0), dtype=np.uintp)
    return np.asarray(cistring.gen_occslst(range(norb), nocc), dtype=np.uintp)


@cache
def occ_indicator_rows(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    out = np.zeros((len(occ), norb), dtype=np.uint8)
    if occ.size:
        out[np.arange(len(occ))[:, None], occ] = 1
    return out


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
