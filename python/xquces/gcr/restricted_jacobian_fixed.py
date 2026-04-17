from __future__ import annotations

from functools import cache

import numpy as np

import xquces.gcr.restricted_jacobian as _restricted_jacobian
from xquces.basis import occ_rows


@cache
def _sector_rep_index(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    if nocc == 0:
        return np.zeros((2, 1, 1, 0, 0), dtype=np.int64)
    dim = len(occ)
    rows = np.broadcast_to(occ[:, None, :, None], (dim, dim, nocc, nocc))
    cols = np.broadcast_to(occ[None, :, None, :], (dim, dim, nocc, nocc))
    return np.stack([rows, cols], axis=0)


def _sector_representation(u: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    if nocc == 0:
        return np.ones((1, 1), dtype=np.complex128)
    index = _sector_rep_index(norb, nocc)
    submats = u[index[0], index[1]]
    return np.linalg.det(submats)


_restricted_jacobian._sector_rep_index = _sector_rep_index
_restricted_jacobian._sector_representation = _sector_representation

make_restricted_gcr_jacobian = _restricted_jacobian.make_restricted_gcr_jacobian

__all__ = ["make_restricted_gcr_jacobian"]
