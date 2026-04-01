from __future__ import annotations

import numpy as np

from xquces.basis import sector_dim


def hartree_fock_state(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    dim = sector_dim(norb, nelec[0]) * sector_dim(norb, nelec[1])
    vec = np.zeros(dim, dtype=np.complex128)
    vec[0] = 1.0
    return vec