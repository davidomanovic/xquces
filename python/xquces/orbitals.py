from __future__ import annotations

import numpy as np
import scipy.linalg

from .basis import flatten_state, occ_rows, reshape_state


def canonicalize_unitary(u: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("unitary must be square")
    out = np.array(u, copy=True)
    norb = out.shape[0]
    phases = np.ones(norb, dtype=np.complex128)
    for j in range(norb):
        col = out[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > tol:
            phases[j] = np.exp(-1j * np.angle(val))
    return out @ np.diag(phases)


def unitary_from_generator(kappa: np.ndarray, gauge_fix: bool = True) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=np.complex128)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError("generator must be square")
    if not np.allclose(kappa.conj().T, -kappa):
        raise ValueError("generator must be anti-Hermitian")
    u = scipy.linalg.expm(kappa)
    return canonicalize_unitary(u) if gauge_fix else u


def ov_generator_from_t1(t1: np.ndarray) -> np.ndarray:
    t1 = np.asarray(t1, dtype=np.complex128)
    if t1.ndim != 2:
        raise ValueError("t1 must be a matrix")
    nocc, nvirt = t1.shape
    norb = nocc + nvirt
    kappa = np.zeros((norb, norb), dtype=np.complex128)
    kappa[nocc:, :nocc] = t1
    kappa[:nocc, nocc:] = -t1.conj().T
    return kappa


def ov_unitary_from_t1(t1: np.ndarray, gauge_fix: bool = True) -> np.ndarray:
    return unitary_from_generator(ov_generator_from_t1(t1), gauge_fix=gauge_fix)


def _slater_transform_matrix(u: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    dim = len(occ)
    out = np.zeros((dim, dim), dtype=np.complex128)
    for i, bra in enumerate(occ):
        for j, ket in enumerate(occ):
            out[i, j] = np.linalg.det(u[np.ix_(bra, ket)])
    return out


def apply_orbital_rotation(
    vec: np.ndarray,
    orbital_rotation: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    mat = reshape_state(arr, norb, nelec)
    nalpha, nbeta = nelec
    ta = _slater_transform_matrix(orbital_rotation, norb, nalpha)
    tb = _slater_transform_matrix(orbital_rotation, norb, nbeta)
    rotated = ta @ mat @ tb.T
    return flatten_state(rotated)