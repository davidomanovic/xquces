from __future__ import annotations

import numpy as np
import scipy.linalg
import ffsim


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
    out = np.zeros((norb, norb), dtype=np.complex128)
    out[nocc:, :nocc] = t1
    out[:nocc, nocc:] = -t1.conj().T
    return out


def ov_generator_from_params(params: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    nvirt = norb - nocc
    ncomplex = nocc * nvirt
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (2 * ncomplex,):
        raise ValueError("wrong OV parameter vector size")
    z = params[:ncomplex].reshape(nvirt, nocc) + 1j * params[ncomplex:].reshape(nvirt, nocc)
    out = np.zeros((norb, norb), dtype=np.complex128)
    out[nocc:, :nocc] = z
    out[:nocc, nocc:] = -z.conj().T
    return out


def ov_params_from_t1(t1: np.ndarray) -> np.ndarray:
    t1 = np.asarray(t1, dtype=np.complex128)
    return np.concatenate([t1.real.reshape(-1), t1.imag.reshape(-1)])


def orbital_rotation_from_ov_params(
    params: np.ndarray,
    norb: int,
    nocc: int,
    gauge_fix: bool = True,
) -> np.ndarray:
    return unitary_from_generator(
        ov_generator_from_params(params, norb, nocc),
        gauge_fix=gauge_fix,
    )


def apply_orbital_rotation(
    vec: np.ndarray,
    orbital_rotation: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    return ffsim.apply_orbital_rotation(
        np.asarray(vec, dtype=np.complex128),
        np.asarray(orbital_rotation, dtype=np.complex128),
        norb=norb,
        nelec=nelec,
        copy=copy,
    )