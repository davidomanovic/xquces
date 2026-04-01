from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from xquces.orbitals import canonicalize_unitary


def _assert_square_matrix(a: np.ndarray, name: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def antihermitian_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (norb**2,):
        raise ValueError(f"Expected {(norb**2,)} parameters, got {params.shape}.")
    k = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for p in range(norb):
        k[p, p] = 1j * params[idx]
        idx += 1
    for p in range(norb):
        for q in range(p + 1, norb):
            a = params[idx]
            b = params[idx + 1]
            idx += 2
            z = a + 1j * b
            k[p, q] = z
            k[q, p] = -np.conjugate(z)
    return k


def parameters_from_antihermitian(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.complex128)
    _assert_square_matrix(k, "k")
    norb = k.shape[0]
    if not np.allclose(k.conj().T, -k, atol=1e-10):
        raise ValueError("k must be antihermitian")
    out = np.zeros(norb**2, dtype=np.float64)
    idx = 0
    for p in range(norb):
        out[idx] = float(np.imag(k[p, p]))
        idx += 1
    for p in range(norb):
        for q in range(p + 1, norb):
            z = k[p, q]
            out[idx] = float(np.real(z))
            out[idx + 1] = float(np.imag(z))
            idx += 2
    return out


def unitary_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    k = antihermitian_from_parameters(params, norb)
    u = scipy.linalg.expm(k)
    return canonicalize_unitary(np.asarray(u, dtype=np.complex128))


def parameters_from_unitary(u: np.ndarray) -> np.ndarray:
    u = canonicalize_unitary(np.asarray(u, dtype=np.complex128))
    _assert_square_matrix(u, "u")
    norb = u.shape[0]
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")
    k = scipy.linalg.logm(u)
    k = 0.5 * (k - k.conj().T)
    for p in range(norb):
        k[p, p] = 1j * np.imag(k[p, p])
    return parameters_from_antihermitian(k)


def _givens_pairs(norb: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for col in range(norb):
        for row in range(norb - 1, col, -1):
            pairs.append((row - 1, row))
    return pairs


def _givens_zeroing_matrix(a: complex, b: complex, tol: float = 1e-14) -> tuple[float, float, np.ndarray]:
    if abs(b) < tol:
        c = 1.0
        s = 0.0j
    elif abs(a) < tol:
        c = 0.0
        s = np.exp(-1j * np.angle(b))
    else:
        r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        c = abs(a) / r
        s = a * np.conjugate(b) / (abs(a) * r)
    theta = float(np.arctan2(abs(s), c))
    phi = 0.0 if abs(s) < tol else float(np.angle(s))
    g = np.array([[c, s], [-np.conjugate(s), c]], dtype=np.complex128)
    return theta, phi, g


def _givens_dagger_from_angles(theta: float, phi: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.exp(1j * phi) * np.sin(theta)
    return np.array([[c, -s], [np.conjugate(s), c]], dtype=np.complex128)


def exact_internal_gauge_fixed_parameters_from_unitary(u: np.ndarray) -> np.ndarray:
    u = canonicalize_unitary(np.asarray(u, dtype=np.complex128))
    _assert_square_matrix(u, "u")
    norb = u.shape[0]
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")
    r = np.array(u, copy=True)
    params = np.zeros(norb * (norb - 1), dtype=np.float64)
    idx = 0
    for col in range(norb):
        for row in range(norb - 1, col, -1):
            theta, phi, g = _givens_zeroing_matrix(r[row - 1, col], r[row, col])
            r[[row - 1, row], :] = g @ r[[row - 1, row], :]
            params[idx] = theta
            params[idx + 1] = phi
            idx += 2
    return params


def exact_internal_gauge_fixed_unitary_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    expected = norb * (norb - 1)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)} parameters, got {params.shape}.")
    q = np.eye(norb, dtype=np.complex128)
    idx = 0
    for p, qrow in _givens_pairs(norb):
        theta = float(params[idx])
        phi = float(params[idx + 1])
        idx += 2
        gdag = _givens_dagger_from_angles(theta, phi)
        block = np.eye(norb, dtype=np.complex128)
        block[np.ix_([p, qrow], [p, qrow])] = gdag
        q = q @ block
    return canonicalize_unitary(q)


def ov_kappa_from_parameters(params: np.ndarray, nocc: int, nvirt: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    expected = 2 * nocc * nvirt
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)} parameters, got {params.shape}.")
    t = np.zeros((nvirt, nocc), dtype=np.complex128)
    idx = 0
    for a in range(nvirt):
        for i in range(nocc):
            t[a, i] = params[idx] + 1j * params[idx + 1]
            idx += 2
    kappa = np.zeros((nocc + nvirt, nocc + nvirt), dtype=np.complex128)
    kappa[nocc:, :nocc] = t
    kappa[:nocc, nocc:] = -t.conj().T
    return kappa


def ov_unitary_from_parameters(params: np.ndarray, nocc: int, nvirt: int) -> np.ndarray:
    kappa = ov_kappa_from_parameters(params, nocc, nvirt)
    u = scipy.linalg.expm(kappa)
    return canonicalize_unitary(np.asarray(u, dtype=np.complex128))


def exact_ov_parameters_from_unitary(u: np.ndarray, nocc: int, nvirt: int) -> np.ndarray:
    u = canonicalize_unitary(np.asarray(u, dtype=np.complex128))
    _assert_square_matrix(u, "u")
    norb = nocc + nvirt
    if u.shape != (norb, norb):
        raise ValueError(f"u must have shape {(norb, norb)}")
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")
    c_occ = u[:, :nocc]
    a = c_occ[:nocc, :]
    b = c_occ[nocc:, :]
    if np.linalg.cond(a) > 1e12:
        raise ValueError("occupied block is too ill-conditioned for OV gauge fixing")
    z = b @ np.linalg.inv(a)
    umat, svals, vh = np.linalg.svd(z, full_matrices=False)
    theta = np.arctan(svals)
    t = umat @ np.diag(theta) @ vh
    params = np.zeros(2 * nocc * nvirt, dtype=np.float64)
    idx = 0
    for aidx in range(nvirt):
        for i in range(nocc):
            params[idx] = float(np.real(t[aidx, i]))
            params[idx + 1] = float(np.imag(t[aidx, i]))
            idx += 2
    return params


@dataclass(frozen=True)
class AntiHermitianUnitaryChart:
    def n_params(self, norb: int) -> int:
        return norb**2

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return unitary_from_parameters(params, norb)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return parameters_from_unitary(u)


@dataclass(frozen=True)
class GaugeFixedInternalUnitaryChart:
    def n_params(self, norb: int) -> int:
        return norb * (norb - 1)

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return exact_internal_gauge_fixed_unitary_from_parameters(params, norb)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return exact_internal_gauge_fixed_parameters_from_unitary(u)


@dataclass(frozen=True)
class OccupiedVirtualUnitaryChart:
    nocc: int
    nvirt: int

    def __post_init__(self):
        if self.nocc < 0 or self.nvirt < 0:
            raise ValueError("nocc and nvirt must be nonnegative")

    @property
    def norb(self) -> int:
        return self.nocc + self.nvirt

    def n_params(self, norb: int | None = None) -> int:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return 2 * self.nocc * self.nvirt

    def unitary_from_parameters(self, params: np.ndarray, norb: int | None = None) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return ov_unitary_from_parameters(params, self.nocc, self.nvirt)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return exact_ov_parameters_from_unitary(u, self.nocc, self.nvirt)