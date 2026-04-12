from __future__ import annotations

import cmath
import math
from functools import cache

import numpy as np
import scipy.linalg
from pyscf.fci import cistring

from xquces._lib import apply_givens_rotation_in_place, apply_phase_shift_in_place

try:
    from ffsim import apply_orbital_rotation as _ffsim_apply_orbital_rotation
except Exception:  # pragma: no cover - optional acceleration path
    _ffsim_apply_orbital_rotation = None


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


def _zrotg(a: complex, b: complex, tol: float = 1e-12) -> tuple[float, complex]:
    if cmath.isclose(a, 0.0, abs_tol=tol):
        if cmath.isclose(b, 0.0, abs_tol=tol):
            return 1.0, 0.0j
        return 0.0, np.exp(-1j * np.angle(b))
    if cmath.isclose(b, 0.0, abs_tol=tol):
        return 1.0, 0.0j
    r = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
    c = abs(a) / r
    alpha = a / abs(a)
    s = alpha * np.conjugate(b) / r
    return float(c), complex(s)


def _zrot(x: np.ndarray, y: np.ndarray, c: float, s: complex) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    x_new = c * x + s * y
    y_new = c * y - np.conjugate(s) * x
    return x_new, y_new


def givens_decomposition(mat: np.ndarray) -> tuple[list[tuple[float, complex, int, int]], np.ndarray]:
    mat = np.asarray(mat, dtype=np.complex128)
    n, m = mat.shape
    if n != m:
        raise ValueError("orbital_rotation must be square")
    current = mat.copy()
    left_rotations: list[tuple[float, complex, int, int]] = []
    right_rotations: list[tuple[float, complex, int, int]] = []

    for i in range(n - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                if not cmath.isclose(current[row, target_index], 0.0):
                    c, s = _zrotg(
                        current[row, target_index + 1],
                        current[row, target_index],
                    )
                    right_rotations.append((c, s, target_index + 1, target_index))
                    col1, col2 = _zrot(
                        current[:, target_index + 1],
                        current[:, target_index],
                        c,
                        s,
                    )
                    current[:, target_index + 1] = col1
                    current[:, target_index] = col2
        else:
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                if not cmath.isclose(current[target_index, col], 0.0):
                    c, s = _zrotg(
                        current[target_index - 1, col],
                        current[target_index, col],
                    )
                    left_rotations.append((c, s, target_index - 1, target_index))
                    row1, row2 = _zrot(
                        current[target_index - 1],
                        current[target_index],
                        c,
                        s,
                    )
                    current[target_index - 1] = row1
                    current[target_index] = row2

    for c, s, i, j in reversed(left_rotations):
        c, s = _zrotg(c * current[j, j], np.conjugate(s) * current[i, i])
        right_rotations.append((c, -np.conjugate(s), i, j))
        givens_mat = np.array([[c, -s], [np.conjugate(s), c]], dtype=np.complex128)
        givens_mat[:, 0] *= current[i, i]
        givens_mat[:, 1] *= current[j, j]
        c, s = _zrotg(givens_mat[1, 1], givens_mat[1, 0])
        new_givens_mat = np.array([[c, s], [-np.conjugate(s), c]], dtype=np.complex128)
        phase_matrix = givens_mat @ new_givens_mat
        current[i, i] = phase_matrix[0, 0]
        current[j, j] = phase_matrix[1, 1]

    return right_rotations, np.diagonal(current).copy()


@cache
def _shifted_orbitals(norb: int, target_orbs: tuple[int, ...]) -> np.ndarray:
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals


@cache
def _zero_one_subspace_indices(
    norb: int,
    nocc: int,
    target_orbs: tuple[int, int],
) -> np.ndarray:
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n00 = math.comb(norb - 2, nocc)
    n11 = math.comb(norb - 2, nocc - 2) if nocc >= 2 else 0
    return indices[n00 : len(indices) - n11].astype(np.uintp, copy=False)


@cache
def _one_subspace_indices(
    norb: int,
    nocc: int,
    target_orbs: tuple[int, ...],
) -> np.ndarray:
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = math.comb(norb, nocc)
    if nocc >= len(target_orbs):
        n0 -= math.comb(norb - len(target_orbs), nocc - len(target_orbs))
    return indices[n0:].astype(np.uintp, copy=False)


def _apply_orbital_rotation_adjacent_spin_in_place(
    vec: np.ndarray,
    c: float,
    s: complex,
    target_orbs: tuple[int, int],
    norb: int,
    nocc: int,
) -> None:
    i, j = target_orbs
    if abs(i - j) != 1:
        raise ValueError("target orbitals must be adjacent")
    indices = _zero_one_subspace_indices(norb, nocc, target_orbs)
    half = len(indices) // 2
    slice1 = np.ascontiguousarray(indices[:half], dtype=np.uintp)
    slice2 = np.ascontiguousarray(indices[half:], dtype=np.uintp)
    apply_givens_rotation_in_place(
        vec,
        c,
        float(np.real(s)),
        float(np.imag(s)),
        slice1,
        slice2,
    )


def _get_givens_decomposition(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
):
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        decomp = givens_decomposition(mat)
        return decomp, decomp
    mat_a, mat_b = mat
    decomp_a = None if mat_a is None else givens_decomposition(mat_a)
    decomp_b = None if mat_b is None else givens_decomposition(mat_b)
    return decomp_a, decomp_b


def _apply_orbital_rotation_spinless(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: int,
) -> np.ndarray:
    givens_rotations, phase_shifts = givens_decomposition(mat)
    vec = np.ascontiguousarray(vec.reshape((-1, 1)))
    for c, s, i, j in givens_rotations:
        _apply_orbital_rotation_adjacent_spin_in_place(
            vec,
            c,
            np.conjugate(s),
            (i, j),
            norb,
            nelec,
        )
    for i, phase_shift in enumerate(phase_shifts):
        indices = np.ascontiguousarray(_one_subspace_indices(norb, nelec, (i,)), dtype=np.uintp)
        apply_phase_shift_in_place(
            vec,
            float(np.real(phase_shift)),
            float(np.imag(phase_shift)),
            indices,
        )
    return vec.reshape(-1)


def _apply_orbital_rotation_spinful(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    givens_decomp_a, givens_decomp_b = _get_givens_decomposition(mat)
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    vec = np.ascontiguousarray(vec.reshape((dim_a, dim_b)))

    if givens_decomp_a is not None:
        givens_rotations, phase_shifts = givens_decomp_a
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec,
                c,
                np.conjugate(s),
                (i, j),
                norb,
                n_alpha,
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = np.ascontiguousarray(_one_subspace_indices(norb, n_alpha, (i,)), dtype=np.uintp)
            apply_phase_shift_in_place(
                vec,
                float(np.real(phase_shift)),
                float(np.imag(phase_shift)),
                indices,
            )

    if givens_decomp_b is not None:
        vec = np.ascontiguousarray(vec.T)
        givens_rotations, phase_shifts = givens_decomp_b
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec,
                c,
                np.conjugate(s),
                (i, j),
                norb,
                n_beta,
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = np.ascontiguousarray(_one_subspace_indices(norb, n_beta, (i,)), dtype=np.uintp)
            apply_phase_shift_in_place(
                vec,
                float(np.real(phase_shift)),
                float(np.imag(phase_shift)),
                indices,
            )
        vec = vec.T

    return vec.reshape(-1)


def apply_orbital_rotation(
    vec: np.ndarray,
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: int | tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    if _ffsim_apply_orbital_rotation is not None:
        return _ffsim_apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            copy=copy,
        )
    vec = np.asarray(vec, dtype=np.complex128)
    if copy:
        vec = vec.copy()
    if isinstance(nelec, int):
        if not isinstance(orbital_rotation, np.ndarray):
            raise ValueError("spinless orbital rotation must be a single matrix")
        return _apply_orbital_rotation_spinless(vec, orbital_rotation, norb, nelec)
    return _apply_orbital_rotation_spinful(vec, orbital_rotation, norb, nelec)
