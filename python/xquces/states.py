from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from xquces.basis import occ_rows, sector_shape


def hartree_fock_state(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    dim = dim_a * dim_b
    vec = np.zeros(dim, dtype=np.complex128)
    vec[0] = 1.0
    return vec


def determinant_index(norb: int, nelec: tuple[int, int], alpha_occ, beta_occ) -> int:
    alpha_occ = tuple(int(i) for i in alpha_occ)
    beta_occ = tuple(int(i) for i in beta_occ)
    if len(alpha_occ) != nelec[0] or len(beta_occ) != nelec[1]:
        raise ValueError("occupation lengths must match nelec")

    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    alpha_matches = np.flatnonzero(np.all(occ_a == alpha_occ, axis=1))
    beta_matches = np.flatnonzero(np.all(occ_b == beta_occ, axis=1))
    if alpha_matches.size != 1 or beta_matches.size != 1:
        raise ValueError("invalid determinant occupation")
    return int(alpha_matches[0]) * len(occ_b) + int(beta_matches[0])


def determinant_state(
    norb: int,
    nelec: tuple[int, int],
    alpha_occ,
    beta_occ,
) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    vec = np.zeros(dim_a * dim_b, dtype=np.complex128)
    vec[determinant_index(norb, nelec, alpha_occ, beta_occ)] = 1.0
    return vec


def linear_combination_state(
    norb: int,
    nelec: tuple[int, int],
    terms,
) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    vec = np.zeros(dim_a * dim_b, dtype=np.complex128)
    for coeff, alpha_occ, beta_occ in terms:
        idx = determinant_index(norb, nelec, alpha_occ, beta_occ)
        vec[idx] += complex(coeff)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("linear combination has zero norm")
    return vec / norm


def open_shell_singlet_state(
    norb: int,
    nelec: tuple[int, int],
    closed_orbitals,
    open_orbitals,
    *,
    relative_sign: float = 1.0,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("open-shell singlet reference requires n_alpha == n_beta")
    closed = tuple(int(i) for i in closed_orbitals)
    open_pair = tuple(int(i) for i in open_orbitals)
    if len(open_pair) != 2:
        raise ValueError("open_orbitals must contain exactly two orbitals")
    if len(closed) + 1 != nelec[0]:
        raise ValueError("closed_orbitals must contain nocc - 1 orbitals")
    p, q = open_pair
    return linear_combination_state(
        norb,
        nelec,
        [
            (1.0, closed + (p,), closed + (q,)),
            (relative_sign, closed + (q,), closed + (p,)),
        ],
    )


@cache
def _doci_spatial_basis(norb: int, npair: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(x) for x in itertools.combinations(range(norb), npair))


@cache
def _doci_subspace_indices(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("DOCI requires n_alpha == n_beta")
    npair = nelec[0]
    sector = tuple(tuple(int(x) for x in row) for row in occ_rows(norb, npair))
    sector_index = {occ: i for i, occ in enumerate(sector)}
    dim_beta = len(sector)
    basis = _doci_spatial_basis(norb, npair)
    return np.asarray(
        [sector_index[occ] * dim_beta + sector_index[occ] for occ in basis],
        dtype=np.intp,
    )


def doci_dimension(norb: int, nelec: tuple[int, int]) -> int:
    if nelec[0] != nelec[1]:
        raise ValueError("DOCI requires n_alpha == n_beta")
    return len(_doci_spatial_basis(norb, nelec[0]))


def _canonicalize_real_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=np.float64)
    if amps.ndim != 1:
        raise ValueError("DOCI amplitudes must be one-dimensional")
    norm = float(np.linalg.norm(amps))
    if norm == 0.0:
        raise ValueError("DOCI amplitudes must be nonzero")
    out = amps / norm
    for value in out:
        if abs(value) > 1e-14:
            if value < 0.0:
                out = -out
            break
    return out


def _real_amplitudes_from_complex_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError("DOCI vector must be one-dimensional")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("DOCI vector must be nonzero")
    out = arr / norm
    nz = np.flatnonzero(np.abs(out) > 1e-14)
    if nz.size:
        phase = np.exp(-1j * np.angle(out[int(nz[0])]))
        out = out * phase
    if np.linalg.norm(np.imag(out)) > 1e-10:
        raise ValueError("this DOCI parameterization only supports real DOCI amplitudes")
    return _canonicalize_real_amplitudes(np.real(out))


def doci_amplitudes_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
) -> np.ndarray:
    dim = doci_dimension(norb, nelec)
    expected = dim - 1
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    if dim == 1:
        return np.ones(1, dtype=np.float64)
    amps = np.empty(dim, dtype=np.float64)
    running = 1.0
    for k, theta in enumerate(params):
        amps[k] = running * np.cos(theta)
        running *= np.sin(theta)
    amps[-1] = running
    return amps


def doci_parameters_from_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    state = _canonicalize_real_amplitudes(amplitudes)
    dim = state.size
    if dim == 1:
        return np.zeros(0, dtype=np.float64)
    params = np.zeros(dim - 1, dtype=np.float64)
    for k in range(dim - 2):
        tail_norm = float(np.linalg.norm(state[k + 1 :]))
        if abs(state[k]) < 1e-14 and tail_norm < 1e-14:
            params[k] = 0.0
        else:
            params[k] = float(np.arctan2(tail_norm, state[k]))
    params[-1] = float(np.arctan2(state[-1], state[-2]))
    return params


def doci_amplitudes_from_state(
    state: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    vec = np.asarray(state, dtype=np.complex128)
    indices = _doci_subspace_indices(norb, nelec)
    dim = int(np.prod(sector_shape(norb, nelec)))
    if vec.shape != (dim,):
        raise ValueError(f"Expected {(dim,)}, got {vec.shape}.")
    mask = np.ones(dim, dtype=bool)
    mask[indices] = False
    if np.linalg.norm(vec[mask]) > 1e-10:
        raise ValueError("state has support outside the DOCI subspace")
    return _real_amplitudes_from_complex_vector(vec[indices])


def doci_params_from_state(
    state: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    return doci_parameters_from_amplitudes(doci_amplitudes_from_state(state, norb, nelec))


def doci_state(
    norb: int,
    nelec: tuple[int, int],
    *,
    params: np.ndarray | None = None,
    amplitudes: np.ndarray | None = None,
) -> np.ndarray:
    if params is not None and amplitudes is not None:
        raise ValueError("pass either params or amplitudes, not both")
    dim_a, dim_b = sector_shape(norb, nelec)
    vec = np.zeros(dim_a * dim_b, dtype=np.complex128)
    indices = _doci_subspace_indices(norb, nelec)
    if params is None and amplitudes is None:
        amps = np.zeros(doci_dimension(norb, nelec), dtype=np.float64)
        amps[0] = 1.0
    elif params is not None:
        amps = doci_amplitudes_from_parameters(norb, nelec, params)
    else:
        amps = _real_amplitudes_from_complex_vector(np.asarray(amplitudes, dtype=np.complex128))
        expected = doci_dimension(norb, nelec)
        if amps.shape != (expected,):
            raise ValueError(f"Expected {(expected,)}, got {amps.shape}.")
    vec[indices] = amps
    return vec


def _doci_unitary_from_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    target = _canonicalize_real_amplitudes(amplitudes)
    dim = target.size
    unitary = np.eye(dim, dtype=np.complex128)
    if dim == 1:
        return unitary
    e0 = np.zeros(dim, dtype=np.float64)
    e0[0] = 1.0
    diff = e0 - target
    norm = float(np.linalg.norm(diff))
    if norm < 1e-14:
        return unitary
    u = diff / norm
    return unitary - 2.0 * np.outer(u, u).astype(np.complex128)


def apply_doci_unitary(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: np.ndarray | None = None,
    amplitudes: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    if params is not None and amplitudes is not None:
        raise ValueError("pass either params or amplitudes, not both")
    if params is None and amplitudes is None:
        raise ValueError("pass params or amplitudes")
    if params is not None:
        amps = doci_amplitudes_from_parameters(norb, nelec, params)
    else:
        amps = _real_amplitudes_from_complex_vector(np.asarray(amplitudes, dtype=np.complex128))
        expected = doci_dimension(norb, nelec)
        if amps.shape != (expected,):
            raise ValueError(f"Expected {(expected,)}, got {amps.shape}.")
    out = np.array(vec, dtype=np.complex128, copy=copy)
    indices = _doci_subspace_indices(norb, nelec)
    unitary = _doci_unitary_from_amplitudes(amps)
    out[indices] = unitary @ np.asarray(out[indices], dtype=np.complex128)
    return out
