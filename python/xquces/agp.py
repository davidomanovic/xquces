from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.states import (
    _doci_spatial_basis,
    _doci_subspace_indices,
    doci_amplitudes_from_state,
    doci_dimension,
)


def _canonicalize_real_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("vector must be nonzero")
    out = arr / norm
    for value in out:
        if abs(value) > 1e-14:
            if value < 0.0:
                out = -out
            break
    return out


def _real_unit_vector_from_parameters(dim: int, params: np.ndarray) -> np.ndarray:
    expected = dim - 1
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    if dim == 1:
        return np.ones(1, dtype=np.float64)
    out = np.empty(dim, dtype=np.float64)
    running = 1.0
    for k, theta in enumerate(params):
        out[k] = running * np.cos(theta)
        running *= np.sin(theta)
    out[-1] = running
    return out


def _real_unit_vector_jacobian_from_parameters(
    dim: int, params: np.ndarray
) -> np.ndarray:
    expected = dim - 1
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    jac = np.zeros((dim, expected), dtype=np.float64)
    if dim == 1:
        return jac
    s = np.sin(params)
    c = np.cos(params)
    prefix = np.ones(dim, dtype=np.float64)
    for k in range(1, dim):
        prefix[k] = prefix[k - 1] * s[k - 1]
    for m in range(expected):
        jac[m, m] = -prefix[m] * s[m]
        tail = prefix[m] * c[m]
        for k in range(m + 1, expected):
            jac[k, m] = tail * c[k]
            tail *= s[k]
        jac[dim - 1, m] = tail
    return jac


def _parameters_from_real_unit_vector(vec: np.ndarray) -> np.ndarray:
    state = _canonicalize_real_vector(vec)
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


def agp_eta_from_parameters(norb: int, params: np.ndarray) -> np.ndarray:
    return _real_unit_vector_from_parameters(norb, params)


def agp_eta_jacobian_from_parameters(norb: int, params: np.ndarray) -> np.ndarray:
    return _real_unit_vector_jacobian_from_parameters(norb, params)


def agp_parameters_from_eta(eta: np.ndarray) -> np.ndarray:
    return _parameters_from_real_unit_vector(eta)


def agp_amplitudes_from_eta(
    norb: int,
    nelec: tuple[int, int],
    eta: np.ndarray,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("AGP/PBCS requires n_alpha == n_beta")
    eta = _canonicalize_real_vector(eta)
    if eta.shape != (norb,):
        raise ValueError(f"Expected {(norb,)}, got {eta.shape}.")
    npair = nelec[0]
    basis = _doci_spatial_basis(norb, npair)
    amps = np.empty(len(basis), dtype=np.float64)
    for i, occ in enumerate(basis):
        value = 1.0
        for p in occ:
            value *= eta[p]
        amps[i] = value
    norm = float(np.linalg.norm(amps))
    if norm == 0.0:
        raise ValueError("AGP amplitudes have zero norm")
    return amps / norm


def agp_amplitudes_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
) -> np.ndarray:
    return agp_amplitudes_from_eta(norb, nelec, agp_eta_from_parameters(norb, params))


def agp_amplitudes_jacobian_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("AGP/PBCS requires n_alpha == n_beta")
    eta = agp_eta_from_parameters(norb, params)
    eta_jac = agp_eta_jacobian_from_parameters(norb, params)
    npair = nelec[0]
    basis = _doci_spatial_basis(norb, npair)
    unnorm = np.empty(len(basis), dtype=np.float64)
    d_unnorm_d_eta = np.zeros((len(basis), norb), dtype=np.float64)
    for i, occ in enumerate(basis):
        value = 1.0
        for p in occ:
            value *= eta[p]
        unnorm[i] = value
        for p in occ:
            deriv = 1.0
            for q in occ:
                if q != p:
                    deriv *= eta[q]
            d_unnorm_d_eta[i, p] = deriv
    d_unnorm = d_unnorm_d_eta @ eta_jac
    norm = float(np.linalg.norm(unnorm))
    if norm == 0.0:
        raise ValueError("AGP amplitudes have zero norm")
    amps = unnorm / norm
    proj = amps @ d_unnorm
    return (d_unnorm - np.outer(amps, proj)) / norm


def agp_state(
    norb: int,
    nelec: tuple[int, int],
    *,
    params: np.ndarray | None = None,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    if params is not None and eta is not None:
        raise ValueError("pass either params or eta, not both")
    if params is None and eta is None:
        eta = np.zeros(norb, dtype=np.float64)
        eta[: nelec[0]] = 1.0
    if params is not None:
        amps = agp_amplitudes_from_parameters(norb, nelec, params)
    else:
        amps = agp_amplitudes_from_eta(norb, nelec, np.asarray(eta, dtype=np.float64))
    dim = len(_doci_spatial_basis(norb, nelec[0])) ** 2
    vec = np.zeros(dim, dtype=np.complex128)
    vec[_doci_subspace_indices(norb, nelec)] = amps
    return vec


def agp_state_jacobian(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
) -> np.ndarray:
    amp_jac = agp_amplitudes_jacobian_from_parameters(norb, nelec, params)
    dim = len(_doci_spatial_basis(norb, nelec[0])) ** 2
    out = np.zeros((dim, amp_jac.shape[1]), dtype=np.complex128)
    out[_doci_subspace_indices(norb, nelec), :] = amp_jac
    return out


def agp_eta_from_amplitudes(
    amplitudes: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("AGP/PBCS requires n_alpha == n_beta")
    amps = np.asarray(amplitudes, dtype=np.float64)
    expected = doci_dimension(norb, nelec)
    if amps.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {amps.shape}.")
    basis = _doci_spatial_basis(norb, nelec[0])
    rows = []
    rhs = []
    for coeff, occ in zip(amps, basis):
        mag = abs(float(coeff))
        if mag <= tol:
            continue
        row = np.zeros(norb + 1, dtype=np.float64)
        row[0] = 1.0
        for p in occ:
            row[1 + p] = 1.0
        rows.append(row)
        rhs.append(np.log(mag))
    if not rows:
        eta = np.zeros(norb, dtype=np.float64)
        eta[: nelec[0]] = 1.0
        return _canonicalize_real_vector(eta)
    design = np.vstack(rows)
    target = np.asarray(rhs, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    eta = np.exp(sol[1:])
    return _canonicalize_real_vector(eta)


def agp_parameters_from_amplitudes(
    amplitudes: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    return agp_parameters_from_eta(agp_eta_from_amplitudes(amplitudes, norb, nelec))


@dataclass(frozen=True)
class AGPStateParameterization:
    norb: int
    nelec: tuple[int, int]

    def __post_init__(self):
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("AGP/PBCS reference requires n_alpha == n_beta")

    @property
    def n_params(self) -> int:
        return self.norb - 1

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return agp_state(self.norb, self.nelec, params=params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return agp_state_jacobian(self.norb, self.nelec, params)

    def parameters_from_eta(self, eta: np.ndarray) -> np.ndarray:
        return agp_parameters_from_eta(eta)

    def parameters_from_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        return agp_parameters_from_amplitudes(amplitudes, self.norb, self.nelec)

    def parameters_from_state(self, state: np.ndarray) -> np.ndarray:
        doci_amplitudes = doci_amplitudes_from_state(state, self.norb, self.nelec)
        return self.parameters_from_amplitudes(doci_amplitudes)
