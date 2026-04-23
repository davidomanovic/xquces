from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np
import scipy.linalg

from xquces._lib import apply_pair_uccd_doci_unitary_in_place
from xquces.states import _doci_spatial_basis, _doci_subspace_indices, hartree_fock_state


@cache
def _pair_uccd_ov_pairs(norb: int, nocc: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, a) for i in range(nocc) for a in range(nocc, norb))


@cache
def _pair_uccd_generator_basis(
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    if nelec[0] != nelec[1]:
        raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")
    npair = nelec[0]
    basis = _doci_spatial_basis(norb, npair)
    dim = len(basis)
    basis_index = {occ: i for i, occ in enumerate(basis)}
    generators: list[np.ndarray] = []
    for i, a in _pair_uccd_ov_pairs(norb, npair):
        gen = np.zeros((dim, dim), dtype=np.float64)
        for col, occ in enumerate(basis):
            occ_set = set(occ)
            if i in occ_set and a not in occ_set:
                target = tuple(sorted((occ_set - {i}) | {a}))
                gen[basis_index[target], col] += 1.0
            elif a in occ_set and i not in occ_set:
                target = tuple(sorted((occ_set - {a}) | {i}))
                gen[basis_index[target], col] -= 1.0
        generators.append(gen)
    return tuple(generators)


def pair_uccd_parameters_from_t2(
    t2: np.ndarray,
    *,
    scale: float = 0.5,
) -> np.ndarray:
    t2 = np.asarray(t2)
    if t2.ndim != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc_i, nocc_j, nvirt_a, nvirt_b = t2.shape
    if nocc_i != nocc_j or nvirt_a != nvirt_b:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    out = np.zeros(nocc_i * nvirt_a, dtype=np.float64)
    k = 0
    for i in range(nocc_i):
        for a in range(nvirt_a):
            out[k] = float(scale) * float(np.real(t2[i, i, a, a]))
            k += 1
    return out


def pair_uccd_generator_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    generators = _pair_uccd_generator_basis(norb, nelec)
    expected = len(generators)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    out = np.zeros_like(generators[0]) if generators else np.zeros((1, 1), dtype=np.float64)
    for theta, gen in zip(time * params, generators):
        if theta != 0.0:
            out += float(theta) * gen
    return out


def pair_uccd_unitary_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    generator = pair_uccd_generator_from_parameters(norb, nelec, params, time=time)
    return np.asarray(scipy.linalg.expm(generator), dtype=np.complex128)


def apply_pair_uccd_reference_global(
    vec: np.ndarray,
    reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
    unitary: np.ndarray | None = None,
) -> np.ndarray:
    out = np.array(vec, dtype=np.complex128, copy=copy)
    indices = np.asarray(_doci_subspace_indices(norb, nelec), dtype=np.uintp)
    if indices.size == 0:
        return out
    if unitary is None:
        unitary = pair_uccd_unitary_from_parameters(
            norb,
            nelec,
            reference_params,
            time=time,
        )
    apply_pair_uccd_doci_unitary_in_place(
        out,
        np.asarray(unitary, dtype=np.complex128),
        indices,
    )
    return out


def pair_uccd_state(
    norb: int,
    nelec: tuple[int, int],
    *,
    params: np.ndarray | None = None,
    time: float = 1.0,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")
    if params is None:
        params = np.zeros(len(_pair_uccd_ov_pairs(norb, nelec[0])), dtype=np.float64)
    reference = hartree_fock_state(norb, nelec)
    return apply_pair_uccd_reference_global(
        reference,
        np.asarray(params, dtype=np.float64),
        norb,
        nelec,
        time=time,
        copy=True,
    )


def pair_uccd_state_jacobian(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")
    params = np.asarray(params, dtype=np.float64)
    generators = _pair_uccd_generator_basis(norb, nelec)
    expected = len(generators)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    indices = _doci_subspace_indices(norb, nelec)
    full_dim = hartree_fock_state(norb, nelec).size
    doci_dim = len(indices)
    e0 = np.zeros(doci_dim, dtype=np.float64)
    if doci_dim:
        e0[0] = 1.0
    generator = pair_uccd_generator_from_parameters(norb, nelec, params, time=time)
    out = np.zeros((full_dim, expected), dtype=np.complex128)
    for k, basis_gen in enumerate(generators):
        frechet = scipy.linalg.expm_frechet(
            generator,
            time * basis_gen,
            compute_expm=False,
        )
        out[indices, k] = frechet @ e0
    return out


@dataclass(frozen=True)
class PairUCCDStateParameterization:
    norb: int
    nelec: tuple[int, int]

    def __post_init__(self):
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return _pair_uccd_ov_pairs(self.norb, self.nelec[0])

    @property
    def n_params(self) -> int:
        return len(self.pair_indices)

    def parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        params = pair_uccd_parameters_from_t2(t2, scale=scale)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return params

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return pair_uccd_state(self.norb, self.nelec, params=params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return pair_uccd_state_jacobian(self.norb, self.nelec, params)

    def params_to_state(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func
