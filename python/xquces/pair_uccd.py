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


@cache
def _pair_uccd_rotation_blocks(
    norb: int,
    nelec: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if nelec[0] != nelec[1]:
        raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")
    npair = nelec[0]
    basis = _doci_spatial_basis(norb, npair)
    basis_index = {occ: i for i, occ in enumerate(basis)}
    all_blocks: list[tuple[tuple[int, int], ...]] = []
    for i, a in _pair_uccd_ov_pairs(norb, npair):
        blocks: list[tuple[int, int]] = []
        for left, occ in enumerate(basis):
            occ_set = set(occ)
            if i in occ_set and a not in occ_set:
                target = tuple(sorted((occ_set - {i}) | {a}))
                blocks.append((left, basis_index[target]))
        all_blocks.append(tuple(blocks))
    return tuple(all_blocks)


def _apply_pair_rotation_blocks_in_place(
    coeffs: np.ndarray,
    blocks: tuple[tuple[int, int], ...],
    theta: float,
) -> None:
    if theta == 0.0:
        return
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    for left, right in blocks:
        x = coeffs[left]
        y = coeffs[right]
        coeffs[left] = c * x - s * y
        coeffs[right] = s * x + c * y


def _apply_pair_rotation_blocks_to_matrix_in_place(
    mat: np.ndarray,
    blocks: tuple[tuple[int, int], ...],
    theta: float,
) -> None:
    if theta == 0.0:
        return
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    for left, right in blocks:
        x = np.array(mat[left], copy=True)
        y = np.array(mat[right], copy=True)
        mat[left] = c * x - s * y
        mat[right] = s * x + c * y


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


def product_pair_uccd_unitary_from_parameters(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    blocks = _pair_uccd_rotation_blocks(norb, nelec)
    expected = len(blocks)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    dim = len(_doci_spatial_basis(norb, nelec[0]))
    out = np.eye(dim, dtype=np.complex128)
    for theta, pair_blocks in zip(time * params, blocks):
        _apply_pair_rotation_blocks_to_matrix_in_place(out, pair_blocks, float(theta))
    return out


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


def apply_product_pair_uccd_reference_global(
    vec: np.ndarray,
    reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    params = np.asarray(reference_params, dtype=np.float64)
    blocks = _pair_uccd_rotation_blocks(norb, nelec)
    expected = len(blocks)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    out = np.array(vec, dtype=np.complex128, copy=copy)
    indices = np.asarray(_doci_subspace_indices(norb, nelec), dtype=np.uintp)
    if indices.size == 0:
        return out
    coeffs = np.array(out[indices], dtype=np.complex128, copy=True)
    for theta, pair_blocks in zip(time * params, blocks):
        _apply_pair_rotation_blocks_in_place(coeffs, pair_blocks, float(theta))
    out[indices] = coeffs
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


def product_pair_uccd_state(
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
    return apply_product_pair_uccd_reference_global(
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


def product_pair_uccd_state_jacobian(
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
    blocks = _pair_uccd_rotation_blocks(norb, nelec)
    expected = len(blocks)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    indices = _doci_subspace_indices(norb, nelec)
    full_dim = hartree_fock_state(norb, nelec).size
    doci_dim = len(indices)
    e0 = np.zeros(doci_dim, dtype=np.complex128)
    if doci_dim:
        e0[0] = 1.0
    forward: list[np.ndarray] = [e0]
    current = np.array(e0, copy=True)
    for theta, pair_blocks in zip(time * params, blocks):
        current = np.array(current, copy=True)
        _apply_pair_rotation_blocks_in_place(current, pair_blocks, float(theta))
        forward.append(current)
    out = np.zeros((full_dim, expected), dtype=np.complex128)
    for k in range(expected):
        vec = time * (generators[k] @ forward[k + 1])
        vec = np.asarray(vec, dtype=np.complex128)
        for theta, pair_blocks in zip(time * params[k + 1 :], blocks[k + 1 :]):
            _apply_pair_rotation_blocks_in_place(vec, pair_blocks, float(theta))
        out[indices, k] = vec
    return out


def product_pair_uccd_state_vjp(
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    v: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    """VJP of product_pair_uccd_state: returns 2 Re(J† v[doci_indices]).

    Backward pass through the stored forward states — O(n_params × dim_DOCI)
    with zero H-applications. `v` may live in the full Hilbert space; only the
    DOCI-subspace components are used.
    """
    if nelec[0] != nelec[1]:
        raise ValueError("Pair-UCCD reference requires n_alpha == n_beta")
    params = np.asarray(params, dtype=np.float64)
    generators = _pair_uccd_generator_basis(norb, nelec)
    blocks = _pair_uccd_rotation_blocks(norb, nelec)
    expected = len(blocks)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    indices = np.asarray(_doci_subspace_indices(norb, nelec), dtype=np.intp)
    doci_dim = len(indices)

    e0 = np.zeros(doci_dim, dtype=np.complex128)
    if doci_dim:
        e0[0] = 1.0

    # Forward pass — store all intermediate DOCI states
    forward: list[np.ndarray] = [e0]
    current = np.array(e0, copy=True)
    for theta, pair_blocks in zip(time * params, blocks):
        current = np.array(current, copy=True)
        _apply_pair_rotation_blocks_in_place(current, pair_blocks, float(theta))
        forward.append(current)

    # Backward pass — λ starts as v projected to DOCI
    v = np.asarray(v, dtype=np.complex128)
    lam = np.array(v[indices], dtype=np.complex128, copy=True)

    grad = np.zeros(expected)
    for k in reversed(range(expected)):
        # ∂E/∂p_k = 2t Re(⟨G_k forward[k+1] | λ[k+1]⟩)
        # where G_k is real antisymmetric so G_k† = -G_k and
        # Re(⟨G_k f | λ⟩) = -Re(⟨f | G_k λ⟩) = Re(vdot(G_k f, λ))
        Gf = generators[k] @ forward[k + 1]
        grad[k] = float(2.0 * time * np.real(np.vdot(Gf, lam)))
        # Propagate λ backward: λ[k] = R_k^{-1} λ[k+1]
        _apply_pair_rotation_blocks_in_place(lam, blocks[k], -time * float(params[k]))

    return grad


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


@dataclass(frozen=True)
class ProductPairUCCDStateParameterization:
    norb: int
    nelec: tuple[int, int]

    def __post_init__(self):
        if self.nelec[0] != self.nelec[1]:
            raise ValueError("Product pair-UCCD reference requires n_alpha == n_beta")

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
        return product_pair_uccd_state(self.norb, self.nelec, params=params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return product_pair_uccd_state_jacobian(self.norb, self.nelec, params)

    def params_to_state(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func
