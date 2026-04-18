from __future__ import annotations

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
