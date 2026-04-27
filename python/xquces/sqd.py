from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from xquces._lib import (
    estimate_spin_orbital_occupancies,
    postselect_spin_bitstrings,
    recover_spin_bitstrings,
    sample_indices_from_probabilities,
    subsample_batches,
)
from xquces.basis import occ_rows
from xquces.hamiltonians import MolecularHamiltonianLinearOperator


def determinant_index_to_spin_bitstring(
    det_index: int,
    norb: int,
    nelec: tuple[int, int],
) -> int:
    alpha_rows = occ_rows(norb, nelec[0])
    beta_rows = occ_rows(norb, nelec[1])
    dim_b = len(beta_rows)

    ia = int(det_index) // dim_b
    ib = int(det_index) % dim_b

    occ_a = alpha_rows[ia]
    occ_b = beta_rows[ib]

    alpha_bits = 0
    beta_bits = 0

    for p in occ_a:
        alpha_bits |= 1 << int(p)
    for p in occ_b:
        beta_bits |= 1 << int(p)

    return alpha_bits | (beta_bits << norb)


def sector_state_to_spin_bitstrings(
    statevector: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    vec = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    probs = np.abs(vec) ** 2
    probs = probs / probs.sum()

    det_indices = sample_indices_from_probabilities(
        probs,
        n_samples=n_samples,
        seed=seed,
    )

    bitstrings = [
        determinant_index_to_spin_bitstring(int(idx), norb=norb, nelec=nelec)
        for idx in det_indices
    ]
    return np.asarray(bitstrings, dtype=np.uint64)


def bitstring_to_determinant_index(
    bitstring: int,
    norb: int,
    nelec: tuple[int, int],
    alpha_index: dict[tuple[int, ...], int],
    beta_index: dict[tuple[int, ...], int],
) -> int | None:
    alpha_bits = bitstring & ((1 << norb) - 1)
    beta_bits = (bitstring >> norb) & ((1 << norb) - 1)

    occ_a = tuple(i for i in range(norb) if (alpha_bits >> i) & 1)
    occ_b = tuple(i for i in range(norb) if (beta_bits >> i) & 1)

    if len(occ_a) != nelec[0] or len(occ_b) != nelec[1]:
        return None

    ia = alpha_index.get(occ_a)
    ib = beta_index.get(occ_b)
    if ia is None or ib is None:
        return None

    dim_b = len(beta_index)
    return ia * dim_b + ib


def determinant_indices_from_bitstrings(
    bitstrings: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    alpha_rows = occ_rows(norb, nelec[0])
    beta_rows = occ_rows(norb, nelec[1])

    alpha_index = {tuple(map(int, row)): i for i, row in enumerate(alpha_rows)}
    beta_index = {tuple(map(int, row)): i for i, row in enumerate(beta_rows)}

    out = []
    for x in np.asarray(bitstrings, dtype=np.uint64):
        idx = bitstring_to_determinant_index(
            int(x), norb, nelec, alpha_index, beta_index
        )
        if idx is not None:
            out.append(idx)
    return np.asarray(sorted(set(out)), dtype=np.int64)


def projected_hamiltonian_from_dim(
    det_indices: np.ndarray,
    hamiltonian: MolecularHamiltonianLinearOperator,
    full_dim: int,
) -> np.ndarray:
    det_indices = np.asarray(det_indices, dtype=np.int64)
    n = len(det_indices)
    out = np.zeros((n, n), dtype=np.complex128)
    if n == 0:
        return out

    for j, idx_j in enumerate(det_indices):
        e = np.zeros(full_dim, dtype=np.complex128)
        e[int(idx_j)] = 1.0
        sigma = hamiltonian.matvec(e)
        out[:, j] = sigma[det_indices]

    return out


def orbital_occupancies_from_projected_state(
    det_indices: np.ndarray,
    coeffs: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    alpha_rows = occ_rows(norb, nelec[0])
    beta_rows = occ_rows(norb, nelec[1])
    dim_b = len(beta_rows)

    occ_a = np.zeros(norb, dtype=np.float64)
    occ_b = np.zeros(norb, dtype=np.float64)

    for idx, c in zip(det_indices, coeffs):
        w = float(np.abs(c) ** 2)
        ia = int(idx) // dim_b
        ib = int(idx) % dim_b
        for p in alpha_rows[ia]:
            occ_a[int(p)] += w
        for p in beta_rows[ib]:
            occ_b[int(p)] += w

    return occ_a, occ_b


@dataclass(frozen=True)
class SQDSubspaceResult:
    energy: float
    det_indices: np.ndarray
    eigenvector: np.ndarray
    occ_alpha: np.ndarray
    occ_beta: np.ndarray


@dataclass(frozen=True)
class SQDRunResult:
    energy: float
    iterations: list[list[SQDSubspaceResult]]
    best: SQDSubspaceResult


def run_sqd_from_statevector(
    statevector: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    hamiltonian: MolecularHamiltonianLinearOperator,
    n_samples: int = 20000,
    batch_size: int = 256,
    num_batches: int = 3,
    max_iterations: int = 5,
    seed: int = 0,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
) -> SQDRunResult:
    samples = sector_state_to_spin_bitstrings(
        statevector,
        norb=norb,
        nelec=nelec,
        n_samples=n_samples,
        seed=seed,
    )

    valid = np.asarray(
        postselect_spin_bitstrings(
            samples, norb=norb, n_alpha=nelec[0], n_beta=nelec[1]
        ),
        dtype=np.uint64,
    )

    if initial_occupancies is None:
        if len(valid) == 0:
            occ_a = np.array(
                [1.0] * nelec[0] + [0.0] * (norb - nelec[0]), dtype=np.float64
            )
            occ_b = np.array(
                [1.0] * nelec[1] + [0.0] * (norb - nelec[1]), dtype=np.float64
            )
        else:
            occ_a, occ_b = estimate_spin_orbital_occupancies(valid, norb=norb)
            occ_a = np.asarray(occ_a, dtype=np.float64)
            occ_b = np.asarray(occ_b, dtype=np.float64)
    else:
        occ_a = np.asarray(initial_occupancies[0], dtype=np.float64)
        occ_b = np.asarray(initial_occupancies[1], dtype=np.float64)

    full_dim = len(np.asarray(statevector).reshape(-1))
    history: list[list[SQDSubspaceResult]] = []
    best_global: SQDSubspaceResult | None = None

    for it in range(max_iterations):
        repaired = np.asarray(
            recover_spin_bitstrings(
                samples,
                norb=norb,
                n_alpha=nelec[0],
                n_beta=nelec[1],
                occ_alpha=occ_a,
                occ_beta=occ_b,
                seed=seed + 1000 + it,
            ),
            dtype=np.uint64,
        )

        merged = np.concatenate([valid, repaired])
        valid_merged = np.asarray(
            postselect_spin_bitstrings(
                merged,
                norb=norb,
                n_alpha=nelec[0],
                n_beta=nelec[1],
            ),
            dtype=np.uint64,
        )

        batches = subsample_batches(
            valid_merged,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed + 2000 + it,
        )

        iter_results: list[SQDSubspaceResult] = []

        for batch in batches:
            batch = np.asarray(batch, dtype=np.uint64)
            det_indices = determinant_indices_from_bitstrings(
                batch, norb=norb, nelec=nelec
            )
            if len(det_indices) == 0:
                continue

            h_sub = projected_hamiltonian_from_dim(
                det_indices, hamiltonian, full_dim=full_dim
            )
            evals, evecs = scipy.linalg.eigh(h_sub)
            k = int(np.argmin(evals))
            energy = float(np.real(evals[k]) + hamiltonian.ecore)
            eigvec = evecs[:, k]

            sub_occ_a, sub_occ_b = orbital_occupancies_from_projected_state(
                det_indices,
                eigvec,
                norb=norb,
                nelec=nelec,
            )

            result = SQDSubspaceResult(
                energy=energy,
                det_indices=det_indices,
                eigenvector=eigvec,
                occ_alpha=sub_occ_a,
                occ_beta=sub_occ_b,
            )
            iter_results.append(result)

        if not iter_results:
            history.append([])
            continue

        best_iter = min(iter_results, key=lambda x: x.energy)
        occ_a = best_iter.occ_alpha
        occ_b = best_iter.occ_beta
        history.append(iter_results)

        if best_global is None or best_iter.energy < best_global.energy:
            best_global = best_iter

    if best_global is None:
        raise RuntimeError("SQD failed to produce any valid projected subspace")

    return SQDRunResult(
        energy=best_global.energy,
        iterations=history,
        best=best_global,
    )
