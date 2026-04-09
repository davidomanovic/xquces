from __future__ import annotations

import itertools
import math
from dataclasses import InitVar, dataclass
from typing import cast

import numpy as np
import pyscf.ci
import scipy.linalg

from xquces import gates
from xquces.orbitals import apply_orbital_rotation


def validate_interaction_pairs(
    interaction_pairs: list[tuple[int, int]] | None, ordered: bool
) -> None:
    if interaction_pairs is None:
        return
    if len(set(interaction_pairs)) != len(interaction_pairs):
        raise ValueError(f"Duplicate interaction pairs encountered: {interaction_pairs}.")
    if not ordered:
        for i, j in interaction_pairs:
            if i > j:
                raise ValueError(
                    "When specifying spinless, alpha-alpha or beta-beta interaction pairs, "
                    f"you must provide only upper triangular pairs. Got {(i, j)}."
                )


def orbital_rotation_from_t1_amplitudes(t1: np.ndarray) -> np.ndarray:
    nocc, nvrt = t1.shape
    norb = nocc + nvrt
    generator = np.zeros((norb, norb), dtype=np.asarray(t1).dtype)
    generator[:nocc, nocc:] = -np.asarray(t1).conj()
    generator[nocc:, :nocc] = np.asarray(t1).T
    return scipy.linalg.expm(generator)


def interaction_pairs_spin_balanced(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    if connectivity == "all-to-all":
        return None, None
    if connectivity == "square":
        return [(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)]
    if connectivity == "hex":
        return [(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb) if p % 2 == 0]
    if connectivity == "heavy-hex":
        return [(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb) if p % 4 == 0]
    raise ValueError(f"Invalid connectivity: {connectivity}")


def is_unitary(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    mat = np.asarray(mat)
    return mat.ndim == 2 and mat.shape[0] == mat.shape[1] and np.allclose(mat.conj().T @ mat, np.eye(mat.shape[0]), rtol=rtol, atol=atol)


def is_real_symmetric(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    mat = np.asarray(mat)
    return np.isrealobj(mat) and mat.ndim == 2 and mat.shape[0] == mat.shape[1] and np.allclose(mat, mat.T, rtol=rtol, atol=atol)


def antihermitian_to_parameters(mat: np.ndarray, real: bool = False) -> np.ndarray:
    dim = mat.shape[0]
    n_triu = dim * (dim - 1) // 2
    n_params = n_triu if real else dim**2
    params = np.zeros(n_params, dtype=float)
    rows, cols = np.triu_indices(dim, k=1)
    params[:n_triu] = np.asarray(mat)[rows, cols].real
    if not real:
        rows, cols = np.triu_indices(dim)
        params[n_triu:] = np.asarray(mat)[rows, cols].imag
    return params


def antihermitian_from_parameters(params: np.ndarray, dim: int, real: bool = False) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    n_triu = dim * (dim - 1) // 2
    n_params = n_triu if real else dim**2
    if params.shape != (n_params,):
        raise ValueError(f"Expected {(n_params,)}, got {params.shape}.")
    mat = np.zeros((dim, dim), dtype=float if real else complex)
    if not real:
        rows, cols = np.triu_indices(dim)
        vals = 1j * params[n_triu:]
        mat[rows, cols] = vals
        mat[cols, rows] = vals
    rows, cols = np.triu_indices(dim, k=1)
    vals = params[:n_triu]
    mat[rows, cols] += vals
    mat[cols, rows] -= vals
    return mat


def unitary_to_parameters(mat: np.ndarray, real: bool = False) -> np.ndarray:
    return antihermitian_to_parameters(scipy.linalg.logm(np.asarray(mat)), real=real)


def unitary_from_parameters(params: np.ndarray, dim: int, real: bool = False) -> np.ndarray:
    return scipy.linalg.expm(antihermitian_from_parameters(params, dim, real=real))


def _truncated_eigh(
    mat: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = scipy.linalg.eigh(mat)
    if max_vecs is None:
        max_vecs = len(eigs)
    indices = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[indices]
    vecs = vecs[:, indices]
    n_discard = int(np.searchsorted(np.cumsum(np.abs(eigs[::-1])), tol))
    n_vecs = cast(int, min(max_vecs, len(eigs) - n_discard))
    return eigs[:n_vecs], vecs[:, :n_vecs]


def _quadrature(mat: np.ndarray, sign: int) -> np.ndarray:
    return 0.5 * (1 - sign * 1j) * (mat + sign * 1j * mat.T.conj())


def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    regularization: float = 0.0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    if optimize:
        raise NotImplementedError("optimize=True is not implemented in xquces native UCJ utilities.")
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt
    t2_mat = np.asarray(t2_amplitudes).transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)
    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol)
    n_vecs = len(outer_eigs)
    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc), range(nocc, nocc + nvrt)))
        mat[row, col] = outer_vec
        one_body_tensor[0] = _quadrature(mat, sign=1)
        one_body_tensor[1] = _quadrature(mat, sign=-1)
    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    coeffs = np.array([1, -1]) * outer_eigs[:, None]
    diag_coulomb_mats = coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:max_terms]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:max_terms]
    if diag_coulomb_indices is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        if diag_coulomb_indices:
            rows, cols = zip(*diag_coulomb_indices)
            mask[rows, cols] = True
            mask[cols, rows] = True
        diag_coulomb_mats = diag_coulomb_mats * mask
    return np.asarray(diag_coulomb_mats, dtype=float), np.asarray(orbital_rotations, dtype=complex)


def reconstruct_t2(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    nocc: int,
) -> np.ndarray:
    return (
        1j
        * np.einsum(
            "kpq,kap,kip,kbq,kjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )


@dataclass(frozen=True)
class UCJOpSpinBalanced:
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    final_orbital_rotation: np.ndarray | None = None
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if not validate:
            return
        if self.diag_coulomb_mats.ndim != 4 or self.diag_coulomb_mats.shape[1] != 2:
            raise ValueError(
                "diag_coulomb_mats should have shape (n_reps, 2, norb, norb). "
                f"Got shape {self.diag_coulomb_mats.shape}."
            )
        if self.orbital_rotations.ndim != 3:
            raise ValueError(
                "orbital_rotations should have shape (n_reps, norb, norb). "
                f"Got shape {self.orbital_rotations.shape}."
            )
        if self.final_orbital_rotation is not None and self.final_orbital_rotation.ndim != 2:
            raise ValueError(
                "final_orbital_rotation should have shape (norb, norb). "
                f"Got shape {self.final_orbital_rotation.shape}."
            )
        if self.diag_coulomb_mats.shape[0] != self.orbital_rotations.shape[0]:
            raise ValueError("diag_coulomb_mats and orbital_rotations should have the same first dimension.")
        if not all(is_real_symmetric(mats[0], rtol=rtol, atol=atol) and is_real_symmetric(mats[1], rtol=rtol, atol=atol) for mats in self.diag_coulomb_mats):
            raise ValueError("Diagonal Coulomb matrices were not all real symmetric.")
        if not all(is_unitary(orbital_rotation, rtol=rtol, atol=atol) for orbital_rotation in self.orbital_rotations):
            raise ValueError("Orbital rotations were not all unitary.")
        if self.final_orbital_rotation is not None and not is_unitary(self.final_orbital_rotation, rtol=rtol, atol=atol):
            raise ValueError("Final orbital rotation was not unitary.")

    @property
    def norb(self) -> int:
        return self.diag_coulomb_mats.shape[-1]

    @property
    def n_reps(self) -> int:
        return self.diag_coulomb_mats.shape[0]

    @staticmethod
    def n_params(
        norb: int,
        n_reps: int,
        *,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False)
        validate_interaction_pairs(pairs_ab, ordered=False)
        n_triu_indices = norb * (norb + 1) // 2
        n_params_aa = n_triu_indices if pairs_aa is None else len(pairs_aa)
        n_params_ab = n_triu_indices if pairs_ab is None else len(pairs_ab)
        return n_reps * (n_params_aa + n_params_ab + norb**2) + with_final_orbital_rotation * norb**2

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> "UCJOpSpinBalanced":
        n_params = UCJOpSpinBalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        params = np.asarray(params, dtype=float)
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        triu_indices = cast(list[tuple[int, int]], list(itertools.combinations_with_replacement(range(norb), 2)))
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = triu_indices
        diag_coulomb_mats = np.zeros((n_reps, 2, norb, norb))
        orbital_rotations = np.zeros((n_reps, norb, norb), dtype=complex)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(orbital_rotations, diag_coulomb_mats):
            orbital_rotation[:] = unitary_from_parameters(params[index : index + norb**2], norb)
            index += norb**2
            for indices, this_diag_coulomb_mat in zip((pairs_aa, pairs_ab), diag_coulomb_mat):
                if indices:
                    n_block = len(indices)
                    rows, cols = zip(*indices)
                    vals = params[index : index + n_block]
                    this_diag_coulomb_mat[cols, rows] = vals
                    this_diag_coulomb_mat[rows, cols] = vals
                    index += n_block
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters(params[index:], norb)
        return UCJOpSpinBalanced(diag_coulomb_mats=diag_coulomb_mats, orbital_rotations=orbital_rotations, final_orbital_rotation=final_orbital_rotation)

    def to_parameters(
        self,
        *,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
    ) -> np.ndarray:
        n_reps, _, norb, _ = self.diag_coulomb_mats.shape
        n_params = UCJOpSpinBalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        triu_indices = cast(list[tuple[int, int]], list(itertools.combinations_with_replacement(range(norb), 2)))
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = triu_indices
        params = np.zeros(n_params, dtype=float)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(self.orbital_rotations, self.diag_coulomb_mats):
            params[index : index + norb**2] = unitary_to_parameters(orbital_rotation)
            index += norb**2
            for indices, this_diag_coulomb_mat in zip((pairs_aa, pairs_ab), diag_coulomb_mat):
                if indices:
                    n_block = len(indices)
                    params[index : index + n_block] = this_diag_coulomb_mat[tuple(zip(*indices))]
                    index += n_block
        if self.final_orbital_rotation is not None:
            params[index:] = unitary_to_parameters(self.final_orbital_rotation)
        return params

    @staticmethod
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback=None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> "UCJOpSpinBalanced":
        if isinstance(n_reps, int) and n_reps <= 0:
            raise ValueError(f"n_reps must be at least 1. Got {n_reps}.")
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False)
        validate_interaction_pairs(pairs_ab, ordered=False)
        nocc, _, nvrt, _ = np.asarray(t2).shape
        norb = nocc + nvrt
        if pairs_aa is None and pairs_ab is None:
            diag_coulomb_indices = None
        else:
            diag_coulomb_indices = list(set((pairs_aa or []) + (pairs_ab or [])))
        diag_coulomb_mats, orbital_rotations = double_factorized_t2(
            np.asarray(t2, dtype=float),
            tol=tol,
            max_terms=n_reps,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            diag_coulomb_indices=diag_coulomb_indices,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            return_optimize_result=False,
        )
        diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
        n_vecs = diag_coulomb_mats.shape[0]
        if n_reps is not None and n_vecs < n_reps:
            diag_coulomb_mats = np.concatenate([diag_coulomb_mats, np.zeros((n_reps - n_vecs, 2, norb, norb))])
            eye = np.eye(norb)
            orbital_rotations = np.concatenate([orbital_rotations, np.stack([eye for _ in range(n_reps - n_vecs)])])
        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation = orbital_rotation_from_t1_amplitudes(np.asarray(t1))
        if pairs_aa is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            if pairs_aa:
                rows, cols = zip(*pairs_aa)
                mask[rows, cols] = True
                mask[cols, rows] = True
            diag_coulomb_mats[:, 0] *= mask
        if pairs_ab is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            if pairs_ab:
                rows, cols = zip(*pairs_ab)
                mask[rows, cols] = True
                mask[cols, rows] = True
            diag_coulomb_mats[:, 1] *= mask
        return UCJOpSpinBalanced(diag_coulomb_mats=diag_coulomb_mats, orbital_rotations=orbital_rotations, final_orbital_rotation=final_orbital_rotation)

    @staticmethod
    def from_cisd_vec(
        cisd_vec: np.ndarray,
        *,
        norb: int,
        nocc: int,
        c0_threshold: float = 1e-12,
        n_reps: int | None = None,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        tol: float = 1e-8,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback=None,
        options: dict | None = None,
        regularization: float = 0.0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> "UCJOpSpinBalanced":
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisd_vec, norb, nocc, copy=False)
        if math.isclose(c0, 0.0, abs_tol=c0_threshold):
            raise ValueError(f"CISD reference coefficient c0={c0} is smaller than the specified threshold, c0_tol={c0_threshold}.")
        t1 = c1 / c0
        t2 = c2 / c0 - 0.5 * np.einsum("ia,jb->ijab", t1, t1)
        return UCJOpSpinBalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            tol=tol,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = np.array(vec, dtype=np.complex128, copy=copy)
        current_basis = np.eye(self.norb, dtype=np.complex128)
        for (same_spin, mixed_spin), orbital_rotation in zip(self.diag_coulomb_mats, self.orbital_rotations):
            out = apply_orbital_rotation(
                out,
                orbital_rotation.T.conj() @ current_basis,
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
            out = gates.apply_ucj_spin_balanced(
                out,
                same_spin_params=same_spin,
                mixed_spin_params=mixed_spin,
                norb=self.norb,
                nelec=nelec,
                time=-1.0,
                orbital_rotation=None,
                copy=False,
            )
            current_basis = orbital_rotation
        if self.final_orbital_rotation is None:
            out = apply_orbital_rotation(out, current_basis, norb=self.norb, nelec=nelec, copy=False)
        else:
            out = apply_orbital_rotation(
                out,
                self.final_orbital_rotation @ current_basis,
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return out

    def _apply_unitary_(self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        return self.apply(vec, nelec=nelec, copy=copy)

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if not isinstance(other, UCJOpSpinBalanced):
            return NotImplemented
        if not np.allclose(self.diag_coulomb_mats, other.diag_coulomb_mats, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.orbital_rotations, other.orbital_rotations, rtol=rtol, atol=atol):
            return False
        if (self.final_orbital_rotation is None) != (other.final_orbital_rotation is None):
            return False
        if self.final_orbital_rotation is not None:
            return np.allclose(cast(np.ndarray, self.final_orbital_rotation), cast(np.ndarray, other.final_orbital_rotation), rtol=rtol, atol=atol)
        return True
