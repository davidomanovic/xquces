from __future__ import annotations

import itertools
from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import occ_rows, reshape_state
from xquces.gcr.igcr2 import (
    IGCR2BlockDiagLeftUnitaryChart,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    IGCR2SpinRestrictedParameterization,
)
from xquces.gcr.igcr3 import (
    IGCR3SpinRestrictedParameterization,
    _default_tau_indices,
    _default_triple_indices,
)
from xquces.gcr.igcr4 import (
    IGCR4SpinRestrictedParameterization,
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
)
from xquces.gcr.spin_balanced_igcr4 import (
    IGCR4SpinBalancedFixedSectorParameterization,
    IGCR4SpinSeparatedFixedSectorParameterization,
)
from xquces.orbitals import ov_generator_from_params


def _antihermitian_basis_from_pairs(
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    basis = np.zeros((2 * len(pairs), norb, norb), dtype=np.complex128)
    for k, (p, q) in enumerate(pairs):
        basis[2 * k, p, q] = 1.0
        basis[2 * k, q, p] = -1.0
        basis[2 * k + 1, p, q] = 1j
        basis[2 * k + 1, q, p] = 1j
    return basis


def _left_chart_basis(chart: object, norb: int) -> np.ndarray:
    if isinstance(chart, IGCR2LeftUnitaryChart):
        pairs = list(itertools.combinations(range(norb), 2))
        return _antihermitian_basis_from_pairs(norb, pairs)
    if isinstance(chart, IGCR2BlockDiagLeftUnitaryChart):
        pairs = list(itertools.combinations(range(chart.nocc), 2))
        pairs += [
            (chart.nocc + p, chart.nocc + q)
            for p, q in itertools.combinations(range(chart.nvirt), 2)
        ]
        return _antihermitian_basis_from_pairs(norb, pairs)
    raise NotImplementedError(type(chart).__name__)


def _left_chart_kappa(
    chart: object,
    params: np.ndarray,
    norb: int,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.size == 0:
        return np.zeros((norb, norb), dtype=np.complex128)
    if basis is None:
        basis = _left_chart_basis(chart, norb)
    return np.tensordot(params, basis, axes=(0, 0))


def _right_chart_basis(chart: object) -> np.ndarray:
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        norb = chart.norb
        nocc = chart.nocc
        nvirt = chart.nvirt
        ncomplex = nocc * nvirt
        basis = np.zeros((2 * ncomplex, norb, norb), dtype=np.complex128)
        for a in range(nvirt):
            for i in range(nocc):
                idx = a * nocc + i
                p = nocc + a
                q = i
                basis[idx, p, q] = 1.0
                basis[idx, q, p] = -1.0
                basis[idx + ncomplex, p, q] = 1j
                basis[idx + ncomplex, q, p] = 1j
        return basis
    if isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        norb = chart.norb
        nocc = chart.nocc
        nvirt = chart.nvirt
        nreal = nocc * nvirt
        basis = np.zeros((nreal, norb, norb), dtype=np.complex128)
        for a in range(nvirt):
            for i in range(nocc):
                idx = a * nocc + i
                p = nocc + a
                q = i
                basis[idx, p, q] = 1.0
                basis[idx, q, p] = -1.0
        return basis
    raise NotImplementedError(type(chart).__name__)


def _right_chart_kappa(chart: object, params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        return ov_generator_from_params(params, norb, chart.nocc)
    if isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        full = np.concatenate([params, np.zeros_like(params)])
        return ov_generator_from_params(full, norb, chart.nocc)
    raise NotImplementedError(type(chart).__name__)


def _generator_batch_from_kappa(
    kappa: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    if basis.shape[0] == 0:
        return np.zeros_like(basis)
    herm = -1j * np.asarray(kappa, dtype=np.complex128)
    eigvals, vecs = np.linalg.eigh(herm)
    delta = 1j * (eigvals[:, None] - eigvals[None, :])
    phi = np.ones_like(delta, dtype=np.complex128)
    mask = np.abs(delta) > 1e-12
    phi[mask] = np.expm1(delta[mask]) / delta[mask]
    basis_eig = np.einsum(
        "pa,jpq,qb->jab",
        vecs.conj(),
        basis,
        vecs,
        optimize=True,
    )
    gen_eig = phi[None, :, :] * basis_eig
    return np.einsum(
        "pa,jab,qb->jpq",
        vecs,
        gen_eig,
        vecs.conj(),
        optimize=True,
    )


@cache
def _bitstrings(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    bits = np.zeros(len(occ), dtype=np.uint64)
    for i, row in enumerate(occ):
        value = 0
        for p in row:
            value |= 1 << int(p)
        bits[i] = value
    return bits


@cache
def _one_body_tensor(norb: int, nocc: int) -> np.ndarray:
    bits = _bitstrings(norb, nocc)
    dim = len(bits)
    index = {int(bit): i for i, bit in enumerate(bits)}
    tensor = np.zeros((norb, norb, dim, dim), dtype=np.complex128)
    for col, bit in enumerate(bits):
        det = int(bit)
        for q in range(norb):
            if ((det >> q) & 1) == 0:
                continue
            sign1 = -1.0 if ((det & ((1 << q) - 1)).bit_count() & 1) else 1.0
            det1 = det ^ (1 << q)
            for p in range(norb):
                if ((det1 >> p) & 1) != 0:
                    continue
                sign2 = -1.0 if ((det1 & ((1 << p) - 1)).bit_count() & 1) else 1.0
                row = index[det1 | (1 << p)]
                tensor[p, q, row, col] += sign1 * sign2
    return tensor


@cache
def _sector_rep_index(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    if nocc == 0:
        return np.zeros((2, 1, 1, 0, 0), dtype=np.int64)
    dim = len(occ)
    rows = np.broadcast_to(occ[:, None, :, None], (dim, dim, nocc, nocc))
    cols = np.broadcast_to(occ[None, :, None, :], (dim, dim, nocc, nocc))
    return np.stack([rows, cols], axis=0)


def _sector_representation(u: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    if nocc == 0:
        return np.ones((1, 1), dtype=np.complex128)
    index = _sector_rep_index(norb, nocc)
    submats = u[index[0], index[1]]
    return np.linalg.det(submats)


def _one_body_batch_to_sector(g_batch: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    if g_batch.shape[0] == 0:
        dim = tensor.shape[-1]
        return np.zeros((0, dim, dim), dtype=np.complex128)
    return np.einsum("jpq,pqmn->jmn", g_batch, tensor, optimize=True)


def _apply_batch_transform(
    left: np.ndarray,
    right: np.ndarray,
    mats: np.ndarray,
) -> np.ndarray:
    if mats.shape[0] == 0:
        return mats
    tmp = np.einsum("am,jmb->jab", left, mats, optimize=True)
    return np.einsum("jab,cb->jac", tmp, right, optimize=True)


def _batch_row_and_col(
    left_batch: np.ndarray,
    right_batch: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    if left_batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    row = np.einsum("jmn,nb->jmb", left_batch, mat, optimize=True)
    col = np.einsum("an,jbn->jab", mat, right_batch, optimize=True)
    return row + col


@cache
def _number_arrays(norb: int, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    dim_a = len(occ_a)
    dim_b = len(occ_b)
    n_a = np.zeros((dim_a, norb), dtype=np.float64)
    n_b = np.zeros((dim_b, norb), dtype=np.float64)
    if occ_a.size:
        n_a[np.arange(dim_a)[:, None], occ_a] = 1.0
    if occ_b.size:
        n_b[np.arange(dim_b)[:, None], occ_b] = 1.0
    n = n_a[:, None, :] + n_b[None, :, :]
    d = n_a[:, None, :] * n_b[None, :, :]
    return n.reshape(dim_a * dim_b, norb), d.reshape(dim_a * dim_b, norb)


def _igcr2_feature_matrix(
    parameterization: IGCR2SpinRestrictedParameterization,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, _ = _number_arrays(parameterization.norb, nelec)
    if not parameterization.pair_indices:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    rows, cols = zip(*parameterization.pair_indices)
    return n[:, rows] * n[:, cols]


def _igcr3_feature_matrix(
    parameterization: IGCR3SpinRestrictedParameterization,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, d = _number_arrays(parameterization.norb, nelec)
    blocks = []
    if parameterization.pair_indices:
        rows, cols = zip(*parameterization.pair_indices)
        blocks.append(n[:, rows] * n[:, cols])
    if parameterization.uses_reduced_cubic_chart:
        tau_idx = _default_tau_indices(parameterization.norb)
        omega_idx = _default_triple_indices(parameterization.norb)
        tau_rows, tau_cols = zip(*tau_idx)
        tau_feat = d[:, tau_rows] * n[:, tau_cols]
        p, q, r = zip(*omega_idx)
        omega_feat = n[:, p] * n[:, q] * n[:, r]
        full = np.concatenate([tau_feat, omega_feat], axis=1)
        blocks.append(full @ parameterization.cubic_reduction.physical_cubic_basis)
    else:
        if parameterization.tau_indices:
            rows, cols = zip(*parameterization.tau_indices)
            blocks.append(d[:, rows] * n[:, cols])
        if parameterization.omega_indices:
            p, q, r = zip(*parameterization.omega_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r])
    if not blocks:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def _igcr4_feature_matrix(
    parameterization: IGCR4SpinRestrictedParameterization,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, d = _number_arrays(parameterization.norb, nelec)
    blocks = []
    if parameterization.pair_indices:
        rows, cols = zip(*parameterization.pair_indices)
        blocks.append(n[:, rows] * n[:, cols])
    if parameterization.uses_reduced_cubic_chart:
        tau_idx = _default_tau_indices(parameterization.norb)
        omega_idx = _default_triple_indices(parameterization.norb)
        tau_rows, tau_cols = zip(*tau_idx)
        tau_feat = d[:, tau_rows] * n[:, tau_cols]
        p, q, r = zip(*omega_idx)
        omega_feat = n[:, p] * n[:, q] * n[:, r]
        full_cubic = np.concatenate([tau_feat, omega_feat], axis=1)
        blocks.append(
            full_cubic @ parameterization.cubic_reduction.physical_cubic_basis
        )
    else:
        if parameterization.tau_indices:
            rows, cols = zip(*parameterization.tau_indices)
            blocks.append(d[:, rows] * n[:, cols])
        if parameterization.omega_indices:
            p, q, r = zip(*parameterization.omega_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r])
    if parameterization.uses_reduced_quartic_chart:
        eta_idx = _default_eta_indices(parameterization.norb)
        rho_idx = _default_rho_indices(parameterization.norb)
        sigma_idx = _default_sigma_indices(parameterization.norb)
        eta_p, eta_q = zip(*eta_idx)
        eta_feat = d[:, eta_p] * d[:, eta_q]
        rho_p, rho_q, rho_r = zip(*rho_idx)
        rho_feat = d[:, rho_p] * n[:, rho_q] * n[:, rho_r]
        sigma_p, sigma_q, sigma_r, sigma_s = zip(*sigma_idx)
        sigma_feat = n[:, sigma_p] * n[:, sigma_q] * n[:, sigma_r] * n[:, sigma_s]
        full_quartic = np.concatenate([eta_feat, rho_feat, sigma_feat], axis=1)
        blocks.append(
            full_quartic @ parameterization.quartic_reduction.physical_quartic_basis
        )
    else:
        if parameterization.eta_indices:
            p, q = zip(*parameterization.eta_indices)
            blocks.append(d[:, p] * d[:, q])
        if parameterization.rho_indices:
            p, q, r = zip(*parameterization.rho_indices)
            blocks.append(d[:, p] * n[:, q] * n[:, r])
        if parameterization.sigma_indices:
            p, q, r, s = zip(*parameterization.sigma_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r] * n[:, s])
    if not blocks:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def _diag_feature_matrix(
    parameterization: object,
    nelec: tuple[int, int],
) -> np.ndarray:
    if isinstance(parameterization, IGCR2SpinRestrictedParameterization):
        return _igcr2_feature_matrix(parameterization, nelec)
    if isinstance(parameterization, IGCR3SpinRestrictedParameterization):
        return _igcr3_feature_matrix(parameterization, nelec)
    if isinstance(parameterization, IGCR4SpinRestrictedParameterization):
        return _igcr4_feature_matrix(parameterization, nelec)
    if isinstance(parameterization, IGCR4SpinBalancedFixedSectorParameterization):
        if tuple(nelec) != tuple(parameterization.nelec):
            raise ValueError(
                "spin-balanced fixed-sector parameterization got wrong nelec"
            )
        return parameterization.diagonal_basis.features
    if isinstance(parameterization, IGCR4SpinSeparatedFixedSectorParameterization):
        if tuple(nelec) != tuple(parameterization.nelec):
            raise ValueError(
                "spin-separated fixed-sector parameterization got wrong nelec"
            )
        return parameterization.diagonal_basis.features
    raise TypeError(type(parameterization).__name__)


def _public_to_native_matrix(parameterization: object) -> np.ndarray | None:
    scale = getattr(parameterization, "_left_right_ov_transform_scale", None)
    if scale is None:
        return None
    n = parameterization.n_params
    eye = np.eye(n, dtype=np.float64)
    return np.column_stack(
        [parameterization._native_parameters_from_public(eye[:, k]) for k in range(n)]
    )


def make_restricted_gcr_jacobian(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
        | IGCR4SpinBalancedFixedSectorParameterization
        | IGCR4SpinSeparatedFixedSectorParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(parameterization, IGCR4SpinSeparatedFixedSectorParameterization):
        return _make_spin_separated_gcr_jacobian(parameterization, reference_vec, nelec)

    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _diag_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        native = parameterization._native_parameters_from_public(params)

        n_left = parameterization.n_left_orbital_rotation_params
        right_start = parameterization._right_orbital_rotation_start
        n_right = parameterization.n_right_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left:right_start]
        right_params = native[right_start : right_start + n_right]

        u_left = left_chart.unitary_from_parameters(left_params, norb)
        u_final = right_chart.unitary_from_parameters(right_params, norb)
        u_right = u_left.conj().T @ u_final

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_final = _right_chart_kappa(right_chart, right_params, norb)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_right_a = _sector_representation(u_right, norb, nelec[0])
        rep_right_b = _sector_representation(u_right, norb, nelec[1])

        rotated_right = rep_right_a @ reference_mat @ rep_right_b.T

        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * rotated_right
        state = rep_left_a @ diagonalized @ rep_left_b.T

        blocks = []

        if n_left:
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            gen_right_from_left = -np.matmul(
                u_left.conj().T,
                np.matmul(gen_left, u_left),
            )

            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            right_a = _one_body_batch_to_sector(gen_right_from_left, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_left, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _batch_row_and_col(left_a, left_b, state)
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_left, dim_a * dim_b).T)

        if diag_params.size:
            d_diagonalized = (
                1j
                * diag_features.T.reshape(diag_params.size, dim_a, dim_b)
                * diagonalized[None, :, :]
            )
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(diag_params.size, dim_a * dim_b).T)

        if n_right:
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            gen_right_from_final = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )

            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_right, dim_a * dim_b).T)

        if blocks:
            out = np.hstack(blocks)
        else:
            out = np.zeros((dim_a * dim_b, 0), dtype=np.complex128)

        if transform is not None:
            out = out @ transform
        return out

    return jac


def _batch_left_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("jmn,nb->jmb", batch, mat, optimize=True)


def _batch_right_transpose_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("an,jbn->jab", mat, batch, optimize=True)


def _make_spin_separated_gcr_jacobian(
    parameterization: IGCR4SpinSeparatedFixedSectorParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    if tuple(nelec) != tuple(parameterization.nelec):
        raise ValueError("spin-separated fixed-sector parameterization got wrong nelec")

    norb = parameterization.norb
    left_chart_alpha = parameterization.left_orbital_chart_alpha
    left_chart_beta = parameterization.left_orbital_chart_beta
    right_chart_alpha = parameterization.right_orbital_chart_alpha
    right_chart_beta = parameterization.right_orbital_chart_beta

    left_basis_alpha = _left_chart_basis(left_chart_alpha, norb)
    left_basis_beta = _left_chart_basis(left_chart_beta, norb)
    right_basis_alpha = _right_chart_basis(right_chart_alpha)
    right_basis_beta = _right_chart_basis(right_chart_beta)

    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = parameterization.diagonal_basis.features

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )

        (
            left_alpha_params,
            left_beta_params,
            diag_params,
            right_alpha_params,
            right_beta_params,
        ) = parameterization._split_params(params)

        u_left_alpha = left_chart_alpha.unitary_from_parameters(
            left_alpha_params,
            norb,
        )
        u_left_beta = left_chart_beta.unitary_from_parameters(left_beta_params, norb)
        u_final_alpha = right_chart_alpha.unitary_from_parameters(
            right_alpha_params,
            norb,
        )
        u_final_beta = right_chart_beta.unitary_from_parameters(
            right_beta_params,
            norb,
        )
        u_right_alpha = u_left_alpha.conj().T @ u_final_alpha
        u_right_beta = u_left_beta.conj().T @ u_final_beta

        kappa_left_alpha = _left_chart_kappa(
            left_chart_alpha,
            left_alpha_params,
            norb,
            basis=left_basis_alpha,
        )
        kappa_left_beta = _left_chart_kappa(
            left_chart_beta,
            left_beta_params,
            norb,
            basis=left_basis_beta,
        )
        kappa_final_alpha = _right_chart_kappa(
            right_chart_alpha,
            right_alpha_params,
            norb,
        )
        kappa_final_beta = _right_chart_kappa(
            right_chart_beta,
            right_beta_params,
            norb,
        )

        rep_left_alpha = _sector_representation(u_left_alpha, norb, nelec[0])
        rep_left_beta = _sector_representation(u_left_beta, norb, nelec[1])
        rep_right_alpha = _sector_representation(u_right_alpha, norb, nelec[0])
        rep_right_beta = _sector_representation(u_right_beta, norb, nelec[1])

        rotated_right = rep_right_alpha @ reference_mat @ rep_right_beta.T
        phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        diagonalized = phase * rotated_right
        state = rep_left_alpha @ diagonalized @ rep_left_beta.T

        blocks = []

        if left_alpha_params.size:
            gen_left = _generator_batch_from_kappa(
                kappa_left_alpha,
                left_basis_alpha,
            )
            gen_right_from_left = -np.matmul(
                u_left_alpha.conj().T,
                np.matmul(gen_left, u_left_alpha),
            )
            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            right_a = _one_body_batch_to_sector(gen_right_from_left, tensor_a)
            d_rotated_right = _batch_left_multiply(right_a, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _batch_left_multiply(left_a, state)
            d_state += _apply_batch_transform(
                rep_left_alpha,
                rep_left_beta,
                d_diagonalized,
            )
            blocks.append(d_state.reshape(left_alpha_params.size, dim_a * dim_b).T)

        if left_beta_params.size:
            gen_left = _generator_batch_from_kappa(
                kappa_left_beta,
                left_basis_beta,
            )
            gen_right_from_left = -np.matmul(
                u_left_beta.conj().T,
                np.matmul(gen_left, u_left_beta),
            )
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            right_b = _one_body_batch_to_sector(gen_right_from_left, tensor_b)
            d_rotated_right = _batch_right_transpose_multiply(
                right_b,
                rotated_right,
            )
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _batch_right_transpose_multiply(left_b, state)
            d_state += _apply_batch_transform(
                rep_left_alpha,
                rep_left_beta,
                d_diagonalized,
            )
            blocks.append(d_state.reshape(left_beta_params.size, dim_a * dim_b).T)

        if diag_params.size:
            d_diagonalized = (
                1j
                * diag_features.T.reshape(diag_params.size, dim_a, dim_b)
                * diagonalized[None, :, :]
            )
            d_state = _apply_batch_transform(
                rep_left_alpha,
                rep_left_beta,
                d_diagonalized,
            )
            blocks.append(d_state.reshape(diag_params.size, dim_a * dim_b).T)

        if right_alpha_params.size:
            gen_final = _generator_batch_from_kappa(
                kappa_final_alpha,
                right_basis_alpha,
            )
            gen_right_from_final = np.matmul(
                u_left_alpha.conj().T,
                np.matmul(gen_final, u_left_alpha),
            )
            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            d_rotated_right = _batch_left_multiply(right_a, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _apply_batch_transform(
                rep_left_alpha,
                rep_left_beta,
                d_diagonalized,
            )
            blocks.append(d_state.reshape(right_alpha_params.size, dim_a * dim_b).T)

        if right_beta_params.size:
            gen_final = _generator_batch_from_kappa(
                kappa_final_beta,
                right_basis_beta,
            )
            gen_right_from_final = np.matmul(
                u_left_beta.conj().T,
                np.matmul(gen_final, u_left_beta),
            )
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)
            d_rotated_right = _batch_right_transpose_multiply(right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _apply_batch_transform(
                rep_left_alpha,
                rep_left_beta,
                d_diagonalized,
            )
            blocks.append(d_state.reshape(right_beta_params.size, dim_a * dim_b).T)

        if not blocks:
            return np.zeros((dim_a * dim_b, 0), dtype=np.complex128)
        return np.hstack(blocks)

    return jac
