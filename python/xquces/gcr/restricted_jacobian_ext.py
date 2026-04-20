from __future__ import annotations

from typing import Callable

import numpy as np

from xquces.basis import reshape_state
from xquces.gcr.commutator_gcr2 import (
    _diag2_features,
    _edge_coloring_matchings,
    _pair_hop_gate_arrays,
)
from xquces.gcr.pair_reference_gcr2 import (
    GCR2PairReferenceParameterization,
    _apply_pair_reference_product_batch,
    _pair_reference_derivative_vectors,
    apply_pair_reference_product,
)
from xquces.gcr.restricted_jacobian import (
    _apply_batch_transform,
    _batch_row_and_col,
    _generator_batch_from_kappa,
    _left_chart_basis,
    _left_chart_kappa,
    _one_body_batch_to_sector,
    _one_body_tensor,
    _public_to_native_matrix,
    _right_chart_basis,
    _right_chart_kappa,
    _sector_representation,
    make_restricted_gcr_jacobian as _base_make_restricted_gcr_jacobian,
)


def make_pair_reference_gcr_jacobian(
    parameterization: GCR2PairReferenceParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
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
    diag_features = _diag2_features(norb, nelec, parameterization.pair_indices)
    transform = _public_to_native_matrix(parameterization)
    source, target, sign, starts = _pair_hop_gate_arrays(
        norb, nelec, parameterization.pair_indices
    )
    order = _edge_coloring_matchings(norb, parameterization.pair_indices)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        native = parameterization._native_parameters_from_public(params)

        n_left = parameterization.n_left_orbital_rotation_params
        n_pair = parameterization.n_pair_params
        n_pair_ref = parameterization.n_pair_reference_params
        right_start = parameterization._right_orbital_rotation_start
        n_right = parameterization.n_right_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left : n_left + n_pair]
        pair_reference_params = native[n_left + n_pair : right_start]
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
        rotated_right_vec = rotated_right.reshape(-1)
        pair_state_vec = apply_pair_reference_product(
            rotated_right_vec,
            pair_reference_params,
            norb,
            nelec,
            parameterization.pair_indices,
            copy=True,
        )
        pair_state = pair_state_vec.reshape(dim_a, dim_b)

        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * pair_state
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
            d_pair_state = _apply_pair_reference_product_batch(
                d_rotated_right.reshape(n_left, dim_a * dim_b),
                pair_reference_params,
                norb,
                parameterization.pair_indices,
                source,
                target,
                sign,
                starts,
                order=order,
            ).reshape(n_left, dim_a, dim_b)
            d_diagonalized = phase[None, :, :] * d_pair_state
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

        if n_pair_ref:
            d_pair_state = _pair_reference_derivative_vectors(
                rotated_right_vec,
                pair_reference_params,
                norb,
                parameterization.pair_indices,
                source,
                target,
                sign,
                starts,
                order=order,
            ).reshape(n_pair_ref, dim_a, dim_b)
            d_diagonalized = phase[None, :, :] * d_pair_state
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_pair_ref, dim_a * dim_b).T)

        if n_right:
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            gen_right_from_final = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )

            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_pair_state = _apply_pair_reference_product_batch(
                d_rotated_right.reshape(n_right, dim_a * dim_b),
                pair_reference_params,
                norb,
                parameterization.pair_indices,
                source,
                target,
                sign,
                starts,
                order=order,
            ).reshape(n_right, dim_a, dim_b)
            d_diagonalized = phase[None, :, :] * d_pair_state
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


def make_restricted_gcr_jacobian(
    parameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
):
    if isinstance(parameterization, GCR2PairReferenceParameterization):
        return make_pair_reference_gcr_jacobian(parameterization, reference_vec, nelec)
    return _base_make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)
