from __future__ import annotations

from typing import Callable

import numpy as np

from xquces.basis import reshape_state
from xquces.gcr.commutator_gcr2 import _diag2_features
from xquces.gcr.pair_reference_gcr2 import (
    GCR2PairReferenceParameterization,
    _doci_unitary_from_params,
    apply_pair_reference_global,
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
    mid_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    mid_basis = _left_chart_basis(mid_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    diag_features = _diag2_features(norb, nelec, parameterization.pair_indices)
    transform = _public_to_native_matrix(parameterization)

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
        u_mid = mid_chart.unitary_from_parameters(right_params, norb)

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_mid = _left_chart_kappa(mid_chart, right_params, norb, basis=mid_basis)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_mid_a = _sector_representation(u_mid, norb, nelec[0])
        rep_mid_b = _sector_representation(u_mid, norb, nelec[1])

        pair_reference_unitary = _doci_unitary_from_params(
            pair_reference_params,
            norb,
            nelec,
        )

        doci_state_vec = apply_pair_reference_global(
            reference_vec,
            pair_reference_params,
            norb,
            nelec,
            parameterization.pair_reference_indices,
            copy=True,
            unitary=pair_reference_unitary,
        )
        doci_state = reshape_state(doci_state_vec, norb, nelec)
        rotated_mid = rep_mid_a @ doci_state @ rep_mid_b.T

        dim_a, dim_b = rotated_mid.shape

        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * rotated_mid
        state = rep_left_a @ diagonalized @ rep_left_b.T

        blocks = []

        if n_left:
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            d_state = _batch_row_and_col(left_a, left_b, state)
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
            eps = 1e-7
            d_pair_state = np.zeros((n_pair_ref, dim_a, dim_b), dtype=np.complex128)
            for k in range(n_pair_ref):
                step = np.zeros_like(pair_reference_params)
                step[k] = eps
                plus = apply_pair_reference_global(
                    reference_vec,
                    pair_reference_params + step,
                    norb,
                    nelec,
                    parameterization.pair_reference_indices,
                    copy=True,
                )
                minus = apply_pair_reference_global(
                    reference_vec,
                    pair_reference_params - step,
                    norb,
                    nelec,
                    parameterization.pair_reference_indices,
                    copy=True,
                )
                plus_mat = reshape_state(plus, norb, nelec)
                minus_mat = reshape_state(minus, norb, nelec)
                plus_rot = rep_mid_a @ plus_mat @ rep_mid_b.T
                minus_rot = rep_mid_a @ minus_mat @ rep_mid_b.T
                d_pair_state[k] = (plus_rot - minus_rot) / (2 * eps)
            d_diagonalized = phase[None, :, :] * d_pair_state
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_pair_ref, dim_a * dim_b).T)

        if n_right:
            gen_mid = _generator_batch_from_kappa(kappa_mid, mid_basis)
            mid_a = _one_body_batch_to_sector(gen_mid, tensor_a)
            mid_b = _one_body_batch_to_sector(gen_mid, tensor_b)
            d_rotated_mid = _batch_row_and_col(mid_a, mid_b, rotated_mid)
            d_diagonalized = phase[None, :, :] * d_rotated_mid
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