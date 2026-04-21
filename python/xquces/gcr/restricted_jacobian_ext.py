from __future__ import annotations

from typing import Callable

import numpy as np

from xquces.basis import reshape_state
from xquces.gcr.doci_reference_gcr2 import (
    GCR2DOCIReferenceParameterization,
    _doci_unitary_from_params,
    apply_doci_reference_global,
)
from xquces.gcr.commutator_gcr2 import _diag2_features
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


def make_doci_reference_gcr_jacobian(
    parameterization: GCR2DOCIReferenceParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    diag_features = _diag2_features(norb, nelec, parameterization.pair_indices)
    transform = _public_to_native_matrix(parameterization)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(f"Expected {(parameterization.n_params,)}, got {params.shape}.")
        native = np.asarray(params, dtype=np.float64)

        n_left = parameterization.n_left_orbital_rotation_params
        n_diag = parameterization.n_diag_params
        n_doci = parameterization.n_doci_reference_params
        middle_start = parameterization._right_orbital_rotation_start
        n_middle = parameterization.n_middle_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left : n_left + n_diag]
        doci_params = native[n_left + n_diag : middle_start]
        middle_params = native[middle_start : middle_start + n_middle]

        u_left = left_chart.unitary_from_parameters(left_params, norb)
        u_middle = middle_chart.unitary_from_parameters(middle_params, norb)

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_middle = _left_chart_kappa(middle_chart, middle_params, norb, basis=middle_basis)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_middle_a = _sector_representation(u_middle, norb, nelec[0])
        rep_middle_b = _sector_representation(u_middle, norb, nelec[1])

        doci_unitary = _doci_unitary_from_params(doci_params, norb, nelec)
        doci_state_vec = apply_doci_reference_global(
            reference_vec,
            doci_params,
            norb,
            nelec,
            copy=True,
            unitary=doci_unitary,
        )
        doci_state = reshape_state(doci_state_vec, norb, nelec)
        rotated_middle = rep_middle_a @ doci_state @ rep_middle_b.T

        dim_a, dim_b = rotated_middle.shape
        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * rotated_middle
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

        if n_doci:
            eps = 1e-7
            d_doci_state = np.zeros((n_doci, dim_a, dim_b), dtype=np.complex128)
            for k in range(n_doci):
                step = np.zeros_like(doci_params)
                step[k] = eps
                plus = apply_doci_reference_global(
                    reference_vec,
                    doci_params + step,
                    norb,
                    nelec,
                    copy=True,
                )
                minus = apply_doci_reference_global(
                    reference_vec,
                    doci_params - step,
                    norb,
                    nelec,
                    copy=True,
                )
                plus_mat = reshape_state(plus, norb, nelec)
                minus_mat = reshape_state(minus, norb, nelec)
                plus_rot = rep_middle_a @ plus_mat @ rep_middle_b.T
                minus_rot = rep_middle_a @ minus_mat @ rep_middle_b.T
                d_doci_state[k] = (plus_rot - minus_rot) / (2 * eps)
            d_diagonalized = phase[None, :, :] * d_doci_state
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_doci, dim_a * dim_b).T)

        if n_middle:
            gen_middle = _generator_batch_from_kappa(kappa_middle, middle_basis)
            middle_a = _one_body_batch_to_sector(gen_middle, tensor_a)
            middle_b = _one_body_batch_to_sector(gen_middle, tensor_b)
            d_rotated_middle = _batch_row_and_col(middle_a, middle_b, rotated_middle)
            d_diagonalized = phase[None, :, :] * d_rotated_middle
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_middle, dim_a * dim_b).T)

        if blocks:
            out = np.hstack(blocks)
        else:
            out = np.zeros((dim_a * dim_b, 0), dtype=np.complex128)

        if transform is not None:
            out = out @ transform
        return out

    return jac


def make_restricted_gcr_jacobian(parameterization, reference_vec: np.ndarray, nelec: tuple[int, int]):
    if isinstance(parameterization, GCR2DOCIReferenceParameterization):
        return make_doci_reference_gcr_jacobian(parameterization, reference_vec, nelec)
    return _base_make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)


__all__ = ["make_restricted_gcr_jacobian"]
