from __future__ import annotations

from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import reshape_state
from xquces.gcr.controlled_orbital_gcr2 import (
    GCR2SpectatorOrbitalParameterization,
    _apply_spectator_controlled_rotation,
    _pair_rotation_unitary,
    _spectator_sector_indices,
)
from xquces.gcr.doci_reference_gcr2 import (
    GCR2DOCIReferenceParameterization,
    _doci_unitary_from_params,
    apply_doci_reference_global,
)
from xquces.gcr.bridge_gcr2 import GCR2FullUnitaryChart
from xquces.gcr.doci_reference_gcr3 import GCR3DOCIReferenceParameterization
from xquces.gcr.doci_reference_gcr4 import GCR4DOCIReferenceParameterization
from xquces.gcr.restricted_jacobian import (
    _apply_batch_transform,
    _batch_row_and_col,
    _diag_feature_matrix,
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
    make_restricted_gcr_subspace_jacobian as _base_make_restricted_gcr_subspace_jacobian,
)
from xquces.orbitals import apply_orbital_rotation


def make_doci_reference_gcr_jacobian(
    parameterization,
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
    diag_features = _diag_feature_matrix(parameterization._base, nelec)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )

        n_left = parameterization.n_left_orbital_rotation_params
        n_diag = parameterization.n_diag_params
        n_doci = parameterization.n_doci_reference_params
        middle_start = parameterization._right_orbital_rotation_start
        n_middle = parameterization.n_middle_orbital_rotation_params

        left_params = params[:n_left]
        diag_params = params[n_left : n_left + n_diag]
        doci_params = params[n_left + n_diag : middle_start]
        middle_params = params[middle_start : middle_start + n_middle]

        u_left = left_chart.unitary_from_parameters(left_params, norb)
        u_middle = middle_chart.unitary_from_parameters(middle_params, norb)

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_middle = _left_chart_kappa(
            middle_chart, middle_params, norb, basis=middle_basis
        )

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
            return np.hstack(blocks)
        return np.zeros((dim_a * dim_b, 0), dtype=np.complex128)

    return jac


@cache
def _pair_generator_sector_ops(
    norb: int,
    nelec: tuple[int, int],
    p: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    gen = np.zeros((1, norb, norb), dtype=np.complex128)
    gen[0, p, q] = -1.0
    gen[0, q, p] = 1.0
    op_a = _one_body_batch_to_sector(gen, _one_body_tensor(norb, nelec[0]))[0]
    op_b = _one_body_batch_to_sector(gen, _one_body_tensor(norb, nelec[1]))[0]
    return op_a, op_b


def _apply_spectator_sequence(
    vec: np.ndarray,
    spectator_params: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    norb: int,
    nelec: tuple[int, int],
    start: int = 0,
) -> np.ndarray:
    out = np.array(vec, dtype=np.complex128, copy=True)
    for theta, (r, p, q) in zip(spectator_params[start:], triples[start:]):
        out = _apply_spectator_controlled_rotation(
            out,
            float(theta),
            r,
            p,
            q,
            norb,
            nelec,
            copy=False,
        )
    return out


def _apply_spectator_controlled_rotation_derivative(
    vec: np.ndarray,
    theta: float,
    spectator: int,
    p: int,
    q: int,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    out = np.zeros_like(np.asarray(vec, dtype=np.complex128))
    sector00, sector22 = _spectator_sector_indices(norb, nelec, spectator)
    if sector00.size == 0 and sector22.size == 0:
        return out
    op_a, op_b = _pair_generator_sector_ops(norb, nelec, p, q)
    if sector00.size:
        tmp = np.zeros_like(out)
        tmp[sector00] = np.asarray(vec, dtype=np.complex128)[sector00]
        rotated = apply_orbital_rotation(
            tmp,
            _pair_rotation_unitary(norb, p, q, -theta),
            norb,
            nelec,
            copy=False,
        )
        rotated_mat = reshape_state(rotated, norb, nelec)
        out -= _batch_row_and_col(op_a[None, :, :], op_b[None, :, :], rotated_mat)[
            0
        ].reshape(-1)
    if sector22.size:
        tmp = np.zeros_like(out)
        tmp[sector22] = np.asarray(vec, dtype=np.complex128)[sector22]
        rotated = apply_orbital_rotation(
            tmp,
            _pair_rotation_unitary(norb, p, q, theta),
            norb,
            nelec,
            copy=False,
        )
        rotated_mat = reshape_state(rotated, norb, nelec)
        out += _batch_row_and_col(op_a[None, :, :], op_b[None, :, :], rotated_mat)[
            0
        ].reshape(-1)
    return out


def make_spectator_orbital_gcr_jacobian(
    parameterization: GCR2SpectatorOrbitalParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    norb = parameterization.norb
    base = parameterization._base
    left_chart = base._left_orbital_chart
    right_chart = base.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    dim = dim_a * dim_b
    diag_features = _diag_feature_matrix(base, nelec)
    base_transform = _public_to_native_matrix(base)
    triples = parameterization.triple_indices
    spectator_transform = parameterization.spectator_transform

    n_left_public = parameterization.n_left_orbital_rotation_params
    n_diag_public = parameterization.n_pair_params
    n_spectator = parameterization.n_spectator_params
    n_right_public = parameterization.n_right_orbital_rotation_params
    n_full_spectator = parameterization.n_full_spectator_terms

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )

        left_public, _, spectator_reduced, right_public = parameterization._split(
            params
        )
        spectator_full = parameterization.full_spectator_params_from_reduced(
            spectator_reduced
        )
        base_public = np.concatenate(
            [
                left_public,
                params[n_left_public : n_left_public + n_diag_public],
                right_public,
            ]
        )
        native = base._native_parameters_from_public(base_public)

        n_left_native = base.n_left_orbital_rotation_params
        right_start = base._right_orbital_rotation_start
        n_right_native = base.n_right_orbital_rotation_params

        left_native = native[:n_left_native]
        diag_native = native[n_left_native:right_start]
        right_native = native[right_start : right_start + n_right_native]

        u_left = left_chart.unitary_from_parameters(left_native, norb)
        u_final = right_chart.unitary_from_parameters(right_native, norb)
        u_right = u_left.conj().T @ u_final

        kappa_left = _left_chart_kappa(left_chart, left_native, norb, basis=left_basis)
        kappa_final = _right_chart_kappa(right_chart, right_native, norb)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_right_a = _sector_representation(u_right, norb, nelec[0])
        rep_right_b = _sector_representation(u_right, norb, nelec[1])

        rotated_right = rep_right_a @ reference_mat @ rep_right_b.T

        if diag_features.shape[1]:
            phase_half = np.exp(0.5j * (diag_features @ diag_native)).reshape(
                dim_a, dim_b
            )
        else:
            phase_half = np.ones((dim_a, dim_b), dtype=np.complex128)

        before_middle = phase_half * rotated_right
        prefix = [before_middle.reshape(-1)]
        for theta, (r, p, q) in zip(spectator_full, triples):
            prefix.append(
                _apply_spectator_controlled_rotation(
                    prefix[-1],
                    float(theta),
                    r,
                    p,
                    q,
                    norb,
                    nelec,
                    copy=True,
                )
            )

        after_middle = reshape_state(prefix[-1], norb, nelec)
        before_left = phase_half * after_middle
        state = rep_left_a @ before_left @ rep_left_b.T

        native_blocks = []

        if n_left_native:
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            gen_right_from_left = -np.matmul(
                u_left.conj().T,
                np.matmul(gen_left, u_left),
            )
            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            right_a_from_left = _one_body_batch_to_sector(gen_right_from_left, tensor_a)
            right_b_from_left = _one_body_batch_to_sector(gen_right_from_left, tensor_b)
            d_rotated_right_left = _batch_row_and_col(
                right_a_from_left,
                right_b_from_left,
                rotated_right,
            )
            direct_left = _batch_row_and_col(left_a, left_b, state)
            left_cols = np.empty((dim, n_left_native), dtype=np.complex128)
            for j in range(n_left_native):
                vec0 = (phase_half * d_rotated_right_left[j]).reshape(-1)
                vec1 = _apply_spectator_sequence(
                    vec0,
                    spectator_full,
                    triples,
                    norb,
                    nelec,
                )
                mat = phase_half * reshape_state(vec1, norb, nelec)
                propagated = rep_left_a @ mat @ rep_left_b.T
                left_cols[:, j] = (direct_left[j] + propagated).reshape(-1)
            native_blocks.append(left_cols)
        else:
            native_blocks.append(np.zeros((dim, 0), dtype=np.complex128))

        if diag_native.size:
            diag_cols = np.empty((dim, diag_native.size), dtype=np.complex128)
            for j in range(diag_native.size):
                feature = diag_features[:, j].reshape(dim_a, dim_b)
                first_half_vec = (0.5j * feature * before_middle).reshape(-1)
                first_half_vec = _apply_spectator_sequence(
                    first_half_vec,
                    spectator_full,
                    triples,
                    norb,
                    nelec,
                )
                first_half = phase_half * reshape_state(first_half_vec, norb, nelec)
                second_half = 0.5j * feature * before_left
                total = rep_left_a @ (first_half + second_half) @ rep_left_b.T
                diag_cols[:, j] = total.reshape(-1)
            native_blocks.append(diag_cols)
        else:
            native_blocks.append(np.zeros((dim, 0), dtype=np.complex128))

        if n_right_native:
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            gen_right = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )
            right_a = _one_body_batch_to_sector(gen_right, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right, tensor_b)
            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            right_cols = np.empty((dim, n_right_native), dtype=np.complex128)
            for j in range(n_right_native):
                vec0 = (phase_half * d_rotated_right[j]).reshape(-1)
                vec1 = _apply_spectator_sequence(
                    vec0,
                    spectator_full,
                    triples,
                    norb,
                    nelec,
                )
                mat = phase_half * reshape_state(vec1, norb, nelec)
                total = rep_left_a @ mat @ rep_left_b.T
                right_cols[:, j] = total.reshape(-1)
            native_blocks.append(right_cols)
        else:
            native_blocks.append(np.zeros((dim, 0), dtype=np.complex128))

        d_base_native = np.hstack(native_blocks)
        d_base_public = (
            d_base_native if base_transform is None else d_base_native @ base_transform
        )

        if n_full_spectator:
            spectator_cols_full = np.empty((dim, n_full_spectator), dtype=np.complex128)
            for j, ((r, p, q), theta) in enumerate(zip(triples, spectator_full)):
                d_mid = _apply_spectator_controlled_rotation_derivative(
                    prefix[j],
                    float(theta),
                    r,
                    p,
                    q,
                    norb,
                    nelec,
                )
                d_mid = _apply_spectator_sequence(
                    d_mid,
                    spectator_full,
                    triples,
                    norb,
                    nelec,
                    start=j + 1,
                )
                mat = phase_half * reshape_state(d_mid, norb, nelec)
                total = rep_left_a @ mat @ rep_left_b.T
                spectator_cols_full[:, j] = total.reshape(-1)
            spectator_cols = spectator_cols_full @ spectator_transform
        else:
            spectator_cols = np.zeros((dim, n_spectator), dtype=np.complex128)

        left_public_block = d_base_public[:, :n_left_public]
        diag_public_block = d_base_public[
            :, n_left_public : n_left_public + n_diag_public
        ]
        right_public_block = d_base_public[
            :,
            n_left_public + n_diag_public : n_left_public
            + n_diag_public
            + n_right_public,
        ]
        return np.hstack(
            [
                left_public_block,
                diag_public_block,
                spectator_cols,
                right_public_block,
            ]
        )

    return jac


def make_restricted_gcr_jacobian(
    parameterization, reference_vec: np.ndarray, nelec: tuple[int, int]
):
    if isinstance(
        parameterization,
        (
            GCR2DOCIReferenceParameterization,
            GCR3DOCIReferenceParameterization,
            GCR4DOCIReferenceParameterization,
        ),
    ):
        return make_doci_reference_gcr_jacobian(parameterization, reference_vec, nelec)
    if isinstance(parameterization, GCR2SpectatorOrbitalParameterization):
        return make_spectator_orbital_gcr_jacobian(
            parameterization, reference_vec, nelec
        )
    return _base_make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)


def make_restricted_gcr_subspace_jacobian(
    parameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
):
    return _base_make_restricted_gcr_subspace_jacobian(
        parameterization,
        reference_vec,
        nelec,
    )


__all__ = [
    "make_restricted_gcr_jacobian",
    "make_restricted_gcr_subspace_jacobian",
]
