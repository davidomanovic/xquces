from __future__ import annotations

import itertools
from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import occ_rows, reshape_state
from xquces.gcr.charts import (
    GCR2FullUnitaryChart,
    GCR2TraceFixedFullUnitaryChart,
    IGCR2BlockDiagLeftUnitaryChart,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
)
from xquces.gcr.igcr import (
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCR4SpinRestrictedParameterization,
)
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
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


def _full_antihermitian_basis(norb: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(norb), 2))
    basis = np.zeros((norb * norb, norb, norb), dtype=np.complex128)

    idx = 0
    for p in range(norb):
        basis[idx, p, p] = 1j
        idx += 1

    offdiag = _antihermitian_basis_from_pairs(norb, pairs)
    basis[idx:] = offdiag
    return basis


def _trace_fixed_full_antihermitian_basis(norb: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(norb), 2))
    basis = np.zeros((norb * norb - 1, norb, norb), dtype=np.complex128)

    idx = 0
    for p in range(max(0, norb - 1)):
        basis[idx, p, p] = 1j
        basis[idx, norb - 1, norb - 1] = -1j
        idx += 1

    offdiag = _antihermitian_basis_from_pairs(norb, pairs)
    basis[idx:] = offdiag
    return basis


def _left_chart_basis(chart: object, norb: int) -> np.ndarray:
    if isinstance(chart, GCR2FullUnitaryChart):
        return _full_antihermitian_basis(norb)
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        return _trace_fixed_full_antihermitian_basis(norb)
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


def _right_chart_basis(chart: object, norb: int) -> np.ndarray:
    if isinstance(chart, GCR2FullUnitaryChart):
        return _full_antihermitian_basis(norb)
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        return _trace_fixed_full_antihermitian_basis(norb)
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
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
    if isinstance(chart, (IGCR2LeftUnitaryChart, IGCR2BlockDiagLeftUnitaryChart)):
        return _left_chart_basis(chart, norb)
    raise NotImplementedError(type(chart).__name__)


def _right_chart_kappa(chart: object, params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if isinstance(chart, GCR2FullUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        basis = _full_antihermitian_basis(norb)
        return np.tensordot(params, basis, axes=(0, 0))
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        basis = _trace_fixed_full_antihermitian_basis(norb)
        return np.tensordot(params, basis, axes=(0, 0))
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        return ov_generator_from_params(params, norb, chart.nocc)
    if isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        full = np.concatenate([params, np.zeros_like(params)])
        return ov_generator_from_params(full, norb, chart.nocc)
    if isinstance(chart, (IGCR2LeftUnitaryChart, IGCR2BlockDiagLeftUnitaryChart)):
        return _left_chart_kappa(chart, params, norb)
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


def _finite_difference_restricted_gcr_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    step: float = 1e-6,
) -> Callable[[np.ndarray], np.ndarray]:
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    dim = reference_vec.size

    def state(params: np.ndarray) -> np.ndarray:
        return parameterization.ansatz_from_parameters(params).apply(
            reference_vec, nelec=nelec, copy=True
        )

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        out = np.empty((dim, parameterization.n_params), dtype=np.complex128)
        for idx in range(parameterization.n_params):
            h = step * max(1.0, abs(float(params[idx])))
            plus = params.copy()
            minus = params.copy()
            plus[idx] += h
            minus[idx] -= h
            out[:, idx] = (state(plus) - state(minus)) / (2.0 * h)
        return out

    return jac


def _finite_difference_restricted_gcr_subspace_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    step: float = 1e-6,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    dim = reference_vec.size

    def state(params: np.ndarray) -> np.ndarray:
        return parameterization.ansatz_from_parameters(params).apply(
            reference_vec, nelec=nelec, copy=True
        )

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        out = np.zeros((dim, directions.shape[1]), dtype=np.complex128)
        param_scale = max(1.0, float(np.linalg.norm(params)))
        for idx in range(directions.shape[1]):
            direction = directions[:, idx]
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0.0:
                continue
            h = step * param_scale / direction_norm
            out[:, idx] = (
                state(params + h * direction) - state(params - h * direction)
            ) / (2.0 * h)
        return out

    return subspace_jac


def _parse_layered_igcr2_native(
    parameterization: IGCR2SpinRestrictedParameterization,
    native: np.ndarray,
):
    idx = 0
    n_left = parameterization.n_left_orbital_rotation_params
    left_params = native[idx : idx + n_left]
    idx += n_left

    n_pair = parameterization.n_pair_params_per_layer
    if parameterization.shared_diagonal:
        pair_params = [native[idx : idx + n_pair]] * parameterization.layers
        idx += n_pair
    else:
        pair_params = []
        for _ in range(parameterization.layers):
            pair_params.append(native[idx : idx + n_pair])
            idx += n_pair

    n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
    middle_params = []
    for _ in range(parameterization.layers - 1):
        middle_params.append(native[idx : idx + n_middle])
        idx += n_middle

    n_right = parameterization.n_right_orbital_rotation_params
    right_params = native[idx : idx + n_right]
    return left_params, pair_params, middle_params, right_params


def _layered_igcr2_runtime(
    parameterization: IGCR2SpinRestrictedParameterization,
    native: np.ndarray,
    reference_mat: np.ndarray,
    nelec: tuple[int, int],
    diag_features: np.ndarray,
    left_chart: object,
    middle_chart: object,
    right_chart: object,
):
    norb = parameterization.norb
    layers = parameterization.layers
    left_params, pair_params, middle_params, right_params = (
        _parse_layered_igcr2_native(parameterization, native)
    )

    rotations = [
        left_chart.unitary_from_parameters(left_params, norb),
        *[
            middle_chart.unitary_from_parameters(params, norb)
            for params in middle_params
        ],
    ]
    prefix = np.eye(norb, dtype=np.complex128)
    for rotation in rotations:
        prefix = prefix @ np.asarray(rotation, dtype=np.complex128)
    final = right_chart.unitary_from_parameters(right_params, norb)
    rotations.append(prefix.conj().T @ final)

    suffix_after = [None] * layers
    suffix = np.eye(norb, dtype=np.complex128)
    for idx in range(layers - 1, -1, -1):
        suffix_after[idx] = suffix
        suffix = np.asarray(rotations[idx], dtype=np.complex128) @ suffix

    rep_a = [_sector_representation(u, norb, nelec[0]) for u in rotations]
    rep_b = [_sector_representation(u, norb, nelec[1]) for u in rotations]

    dim_a, dim_b = reference_mat.shape
    phases = [None] * layers
    after_diag = [None] * layers
    after_rotation = [None] * (layers + 1)

    current = rep_a[layers] @ reference_mat @ rep_b[layers].T
    after_rotation[layers] = current
    for idx in range(layers - 1, -1, -1):
        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ pair_params[idx])).reshape(
                dim_a, dim_b
            )
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)
        phases[idx] = phase
        current = phase * current
        after_diag[idx] = current
        current = rep_a[idx] @ current @ rep_b[idx].T
        after_rotation[idx] = current

    return {
        "left_params": left_params,
        "middle_params": middle_params,
        "right_params": right_params,
        "rotations": rotations,
        "prefix": prefix,
        "suffix_after": suffix_after,
        "rep_a": rep_a,
        "rep_b": rep_b,
        "phases": phases,
        "after_diag": after_diag,
        "after_rotation": after_rotation,
    }


def _apply_orbital_generator_batch(
    generator_batch: np.ndarray,
    mat: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    left = _one_body_batch_to_sector(generator_batch, tensor_a)
    right = _one_body_batch_to_sector(generator_batch, tensor_b)
    return _batch_row_and_col(left, right, mat)


def _propagate_layered_after_rotation(runtime: dict, start: int, mats: np.ndarray):
    out = mats
    for idx in range(start - 1, -1, -1):
        out = runtime["phases"][idx][None, :, :] * out
        out = _apply_batch_transform(
            runtime["rep_a"][idx],
            runtime["rep_b"][idx],
            out,
        )
    return out


def _propagate_layered_after_diagonal(runtime: dict, idx: int, mats: np.ndarray):
    out = _apply_batch_transform(
        runtime["rep_a"][idx],
        runtime["rep_b"][idx],
        mats,
    )
    return _propagate_layered_after_rotation(runtime, idx, out)


def _conjugate_generator_batch(batch: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.matmul(u.conj().T, np.matmul(batch, u))


def _layered_prefix_rotation_block(
    runtime: dict,
    rot_idx: int,
    generator_batch: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    if generator_batch.shape[0] == 0:
        dim_a, dim_b = runtime["after_rotation"][0].shape
        return np.zeros((0, dim_a, dim_b), dtype=np.complex128)

    direct = _apply_orbital_generator_batch(
        generator_batch,
        runtime["after_rotation"][rot_idx],
        tensor_a,
        tensor_b,
    )
    direct = _propagate_layered_after_rotation(runtime, rot_idx, direct)

    u = runtime["rotations"][rot_idx]
    suffix = runtime["suffix_after"][rot_idx]
    right_generator = -_conjugate_generator_batch(
        _conjugate_generator_batch(generator_batch, u),
        suffix,
    )
    right = _apply_orbital_generator_batch(
        right_generator,
        runtime["after_rotation"][-1],
        tensor_a,
        tensor_b,
    )
    right = _propagate_layered_after_rotation(
        runtime, len(runtime["after_diag"]), right
    )
    return direct + right


def _layered_final_rotation_block(
    runtime: dict,
    generator_batch: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    if generator_batch.shape[0] == 0:
        dim_a, dim_b = runtime["after_rotation"][0].shape
        return np.zeros((0, dim_a, dim_b), dtype=np.complex128)
    right_generator = _conjugate_generator_batch(generator_batch, runtime["prefix"])
    out = _apply_orbital_generator_batch(
        right_generator,
        runtime["after_rotation"][-1],
        tensor_a,
        tensor_b,
    )
    return _propagate_layered_after_rotation(
        runtime, len(runtime["after_diag"]), out
    )


def make_layered_igcr2_jacobian(
    parameterization: IGCR2SpinRestrictedParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _igcr2_feature_matrix(parameterization, nelec)
    diag_feature_tensor = diag_features.T.reshape(
        diag_features.shape[1], dim_a, dim_b
    )
    transform = _public_to_native_matrix(parameterization)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        native = parameterization._native_parameters_from_public(params)
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        blocks = []

        n_left = parameterization.n_left_orbital_rotation_params
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            blocks.append(
                _layered_prefix_rotation_block(
                    runtime, 0, gen_left, tensor_a, tensor_b
                ).reshape(n_left, dim_a * dim_b).T
            )

        n_pair = parameterization.n_pair_params_per_layer
        if n_pair:
            pair_blocks = []
            for idx in range(layers):
                d_after_diag = (
                    1j
                    * diag_feature_tensor
                    * runtime["after_diag"][idx][None, :, :]
                )
                pair_blocks.append(
                    _propagate_layered_after_diagonal(
                        runtime, idx, d_after_diag
                    )
                )
            if parameterization.shared_diagonal:
                blocks.append(
                    np.sum(pair_blocks, axis=0).reshape(n_pair, dim_a * dim_b).T
                )
            else:
                blocks.extend(
                    block.reshape(n_pair, dim_a * dim_b).T
                    for block in pair_blocks
                )

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        if n_middle:
            for idx, middle_params in enumerate(runtime["middle_params"], start=1):
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis
                )
                blocks.append(
                    _layered_prefix_rotation_block(
                        runtime, idx, gen_middle, tensor_a, tensor_b
                    ).reshape(n_middle, dim_a * dim_b).T
                )

        n_right = parameterization.n_right_orbital_rotation_params
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            blocks.append(
                _layered_final_rotation_block(
                    runtime, gen_final, tensor_a, tensor_b
                ).reshape(n_right, dim_a * dim_b).T
            )

        if blocks:
            out = np.hstack(blocks)
        else:
            out = np.zeros((dim_a * dim_b, 0), dtype=np.complex128)
        if transform is not None:
            out = out @ transform
        return out

    return jac


def make_layered_igcr2_subspace_jacobian(
    parameterization: IGCR2SpinRestrictedParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    dim = dim_a * dim_b
    diag_features = _igcr2_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        n_dir = directions.shape[1]
        if n_dir == 0:
            return np.zeros((dim, 0), dtype=np.complex128)

        native = parameterization._native_parameters_from_public(params)
        native_dirs = directions if transform is None else transform @ directions
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        d_state = np.zeros((n_dir, dim_a, dim_b), dtype=np.complex128)
        idx = 0

        n_left = parameterization.n_left_orbital_rotation_params
        left_dirs = native_dirs[idx : idx + n_left]
        idx += n_left
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            left_basis_dirs = np.einsum(
                "kj,kpq->jpq", left_dirs, left_basis, optimize=True
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis_dirs)
            d_state += _layered_prefix_rotation_block(
                runtime, 0, gen_left, tensor_a, tensor_b
            )

        n_pair = parameterization.n_pair_params_per_layer
        if parameterization.shared_diagonal:
            pair_dirs = native_dirs[idx : idx + n_pair]
            idx += n_pair
            if n_pair:
                feature_dirs = diag_features @ pair_dirs
                d_after_diag = [
                    1j
                    * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                    * runtime["after_diag"][layer][None, :, :]
                    for layer in range(layers)
                ]
                for layer, block in enumerate(d_after_diag):
                    d_state += _propagate_layered_after_diagonal(
                        runtime, layer, block
                    )
        else:
            for layer in range(layers):
                pair_dirs = native_dirs[idx : idx + n_pair]
                idx += n_pair
                if n_pair:
                    feature_dirs = diag_features @ pair_dirs
                    block = (
                        1j
                        * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                        * runtime["after_diag"][layer][None, :, :]
                    )
                    d_state += _propagate_layered_after_diagonal(
                        runtime, layer, block
                    )

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        for layer, middle_params in enumerate(runtime["middle_params"], start=1):
            middle_dirs = native_dirs[idx : idx + n_middle]
            idx += n_middle
            if n_middle:
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                middle_basis_dirs = np.einsum(
                    "kj,kpq->jpq",
                    middle_dirs,
                    middle_basis,
                    optimize=True,
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis_dirs
                )
                d_state += _layered_prefix_rotation_block(
                    runtime, layer, gen_middle, tensor_a, tensor_b
                )

        n_right = parameterization.n_right_orbital_rotation_params
        right_dirs = native_dirs[idx : idx + n_right]
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            right_basis_dirs = np.einsum(
                "kj,kpq->jpq", right_dirs, right_basis, optimize=True
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis_dirs)
            d_state += _layered_final_rotation_block(
                runtime, gen_final, tensor_a, tensor_b
            )

        return d_state.reshape(n_dir, dim).T

    return subspace_jac


def _batch_vjp(batch: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return 2.0 * np.einsum("jab,ab->j", batch.conj(), v_mat, optimize=True).real


def make_layered_igcr2_vjp(
    parameterization: IGCR2SpinRestrictedParameterization,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _igcr2_feature_matrix(parameterization, nelec)
    diag_feature_tensor = diag_features.T.reshape(
        diag_features.shape[1], dim_a, dim_b
    )
    transform = _public_to_native_matrix(parameterization)

    def vjp(params: np.ndarray, v: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        v = np.asarray(v, dtype=np.complex128)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        if v.shape != (reference_vec.size,):
            raise ValueError(f"Expected v with shape {(reference_vec.size,)}, got {v.shape}.")
        v_mat = reshape_state(v, norb, nelec)
        native = parameterization._native_parameters_from_public(params)
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        grad_blocks = []

        n_left = parameterization.n_left_orbital_rotation_params
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            block = _layered_prefix_rotation_block(
                runtime, 0, gen_left, tensor_a, tensor_b
            )
            grad_blocks.append(_batch_vjp(block, v_mat))

        n_pair = parameterization.n_pair_params_per_layer
        if n_pair:
            pair_blocks = []
            for idx in range(layers):
                d_after_diag = (
                    1j
                    * diag_feature_tensor
                    * runtime["after_diag"][idx][None, :, :]
                )
                pair_blocks.append(
                    _propagate_layered_after_diagonal(
                        runtime, idx, d_after_diag
                    )
                )
            if parameterization.shared_diagonal:
                grad_blocks.append(_batch_vjp(np.sum(pair_blocks, axis=0), v_mat))
            else:
                grad_blocks.extend(_batch_vjp(block, v_mat) for block in pair_blocks)

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        if n_middle:
            for idx, middle_params in enumerate(runtime["middle_params"], start=1):
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis
                )
                block = _layered_prefix_rotation_block(
                    runtime, idx, gen_middle, tensor_a, tensor_b
                )
                grad_blocks.append(_batch_vjp(block, v_mat))

        n_right = parameterization.n_right_orbital_rotation_params
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            block = _layered_final_rotation_block(
                runtime, gen_final, tensor_a, tensor_b
            )
            grad_blocks.append(_batch_vjp(block, v_mat))

        if grad_blocks:
            grad = np.concatenate(grad_blocks)
        else:
            grad = np.zeros(0, dtype=np.float64)
        if transform is not None:
            grad = transform.T @ grad
        return grad

    return vjp


def make_restricted_gcr_vjp(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        and parameterization.layers != 1
    ):
        return make_layered_igcr2_vjp(parameterization, reference_vec, nelec)

    jac = make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)

    def vjp(params: np.ndarray, v: np.ndarray) -> np.ndarray:
        J = jac(params)
        return 2.0 * (J.conj().T @ np.asarray(v, dtype=np.complex128)).real

    return vjp


def make_restricted_gcr_jacobian(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        and parameterization.layers != 1
    ):
        return make_layered_igcr2_jacobian(
            parameterization, reference_vec, nelec
        )
    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
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


def make_restricted_gcr_subspace_jacobian(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function computing ``J(params) @ directions`` analytically.

    Unlike :func:`make_restricted_gcr_jacobian`, this does not materialise the
    full tangent matrix. Its cost scales with the number of requested directions.
    """
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        and parameterization.layers != 1
    ):
        return make_layered_igcr2_subspace_jacobian(
            parameterization, reference_vec, nelec
        )
    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    dim = dim_a * dim_b
    diag_features = _diag_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        n_dir = directions.shape[1]
        if n_dir == 0:
            return np.zeros((dim, 0), dtype=np.complex128)

        native = parameterization._native_parameters_from_public(params)
        native_dirs = directions if transform is None else transform @ directions

        n_left = parameterization.n_left_orbital_rotation_params
        right_start = parameterization._right_orbital_rotation_start
        n_right = parameterization.n_right_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left:right_start]
        right_params = native[right_start : right_start + n_right]

        left_dirs = native_dirs[:n_left]
        diag_dirs = native_dirs[n_left:right_start]
        right_dirs = native_dirs[right_start : right_start + n_right]

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
        d_state = np.zeros((n_dir, dim_a, dim_b), dtype=np.complex128)

        if n_left:
            left_basis_dirs = np.einsum(
                "kj,kpq->jpq",
                left_dirs,
                left_basis,
                optimize=True,
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis_dirs)
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
            d_left = _batch_row_and_col(left_a, left_b, state)
            d_left += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            d_state += d_left

        if diag_params.size:
            feature_dirs = diag_features @ diag_dirs
            d_diagonalized = (
                1j
                * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                * diagonalized[None, :, :]
            )
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)

        if n_right:
            right_basis_dirs = np.einsum(
                "kj,kpq->jpq",
                right_dirs,
                right_basis,
                optimize=True,
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis_dirs)
            gen_right_from_final = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )

            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)

        return d_state.reshape(n_dir, dim).T

    return subspace_jac


def _batch_left_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("jmn,nb->jmb", batch, mat, optimize=True)


def _batch_right_transpose_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("an,jbn->jab", mat, batch, optimize=True)


__all__ = [
    "make_restricted_gcr_jacobian",
    "make_restricted_gcr_subspace_jacobian",
    "make_restricted_gcr_vjp",
]
