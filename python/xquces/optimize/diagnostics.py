from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from xquces.optimize.metric_bfgs import real_jacobian


@dataclass(frozen=True)
class TangentResidualProjection:
    residual_norm: float
    projected_norm: float
    projected_fraction: float
    orthogonal_norm: float
    orthogonal_fraction: float
    rank: int
    condition: float


def energy_and_residual(psi: np.ndarray, hamiltonian) -> tuple[float, np.ndarray]:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    hpsi = np.asarray(hamiltonian @ psi, dtype=np.complex128).reshape(-1)
    energy = float(np.real(np.vdot(psi, hpsi)))
    return energy, hpsi - energy * psi


def tangent_residual_projection(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> TangentResidualProjection:
    Jr = real_jacobian(jacobian)
    rr = np.concatenate(
        [
            np.asarray(residual, dtype=np.complex128).real.reshape(-1),
            np.asarray(residual, dtype=np.complex128).imag.reshape(-1),
        ]
    )
    residual_norm = float(np.linalg.norm(rr))
    if residual_norm == 0.0:
        return TangentResidualProjection(0.0, 0.0, 0.0, 0.0, 0.0, 0, float("nan"))

    U, svals, _ = np.linalg.svd(Jr, full_matrices=False)
    if svals.size == 0:
        return TangentResidualProjection(
            residual_norm, 0.0, 0.0, residual_norm, 1.0, 0, float("nan")
        )

    cutoff = max(float(atol), float(rtol) * float(svals[0]))
    rank = int(np.sum(svals > cutoff))
    if rank == 0:
        return TangentResidualProjection(
            residual_norm, 0.0, 0.0, residual_norm, 1.0, 0, float("inf")
        )

    coeff = U[:, :rank].T @ rr
    projected_norm = float(np.linalg.norm(coeff))
    orthogonal_sq = max(residual_norm**2 - projected_norm**2, 0.0)
    orthogonal_norm = float(np.sqrt(orthogonal_sq))
    condition = float(svals[0] / svals[rank - 1])
    return TangentResidualProjection(
        residual_norm=residual_norm,
        projected_norm=projected_norm,
        projected_fraction=projected_norm / residual_norm,
        orthogonal_norm=orthogonal_norm,
        orthogonal_fraction=orthogonal_norm / residual_norm,
        rank=rank,
        condition=condition,
    )


def sector_residual_projections(
    jacobian: np.ndarray,
    residual: np.ndarray,
    sectors: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> dict[str, TangentResidualProjection]:
    out = {}
    for name, indices in sectors.items():
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size:
            out[name] = tangent_residual_projection(
                np.asarray(jacobian)[:, idx],
                residual,
                rtol=rtol,
                atol=atol,
            )
    return out
