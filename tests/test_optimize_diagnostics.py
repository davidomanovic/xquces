import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from xquces.optimize import (
    energy_and_residual,
    minimize_svd_metric_bfgs,
    tangent_residual_projection,
    tangent_svd_preconditioner,
)


def test_tangent_residual_projection_keeps_only_column_space():
    jacobian = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j],
        ]
    )
    residual = np.array([3.0 + 4.0j, 5.0 + 12.0j])

    projection = tangent_residual_projection(jacobian, residual)

    expected_projected = np.linalg.norm([3.0, 5.0])
    expected_residual = np.linalg.norm([3.0, 5.0, 4.0, 12.0])
    assert projection.rank == 2
    assert projection.projected_norm == pytest.approx(expected_projected)
    assert projection.projected_fraction == pytest.approx(
        expected_projected / expected_residual
    )


def test_tangent_svd_preconditioner_drops_near_null_modes():
    def state_jacobian(_x):
        return np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0e-7 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        )

    transform, info = tangent_svd_preconditioner(
        state_jacobian, np.zeros(3), rtol=1e-10
    )

    assert transform.shape == (3, 1)
    assert info["active_modes"] == 1
    assert info["dropped_modes"] == 2


def test_minimize_svd_metric_bfgs_returns_full_coordinate_vector():
    def state_jacobian(_x):
        return np.array([[1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]])

    def fun(x):
        return float((x[0] + x[1] - 2.0) ** 2 + 0.5 * x[2] ** 2)

    def jac(x):
        return np.array([2.0 * (x[0] + x[1] - 2.0)] * 2 + [x[2]])

    result = minimize_svd_metric_bfgs(
        fun,
        jac,
        np.zeros(3),
        state_jacobian,
        rtol=1e-12,
        gtol=1e-10,
        maxiter=100,
    )

    assert result.x.shape == (3,)
    assert result.metric_preconditioner["active_modes"] == 1
    assert result.x[0] + result.x[1] == pytest.approx(2.0)
    assert result.x[2] == pytest.approx(0.0)


def test_energy_and_residual_uses_hamiltonian_action():
    hamiltonian = LinearOperator(
        (2, 2),
        matvec=lambda vec: np.array([vec[0], 3.0 * vec[1]], dtype=np.complex128),
        dtype=np.complex128,
    )
    psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

    energy, residual = energy_and_residual(psi, hamiltonian)

    assert energy == pytest.approx(2.0)
    assert residual == pytest.approx(np.array([-1.0, 1.0]) / np.sqrt(2.0))
