import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from xquces.optimize import (
    energy_and_residual,
    make_expectation_penalty_state_objective,
    make_projector_penalty_state_objective,
    minimize_subspace_linear_method,
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


def test_minimize_subspace_linear_method_decreases_energy_with_analytic_jacobian():
    hamiltonian = np.diag([0.0, 1.0])

    def params_to_state(x):
        theta = x[0] + 0.5 * x[1]
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)

    def state_jacobian(x):
        theta = x[0] + 0.5 * x[1]
        base = np.array([-np.sin(theta), np.cos(theta)], dtype=np.complex128)
        return np.column_stack([base, 0.5 * base])

    def energy_gradient(x):
        psi = params_to_state(x)
        hpsi = hamiltonian @ psi
        energy = float(np.vdot(psi, hpsi).real)
        grad = 2.0 * np.real(state_jacobian(x).conj().T @ (hpsi - energy * psi))
        return energy, grad

    x0 = np.array([0.8, 0.2])
    e0 = energy_gradient(x0)[0]
    accepted_energies = []

    def callback(result):
        if getattr(result, "accepted", False):
            accepted_energies.append(float(result.fun))

    result = minimize_subspace_linear_method(
        params_to_state,
        hamiltonian,
        x0,
        energy_gradient=energy_gradient,
        jac=state_jacobian,
        subspace_dim=1,
        maxiter=8,
        regularization=1e-4,
        regularization_attempts=4,
        callback=callback,
    )

    assert result.fun <= e0
    assert accepted_energies
    assert all(
        right <= left + 1e-12
        for left, right in zip([e0] + accepted_energies[:-1], accepted_energies)
    )


def test_minimize_subspace_linear_method_supports_finite_difference_subspace_jacobian():
    hamiltonian = np.diag([0.0, 1.0])

    def params_to_state(x):
        theta = x[0]
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)

    def energy_gradient(x):
        psi = params_to_state(x)
        hpsi = hamiltonian @ psi
        energy = float(np.vdot(psi, hpsi).real)
        grad = np.array([2.0 * np.sin(x[0]) * np.cos(x[0])])
        return energy, grad

    x0 = np.array([0.7])
    e0 = energy_gradient(x0)[0]
    result = minimize_subspace_linear_method(
        params_to_state,
        hamiltonian,
        x0,
        energy_gradient=energy_gradient,
        subspace_jacobian="finite-difference",
        subspace_dim=1,
        maxiter=4,
        regularization=1e-4,
        regularization_attempts=4,
    )

    assert result.fun <= e0
    assert abs(result.x[0]) < abs(x0[0])


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


def test_projector_penalty_state_objective_adds_population_gradient():
    hamiltonian = np.diag([0.0, 1.0])
    projector = np.array([1.0, 0.0], dtype=np.complex128)

    def params_to_state(x):
        theta = x[0]
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)

    def state_jacobian(x):
        theta = x[0]
        return np.array([[-np.sin(theta)], [np.cos(theta)]], dtype=np.complex128)

    fun, jac, cache = make_projector_penalty_state_objective(
        params_to_state,
        state_jacobian,
        hamiltonian,
        projector,
        penalty_weight=2.0,
    )
    x = np.array([0.3])
    theta = x[0]

    assert fun(x) == pytest.approx(np.sin(theta) ** 2 + 2.0 * np.cos(theta) ** 2)
    assert jac(x)[0] == pytest.approx(-2.0 * np.sin(theta) * np.cos(theta))
    assert cache["energy"] == pytest.approx(np.sin(theta) ** 2)
    assert cache["projector_population"] == pytest.approx(np.cos(theta) ** 2)


def test_expectation_penalty_state_objective_adds_operator_gradient():
    hamiltonian = np.diag([0.0, 1.0])
    operator = np.diag([0.0, 2.0])

    def params_to_state(x):
        theta = x[0]
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)

    def state_jacobian(x):
        theta = x[0]
        return np.array([[-np.sin(theta)], [np.cos(theta)]], dtype=np.complex128)

    fun, jac, cache = make_expectation_penalty_state_objective(
        params_to_state,
        state_jacobian,
        hamiltonian,
        lambda psi: operator @ psi,
        penalty_weight=0.25,
        target=0.0,
    )
    x = np.array([0.3])
    theta = x[0]
    expectation = 2.0 * np.sin(theta) ** 2
    expected_fun = np.sin(theta) ** 2 + 0.25 * expectation**2
    expected_jac = (
        2.0 * np.sin(theta) * np.cos(theta)
        + 0.5 * expectation * 4.0 * np.sin(theta) * np.cos(theta)
    )

    assert fun(x) == pytest.approx(expected_fun)
    assert jac(x)[0] == pytest.approx(expected_jac)
    assert cache["energy"] == pytest.approx(np.sin(theta) ** 2)
    assert cache["operator_expectation"] == pytest.approx(expectation)
