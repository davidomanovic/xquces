from __future__ import annotations

import numpy as np
import pytest

from xquces.gcr.pair_uccd_reference import (
    GCR2PairUCCDParameterization,
    GCR3PairUCCDParameterization,
    GCR4PairUCCDParameterization,
)


def _random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return 0.5 * (mat + mat.conj().T)


def _energy(param, hamiltonian: np.ndarray, params: np.ndarray) -> float:
    psi = param.state_from_parameters(params)
    return float(np.vdot(psi, hamiltonian @ psi).real)


@pytest.mark.parametrize(
    "parameterization_cls",
    [
        GCR2PairUCCDParameterization,
        GCR3PairUCCDParameterization,
        GCR4PairUCCDParameterization,
    ],
)
def test_pair_uccd_reference_energy_gradient_matches_jacobian_and_finite_difference(
    parameterization_cls,
):
    param = parameterization_cls(norb=4, nocc=2)
    rng = np.random.default_rng(2001 + param.n_params)
    params = 0.02 * rng.normal(size=param.n_params)
    psi = param.state_from_parameters(params)
    hamiltonian = _random_hermitian(psi.size, rng)

    energy, grad = param.energy_gradient_from_parameters(params, hamiltonian)

    hpsi = hamiltonian @ psi
    expected_energy = float(np.vdot(psi, hpsi).real)
    jac = param.state_jacobian_from_parameters(params)
    expected_grad = 2.0 * np.real(jac.conj().T @ (hpsi - expected_energy * psi))

    assert energy == pytest.approx(expected_energy, abs=1e-12)
    assert np.allclose(grad, expected_grad, atol=1e-10, rtol=1e-10)

    eps = 1e-6
    fd_grad = np.empty_like(grad)
    for k in range(param.n_params):
        step = np.zeros_like(params)
        step[k] = eps
        fd_grad[k] = (
            _energy(param, hamiltonian, params + step)
            - _energy(param, hamiltonian, params - step)
        ) / (2.0 * eps)

    assert np.allclose(grad, fd_grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "parameterization_cls",
    [
        GCR2PairUCCDParameterization,
        GCR3PairUCCDParameterization,
        GCR4PairUCCDParameterization,
    ],
)
def test_pair_uccd_reference_subspace_jacobian_matches_full_jacobian(
    parameterization_cls,
):
    param = parameterization_cls(norb=4, nocc=2)
    rng = np.random.default_rng(3001 + param.n_params)
    params = 0.02 * rng.normal(size=param.n_params)
    directions, _ = np.linalg.qr(rng.normal(size=(param.n_params, 5)))

    full_projected = param.state_jacobian_from_parameters(params) @ directions
    subspace_jac = param.state_subspace_jacobian_from_parameters(params, directions)

    assert np.allclose(subspace_jac, full_projected, atol=1e-10, rtol=1e-10)


def test_nested_higher_order_lift_is_guarded_by_true_energy():
    gcr2 = GCR2PairUCCDParameterization(norb=4, nocc=2)
    gcr3 = GCR3PairUCCDParameterization(norb=4, nocc=2)
    gcr4 = GCR4PairUCCDParameterization(norb=4, nocc=2)
    rng = np.random.default_rng(4001)
    x2 = 0.03 * rng.normal(size=gcr2.n_params)
    psi = gcr2.state_from_parameters(x2)
    hamiltonian = _random_hermitian(psi.size, rng)

    seed3 = gcr3.nested_lift_parameters_from(
        x2,
        gcr2,
        hamiltonian=hamiltonian,
        maxiter=3,
        return_info=True,
    )
    assert seed3.params.shape == (gcr3.n_params,)
    assert seed3.weights.shape == (2,)
    assert seed3.energy <= seed3.baseline_energy + 1e-12

    seed4 = gcr4.nested_lift_parameters_from(
        seed3.params,
        gcr3,
        hamiltonian=hamiltonian,
        maxiter=3,
        return_info=True,
    )
    assert seed4.params.shape == (gcr4.n_params,)
    assert seed4.weights.shape == (3,)
    assert seed4.energy <= seed4.baseline_energy + 1e-12


def test_nested_higher_order_lift_can_return_zero_extension():
    gcr2 = GCR2PairUCCDParameterization(norb=4, nocc=2)
    gcr3 = GCR3PairUCCDParameterization(norb=4, nocc=2)
    rng = np.random.default_rng(4002)
    x2 = 0.02 * rng.normal(size=gcr2.n_params)
    psi = gcr2.state_from_parameters(x2)
    hamiltonian = np.eye(psi.size)

    seed = gcr3.nested_lift_parameters_from(
        x2,
        gcr2,
        hamiltonian=hamiltonian,
        optimize_weights=False,
        return_info=True,
    )

    assert seed.accepted is False
    assert np.allclose(seed.weights, 0.0)
    assert seed.energy == pytest.approx(seed.baseline_energy)
