import numpy as np
import pytest

from xquces.gcr.diagonal_rank import spin_flip_orbit_count
from xquces.gcr.spin_balanced_igcr4 import (
    FixedOrbitalDiagonalModel,
    IGCR4SpinBalancedFixedSectorParameterization,
    IGCR4SpinSeparatedFixedSectorParameterization,
    make_spin_orbital_diagonal_basis,
)
from xquces.gcr.restricted_jacobian import make_restricted_gcr_jacobian
from xquces.states import hartree_fock_state


def test_spin_balanced_basis_has_expected_fixed_sector_rank():
    norb = 4
    nelec = (2, 2)

    basis = make_spin_orbital_diagonal_basis(
        norb,
        nelec,
        max_body=4,
        spin_balanced=True,
    )

    assert basis.n_determinants == 36
    assert basis.n_params == spin_flip_orbit_count(norb, nelec) - 1
    assert basis.features.T @ basis.features == pytest.approx(np.eye(basis.n_params))


def test_fixed_orbital_diagonal_model_jacobian_matches_finite_difference():
    norb = 4
    nelec = (2, 2)
    basis = make_spin_orbital_diagonal_basis(
        norb,
        nelec,
        max_body=4,
        spin_balanced=True,
    )
    model = FixedOrbitalDiagonalModel.from_orbitals(
        basis,
        hartree_fock_state(norb, nelec),
        left=np.eye(norb, dtype=np.complex128),
        right=np.eye(norb, dtype=np.complex128),
        norb=norb,
        nelec=nelec,
    )
    x = np.linspace(-0.2, 0.2, basis.n_params)
    jac = model.jacobian(x)

    eps = 1e-6
    for k in [0, basis.n_params // 2, basis.n_params - 1]:
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (model.state(xp) - model.state(xm)) / (2.0 * eps)
        assert jac[:, k] == pytest.approx(fd, abs=1e-8)


def test_phase_overlap_bound_is_one_for_matching_amplitudes():
    norb = 4
    nelec = (2, 2)
    basis = make_spin_orbital_diagonal_basis(norb, nelec)
    model = FixedOrbitalDiagonalModel.from_orbitals(
        basis,
        hartree_fock_state(norb, nelec),
        left=np.eye(norb, dtype=np.complex128),
        right=np.eye(norb, dtype=np.complex128),
        norb=norb,
        nelec=nelec,
    )
    target = np.exp(1j * np.linspace(0.0, 1.0, model.right_state.size))
    target *= np.abs(model.right_state)

    bound = model.phase_overlap_bound(target)

    assert bound["best_overlap"] == pytest.approx(1.0)
    assert bound["best_overlap_squared"] == pytest.approx(1.0)
    assert bound["amplitude_l2"] == pytest.approx(0.0)


def test_spin_balanced_fixed_sector_parameterization_jacobian_matches_finite_difference():
    norb = 4
    nelec = (2, 2)
    param = IGCR4SpinBalancedFixedSectorParameterization(norb=norb, nelec=nelec)
    phi0 = hartree_fock_state(norb, nelec)
    jacobian = make_restricted_gcr_jacobian(param, phi0, nelec)
    x = np.zeros(param.n_params)
    x[0] = 0.03
    x[param.n_left_orbital_rotation_params] = -0.07
    x[-1] = 0.04
    jac = jacobian(x)

    eps = 1e-6
    for k in [0, param.n_left_orbital_rotation_params, param.n_params - 1]:
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (
            param.ansatz_from_parameters(xp).apply(phi0, nelec=nelec)
            - param.ansatz_from_parameters(xm).apply(phi0, nelec=nelec)
        ) / (2.0 * eps)
        assert jac[:, k] == pytest.approx(fd, abs=2e-7)


def test_spin_separated_fixed_sector_parameterization_jacobian_matches_finite_difference():
    norb = 4
    nelec = (2, 2)
    param = IGCR4SpinSeparatedFixedSectorParameterization(
        norb=norb,
        nelec=nelec,
        spin_balanced=True,
    )
    phi0 = hartree_fock_state(norb, nelec)
    jacobian = make_restricted_gcr_jacobian(param, phi0, nelec)
    x = np.zeros(param.n_params)
    indices = [
        0,
        param.n_left_alpha_orbital_rotation_params,
        param.n_left_orbital_rotation_params,
        param._right_orbital_rotation_start,
        param._right_orbital_rotation_start
        + param.n_right_alpha_orbital_rotation_params,
    ]
    for value, index in zip([0.03, -0.02, 0.04, -0.05, 0.06], indices):
        x[index] = value
    jac = jacobian(x)

    eps = 1e-6
    for k in indices:
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (
            param.ansatz_from_parameters(xp).apply(phi0, nelec=nelec)
            - param.ansatz_from_parameters(xm).apply(phi0, nelec=nelec)
        ) / (2.0 * eps)
        assert jac[:, k] == pytest.approx(fd, abs=3e-7)
