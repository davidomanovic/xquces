from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pytest

from xquces.basis import flatten_state, occ_rows, reshape_state
from xquces.gcr.igcr2 import IGCR2SpinRestrictedParameterization
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJRestrictedProjectedDFSeed


igcr3 = importlib.import_module("xquces.igcr3")


def _random_complex(shape, rng, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _random_real(shape, rng, scale=1.0):
    return scale * rng.normal(size=shape)


def _align_global_phase(psi_ref, psi):
    overlap = np.vdot(psi_ref, psi)
    if abs(overlap) < 1e-14:
        return psi
    return psi * overlap.conjugate() / abs(overlap)


def _assert_same_state_up_to_phase(psi_ref, psi, atol=1e-10):
    psi_aligned = _align_global_phase(psi_ref, psi)
    err = np.linalg.norm(psi_aligned - psi_ref)
    assert err < atol, f"state mismatch {err}"


def _restricted_ucj_seed(seed=456, scale_t2=0.05, scale_t1=0.03):
    rng = np.random.default_rng(seed)
    nocc = 2
    nvirt = 2
    t2 = _random_real((nocc, nocc, nvirt, nvirt), rng, scale=scale_t2)
    t1 = _random_complex((nocc, nvirt), rng, scale=scale_t1)
    ucj = UCJRestrictedProjectedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()
    return ucj, t2, t1


def test_igcr3_module_imports():
    assert igcr3 is not None


@pytest.mark.parametrize("norb,nocc", [(4, 2), (5, 2), (6, 3)])
def test_spin_restricted_parameter_count(norb, nocc):
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    cubic_reduced = (
        norb * (norb - 1)
        + math.comb(norb, 3)
        - norb
        - math.comb(norb, 2)
    )
    expected = (
        norb * (norb - 1)
        + math.comb(norb, 2)
        + cubic_reduced
        + 2 * nocc * (norb - nocc)
    )
    assert param.n_double_params == 0
    assert param.n_pair_params == math.comb(norb, 2)
    assert param.uses_reduced_cubic_chart
    assert param.n_tau_params == cubic_reduced
    assert param.n_omega_params == 0
    assert param.n_params == expected


@pytest.mark.parametrize("norb,nocc", [(4, 2), (6, 3)])
def test_real_right_spin_restricted_parameter_count(norb, nocc):
    param = igcr3.IGCR3SpinRestrictedParameterization(
        norb=norb,
        nocc=nocc,
        real_right_orbital_chart=True,
    )
    cubic_reduced = (
        norb * (norb - 1)
        + math.comb(norb, 3)
        - norb
        - math.comb(norb, 2)
    )
    expected = (
        norb * (norb - 1)
        + math.comb(norb, 2)
        + cubic_reduced
        + nocc * (norb - nocc)
    )
    assert param.n_right_orbital_rotation_params == nocc * (norb - nocc)
    assert param.n_params == expected


@pytest.mark.parametrize("norb,nocc", [(4, 2), (6, 3)])
def test_unreduced_spin_restricted_parameter_count(norb, nocc):
    param = igcr3.IGCR3SpinRestrictedParameterization(
        norb=norb,
        nocc=nocc,
        reduce_cubic_gauge=False,
    )
    expected = (
        norb * (norb - 1)
        + math.comb(norb, 2)
        + norb * (norb - 1)
        + math.comb(norb, 3)
        + 2 * nocc * (norb - nocc)
    )
    assert not param.uses_reduced_cubic_chart
    assert param.n_tau_params == norb * (norb - 1)
    assert param.n_omega_params == math.comb(norb, 3)
    assert param.n_params == expected


def test_flatten_unflatten_roundtrip_small_random():
    rng = np.random.default_rng(301)
    norb = 4
    nocc = 2
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = 0.03 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    x_roundtrip = param.parameters_from_ansatz(ansatz)
    assert np.allclose(x_roundtrip, x, atol=1e-8)


def test_reduced_cubic_chart_preserves_unreduced_state():
    rng = np.random.default_rng(351)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    full = igcr3.IGCR3SpinRestrictedParameterization(
        norb=norb,
        nocc=nocc,
        reduce_cubic_gauge=False,
    )
    reduced = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x_full = 0.025 * rng.normal(size=full.n_params)
    ansatz_full = full.ansatz_from_parameters(x_full)
    x_reduced = reduced.parameters_from_ansatz(ansatz_full)

    phi0 = hartree_fock_state(norb, nelec)
    psi_full = ansatz_full.apply(phi0, nelec=nelec, copy=True)
    psi_reduced = reduced.ansatz_from_parameters(x_reduced).apply(
        phi0,
        nelec=nelec,
        copy=True,
    )
    _assert_same_state_up_to_phase(psi_full, psi_reduced, atol=1e-10)


def test_zero_parameters_prepare_hartree_fock_reference():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    phi0 = hartree_fock_state(norb, nelec)
    psi = param.ansatz_from_parameters(np.zeros(param.n_params)).apply(phi0, nelec=nelec)
    assert np.allclose(psi, phi0)


def test_zero_cubic_sector_reproduces_igcr2_submanifold():
    rng = np.random.default_rng(401)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)

    param2 = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    param3 = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x2 = 0.04 * rng.normal(size=param2.n_params)
    x3 = np.zeros(param3.n_params)
    x3[: param2._right_orbital_rotation_start] = x2[: param2._right_orbital_rotation_start]
    x3[param3._right_orbital_rotation_start :] = x2[param2._right_orbital_rotation_start :]

    psi2 = param2.ansatz_from_parameters(x2).apply(phi0, nelec=nelec, copy=True)
    psi3 = param3.ansatz_from_parameters(x3).apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi2, psi3, atol=1e-10)


def test_zero_cubic_ansatz_converts_to_igcr2_with_nonzero_double_sector():
    rng = np.random.default_rng(451)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    double = 0.1 * rng.normal(size=norb)
    pair_values = 0.1 * rng.normal(size=math.comb(norb, 2))
    ansatz3 = igcr3.IGCR3Ansatz(
        diagonal=igcr3.IGCR3SpinRestrictedSpec(
            double_params=double,
            pair_values=pair_values,
            tau=np.zeros((norb, norb)),
            omega_values=np.zeros(math.comb(norb, 3)),
        ),
        left=np.eye(norb, dtype=np.complex128),
        right=np.eye(norb, dtype=np.complex128),
        nocc=nocc,
    )
    ansatz2 = ansatz3.to_igcr2_ansatz()
    psi3 = ansatz3.apply(phi0, nelec=nelec, copy=True)
    psi2 = ansatz2.apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi2, psi3, atol=1e-10)


def test_from_ucj_with_zero_cubic_reproduces_igcr2_state():
    ucj, _, _ = _restricted_ucj_seed(seed=501)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)

    ig2 = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x2 = ig2.parameters_from_ucj_ansatz(ucj)
    psi2 = ig2.ansatz_from_parameters(x2).apply(phi0, nelec=nelec, copy=True)

    ig3 = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x3 = ig3.parameters_from_ucj_ansatz(ucj)
    psi3 = ig3.ansatz_from_parameters(x3).apply(phi0, nelec=nelec, copy=True)

    _assert_same_state_up_to_phase(psi2, psi3, atol=1e-10)


def test_diagonal_phase_matches_direct_operator_evaluation():
    rng = np.random.default_rng(601)
    norb = 4
    double = 0.1 * rng.normal(size=norb)
    pair_values = 0.1 * rng.normal(size=math.comb(norb, 2))
    tau = 0.1 * rng.normal(size=(norb, norb))
    np.fill_diagonal(tau, 0.0)
    omega = 0.1 * rng.normal(size=math.comb(norb, 3))
    diag = igcr3.IGCR3SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega,
    )
    occ_alpha = np.array([0, 2], dtype=np.uintp)
    occ_beta = np.array([1, 2], dtype=np.uintp)

    n = np.array([1, 1, 2, 0], dtype=float)
    d = np.array([0, 0, 1, 0], dtype=float)
    pair = diag.pair_matrix()
    expected = np.dot(double, d)
    expected += sum(pair[p, q] * n[p] * n[q] for p, q in itertools.combinations(range(norb), 2))
    expected += sum(tau[p, q] * d[p] * n[q] for p in range(norb) for q in range(norb) if p != q)
    expected += sum(
        value * n[p] * n[q] * n[r]
        for value, (p, q, r) in zip(omega, itertools.combinations(range(norb), 3))
    )

    assert np.isclose(diag.phase_from_occupations(occ_alpha, occ_beta), expected)


def test_rust_diagonal_kernel_matches_python_reference():
    rng = np.random.default_rng(651)
    norb = 4
    nelec = (2, 2)
    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    double = 0.1 * rng.normal(size=norb)
    pair_values = 0.1 * rng.normal(size=math.comb(norb, 2))
    tau = 0.1 * rng.normal(size=(norb, norb))
    np.fill_diagonal(tau, 0.0)
    omega = 0.1 * rng.normal(size=math.comb(norb, 3))
    diag = igcr3.IGCR3SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega,
    )

    out = igcr3.apply_igcr3_spin_restricted_diagonal(
        vec,
        diag,
        norb,
        nelec,
        time=0.7,
        copy=True,
    )

    ref = reshape_state(vec, norb, nelec)
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    for ia, alpha in enumerate(occ_alpha):
        for ib, beta in enumerate(occ_beta):
            phase = diag.phase_from_occupations(alpha, beta)
            ref[ia, ib] *= np.exp(0.7j * phase)
    assert np.allclose(out, flatten_state(ref), atol=1e-12)


def test_pure_tau_sector_phase():
    norb = 3
    tau = np.zeros((norb, norb), dtype=np.float64)
    tau[0, 2] = 0.37
    diag = igcr3.IGCR3SpinRestrictedSpec(
        double_params=np.zeros(norb),
        pair_values=np.zeros(math.comb(norb, 2)),
        tau=tau,
        omega_values=np.zeros(math.comb(norb, 3)),
    )
    occ_alpha = np.array([0, 2], dtype=np.uintp)
    occ_beta = np.array([0], dtype=np.uintp)
    assert np.isclose(diag.phase_from_occupations(occ_alpha, occ_beta), 0.37)


def test_pure_omega_sector_phase_counts_spin_choices():
    norb = 3
    diag = igcr3.IGCR3SpinRestrictedSpec(
        double_params=np.zeros(norb),
        pair_values=np.zeros(math.comb(norb, 2)),
        tau=np.zeros((norb, norb)),
        omega_values=np.array([0.25]),
    )
    occ_alpha = np.array([0, 1, 2], dtype=np.uintp)
    occ_beta = np.array([0], dtype=np.uintp)
    assert np.isclose(diag.phase_from_occupations(occ_alpha, occ_beta), 0.5)


def test_small_random_state_remains_normalized():
    rng = np.random.default_rng(701)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = 0.02 * rng.normal(size=param.n_params)
    phi0 = hartree_fock_state(norb, nelec)
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    assert np.isfinite(psi).all()
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_cubic_seed_is_deterministic_and_zero_by_default():
    rng = np.random.default_rng(801)
    pair = rng.normal(size=(4, 4))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    tau0, omega0 = igcr3.spin_restricted_triples_seed_from_pair_params(pair, nocc=2)
    tau1, omega1 = igcr3.spin_restricted_triples_seed_from_pair_params(
        pair,
        nocc=2,
        tau_scale=0.2,
        omega_scale=0.3,
    )
    tau2, omega2 = igcr3.spin_restricted_triples_seed_from_pair_params(
        pair,
        nocc=2,
        tau_scale=0.2,
        omega_scale=0.3,
    )
    assert np.allclose(tau0, 0.0)
    assert np.allclose(omega0, 0.0)
    assert np.allclose(tau1, tau2)
    assert np.allclose(omega1, omega2)


def test_params_to_vec_matches_ansatz_apply():
    rng = np.random.default_rng(901)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = 0.03 * rng.normal(size=param.n_params)
    phi0 = hartree_fock_state(norb, nelec)
    psi_a = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi_b = param.params_to_vec(phi0, nelec)(x)
    _assert_same_state_up_to_phase(psi_a, psi_b, atol=1e-10)


def test_sparse_pair_tau_omega_parameterization_expands_to_canonical_storage():
    rng = np.random.default_rng(951)
    norb = 5
    nocc = 2
    pairs = [(0, 1), (2, 4)]
    tau_indices = [(0, 3), (4, 1)]
    omega_indices = [(0, 1, 2), (1, 3, 4)]
    param = igcr3.IGCR3SpinRestrictedParameterization(
        norb=norb,
        nocc=nocc,
        interaction_pairs=pairs,
        tau_indices_=tau_indices,
        omega_indices_=omega_indices,
    )
    x = 0.02 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    pair = ansatz.diagonal.pair_matrix()
    tau = ansatz.diagonal.tau_matrix()
    omega = {
        triple: value
        for triple, value in zip(ansatz.diagonal.omega_indices, ansatz.diagonal.omega_vector())
    }

    allowed_pairs = set(pairs)
    allowed_tau = set(tau_indices)
    allowed_omega = set(omega_indices)
    for p, q in itertools.combinations(range(norb), 2):
        if (p, q) not in allowed_pairs:
            assert pair[p, q] == 0.0
    for p in range(norb):
        for q in range(norb):
            if p != q and (p, q) not in allowed_tau:
                assert tau[p, q] == 0.0
    for triple, value in omega.items():
        if triple not in allowed_omega:
            assert value == 0.0


def test_particle_number_and_sz_sector_shape_is_preserved():
    rng = np.random.default_rng(1001)
    norb = 5
    nocc = 2
    nelec = (nocc, nocc)
    param = igcr3.IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = 0.02 * rng.normal(size=param.n_params)
    phi0 = hartree_fock_state(norb, nelec)
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    expected_dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    assert psi.shape == (expected_dim,)
