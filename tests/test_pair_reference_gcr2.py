from __future__ import annotations

import numpy as np

from xquces.gcr import (
    GCR2PairReferenceParameterization,
    IGCR2SpinRestrictedParameterization,
    make_restricted_gcr_jacobian,
)
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJRestrictedProjectedDFSeed


def _align_global_phase(psi_ref, psi):
    overlap = np.vdot(psi_ref, psi)
    if abs(overlap) < 1e-14:
        return psi
    return psi * overlap.conjugate() / abs(overlap)


def _assert_same_state_up_to_phase(psi_ref, psi, atol=1e-10):
    psi_aligned = _align_global_phase(psi_ref, psi)
    err = np.linalg.norm(psi_aligned - psi_ref)
    assert err < atol, f"state mismatch {err}"


def test_pair_reference_zero_block_matches_igcr2():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    base = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    pair_ref = GCR2PairReferenceParameterization(norb=norb, nocc=nocc)
    rng = np.random.default_rng(1001)

    x_base = 0.05 * rng.normal(size=base.n_params)
    x_pair = pair_ref.parameters_from_igcr2(x_base, parameterization=base)

    psi_base = base.ansatz_from_parameters(x_base).apply(phi0, nelec=nelec, copy=True)
    psi_pair = pair_ref.ansatz_from_parameters(x_pair).apply(phi0, nelec=nelec, copy=True)

    _assert_same_state_up_to_phase(psi_base, psi_pair, atol=1e-10)


def test_pair_reference_parameters_from_ucj_ansatz_insert_zero_block():
    norb = 4
    nocc = 2
    t1 = np.zeros((nocc, norb - nocc), dtype=np.complex128)
    t2 = np.zeros((nocc, nocc, norb - nocc, norb - nocc), dtype=np.float64)
    ucj = UCJRestrictedProjectedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()
    param = GCR2PairReferenceParameterization(norb=norb, nocc=nocc)

    x = param.parameters_from_ucj_ansatz(ucj)
    _, _, pair_reference, _ = param._split(x)

    assert np.allclose(pair_reference, 0.0)


def test_pair_reference_params_to_vec_matches_ansatz_apply():
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    param = GCR2PairReferenceParameterization(norb=norb, nocc=nocc)
    rng = np.random.default_rng(1002)
    x = 0.05 * rng.normal(size=param.n_params)

    psi_a = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi_b = param.params_to_vec(phi0, nelec)(x)

    _assert_same_state_up_to_phase(psi_a, psi_b, atol=1e-10)


def test_pair_reference_restricted_jacobian_matches_finite_difference():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    param = GCR2PairReferenceParameterization(norb=norb, nocc=nocc)
    rng = np.random.default_rng(1003)
    x = 0.02 * rng.normal(size=param.n_params)

    f = param.params_to_vec(phi0, nelec)
    jac = make_restricted_gcr_jacobian(param, phi0, nelec)(x)

    eps = 1e-7
    fd = np.zeros_like(jac)
    for k in range(param.n_params):
        step = np.zeros(param.n_params, dtype=np.float64)
        step[k] = eps
        fd[:, k] = (f(x + step) - f(x - step)) / (2 * eps)

    err = np.linalg.norm(jac - fd)
    ref = np.linalg.norm(fd)
    assert err / max(ref, 1.0) < 5e-5
