from __future__ import annotations

import importlib
import itertools
import numpy as np
import pytest

from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.orbitals import apply_orbital_rotation
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJBalancedDFSeed, UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec


igcr2 = importlib.import_module("xquces.igcr2")


def _random_complex(shape, rng, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _random_real(shape, rng, scale=1.0):
    return scale * rng.normal(size=shape)


def _random_unitary(norb, rng):
    x = _random_complex((norb, norb), rng)
    q, r = np.linalg.qr(x)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 1e-14, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conjugate(phases))


def _align_global_phase(psi_ref, psi):
    overlap = np.vdot(psi_ref, psi)
    if abs(overlap) < 1e-14:
        return psi
    return psi * overlap.conjugate() / abs(overlap)


def _assert_same_state_up_to_phase(psi_ref, psi, atol=1e-10):
    psi_aligned = _align_global_phase(psi_ref, psi)
    err = np.linalg.norm(psi_aligned - psi_ref)
    assert err < atol, f"state mismatch {err}"


def _assert_unitary_close(u, v, atol=1e-10):
    err = np.linalg.norm(u - v)
    assert err < atol, f"unitary mismatch {err}"


def _balanced_ucj_seed(seed=123, scale_t2=0.05, scale_t1=0.03):
    rng = np.random.default_rng(seed)
    nocc = 2
    nvirt = 2
    t2 = _random_real((nocc, nocc, nvirt, nvirt), rng, scale=scale_t2)
    t1 = _random_complex((nocc, nvirt), rng, scale=scale_t1)
    ucj = UCJBalancedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()
    return ucj, t2, t1


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


def _random_restricted_spec(norb, rng, scale=0.2):
    pair = rng.normal(size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    b = rng.normal(size=norb)
    return SpinRestrictedSpec(
        double_params=scale * b,
        pair_params=scale * pair,
    )


def _random_balanced_spec(norb, rng, scale=0.2):
    same = rng.normal(size=(norb, norb))
    same = 0.5 * (same + same.T)
    mixed = rng.normal(size=(norb, norb))
    mixed = 0.5 * (mixed + mixed.T)
    return SpinBalancedSpec(
        same_spin_params=scale * same,
        mixed_spin_params=scale * mixed,
    )


def _all_upper_pairs(norb):
    return list(itertools.combinations(range(norb), 2))


def test_igcr2_module_imports():
    assert igcr2 is not None


def test_structural_from_ucj_balanced_callable():
    ucj, _, _ = _balanced_ucj_seed()
    ansatz = igcr2.IGCR2Ansatz.from_ucj(ucj, nocc=2)
    assert ansatz is not None


def test_structural_from_ucj_restricted_callable():
    ucj, _, _ = _restricted_ucj_seed()
    ansatz = igcr2.IGCR2Ansatz.from_ucj(ucj, nocc=2)
    assert ansatz is not None


def test_structural_from_gcr_ansatz_callable_balanced():
    ucj, _, _ = _balanced_ucj_seed()
    gcr = gcr_from_ucj_ansatz(ucj)
    ansatz = igcr2.IGCR2Ansatz.from_gcr_ansatz(gcr, nocc=2)
    assert ansatz is not None


def test_structural_balanced_parameterization_callable():
    obj = igcr2.IGCR2SpinBalancedParameterization(norb=4, nocc=2)
    assert obj.norb == 4
    assert obj.nocc == 2


def test_structural_restricted_parameterization_callable():
    obj = igcr2.IGCR2SpinRestrictedParameterization(norb=4, nocc=2)
    assert obj.norb == 4
    assert obj.nocc == 2


@pytest.mark.parametrize("norb,nocc", [(4, 2), (6, 3), (5, 0), (5, 5)])
def test_reference_ov_reduction_exact_on_hf(norb, nocc):
    rng = np.random.default_rng(789 + norb + 10 * nocc)
    nelec = (nocc, nocc)

    u = _random_unitary(norb, rng)
    u_ov = igcr2.exact_reference_ov_unitary(u, nocc=nocc)

    phi0 = hartree_fock_state(norb, nelec)
    psi_u = apply_orbital_rotation(phi0, u, norb=norb, nelec=nelec, copy=True)
    psi_ov = apply_orbital_rotation(phi0, u_ov, norb=norb, nelec=nelec, copy=True)

    _assert_same_state_up_to_phase(psi_u, psi_ov, atol=1e-10)


@pytest.mark.parametrize("seed", [321, 322, 323, 324])
def test_restricted_diagonal_reduction_roundtrip_exact_with_identity_rotations(seed):
    rng = np.random.default_rng(seed)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    full = _random_restricted_spec(norb, rng)
    red = igcr2.reduce_spin_restricted(full)
    back = red.to_standard()

    phi0 = hartree_fock_state(norb, nelec)

    psi_full = igcr2.apply_gcr_spin_restricted(
        phi0,
        full.double_params,
        full.pair_params,
        norb,
        nelec,
        left_orbital_rotation=np.eye(norb, dtype=np.complex128),
        right_orbital_rotation=np.eye(norb, dtype=np.complex128),
        copy=True,
    )

    psi_back = igcr2.apply_gcr_spin_restricted(
        phi0,
        back.double_params,
        back.pair_params,
        norb,
        nelec,
        left_orbital_rotation=np.eye(norb, dtype=np.complex128),
        right_orbital_rotation=np.eye(norb, dtype=np.complex128),
        copy=True,
    )

    _assert_same_state_up_to_phase(psi_full, psi_back, atol=1e-10)


@pytest.mark.parametrize("seed", [654, 655, 656, 657])
def test_balanced_diagonal_reduction_roundtrip_exact_with_identity_rotations(seed):
    rng = np.random.default_rng(seed)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    full = _random_balanced_spec(norb, rng)
    red = igcr2.reduce_spin_balanced(full)
    back = red.to_standard()

    phi0 = hartree_fock_state(norb, nelec)

    psi_full = igcr2.apply_gcr_spin_balanced(
        phi0,
        full.same_spin_params,
        full.mixed_spin_params,
        norb,
        nelec,
        left_orbital_rotation=np.eye(norb, dtype=np.complex128),
        right_orbital_rotation=np.eye(norb, dtype=np.complex128),
        copy=True,
    )

    psi_back = igcr2.apply_gcr_spin_balanced(
        phi0,
        back.same_spin_params,
        back.mixed_spin_params,
        norb,
        nelec,
        left_orbital_rotation=np.eye(norb, dtype=np.complex128),
        right_orbital_rotation=np.eye(norb, dtype=np.complex128),
        copy=True,
    )

    _assert_same_state_up_to_phase(psi_full, psi_back, atol=1e-10)


def test_balanced_ucj_to_gcr_exact_state():
    ucj, _, _ = _balanced_ucj_seed(seed=9001)
    gcr = gcr_from_ucj_ansatz(ucj)
    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_gcr = gcr.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_gcr, atol=1e-10)


def test_restricted_ucj_to_gcr_exact_state():
    ucj, _, _ = _restricted_ucj_seed(seed=9003)
    gcr = gcr_from_ucj_ansatz(ucj)
    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_gcr = gcr.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_gcr, atol=1e-10)


def test_balanced_gcr_to_ov_reduced_gcr_exact_state():
    ucj, _, _ = _balanced_ucj_seed(seed=9002)
    gcr = gcr_from_ucj_ansatz(ucj)
    right_ov = igcr2.exact_reference_ov_unitary(gcr.right_orbital_rotation, nocc=2)
    gcr_ov = type(gcr)(
        diagonal=gcr.diagonal,
        left_orbital_rotation=gcr.left_orbital_rotation,
        right_orbital_rotation=right_ov,
    )
    phi0 = hartree_fock_state(4, (2, 2))
    psi_gcr = gcr.apply(phi0, nelec=(2, 2), copy=True)
    psi_gcr_ov = gcr_ov.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_gcr, psi_gcr_ov, atol=1e-10)


def test_restricted_gcr_to_ov_reduced_gcr_exact_state():
    ucj, _, _ = _restricted_ucj_seed(seed=9004)
    gcr = gcr_from_ucj_ansatz(ucj)
    right_ov = igcr2.exact_reference_ov_unitary(gcr.right_orbital_rotation, nocc=2)
    gcr_ov = type(gcr)(
        diagonal=gcr.diagonal,
        left_orbital_rotation=gcr.left_orbital_rotation,
        right_orbital_rotation=right_ov,
    )
    phi0 = hartree_fock_state(4, (2, 2))
    psi_gcr = gcr.apply(phi0, nelec=(2, 2), copy=True)
    psi_gcr_ov = gcr_ov.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_gcr, psi_gcr_ov, atol=1e-10)


def test_balanced_ucj_to_igcr2_exact_state():
    ucj, _, _ = _balanced_ucj_seed(seed=123)
    ig = igcr2.IGCR2Ansatz.from_ucj(ucj, nocc=2)
    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_ig = ig.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_ig, atol=1e-10)


def test_restricted_ucj_to_igcr2_exact_state():
    ucj, _, _ = _restricted_ucj_seed(seed=456)
    ig = igcr2.IGCR2Ansatz.from_ucj(ucj, nocc=2)
    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_ig = ig.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_ig, atol=1e-10)


def test_balanced_ccsd_initialization_path_exact():
    rng = np.random.default_rng(111)
    t2 = _random_real((2, 2, 2, 2), rng, scale=0.04)
    t1 = _random_complex((2, 2), rng, scale=0.02)

    ig = igcr2.IGCR2Ansatz.from_t_balanced(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    )

    ucj = UCJBalancedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()

    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_ig = ig.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_ig, atol=1e-10)


def test_restricted_ccsd_initialization_path_exact():
    rng = np.random.default_rng(222)
    t2 = _random_real((2, 2, 2, 2), rng, scale=0.04)
    t1 = _random_complex((2, 2), rng, scale=0.02)

    ig = igcr2.IGCR2Ansatz.from_t_restricted(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    )

    ucj = UCJRestrictedProjectedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()

    phi0 = hartree_fock_state(4, (2, 2))
    psi_ucj = ucj.apply(phi0, nelec=(2, 2), copy=True)
    psi_ig = ig.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_ucj, psi_ig, atol=1e-10)


def test_balanced_gcr_ov_to_final_igcr_exact_state():
    ucj, _, _ = _balanced_ucj_seed(seed=9005)
    gcr = gcr_from_ucj_ansatz(ucj)
    right_ov = igcr2.exact_reference_ov_unitary(gcr.right_orbital_rotation, nocc=2)
    gcr_ov = type(gcr)(
        diagonal=gcr.diagonal,
        left_orbital_rotation=gcr.left_orbital_rotation,
        right_orbital_rotation=right_ov,
    )
    ig = igcr2.IGCR2Ansatz.from_ucj(ucj, nocc=2)
    phi0 = hartree_fock_state(4, (2, 2))
    psi_gcr_ov = gcr_ov.apply(phi0, nelec=(2, 2), copy=True)
    psi_ig = ig.apply(phi0, nelec=(2, 2), copy=True)
    _assert_same_state_up_to_phase(psi_gcr_ov, psi_ig, atol=1e-10)


@pytest.mark.parametrize("seed", [1401, 1402, 1403])
def test_restricted_params_to_vec_matches_ansatz_apply(seed):
    rng = np.random.default_rng(seed)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = rng.normal(size=param.n_params)
    f = param.params_to_vec(phi0, nelec)
    psi_a = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi_b = f(x)
    _assert_same_state_up_to_phase(psi_a, psi_b, atol=1e-10)


@pytest.mark.parametrize("seed", [1501, 1502, 1503])
def test_balanced_params_to_vec_matches_ansatz_apply(seed):
    rng = np.random.default_rng(seed)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    param = igcr2.IGCR2SpinBalancedParameterization(norb=norb, nocc=nocc)
    x = rng.normal(size=param.n_params)
    f = param.params_to_vec(phi0, nelec)
    psi_a = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi_b = f(x)
    _assert_same_state_up_to_phase(psi_a, psi_b, atol=1e-10)


def test_restricted_parameterization_custom_sparse_pairs_forward():
    rng = np.random.default_rng(1601)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    pairs = [(0, 1), (0, 3), (2, 5)]
    param = igcr2.IGCR2SpinRestrictedParameterization(
        norb=norb,
        nocc=nocc,
        interaction_pairs=pairs,
    )
    assert param.pair_indices == pairs
    x = rng.normal(size=param.n_params)
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    assert psi.shape == phi0.shape


def test_balanced_parameterization_custom_sparse_pairs_forward():
    rng = np.random.default_rng(1701)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    same_pairs = [(0, 1), (1, 4), (2, 5)]
    mixed_pairs = [(0, 2), (0, 5), (3, 4)]
    param = igcr2.IGCR2SpinBalancedParameterization(
        norb=norb,
        nocc=nocc,
        same_spin_interaction_pairs=same_pairs,
        mixed_spin_interaction_pairs=mixed_pairs,
    )
    assert param.same_spin_indices == same_pairs
    assert param.mixed_spin_indices == mixed_pairs
    x = rng.normal(size=param.n_params)
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    assert psi.shape == phi0.shape


@pytest.mark.parametrize(
    "pairs",
    [
        [(1, 0)],
        [(0, 0)],
        [(0, 1), (0, 1)],
        [(0, 7)],
    ],
)
def test_validate_pairs_restricted_invalid(pairs):
    with pytest.raises(ValueError):
        igcr2.IGCR2SpinRestrictedParameterization(
            norb=6,
            nocc=3,
            interaction_pairs=pairs,
        )


@pytest.mark.parametrize(
    "same_pairs,mixed_pairs",
    [
        ([(1, 0)], None),
        ([(0, 0)], None),
        ([(0, 1), (0, 1)], None),
        (None, [(0, 7)]),
    ],
)
def test_validate_pairs_balanced_invalid(same_pairs, mixed_pairs):
    with pytest.raises(ValueError):
        igcr2.IGCR2SpinBalancedParameterization(
            norb=6,
            nocc=3,
            same_spin_interaction_pairs=same_pairs,
            mixed_spin_interaction_pairs=mixed_pairs,
        )


@pytest.mark.parametrize("nocc", [-1, 7])
def test_invalid_nocc_restricted_raises(nocc):
    with pytest.raises(ValueError):
        igcr2.IGCR2SpinRestrictedParameterization(norb=6, nocc=nocc)


@pytest.mark.parametrize("nocc", [-1, 7])
def test_invalid_nocc_balanced_raises(nocc):
    with pytest.raises(ValueError):
        igcr2.IGCR2SpinBalancedParameterization(norb=6, nocc=nocc)


def test_exact_reference_ov_unitary_is_identity_for_zero_occ():
    rng = np.random.default_rng(2001)
    norb = 5
    u = _random_unitary(norb, rng)
    u_ov = igcr2.exact_reference_ov_unitary(u, nocc=0)
    _assert_unitary_close(u_ov, np.eye(norb, dtype=np.complex128), atol=1e-10)


def test_exact_reference_ov_unitary_is_identity_for_full_occ():
    rng = np.random.default_rng(2002)
    norb = 5
    u = _random_unitary(norb, rng)
    u_ov = igcr2.exact_reference_ov_unitary(u, nocc=norb)
    _assert_unitary_close(u_ov, np.eye(norb, dtype=np.complex128), atol=1e-10)


def test_restricted_parameters_from_ucj_ansatz_matches_state():
    ucj, _, _ = _restricted_ucj_seed(seed=2101)
    norb = 4
    nocc = 2
    nelec = (2, 2)
    phi0 = hartree_fock_state(norb, nelec)
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    x = param.parameters_from_ucj_ansatz(ucj)
    psi_param = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi_ref = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=nocc).apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi_ref, psi_param, atol=1e-10)


def test_restricted_parameters_from_ansatz_from_ucj_roundtrip_state():
    ucj, _, _ = _restricted_ucj_seed(seed=2102)
    norb = 4
    nocc = 2
    nelec = (2, 2)
    phi0 = hartree_fock_state(norb, nelec)
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=nocc)
    x = param.parameters_from_ansatz(ansatz)
    recovered = param.ansatz_from_parameters(x)
    psi_ref = ansatz.apply(phi0, nelec=nelec, copy=True)
    psi = recovered.apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi_ref, psi, atol=1e-10)


def test_balanced_parameters_from_ansatz_from_ucj_roundtrip_state():
    ucj, _, _ = _balanced_ucj_seed(seed=2202)
    norb = 4
    nocc = 2
    nelec = (2, 2)
    phi0 = hartree_fock_state(norb, nelec)
    param = igcr2.IGCR2SpinBalancedParameterization(norb=norb, nocc=nocc)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=nocc)
    x = param.parameters_from_ansatz(ansatz)
    recovered = param.ansatz_from_parameters(x)
    psi_ref = ansatz.apply(phi0, nelec=nelec, copy=True)
    psi = recovered.apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi_ref, psi, atol=1e-10)


@pytest.mark.parametrize("seed", [2301, 2302, 2303])
def test_from_gcr_ansatz_restricted_state_equivalence(seed):
    rng = np.random.default_rng(seed)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    diag = _random_restricted_spec(norb, rng, scale=0.1)
    left = _random_unitary(norb, rng)
    right = _random_unitary(norb, rng)
    gcr = GCRAnsatz(
        diagonal=diag,
        left_orbital_rotation=left,
        right_orbital_rotation=right,
    )
    ig = igcr2.IGCR2Ansatz.from_gcr_ansatz(gcr, nocc=nocc)
    psi_gcr = gcr.apply(phi0, nelec=nelec, copy=True)
    psi_ig = ig.apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi_gcr, psi_ig, atol=1e-10)


@pytest.mark.parametrize("seed", [2401, 2402, 2403])
def test_from_gcr_ansatz_balanced_state_equivalence(seed):
    rng = np.random.default_rng(seed)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    phi0 = hartree_fock_state(norb, nelec)
    diag = _random_balanced_spec(norb, rng, scale=0.1)
    left = _random_unitary(norb, rng)
    right = _random_unitary(norb, rng)
    gcr = GCRAnsatz(
        diagonal=diag,
        left_orbital_rotation=left,
        right_orbital_rotation=right,
    )
    ig = igcr2.IGCR2Ansatz.from_gcr_ansatz(gcr, nocc=nocc)
    psi_gcr = gcr.apply(phi0, nelec=nelec, copy=True)
    psi_ig = ig.apply(phi0, nelec=nelec, copy=True)
    _assert_same_state_up_to_phase(psi_gcr, psi_ig, atol=1e-10)


def test_restricted_wrong_shape_raises():
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=6, nocc=3)
    with pytest.raises(ValueError):
        param.ansatz_from_parameters(np.zeros(param.n_params + 1))


def test_balanced_wrong_shape_raises():
    param = igcr2.IGCR2SpinBalancedParameterization(norb=6, nocc=3)
    with pytest.raises(ValueError):
        param.ansatz_from_parameters(np.zeros(param.n_params + 1))


def test_restricted_parameters_from_ansatz_wrong_variant_raises():
    ucj, _, _ = _balanced_ucj_seed(seed=2501)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=2)
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=4, nocc=2)
    with pytest.raises(TypeError):
        param.parameters_from_ansatz(ansatz)


def test_balanced_parameters_from_ansatz_wrong_variant_raises():
    ucj, _, _ = _restricted_ucj_seed(seed=2502)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=2)
    param = igcr2.IGCR2SpinBalancedParameterization(norb=4, nocc=2)
    with pytest.raises(TypeError):
        param.parameters_from_ansatz(ansatz)


def test_restricted_parameters_from_ansatz_wrong_norb_raises():
    ucj, _, _ = _restricted_ucj_seed(seed=2601)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=2)
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=6, nocc=3)
    with pytest.raises(ValueError):
        param.parameters_from_ansatz(ansatz)


def test_balanced_parameters_from_ansatz_wrong_norb_raises():
    ucj, _, _ = _balanced_ucj_seed(seed=2602)
    ansatz = igcr2.IGCR2Ansatz.from_ucj_ansatz(ucj, nocc=2)
    param = igcr2.IGCR2SpinBalancedParameterization(norb=6, nocc=3)
    with pytest.raises(ValueError):
        param.parameters_from_ansatz(ansatz)


def test_restricted_default_pair_count_matches_complete_graph():
    param = igcr2.IGCR2SpinRestrictedParameterization(norb=6, nocc=3)
    assert param.n_pair_params == len(_all_upper_pairs(6))


def test_balanced_default_pair_count_matches_complete_graph():
    param = igcr2.IGCR2SpinBalancedParameterization(norb=6, nocc=3)
    assert param.n_same_spin_params == len(_all_upper_pairs(6))
    assert param.n_mixed_spin_params == len(_all_upper_pairs(6))