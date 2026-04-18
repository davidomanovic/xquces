import numpy as np
import ffsim

from xquces.gcr.commutator_gcr2 import (
    GCR2PairHopAnsatz,
    GCR2PairHopParameterization,
    apply_gcr2_pairhop_middle_in_place_num_rep,
    gcr2_pairhop_middle_generator,
)
from xquces.gcr.igcr2 import IGCR2SpinRestrictedParameterization
from xquces.ucj.init import UCJRestrictedProjectedDFSeed


def test_zero_parameters_are_identity_on_reference():
    norb = 4
    nelec = (2, 2)
    param = GCR2PairHopParameterization(norb, nocc=2)
    reference = ffsim.hartree_fock_state(norb, nelec)

    state = param.params_to_vec(reference, nelec)(np.zeros(param.n_params))

    assert np.allclose(state, reference)


def test_pair_hop_generator_is_antihermitian():
    norb = 4
    nelec = (2, 2)
    param = GCR2PairHopParameterization(norb, nocc=2)
    rng = np.random.default_rng(11)
    pairs = param.pair_indices

    generator = gcr2_pairhop_middle_generator(
        np.zeros(len(pairs)),
        rng.normal(size=len(pairs)),
        norb,
        nelec,
        pairs,
    )

    assert np.allclose((generator.getH() + generator).toarray(), 0.0)


def test_pairhop_ansatz_preserves_norm():
    norb = 4
    nelec = (2, 2)
    param = GCR2PairHopParameterization(norb, nocc=2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    rng = np.random.default_rng(13)
    x = 0.05 * rng.normal(size=param.n_params)

    state = param.params_to_vec(reference, nelec)(x)

    assert np.isclose(np.linalg.norm(state), 1.0)


def test_ucj_seed_maps_to_zero_pair_hop_sector():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    param = GCR2PairHopParameterization(norb, nocc)
    t1 = np.zeros((nocc, norb - nocc))
    t2 = np.zeros((nocc, nocc, norb - nocc, norb - nocc))
    ucj = UCJRestrictedProjectedDFSeed(t2=t2, t1=t1, n_reps=1).build_ansatz()

    x = param.parameters_from_ucj_ansatz(ucj)
    _, _, pair_hop, _ = param._split(x)

    assert np.allclose(pair_hop, 0.0)


def test_ucj_seed_state_matches_igcr2_when_pair_hop_is_zero():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    rng = np.random.default_rng(31)
    t1 = 0.02 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    t2 = 0.03 * rng.normal(size=(2, 2, 2, 2))
    ucj = UCJRestrictedProjectedDFSeed(
        t2=t2,
        t1=t1,
        n_reps=1,
        tol=1e-12,
        optimize=False,
    ).build_ansatz()
    reference = ffsim.hartree_fock_state(norb, nelec)

    pairhop_param = GCR2PairHopParameterization(norb, nocc)
    igcr2_param = IGCR2SpinRestrictedParameterization(norb, nocc)
    psi_pairhop = pairhop_param.params_to_vec(reference, nelec)(
        pairhop_param.parameters_from_ucj_ansatz(ucj)
    )
    psi_igcr2 = igcr2_param.params_to_vec(reference, nelec)(
        igcr2_param.parameters_from_ucj_ansatz(ucj)
    )

    overlap = np.vdot(psi_igcr2, psi_pairhop)
    psi_pairhop *= overlap.conjugate() / abs(overlap)
    assert np.linalg.norm(psi_pairhop - psi_igcr2) < 1e-10


def test_rust_pairhop_action_matches_sparse_reference():
    if apply_gcr2_pairhop_middle_in_place_num_rep is None:
        return
    norb = 4
    nelec = (2, 2)
    param = GCR2PairHopParameterization(norb, nocc=2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    rng = np.random.default_rng(41)
    x = 0.07 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    sparse_ansatz = GCR2PairHopAnsatz(
        pair_params=ansatz.pair_params,
        pair_hop_params=ansatz.pair_hop_params,
        left=ansatz.left,
        right=ansatz.right,
        norb=ansatz.norb,
        nocc=ansatz.nocc,
        pairs=ansatz.pairs,
        use_rust=False,
    )

    psi_rust = ansatz.apply(reference, nelec)
    psi_sparse = sparse_ansatz.apply(reference, nelec)

    assert np.linalg.norm(psi_rust - psi_sparse) < 1e-10
