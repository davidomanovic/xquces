import numpy as np

from xquces.orbitals import ov_generator_from_t1, ov_unitary_from_t1
from xquces.ucj import (
    UCJAnsatz,
    UCJLayer,
    UCJRestrictedHeuristicSeed,
    SpinBalancedSpec,
    SpinRestrictedSpec,
    heuristic_restricted_pair_params_from_t2,
)


def random_state(dim, seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def test_ov_generator_from_t1_is_antihermitian():
    t1 = np.array(
        [
            [0.1 + 0.2j, -0.3 + 0.1j],
            [0.05 - 0.1j, 0.2 + 0.0j],
        ],
        dtype=np.complex128,
    )
    kappa = ov_generator_from_t1(t1)
    assert kappa.shape == (4, 4)
    assert np.allclose(kappa.conj().T, -kappa, atol=1e-12)


def test_ov_unitary_from_t1_is_unitary():
    t1 = np.array(
        [
            [0.1 + 0.2j, -0.3 + 0.1j],
            [0.05 - 0.1j, 0.2 + 0.0j],
        ],
        dtype=np.complex128,
    )
    u = ov_unitary_from_t1(t1)
    assert u.shape == (4, 4)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_pair_params_from_t2_shape_symmetry_zero_diag():
    t2 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    t2[0, 1, 0, 1] = 0.7
    pair = heuristic_restricted_pair_params_from_t2(t2)
    assert pair.shape == (4, 4)
    assert np.allclose(pair, pair.T, atol=1e-12)
    assert np.allclose(np.diag(pair), 0.0, atol=1e-12)


def test_pair_params_from_t2_nonzero_when_t2_nonzero():
    t2 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    t2[0, 1, 0, 1] = 0.7
    pair = heuristic_restricted_pair_params_from_t2(t2)
    assert np.max(np.abs(pair)) > 0.0


def test_ucj_identity_spin_restricted_is_identity_action():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 20)
    ansatz = UCJAnsatz(
        layers=(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=np.zeros(norb),
                    pair_params=np.zeros((norb, norb)),
                ),
                orbital_rotation=np.eye(norb, dtype=np.complex128),
            ),
        )
    )
    out = ansatz.apply(vec, nelec)
    assert np.allclose(out, vec, atol=1e-12)


def test_ucj_identity_spin_balanced_is_identity_action():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 21)
    ansatz = UCJAnsatz(
        layers=(
            UCJLayer(
                diagonal=SpinBalancedSpec(
                    same_spin_params=np.zeros((norb, norb)),
                    mixed_spin_params=np.zeros((norb, norb)),
                ),
                orbital_rotation=np.eye(norb, dtype=np.complex128),
            ),
        )
    )
    out = ansatz.apply(vec, nelec)
    assert np.allclose(out, vec, atol=1e-12)


def test_ucj_from_ov_rotation_zero_t1_gives_identity():
    t1 = np.zeros((2, 2), dtype=np.complex128)
    ansatz = UCJRestrictedHeuristicSeed(
        np.zeros((2, 2, 2, 2), dtype=np.float64),
        t1=t1,
        pair_scale=0.0,
    ).build_ansatz()
    vec = random_state(36, 22)
    out = ansatz.apply(vec, nelec=(2, 2))
    assert np.allclose(out, vec, atol=1e-12)


def test_ucj_from_t_amplitudes_builds_valid_ansatz():
    t2 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    t2[0, 1, 0, 1] = 0.3
    t1 = np.array(
        [
            [0.02, -0.01],
            [0.03, 0.04],
        ],
        dtype=np.float64,
    )
    ansatz = UCJRestrictedHeuristicSeed(t2, t1=t1, pair_scale=0.5).build_ansatz()
    assert ansatz.norb == 4
    assert len(ansatz.layers) == 1
    assert isinstance(ansatz.layers[0].diagonal, SpinRestrictedSpec)
    assert ansatz.final_orbital_rotation is not None
    assert np.allclose(
        ansatz.final_orbital_rotation.conj().T @ ansatz.final_orbital_rotation,
        np.eye(4),
        atol=1e-10,
    )


def test_ucj_from_t_amplitudes_zero_t2_zero_t1_is_identity_action():
    t2 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    t1 = np.zeros((2, 2), dtype=np.float64)
    ansatz = UCJRestrictedHeuristicSeed(t2, t1=t1).build_ansatz()
    vec = random_state(36, 23)
    out = ansatz.apply(vec, nelec=(2, 2))
    assert np.allclose(out, vec, atol=1e-12)


def test_ucj_from_t_amplitudes_nonzero_t2_changes_pair_params():
    t2 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    t2[0, 1, 0, 1] = 0.6
    ansatz = UCJRestrictedHeuristicSeed(t2).build_ansatz()
    pair = ansatz.layers[0].diagonal.pair_params
    assert np.max(np.abs(pair)) > 0.0
    assert np.allclose(pair, pair.T, atol=1e-12)
    assert np.allclose(np.diag(pair), 0.0, atol=1e-12)
