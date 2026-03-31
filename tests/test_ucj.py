import numpy as np

from xquces.gates import apply_ucj_spin_balanced, apply_ucj_spin_restricted
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj import UCJAnsatz, UCJLayer, SpinBalancedSpec, SpinRestrictedSpec


def random_state(dim, seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def test_spin_restricted_spec_zeros_diagonal_of_pair():
    norb = 4
    pair = np.ones((norb, norb))
    spec = SpinRestrictedSpec(
        double_params=np.zeros(norb),
        pair_params=pair,
    )
    assert np.allclose(np.diag(spec.pair_params), 0.0)


def test_ucj_layer_rejects_nonunitary():
    norb = 4
    spec = SpinRestrictedSpec(
        double_params=np.zeros(norb),
        pair_params=np.zeros((norb, norb)),
    )
    bad = np.ones((norb, norb), dtype=np.complex128)
    try:
        UCJLayer(diagonal=spec, orbital_rotation=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for nonunitary rotation")


def test_ucj_ansatz_single_layer_spin_restricted_matches_gate_call():
    norb = 4
    nelec = (2, 2)
    dim = 36
    vec = random_state(dim, 10)

    double_params = np.array([0.2, -0.1, 0.05, 0.0])
    pair_params = np.zeros((norb, norb))
    pair_params[0, 1] = pair_params[1, 0] = 0.3

    u = np.eye(norb, dtype=np.complex128)

    spec = SpinRestrictedSpec(double_params=double_params, pair_params=pair_params)
    layer = UCJLayer(diagonal=spec, orbital_rotation=u)
    ansatz = UCJAnsatz(layers=(layer,))

    ref = apply_ucj_spin_restricted(vec, double_params, pair_params, norb, nelec, orbital_rotation=u)
    out = ansatz.apply(vec, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_ucj_ansatz_single_layer_spin_balanced_matches_gate_call():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 11)

    same = np.zeros((norb, norb))
    mixed = np.zeros((norb, norb))
    same[0, 2] = same[2, 0] = 0.2
    mixed[1, 3] = mixed[3, 1] = -0.3
    mixed[0, 0] = 0.4

    u = np.eye(norb, dtype=np.complex128)

    spec = SpinBalancedSpec(same_spin_params=same, mixed_spin_params=mixed)
    layer = UCJLayer(diagonal=spec, orbital_rotation=u)
    ansatz = UCJAnsatz(layers=(layer,))

    ref = apply_ucj_spin_balanced(vec, same, mixed, norb, nelec, orbital_rotation=u)
    out = ansatz.apply(vec, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_ucj_ansatz_two_layers_matches_manual_composition():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 12)

    d1 = SpinRestrictedSpec(
        double_params=np.array([0.1, 0.0, -0.1, 0.2]),
        pair_params=np.array(
            [
                [0.0, 0.2, 0.0, 0.0],
                [0.2, 0.0, -0.1, 0.0],
                [0.0, -0.1, 0.0, 0.05],
                [0.0, 0.0, 0.05, 0.0],
            ]
        ),
    )
    d2 = SpinBalancedSpec(
        same_spin_params=np.array(
            [
                [0.2, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, -0.1],
                [0.1, 0.0, -0.1, 0.0],
                [0.0, -0.1, 0.0, 0.3],
            ]
        ),
        mixed_spin_params=np.array(
            [
                [0.4, 0.1, 0.0, 0.0],
                [0.1, -0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15, 0.1],
            ]
        ),
    )

    u1 = np.eye(norb, dtype=np.complex128)
    u2 = np.eye(norb, dtype=np.complex128)

    ansatz = UCJAnsatz(
        layers=(
            UCJLayer(diagonal=d1, orbital_rotation=u1),
            UCJLayer(diagonal=d2, orbital_rotation=u2),
        )
    )

    manual = apply_ucj_spin_restricted(vec, d1.double_params, d1.pair_params, norb, nelec, orbital_rotation=u1)
    manual = apply_ucj_spin_balanced(manual, d2.same_spin_params, d2.mixed_spin_params, norb, nelec, orbital_rotation=u2)

    out = ansatz.apply(vec, nelec)

    assert np.allclose(out, manual, atol=1e-10)


def test_ucj_ansatz_final_orbital_rotation_applied():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 13)

    spec = SpinRestrictedSpec(
        double_params=np.zeros(norb),
        pair_params=np.zeros((norb, norb)),
    )
    layer = UCJLayer(diagonal=spec, orbital_rotation=np.eye(norb, dtype=np.complex128))

    final = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    ansatz = UCJAnsatz(layers=(layer,), final_orbital_rotation=final)

    ref = apply_orbital_rotation(vec, final, norb, nelec)
    out = ansatz.apply(vec, nelec)

    assert np.allclose(out, ref, atol=1e-10)