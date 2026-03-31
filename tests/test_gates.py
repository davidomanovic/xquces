import numpy as np

from xquces.basis import occ_rows
from xquces.gates import apply_ucj_spin_balanced, apply_ucj_spin_restricted
from xquces.orbitals import unitary_from_generator


def sr_reference(vec, double_params, pair_params, norb, nelec):
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    out = np.array(vec, dtype=np.complex128, copy=True).reshape(len(occ_a), len(occ_b))

    for ia, oa in enumerate(occ_a):
        set_a = set(map(int, oa))
        for ib, ob in enumerate(occ_b):
            set_b = set(map(int, ob))
            phase = 1.0 + 0.0j
            counts = np.zeros(norb, dtype=int)
            for p in set_a:
                counts[p] += 1
            for p in set_b:
                counts[p] += 1

            for p in range(norb):
                if counts[p] == 2:
                    phase *= np.exp(1j * double_params[p])

            for p in range(norb):
                for q in range(p + 1, norb):
                    if counts[p] and counts[q]:
                        phase *= np.exp(1j * pair_params[p, q] * counts[p] * counts[q])

            out[ia, ib] *= phase

    return out.reshape(-1)


def sb_reference(vec, same_spin_params, mixed_spin_params, norb, nelec):
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    out = np.array(vec, dtype=np.complex128, copy=True).reshape(len(occ_a), len(occ_b))

    for ia, oa in enumerate(occ_a):
        set_a = set(map(int, oa))
        for ib, ob in enumerate(occ_b):
            set_b = set(map(int, ob))
            exponent = 0.0

            for p in range(norb):
                if p in set_a:
                    exponent += 0.5 * same_spin_params[p, p]
                if p in set_b:
                    exponent += 0.5 * same_spin_params[p, p]
                if p in set_a and p in set_b:
                    exponent += mixed_spin_params[p, p]

            for p in range(norb):
                for q in range(p + 1, norb):
                    if p in set_a and q in set_a:
                        exponent += same_spin_params[p, q]
                    if p in set_b and q in set_b:
                        exponent += same_spin_params[p, q]
                    if p in set_a and q in set_b:
                        exponent += mixed_spin_params[p, q]
                    if p in set_b and q in set_a:
                        exponent += mixed_spin_params[p, q]

            out[ia, ib] *= np.exp(1j * exponent)

    return out.reshape(-1)


def random_state(dim, seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def test_ucj_spin_restricted_matches_reference():
    norb = 4
    nelec = (2, 2)
    dim = 36
    vec = random_state(dim, 1)

    double_params = np.array([0.2, -0.1, 0.05, 0.3])
    pair_params = np.zeros((norb, norb))
    pair_params[0, 1] = pair_params[1, 0] = 0.4
    pair_params[0, 2] = pair_params[2, 0] = -0.2
    pair_params[1, 3] = pair_params[3, 1] = 0.15

    ref = sr_reference(vec, double_params, pair_params, norb, nelec)
    out = apply_ucj_spin_restricted(vec, double_params, pair_params, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_ucj_spin_restricted_identity_for_zero_params():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 2)

    double_params = np.zeros(norb)
    pair_params = np.zeros((norb, norb))

    out = apply_ucj_spin_restricted(vec, double_params, pair_params, norb, nelec)
    assert np.allclose(out, vec, atol=1e-12)


def test_ucj_spin_balanced_matches_reference():
    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 3)

    same_spin_params = np.zeros((norb, norb))
    mixed_spin_params = np.zeros((norb, norb))

    same_spin_params[0, 0] = 0.3
    same_spin_params[1, 1] = -0.1
    same_spin_params[0, 2] = same_spin_params[2, 0] = 0.2
    same_spin_params[1, 3] = same_spin_params[3, 1] = -0.25

    mixed_spin_params[0, 0] = 0.5
    mixed_spin_params[1, 1] = -0.2
    mixed_spin_params[0, 1] = mixed_spin_params[1, 0] = 0.1
    mixed_spin_params[2, 3] = mixed_spin_params[3, 2] = -0.15

    ref = sb_reference(vec, same_spin_params, mixed_spin_params, norb, nelec)
    out = apply_ucj_spin_balanced(vec, same_spin_params, mixed_spin_params, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_ucj_spin_restricted_with_orbital_rotation_matches_manual_composition():
    from xquces.orbitals import apply_orbital_rotation

    norb = 4
    nelec = (2, 1)
    dim = 24
    vec = random_state(dim, 4)

    a = np.array(
        [
            [0.0, 0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.08],
            [0.0, 0.0, -0.08, 0.0],
        ],
        dtype=np.complex128,
    )
    kappa = a - a.conj().T
    u = unitary_from_generator(kappa)

    double_params = np.array([0.2, -0.1, 0.0, 0.1])
    pair_params = np.zeros((norb, norb))
    pair_params[0, 1] = pair_params[1, 0] = 0.3
    pair_params[2, 3] = pair_params[3, 2] = -0.2

    manual = apply_orbital_rotation(vec, u.conj().T, norb, nelec)
    manual = apply_ucj_spin_restricted(manual, double_params, pair_params, norb, nelec, orbital_rotation=None, copy=False)
    manual = apply_orbital_rotation(manual, u, norb, nelec)

    out = apply_ucj_spin_restricted(
        vec,
        double_params,
        pair_params,
        norb,
        nelec,
        orbital_rotation=u,
    )

    assert np.allclose(out, manual, atol=1e-10)