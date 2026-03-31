import itertools

import numpy as np

from xquces.basis import flatten_state, occ_rows, reshape_state
from xquces.orbitals import apply_orbital_rotation, canonicalize_unitary, unitary_from_generator


def brute_force_orbital_rotation(vec, u, norb, nelec):
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    ta = np.zeros((len(occ_a), len(occ_a)), dtype=np.complex128)
    tb = np.zeros((len(occ_b), len(occ_b)), dtype=np.complex128)

    for i, bra in enumerate(occ_a):
        for j, ket in enumerate(occ_a):
            ta[i, j] = np.linalg.det(u[np.ix_(bra, ket)])

    for i, bra in enumerate(occ_b):
        for j, ket in enumerate(occ_b):
            tb[i, j] = np.linalg.det(u[np.ix_(bra, ket)])

    mat = reshape_state(vec, norb, nelec)
    return flatten_state(ta @ mat @ tb.T)


def test_canonicalize_unitary_preserves_unitarity():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, _ = np.linalg.qr(x)
    u = canonicalize_unitary(q)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_unitary_from_generator_is_unitary():
    rng = np.random.default_rng(456)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    kappa = a - a.conj().T
    u = unitary_from_generator(kappa)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_apply_orbital_rotation_matches_bruteforce():
    rng = np.random.default_rng(789)
    norb = 4
    nelec = (2, 1)

    a = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    kappa = 0.1 * (a - a.conj().T)
    u = unitary_from_generator(kappa)

    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)

    ref = brute_force_orbital_rotation(vec, u, norb, nelec)
    out = apply_orbital_rotation(vec, u, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)