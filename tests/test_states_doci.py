import numpy as np

from xquces.states import (
    apply_doci_unitary,
    doci_dimension,
    doci_params_from_state,
    doci_state,
    hartree_fock_state,
)


def test_doci_default_state_matches_hartree_fock():
    norb = 4
    nelec = (2, 2)
    assert np.allclose(doci_state(norb, nelec), hartree_fock_state(norb, nelec))


def test_doci_state_parameter_roundtrip():
    norb = 4
    nelec = (2, 2)
    params = np.linspace(0.1, 0.5, doci_dimension(norb, nelec) - 1)
    state = doci_state(norb, nelec, params=params)
    recovered = doci_params_from_state(state, norb, nelec)
    rebuilt = doci_state(norb, nelec, params=recovered)
    assert np.allclose(rebuilt, state)


def test_apply_doci_unitary_matches_target_state_from_hf():
    norb = 4
    nelec = (2, 2)
    params = np.linspace(0.2, 0.6, doci_dimension(norb, nelec) - 1)
    hf = hartree_fock_state(norb, nelec)
    out = apply_doci_unitary(hf, norb, nelec, params=params)
    target = doci_state(norb, nelec, params=params)
    assert np.allclose(out, target)
