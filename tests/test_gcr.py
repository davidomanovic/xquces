import numpy as np

from xquces.gcr import (
    GCRSpinBalancedParameterization,
    GCRSpinRestrictedParameterization,
    gcr_from_ucj_ansatz,
)
from xquces.states import hartree_fock_state
from xquces.ucj import UCJSpinBalancedParameterization, UCJSpinRestrictedParameterization


def _random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def test_gcr_spin_restricted_roundtrip_state():
    norb = 4
    nelec = (2, 2)
    param = GCRSpinRestrictedParameterization(
        norb=norb,
        interaction_pairs=[(0, 1), (0, 2), (1, 3), (2, 3)],
    )
    rng = np.random.default_rng(123)
    x = 0.2 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    x_rt = param.parameters_from_ansatz(ansatz)
    ansatz_rt = param.ansatz_from_parameters(x_rt)
    vec = _random_state(hartree_fock_state(norb, nelec).size, seed=456)
    out = ansatz.apply(vec, nelec=nelec)
    out_rt = ansatz_rt.apply(vec, nelec=nelec)
    assert np.allclose(out, out_rt, atol=1e-10)


def test_gcr_spin_balanced_roundtrip_state():
    norb = 4
    nelec = (2, 2)
    param = GCRSpinBalancedParameterization(
        norb=norb,
        same_spin_interaction_pairs=[(0, 1), (0, 2), (1, 3), (2, 3)],
        mixed_spin_interaction_pairs=[(0, 0), (0, 1), (1, 2), (2, 3)],
    )
    rng = np.random.default_rng(321)
    x = 0.2 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    x_rt = param.parameters_from_ansatz(ansatz)
    ansatz_rt = param.ansatz_from_parameters(x_rt)
    vec = _random_state(hartree_fock_state(norb, nelec).size, seed=654)
    out = ansatz.apply(vec, nelec=nelec)
    out_rt = ansatz_rt.apply(vec, nelec=nelec)
    assert np.allclose(out, out_rt, atol=1e-10)


def test_single_layer_ucj_maps_exactly_to_gcr_spin_restricted():
    norb = 4
    nelec = (2, 2)
    param = UCJSpinRestrictedParameterization(
        norb=norb,
        n_layers=1,
        interaction_pairs=[(0, 1), (0, 2), (1, 3), (2, 3)],
        with_final_orbital_rotation=True,
    )
    rng = np.random.default_rng(111)
    x = 0.2 * rng.normal(size=param.n_params)
    ucj = param.ansatz_from_parameters(x)
    gcr = gcr_from_ucj_ansatz(ucj)
    vec = _random_state(hartree_fock_state(norb, nelec).size, seed=222)
    out_ucj = ucj.apply(vec, nelec=nelec)
    out_gcr = gcr.apply(vec, nelec=nelec)
    assert np.allclose(out_ucj, out_gcr, atol=1e-10)


def test_single_layer_ucj_maps_exactly_to_gcr_spin_balanced():
    norb = 4
    nelec = (2, 2)
    param = UCJSpinBalancedParameterization(
        norb=norb,
        n_layers=1,
        same_spin_interaction_pairs=[(0, 1), (0, 2), (1, 3), (2, 3)],
        mixed_spin_interaction_pairs=[(0, 0), (0, 1), (1, 2), (2, 3)],
        with_final_orbital_rotation=True,
    )
    rng = np.random.default_rng(333)
    x = 0.2 * rng.normal(size=param.n_params)
    ucj = param.ansatz_from_parameters(x)
    gcr = gcr_from_ucj_ansatz(ucj)
    vec = _random_state(hartree_fock_state(norb, nelec).size, seed=444)
    out_ucj = ucj.apply(vec, nelec=nelec)
    out_gcr = gcr.apply(vec, nelec=nelec)
    assert np.allclose(out_ucj, out_gcr, atol=1e-10)
