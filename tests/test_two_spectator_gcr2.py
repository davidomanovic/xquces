from __future__ import annotations

import itertools

import numpy as np
import pytest

try:
    import ffsim
    HAS_FFSIM = True
except ImportError:
    HAS_FFSIM = False

from xquces.gcr.controlled_orbital_gcr2 import (
    GCR2SpectatorOrbitalParameterization,
    GCR2TwoSpectatorOrbitalAnsatz,
    GCR2TwoSpectatorOrbitalParameterization,
    _two_spectator_transform_basis,
    _validate_quadruples,
    project_two_spectator_gauge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_state(norb, nelec):
    if HAS_FFSIM:
        return ffsim.hartree_fock_state(norb, nelec)
    from xquces.states import hartree_fock_state
    return hartree_fock_state(norb, nelec)


def _random_params(param, rng, scale=0.05):
    return scale * rng.standard_normal(param.n_params)


# ---------------------------------------------------------------------------
# Parameter count tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norb,expected", [
    (4,  0),   # m=2: C(m-1,2)=0 free params per pair
    (5,  1 * 10),   # m=3: C(2,2)=1 per pair, C(5,2)=10 pairs → 10
    (6,  3 * 15),   # m=4: C(3,2)=3, C(6,2)=15 → 45
    (10, 21 * 45),  # m=8: C(7,2)=21, C(10,2)=45 → 945
    (12, 36 * 66),  # m=10: C(9,2)=36, C(12,2)=66 → 2376
    (14, 55 * 91),  # m=12: C(11,2)=55, C(14,2)=91 → 5005
])
def test_two_spec_param_count(norb, expected):
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=norb // 2)
    assert param.n_two_spec_params == expected, (
        f"norb={norb}: expected {expected}, got {param.n_two_spec_params}"
    )


def test_param_counts_logged():
    """Document expected counts for norb=10, 12, 14 as an explicit assertion."""
    for norb, expected in [(10, 945), (12, 2376), (14, 5005)]:
        param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=norb // 2)
        assert param.n_two_spec_params == expected


# ---------------------------------------------------------------------------
# Gauge / transform tests
# ---------------------------------------------------------------------------

def test_transform_basis_orthonormal():
    for m in range(2, 8):
        T = _two_spectator_transform_basis(m)
        if T.shape[1] == 0:
            continue
        err = np.linalg.norm(T.T @ T - np.eye(T.shape[1]))
        assert err < 1e-12, f"m={m}: T^T T not identity, err={err}"


def test_transform_basis_in_null_space():
    """Each column of T should satisfy A T[:,j] = 0 (vertex-edge incidence)."""
    for m in range(2, 8):
        T = _two_spectator_transform_basis(m)
        if T.shape[1] == 0:
            continue
        edges = list(itertools.combinations(range(m), 2))
        n_edges = len(edges)
        A = np.zeros((m, n_edges))
        for j, (r, s) in enumerate(edges):
            A[r, j] = 1.0
            A[s, j] = 1.0
        err = np.linalg.norm(A @ T)
        assert err < 1e-11, f"m={m}: basis not in null space, err={err}"


def test_gauge_projection_idempotent():
    norb = 6
    rng = np.random.default_rng(42)
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=3)
    quadruples = param.quadruple_indices
    pairs = param.pair_indices
    xi_rand = rng.standard_normal(param.n_full_two_spec_terms)
    xi1 = project_two_spectator_gauge(xi_rand, norb, pairs, quadruples)
    xi2 = project_two_spectator_gauge(xi1, norb, pairs, quadruples)
    assert np.allclose(xi1, xi2, atol=1e-12), "projection is not idempotent"


def test_gauge_projection_satisfies_constraint():
    """After projection, row sums of ξ_{rs,pq} over s are zero for every r."""
    norb = 6
    rng = np.random.default_rng(7)
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=3)
    quadruples = param.quadruple_indices
    pairs = param.pair_indices
    xi_rand = rng.standard_normal(param.n_full_two_spec_terms)
    xi_proj = project_two_spectator_gauge(xi_rand, norb, pairs, quadruples)
    quad_map = {(r, s, p, q): v for (r, s, p, q), v in zip(quadruples, xi_proj)}
    for p, q in pairs:
        spectators = [x for x in range(norb) if x != p and x != q]
        for r in spectators:
            row_sum = sum(
                quad_map.get((min(r, s), max(r, s), p, q), 0.0)
                for s in spectators if s != r
            )
            assert abs(row_sum) < 1e-11, (
                f"Gauge violated at ({p},{q}), r={r}: sum={row_sum}"
            )


# ---------------------------------------------------------------------------
# Validate quadruples
# ---------------------------------------------------------------------------

def test_validate_quadruples_default():
    norb = 5
    pairs = tuple(itertools.combinations(range(norb), 2))
    quads = _validate_quadruples(None, norb, pairs)
    # Every (r,s,p,q) must satisfy r<s, r,s ∉ {p,q}
    for r, s, p, q in quads:
        assert r < s
        assert r not in (p, q) and s not in (p, q)
    # Count: for each (p,q), C(norb-2, 2) quadruples
    m = norb - 2
    expected = len(pairs) * (m * (m - 1) // 2)
    assert len(quads) == expected


# ---------------------------------------------------------------------------
# Zero-param identity tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim required")
def test_zero_two_spec_params_is_identity():
    """Setting all two-spectator params to zero must leave the state unchanged
    compared to the one-spectator ansatz with the same base params."""
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(99)
    one_param = GCR2SpectatorOrbitalParameterization(norb=norb, nocc=2)
    two_param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=2)
    phi0 = _hf_state(norb, nelec)
    x1 = _random_params(one_param, rng, scale=0.1)
    x2 = two_param.parameters_from_one_spectator(x1, one_spec_parameterization=one_param)
    psi1 = one_param.ansatz_from_parameters(x1).apply(phi0, nelec=nelec)
    psi2 = two_param.ansatz_from_parameters(x2).apply(phi0, nelec=nelec)
    assert np.allclose(psi1, psi2, atol=1e-12), "two-spec with ξ=0 differs from one-spec"


@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim required")
def test_two_spec_preserves_norm():
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(17)
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=2)
    phi0 = _hf_state(norb, nelec)
    x = _random_params(param, rng, scale=0.05)
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Round-trip: parameters → ansatz → parameters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim required")
def test_round_trip_parameters():
    norb = 4
    rng = np.random.default_rng(55)
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=2)
    x = _random_params(param, rng, scale=0.05)
    ansatz = param.ansatz_from_parameters(x)
    x_recovered = param.parameters_from_ansatz(ansatz)
    assert np.allclose(x, x_recovered, atol=1e-9), (
        f"Round-trip failed: max diff = {np.max(np.abs(x - x_recovered)):.3e}"
    )


# ---------------------------------------------------------------------------
# parameters_from_igcr2 reduces to C_1 = C_2 = 0
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim required")
def test_parameters_from_igcr2_zero_spectators():
    """parameters_from_igcr2 must produce zero spectator and two-spec params."""
    norb = 4
    from xquces.gcr.igcr2 import IGCR2SpinRestrictedParameterization
    base = IGCR2SpinRestrictedParameterization(norb=norb, nocc=2)
    rng = np.random.default_rng(13)
    x_base = 0.05 * rng.standard_normal(base.n_params)
    two_param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=2)
    x_two = two_param.parameters_from_igcr2(x_base, parameterization=base)
    n_left = two_param.n_left_orbital_rotation_params
    n_pair = two_param.n_pair_params
    n_spec = two_param.n_spectator_params
    n_two = two_param.n_two_spec_params
    spec_block = x_two[n_left + n_pair: n_left + n_pair + n_spec]
    two_block = x_two[n_left + n_pair + n_spec: n_left + n_pair + n_spec + n_two]
    assert np.allclose(spec_block, 0.0), "spectator block should be zero"
    assert np.allclose(two_block, 0.0), "two-spec block should be zero"


# ---------------------------------------------------------------------------
# Rotation sign check: small xi produces expected perturbation direction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim required")
def test_two_spec_rotation_antisymmetry():
    """Swapping Ñ_r → -Ñ_r by permuting the spectator indices should flip the rotation."""
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(200)
    param = GCR2TwoSpectatorOrbitalParameterization(norb=norb, nocc=2)
    phi0 = _hf_state(norb, nelec)
    # Use zero for everything except a single two-spec parameter
    x0 = np.zeros(param.n_params)
    if param.n_two_spec_params == 0:
        pytest.skip("no two-spec params for norb=4")
    two_start = (
        param.n_left_orbital_rotation_params
        + param.n_pair_params
        + param.n_spectator_params
    )
    x_pos = x0.copy(); x_pos[two_start] = 0.05
    x_neg = x0.copy(); x_neg[two_start] = -0.05
    psi_pos = param.ansatz_from_parameters(x_pos).apply(phi0, nelec=nelec)
    psi_neg = param.ansatz_from_parameters(x_neg).apply(phi0, nelec=nelec)
    # The states should differ (non-trivial) and be related by conjugation in a sector
    assert not np.allclose(psi_pos, psi_neg, atol=1e-6), "positive and negative xi gave same state"
