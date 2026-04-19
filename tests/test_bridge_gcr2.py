import numpy as np
import ffsim

from xquces.gcr import (
    GCR2FullUnitaryChart,
    GCR2SplitBridgeParameterization,
    GCR2UntiedSplitBridgeParameterization,
    IGCR2SpinRestrictedParameterization,
)
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz, UCJLayer


def _align_phase(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    overlap = np.vdot(reference, state)
    if abs(overlap) == 0:
        return state
    return state * overlap.conjugate() / abs(overlap)


def test_split_bridge_zero_parameters_are_identity_on_reference():
    norb = 4
    nelec = (2, 2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    param = GCR2SplitBridgeParameterization(norb, nocc=2)

    state = param.params_to_vec(reference, nelec)(np.zeros(param.n_params))

    assert np.allclose(state, reference)


def test_untied_split_bridge_zero_parameters_are_identity_on_reference():
    norb = 4
    nelec = (2, 2)
    reference = ffsim.hartree_fock_state(norb, nelec)
    param = GCR2UntiedSplitBridgeParameterization(norb, nocc=2)

    state = param.params_to_vec(reference, nelec)(np.zeros(param.n_params))

    assert np.allclose(state, reference)


def test_split_bridge_with_zero_middle_matches_igcr2():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    rng = np.random.default_rng(11)
    reference = ffsim.hartree_fock_state(norb, nelec)
    igcr2 = IGCR2SpinRestrictedParameterization(norb, nocc)
    bridge = GCR2SplitBridgeParameterization(norb, nocc)
    x_igcr2 = 0.05 * rng.normal(size=igcr2.n_params)

    psi_igcr2 = igcr2.params_to_vec(reference, nelec)(x_igcr2)
    psi_bridge = bridge.params_to_vec(reference, nelec)(
        bridge.parameters_from_igcr2(x_igcr2, igcr2)
    )

    assert np.linalg.norm(_align_phase(psi_igcr2, psi_bridge) - psi_igcr2) < 1e-10


def test_untied_split_bridge_with_zero_middle_matches_igcr2():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    rng = np.random.default_rng(13)
    reference = ffsim.hartree_fock_state(norb, nelec)
    igcr2 = IGCR2SpinRestrictedParameterization(norb, nocc)
    bridge = GCR2UntiedSplitBridgeParameterization(norb, nocc)
    x_igcr2 = 0.05 * rng.normal(size=igcr2.n_params)

    psi_igcr2 = igcr2.params_to_vec(reference, nelec)(x_igcr2)
    psi_bridge = bridge.params_to_vec(reference, nelec)(
        bridge.parameters_from_igcr2(x_igcr2, igcr2)
    )

    assert np.linalg.norm(_align_phase(psi_igcr2, psi_bridge) - psi_igcr2) < 1e-10


def test_split_bridge_preserves_norm():
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(17)
    reference = ffsim.hartree_fock_state(norb, nelec)
    param = GCR2SplitBridgeParameterization(norb, nocc=2)
    x = 0.07 * rng.normal(size=param.n_params)

    state = param.params_to_vec(reference, nelec)(x)

    assert np.isclose(np.linalg.norm(state), 1.0)


def test_untied_split_bridge_preserves_norm():
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(19)
    reference = ffsim.hartree_fock_state(norb, nelec)
    param = GCR2UntiedSplitBridgeParameterization(norb, nocc=2)
    x = 0.07 * rng.normal(size=param.n_params)

    state = param.params_to_vec(reference, nelec)(x)

    assert np.isclose(np.linalg.norm(state), 1.0)


def test_bridge_parameter_counts_add_only_pair_sized_blocks():
    norb = 6
    nocc = 3
    igcr2 = IGCR2SpinRestrictedParameterization(norb, nocc)
    tied = GCR2SplitBridgeParameterization(norb, nocc)
    untied = GCR2UntiedSplitBridgeParameterization(norb, nocc)

    assert tied.n_params == igcr2.n_params + tied.n_middle_orbital_rotation_params
    assert (
        untied.n_params
        == igcr2.n_params
        + untied.n_middle_orbital_rotation_params
        + untied.n_pair_params
    )


def _two_layer_ucj(norb: int, nocc: int, *, tied_diagonal: bool) -> UCJAnsatz:
    rng = np.random.default_rng(23 if tied_diagonal else 29)
    chart = GCR2SplitBridgeParameterization(norb, nocc).middle_orbital_chart
    u1 = chart.unitary_from_parameters(0.04 * rng.normal(size=chart.n_params(norb)), norb)
    u2 = chart.unitary_from_parameters(0.04 * rng.normal(size=chart.n_params(norb)), norb)
    final = chart.unitary_from_parameters(
        0.03 * rng.normal(size=chart.n_params(norb)),
        norb,
    )
    pair1 = np.zeros((norb, norb), dtype=np.float64)
    pair2 = np.zeros((norb, norb), dtype=np.float64)
    values1 = 0.05 * rng.normal(size=norb * (norb - 1) // 2)
    values2 = values1 if tied_diagonal else 0.05 * rng.normal(size=values1.size)
    idx = 0
    for p in range(norb):
        for q in range(p + 1, norb):
            pair1[p, q] = pair1[q, p] = values1[idx]
            pair2[p, q] = pair2[q, p] = values2[idx]
            idx += 1

    return UCJAnsatz(
        layers=(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=np.zeros(norb),
                    pair_params=pair1,
                ),
                orbital_rotation=u1,
            ),
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=np.zeros(norb),
                    pair_params=pair2,
                ),
                orbital_rotation=u2,
            ),
        ),
        final_orbital_rotation=final,
    )


def test_untied_split_bridge_projects_two_layer_ucj_seed():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    reference = ffsim.hartree_fock_state(norb, nelec)
    ucj = _two_layer_ucj(norb, nocc, tied_diagonal=False)
    base = IGCR2SpinRestrictedParameterization(
        norb,
        nocc,
        left_orbital_chart=GCR2FullUnitaryChart(),
        right_orbital_chart_override=GCR2FullUnitaryChart(),
    )
    bridge = GCR2UntiedSplitBridgeParameterization(
        norb,
        nocc,
        base_parameterization=base,
    )

    psi_ucj = ucj.apply(reference, nelec=nelec, copy=True)
    psi_bridge = bridge.params_to_vec(reference, nelec)(
        bridge.parameters_from_ucj_ansatz(ucj)
    )

    assert np.linalg.norm(_align_phase(psi_ucj, psi_bridge) - psi_ucj) < 1e-10


def test_tied_split_bridge_projects_two_layer_ucj_seed_when_diagonals_match():
    norb = 4
    nelec = (2, 2)
    nocc = nelec[0]
    reference = ffsim.hartree_fock_state(norb, nelec)
    ucj = _two_layer_ucj(norb, nocc, tied_diagonal=True)
    base = IGCR2SpinRestrictedParameterization(
        norb,
        nocc,
        left_orbital_chart=GCR2FullUnitaryChart(),
        right_orbital_chart_override=GCR2FullUnitaryChart(),
    )
    bridge = GCR2SplitBridgeParameterization(
        norb,
        nocc,
        base_parameterization=base,
    )

    psi_ucj = ucj.apply(reference, nelec=nelec, copy=True)
    psi_bridge = bridge.params_to_vec(reference, nelec)(
        bridge.parameters_from_ucj_ansatz(ucj)
    )

    assert np.linalg.norm(_align_phase(psi_ucj, psi_bridge) - psi_ucj) < 1e-10
