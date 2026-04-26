from __future__ import annotations

import numpy as np
import scipy.linalg
import ffsim

from xquces.gcr import (
    GCR2PairUCCDParameterization,
    GCR3PairUCCDParameterization,
    GCR4PairUCCDParameterization,
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCR4SpinRestrictedParameterization,
)
from xquces.orbitals import apply_orbital_rotation


def _assert_same_state_up_to_phase(param_a, x_a, param_b, x_b, atol=1e-10):
    psi_a = param_a.state_from_parameters(x_a)
    psi_b = param_b.state_from_parameters(x_b)
    overlap = np.vdot(psi_a, psi_b)
    if abs(overlap) > 1e-14:
        psi_b = psi_b * overlap.conjugate() / abs(overlap)
    err = np.linalg.norm(psi_a - psi_b)
    assert err < atol, f"state mismatch {err}"


def test_pair_uccd_ansatz_object_roundtrip_preserves_state():
    rng = np.random.default_rng(20260426)
    parameterizations = [
        GCR2PairUCCDParameterization(norb=6, nocc=3),
        GCR3PairUCCDParameterization(norb=6, nocc=3),
        GCR4PairUCCDParameterization(norb=6, nocc=3),
    ]

    for param in parameterizations:
        x = 0.2 * rng.normal(size=param.n_params)
        reference_params, ansatz_params = param.split_parameters(x)
        ansatz = param.ansatz_parameterization.ansatz_from_parameters(ansatz_params)
        roundtrip_ansatz_params = param.ansatz_parameterization.parameters_from_ansatz(
            ansatz
        )
        y = np.concatenate([reference_params, roundtrip_ansatz_params])

        _assert_same_state_up_to_phase(param, x, param, y)


def test_pair_uccd_higher_order_transfer_preserves_nested_state():
    rng = np.random.default_rng(20260427)
    parameterizations = {
        "GCR2": GCR2PairUCCDParameterization(norb=6, nocc=3),
        "GCR3": GCR3PairUCCDParameterization(norb=6, nocc=3),
        "GCR4": GCR4PairUCCDParameterization(norb=6, nocc=3),
    }

    for source_name, target_name in [
        ("GCR2", "GCR3"),
        ("GCR2", "GCR4"),
        ("GCR3", "GCR4"),
    ]:
        source_param = parameterizations[source_name]
        target_param = parameterizations[target_name]
        source_x = 0.2 * rng.normal(size=source_param.n_params)
        target_x = target_param.transfer_parameters_from(
            source_x,
            previous_parameterization=source_param,
        )

        _assert_same_state_up_to_phase(
            source_param,
            source_x,
            target_param,
            target_x,
        )


def test_general_orbital_overlap_transport_preserves_state_locally():
    rng = np.random.default_rng(20260428)
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)
    raw = rng.normal(size=(norb, norb))
    kappa = 0.005 * (raw - raw.T)
    basis_change = scipy.linalg.expm(kappa).astype(np.complex128)
    hf_state = ffsim.hartree_fock_state(norb, nelec)

    for param in [
        IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc),
        IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc),
        IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc),
    ]:
        x = 0.02 * rng.normal(size=param.n_params)
        y = param.transfer_parameters_from(
            x,
            previous_parameterization=param,
            orbital_overlap=basis_change,
        )
        psi_old = param.params_to_vec(hf_state, nelec)(x)
        psi_new = param.params_to_vec(hf_state, nelec)(y)
        target = apply_orbital_rotation(
            psi_old,
            (basis_change.conj().T, basis_change.conj().T),
            norb,
            nelec,
        )
        fidelity = abs(np.vdot(target, psi_new)) ** 2
        assert fidelity > 1.0 - 1e-10

    for param in [
        GCR2PairUCCDParameterization(norb=norb, nocc=nocc),
        GCR3PairUCCDParameterization(norb=norb, nocc=nocc),
        GCR4PairUCCDParameterization(norb=norb, nocc=nocc),
    ]:
        x = 0.02 * rng.normal(size=param.n_params)
        y = param.transfer_parameters_from(
            x,
            previous_parameterization=param,
            orbital_overlap=basis_change,
        )
        psi_old = param.state_from_parameters(x)
        psi_new = param.state_from_parameters(y)
        target = apply_orbital_rotation(
            psi_old,
            (basis_change.conj().T, basis_change.conj().T),
            norb,
            nelec,
        )
        fidelity = abs(np.vdot(target, psi_new)) ** 2
        assert fidelity > 1.0 - 1e-7
