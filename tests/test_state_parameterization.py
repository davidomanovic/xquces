import numpy as np

from xquces.gcr import IGCR2SpinRestrictedParameterization
from xquces.state_parameterization import (
    CompositeReferenceAnsatzParameterization,
    DOCIStateParameterization,
)
from xquces.states import doci_dimension, doci_state


def test_composite_reference_ansatz_matches_manual_application():
    norb = 4
    nocc = 2
    nelec = (2, 2)

    reference_parameterization = DOCIStateParameterization(norb=norb, nelec=nelec)
    ansatz_parameterization = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    composite = CompositeReferenceAnsatzParameterization(
        reference_parameterization=reference_parameterization,
        ansatz_parameterization=ansatz_parameterization,
        nelec=nelec,
    )

    ref_params = np.linspace(0.1, 0.5, doci_dimension(norb, nelec) - 1)
    ansatz_params = np.zeros(ansatz_parameterization.n_params, dtype=np.float64)
    params = np.concatenate([ref_params, ansatz_params])

    manual = ansatz_parameterization.ansatz_from_parameters(ansatz_params).apply(
        doci_state(norb, nelec, params=ref_params),
        nelec=nelec,
        copy=True,
    )
    combined = composite.state_from_parameters(params)
    assert np.allclose(combined, manual)


def test_composite_parameters_from_state_and_ansatz_roundtrip():
    norb = 4
    nocc = 2
    nelec = (2, 2)

    reference_parameterization = DOCIStateParameterization(norb=norb, nelec=nelec)
    ansatz_parameterization = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    composite = CompositeReferenceAnsatzParameterization(
        reference_parameterization=reference_parameterization,
        ansatz_parameterization=ansatz_parameterization,
        nelec=nelec,
    )

    ref_params = np.linspace(0.2, 0.6, doci_dimension(norb, nelec) - 1)
    ansatz_params = np.zeros(ansatz_parameterization.n_params, dtype=np.float64)
    reference_state = doci_state(norb, nelec, params=ref_params)
    ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)

    combined = composite.parameters_from_state_and_ansatz(reference_state, ansatz)
    assert np.allclose(combined[: reference_parameterization.n_params], ref_params)
    assert np.allclose(combined[reference_parameterization.n_params :], ansatz_params)
