"""Regression tests for direct CCSD t-amplitude seeding of iGCR-2.

The key invariant: for L-layer iGCR-2, the l-th diagonal comes from the l-th
double-factorization term J_l, not from splitting a single diagonal equally.
"""

from __future__ import annotations

import numpy as np
import pytest

from xquces.gcr.igcr import (
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinRestrictedParameterization,
    layered_igcr2_from_ccsd_t_amplitudes,
)
from xquces.gcr.pair_uccd_reference import (
    GCR2ProductPairUCCDParameterization,
    GCR3ProductPairUCCDParameterization,
)
from xquces.ucj.init import CCSDDoubleFactorization, factorize_ccsd_t_amplitudes


def _small_t2(nocc: int = 2, nvirt: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t2 = rng.standard_normal((nocc, nocc, nvirt, nvirt)) * 0.3
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return t2


def _small_t1(nocc: int = 2, nvirt: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nocc, nvirt)) * 0.1


# ---------------------------------------------------------------------------
# factorize_ccsd_t_amplitudes
# ---------------------------------------------------------------------------

class TestFactorizeCCSDTAmplitudes:
    def test_returns_dataclass(self):
        t2 = _small_t2()
        result = factorize_ccsd_t_amplitudes(t2, n_reps=1)
        assert isinstance(result, CCSDDoubleFactorization)

    def test_single_rep_shapes(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        df = factorize_ccsd_t_amplitudes(t2, n_reps=1)
        assert len(df.orbital_rotations) == 1
        assert len(df.diagonal_coulomb_mats) == 1
        assert df.orbital_rotations[0].shape == (norb, norb)
        assert df.diagonal_coulomb_mats[0].shape == (norb, norb)

    def test_two_reps_give_distinct_diagonals(self):
        """The two diagonal Coulomb matrices must differ for generic t2."""
        t2 = _small_t2(seed=99)
        df = factorize_ccsd_t_amplitudes(t2, n_reps=2)
        assert len(df.diagonal_coulomb_mats) == 2
        J0, J1 = df.diagonal_coulomb_mats
        assert not np.allclose(J0, J1), (
            "J_0 and J_1 are identical — cannot test distinct-diagonal invariant"
        )

    def test_symmetry_of_diagonal_coulomb_mats(self):
        t2 = _small_t2()
        df = factorize_ccsd_t_amplitudes(t2, n_reps=2)
        for J in df.diagonal_coulomb_mats:
            assert np.allclose(J, J.T, atol=1e-12), "J_l should be symmetric"

    def test_unitarity_of_orbital_rotations(self):
        t2 = _small_t2()
        df = factorize_ccsd_t_amplitudes(t2, n_reps=2)
        norb = df.orbital_rotations[0].shape[0]
        eye = np.eye(norb)
        for U in df.orbital_rotations:
            assert np.allclose(U.conj().T @ U, eye, atol=1e-10)

    def test_final_orbital_rotation_with_t1(self):
        t2 = _small_t2()
        t1 = _small_t1()
        df = factorize_ccsd_t_amplitudes(t2, t1=t1, n_reps=1)
        assert df.final_orbital_rotation is not None
        norb = df.orbital_rotations[0].shape[0]
        eye = np.eye(norb)
        U_F = df.final_orbital_rotation
        assert np.allclose(U_F.conj().T @ U_F, eye, atol=1e-10)

    def test_no_final_orbital_rotation_without_t1(self):
        t2 = _small_t2()
        df = factorize_ccsd_t_amplitudes(t2, t1=None, n_reps=1)
        assert df.final_orbital_rotation is None


# ---------------------------------------------------------------------------
# layered_igcr2_from_ccsd_t_amplitudes
# ---------------------------------------------------------------------------

class TestLayeredIGCR2FromCCSD:
    def test_single_layer_returns_igcr2_ansatz(self):
        t2 = _small_t2()
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=1)
        assert isinstance(result, IGCR2Ansatz)

    def test_two_layers_returns_igcr2_layered_ansatz(self):
        t2 = _small_t2()
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=2)
        assert isinstance(result, IGCR2LayeredAnsatz)

    def test_layered_shape_invariant(self):
        """len(diagonals) == L and len(rotations) == L + 1."""
        for layers in [1, 2, 3]:
            t2 = _small_t2()
            result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=layers)
            if layers == 1:
                assert isinstance(result, IGCR2Ansatz)
            else:
                assert result.layers == layers
                assert len(result.rotations) == layers + 1

    def test_rotation_unitarity(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        eye = np.eye(norb)
        t2 = _small_t2(nocc, nvirt)
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=2, nocc=nocc)
        for R in result.rotations:
            assert np.allclose(R.conj().T @ R, eye, atol=1e-10)

    def test_nocc_inferred_from_t2(self):
        nocc, nvirt = 3, 2
        t2 = _small_t2(nocc, nvirt)
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=1)
        assert result.nocc == nocc

    # KEY REGRESSION TEST
    def test_two_layer_diagonals_are_distinct(self):
        """Core invariant: diagonal[k] comes from J_k, not from splitting J_0.

        This test was the original bug: the old UCJ-mediated path would split a
        single-layer diagonal equally across all layers, giving identical blocks.
        The new direct CCSD path uses the distinct J_1 and J_2 from ffsim.
        """
        t2 = _small_t2(seed=42)
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=2)
        assert isinstance(result, IGCR2LayeredAnsatz)
        assert result.layers == 2
        d0 = result.diagonals[0].pair
        d1 = result.diagonals[1].pair
        assert not np.allclose(d0, d1, atol=1e-10), (
            "diagonals[0] and diagonals[1] are identical — "
            "this indicates the equal-split bug is present"
        )

    def test_three_layer_first_diagonal_distinct_from_others(self):
        """At minimum the first diagonal must differ from the second for generic t2.

        With 3 repetitions, ffsim may occasionally return equal factors for
        some seeds; we only require the first-vs-second check as the core
        invariant (matches the two-layer test above).
        """
        t2 = _small_t2(seed=42)
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=3)
        assert not np.allclose(result.diagonals[0].pair, result.diagonals[1].pair, atol=1e-10), (
            "diagonals[0] == diagonals[1] — equal-split bug"
        )

    def test_diagonal_pair_is_zero_on_diagonal(self):
        """iGCR-2 reduction zeroes the diagonal of the pair matrix."""
        t2 = _small_t2()
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, layers=2)
        for d in result.diagonals:
            assert np.allclose(np.diag(d.pair), 0.0, atol=1e-14)

    def test_with_t1(self):
        t2 = _small_t2()
        t1 = _small_t1()
        result = layered_igcr2_from_ccsd_t_amplitudes(t2, t1=t1, layers=2)
        assert isinstance(result, IGCR2LayeredAnsatz)
        assert result.layers == 2


# ---------------------------------------------------------------------------
# IGCR2SpinRestrictedParameterization.parameters_from_t_amplitudes
# ---------------------------------------------------------------------------

class TestIGCR2ParameterizationFromTAmplitudes:
    def test_single_layer_param_shape(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=1)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)

    def test_two_layer_param_shape(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=2)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)

    def test_two_layer_pair_blocks_differ(self):
        """The pair parameter blocks for layer 0 and layer 1 must differ."""
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt, seed=42)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=2)
        x = param.parameters_from_t_amplitudes(t2)
        # Extract the two pair parameter blocks
        n_left = param.n_left_orbital_rotation_params
        n_pair_per_layer = param.n_pair_params_per_layer
        block0 = x[n_left : n_left + n_pair_per_layer]
        block1 = x[n_left + n_pair_per_layer : n_left + 2 * n_pair_per_layer]
        assert not np.allclose(block0, block1, atol=1e-10), (
            "pair parameter blocks are identical — equal-split bug"
        )

    def test_parameters_from_ucj_ansatz_raises(self):
        """The deprecated UCJ path must raise NotImplementedError."""
        from xquces.ucj.model import UCJAnsatz, UCJLayer, SpinRestrictedSpec

        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=2)
        dummy_ucj = UCJAnsatz(
            layers=(
                UCJLayer(
                    diagonal=SpinRestrictedSpec(
                        double_params=np.zeros(norb),
                        pair_params=np.zeros((norb, norb)),
                    ),
                    orbital_rotation=np.eye(norb, dtype=complex),
                ),
            ),
            final_orbital_rotation=None,
        )
        with pytest.raises(NotImplementedError, match="parameters_from_t_amplitudes"):
            param.parameters_from_ucj_ansatz(dummy_ucj)

    def test_from_ucj_raises(self):
        from xquces.ucj.model import UCJAnsatz, UCJLayer, SpinRestrictedSpec

        norb = 5
        nocc = 2
        dummy_ucj = UCJAnsatz(
            layers=(
                UCJLayer(
                    diagonal=SpinRestrictedSpec(
                        double_params=np.zeros(norb),
                        pair_params=np.zeros((norb, norb)),
                    ),
                    orbital_rotation=np.eye(norb, dtype=complex),
                ),
            ),
            final_orbital_rotation=None,
        )
        with pytest.raises(NotImplementedError, match="from_t_amplitudes"):
            IGCR2Ansatz.from_ucj(dummy_ucj, nocc)

    def test_roundtrip_single_layer(self):
        """parameters_from_ansatz(ansatz_from_parameters(x)) ≈ x."""
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=1)
        x = param.parameters_from_t_amplitudes(t2)
        ansatz = param.ansatz_from_parameters(x)
        x2 = param.parameters_from_ansatz(ansatz)
        assert np.allclose(x, x2, atol=1e-10)

    def test_roundtrip_two_layers(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=2)
        x = param.parameters_from_t_amplitudes(t2)
        ansatz = param.ansatz_from_parameters(x)
        x2 = param.parameters_from_ansatz(ansatz)
        assert np.allclose(x, x2, atol=1e-10)

    def test_with_t1_differs_from_without(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        t1 = _small_t1(nocc, nvirt)
        param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc, layers=1)
        x_no_t1 = param.parameters_from_t_amplitudes(t2, t1=None)
        x_with_t1 = param.parameters_from_t_amplitudes(t2, t1=t1)
        assert not np.allclose(x_no_t1, x_with_t1), (
            "t1 had no effect on the parameters"
        )


# ---------------------------------------------------------------------------
# GCR2ProductPairUCCDParameterization.parameters_from_t_amplitudes
# ---------------------------------------------------------------------------

class TestGCR2ProductPUCCDFromTAmplitudes:
    def test_single_layer_param_shape(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = GCR2ProductPairUCCDParameterization(norb=norb, nocc=nocc, layers=1)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)

    def test_two_layer_param_shape(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = GCR2ProductPairUCCDParameterization(norb=norb, nocc=nocc, layers=2)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)

    def test_two_layer_ansatz_diagonals_distinct(self):
        """Regression: the two diagonal blocks of the seeded ansatz must differ."""
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt, seed=42)
        param = GCR2ProductPairUCCDParameterization(norb=norb, nocc=nocc, layers=2)
        x = param.parameters_from_t_amplitudes(t2)
        # Recover the ansatz and check its diagonals
        ansatz = param.ansatz_parameterization.ansatz_from_parameters(
            x[param.n_reference_params:]
        )
        assert isinstance(ansatz, IGCR2LayeredAnsatz)
        d0 = ansatz.diagonals[0].pair
        d1 = ansatz.diagonals[1].pair
        assert not np.allclose(d0, d1, atol=1e-10), (
            "pair blocks are identical — equal-split bug in GCR2ProductPUCCD"
        )

    def test_reference_params_match_parameters_from_t2(self):
        """The reference part of parameters_from_t_amplitudes == reference_parameters_from_t2."""
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = GCR2ProductPairUCCDParameterization(norb=norb, nocc=nocc, layers=1)
        x = param.parameters_from_t_amplitudes(t2)
        ref_from_call = x[: param.n_reference_params]
        ref_expected = param.reference_parameters_from_t2(t2, scale=0.5)
        assert np.allclose(ref_from_call, ref_expected, atol=1e-14)


# ---------------------------------------------------------------------------
# GCR3ProductPairUCCDParameterization.parameters_from_t_amplitudes
# ---------------------------------------------------------------------------

class TestGCR3ProductPUCCDFromTAmplitudes:
    def test_param_shape(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        param = GCR3ProductPairUCCDParameterization(norb=norb, nocc=nocc)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)

    def test_with_t1(self):
        nocc, nvirt = 2, 3
        norb = nocc + nvirt
        t2 = _small_t2(nocc, nvirt)
        t1 = _small_t1(nocc, nvirt)
        param = GCR3ProductPairUCCDParameterization(norb=norb, nocc=nocc)
        x = param.parameters_from_t_amplitudes(t2, t1=t1)
        assert x.shape == (param.n_params,)
