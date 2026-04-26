from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
    IGCR2SpinRestrictedParameterization,
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _orbital_relabeling_unitary,
    _parameters_from_zero_diag_antihermitian,
    _restricted_irreducible_pair_matrix,
    _restricted_left_phase_vector,
    orbital_relabeling_from_overlap,
    reduce_spin_restricted,
    relabel_igcr2_ansatz_orbitals,
    _zero_diag_antihermitian_from_parameters,
)
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3SpinRestrictedParameterization,
    _default_pair_indices,
    _default_tau_indices,
    _default_triple_indices,
    _values_from_ordered_matrix,
    relabel_igcr3_ansatz_orbitals,
)
from xquces.gcr.igcr4 import (
    IGCR4Ansatz,
    IGCR4SpinRestrictedParameterization,
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
    relabel_igcr4_ansatz_orbitals,
)
from xquces.gcr.pair_uccd_reference import (
    _combined_seed,
    _is_trivial_relabel,
    _make_composite,
    _make_composite_jacobian,
    _transfer_reference_params,
)
from xquces.gcr.model import gcr_from_ucj_ansatz
from xquces.pair_uccd import ProductPairUCCDStateParameterization
from xquces.ucj.model import UCJAnsatz


def _igcr2_product_ansatz_from_ucj(ansatz: UCJAnsatz, nocc: int) -> IGCR2Ansatz:
    gcr = gcr_from_ucj_ansatz(ansatz)
    if not gcr.is_spin_restricted:
        raise TypeError("expected a spin-restricted UCJ-derived GCR ansatz")
    diagonal = reduce_spin_restricted(gcr.diagonal)
    phase_vec = _restricted_left_phase_vector(gcr.diagonal.double_params, nocc)
    left = np.asarray(gcr.left_orbital_rotation, dtype=np.complex128) @ _diag_unitary(phase_vec)
    return IGCR2Ansatz(
        diagonal=diagonal,
        left=left,
        right=np.asarray(gcr.right_orbital_rotation, dtype=np.complex128),
        nocc=nocc,
    )


def _final_unitary_for_transfer(left_param_unitary, right_eff, right_chart, nocc):
    if isinstance(right_chart, (IGCR2ReferenceOVUnitaryChart, IGCR2RealReferenceOVUnitaryChart)):
        return _final_unitary_from_left_and_right(left_param_unitary, right_eff, nocc)
    return np.asarray(left_param_unitary, dtype=np.complex128) @ np.asarray(right_eff, dtype=np.complex128)


def _igcr2_product_transfer_parameters_from_ansatz(
    parameterization: IGCR2SpinRestrictedParameterization,
    ansatz: IGCR2Ansatz,
) -> np.ndarray:
    if ansatz.norb != parameterization.norb:
        raise ValueError("ansatz norb does not match parameterization")
    if not ansatz.is_spin_restricted:
        raise TypeError("expected a spin-restricted ansatz")
    left_chart = parameterization._left_orbital_chart
    if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
        left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(
            np.asarray(ansatz.left, dtype=np.complex128)
        )
    else:
        left_params = left_chart.parameters_from_unitary(
            np.asarray(ansatz.left, dtype=np.complex128)
        )
        right_phase = np.zeros(parameterization.norb, dtype=np.float64)

    pair_eff = ansatz.diagonal.pair
    right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
    out = np.zeros(parameterization.n_params, dtype=np.float64)
    idx = 0

    n = parameterization.n_left_orbital_rotation_params
    out[idx : idx + n] = left_params
    idx += n

    n = parameterization.n_pair_params
    out[idx : idx + n] = np.asarray(
        [pair_eff[p, q] for p, q in parameterization.pair_indices],
        dtype=np.float64,
    )
    idx += n

    n = parameterization.n_right_orbital_rotation_params
    left_param_unitary = parameterization._left_orbital_chart.unitary_from_parameters(
        left_params,
        parameterization.norb,
    )
    final_eff = _final_unitary_for_transfer(
        left_param_unitary,
        right_eff,
        parameterization.right_orbital_chart,
        parameterization.nocc,
    )
    out[idx : idx + n] = parameterization.right_orbital_chart.parameters_from_unitary(final_eff)
    return parameterization._public_parameters_from_native(out)


def _igcr3_product_transfer_parameters_from_ansatz(
    parameterization: IGCR3SpinRestrictedParameterization,
    ansatz: IGCR3Ansatz,
) -> np.ndarray:
    if ansatz.norb != parameterization.norb:
        raise ValueError("ansatz norb does not match parameterization")
    d = ansatz.diagonal
    pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
    tau = d.tau_matrix()
    omega = d.omega_vector()

    cubic_onebody_phase = np.zeros(parameterization.norb, dtype=np.float64)
    reduced_pair_values = None
    reduced_cubic_values = None
    if parameterization.uses_reduced_cubic_chart:
        full_pair_values = np.asarray(
            [pair_eff[p, q] for p, q in _default_pair_indices(parameterization.norb)],
            dtype=np.float64,
        )
        full_cubic = np.concatenate(
            [
                _values_from_ordered_matrix(tau, _default_tau_indices(parameterization.norb)),
                omega,
            ]
        )
        reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = (
            parameterization.cubic_reduction.reduce_full(full_pair_values, full_cubic)
        )

    phase_vec = _restricted_left_phase_vector(d.full_double(), parameterization.nocc) + cubic_onebody_phase
    left_eff = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(phase_vec)
    left_chart = parameterization._left_orbital_chart
    if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
        left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(left_eff)
    else:
        left_params = left_chart.parameters_from_unitary(left_eff)
        right_phase = np.zeros(parameterization.norb, dtype=np.float64)

    right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)

    out = np.zeros(parameterization.n_params, dtype=np.float64)
    idx = 0
    n = parameterization.n_left_orbital_rotation_params
    out[idx : idx + n] = left_params
    idx += n

    n = parameterization.n_pair_params
    out[idx : idx + n] = np.asarray(
        [pair_eff[p, q] for p, q in parameterization.pair_indices],
        dtype=np.float64,
    )
    idx += n

    if parameterization.uses_reduced_cubic_chart:
        out[
            parameterization.n_left_orbital_rotation_params :
            parameterization.n_left_orbital_rotation_params + parameterization.n_pair_params
        ] = reduced_pair_values
        n = parameterization.n_tau_params
        out[idx : idx + n] = reduced_cubic_values
        idx += n
    else:
        n = parameterization.n_tau_params
        out[idx : idx + n] = _values_from_ordered_matrix(tau, parameterization.tau_indices)
        idx += n

        n = parameterization.n_omega_params
        full_omega = {triple: value for value, triple in zip(omega, d.omega_indices)}
        out[idx : idx + n] = np.asarray(
            [full_omega[t] for t in parameterization.omega_indices],
            dtype=np.float64,
        )
        idx += n

    n = parameterization.n_right_orbital_rotation_params
    left_param_unitary = parameterization._left_orbital_chart.unitary_from_parameters(
        left_params,
        parameterization.norb,
    )
    final_eff = _final_unitary_for_transfer(
        left_param_unitary,
        right_eff,
        parameterization.right_orbital_chart,
        parameterization.nocc,
    )
    out[idx : idx + n] = parameterization.right_orbital_chart.parameters_from_unitary(final_eff)
    return parameterization._public_parameters_from_native(out)


def _igcr4_product_transfer_parameters_from_ansatz(
    parameterization: IGCR4SpinRestrictedParameterization,
    ansatz: IGCR4Ansatz,
) -> np.ndarray:
    if ansatz.norb != parameterization.norb:
        raise ValueError("ansatz norb does not match parameterization")

    d = ansatz.diagonal
    pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
    tau = d.tau_matrix()
    omega = d.omega_vector()
    eta = d.eta_vector()
    rho = d.rho_vector()
    sigma = d.sigma_vector()

    full_pair_values = np.asarray(
        [pair_eff[p, q] for p, q in _default_pair_indices(parameterization.norb)],
        dtype=np.float64,
    )
    full_cubic = np.concatenate(
        [
            _values_from_ordered_matrix(tau, _default_tau_indices(parameterization.norb)),
            omega,
        ]
    )
    full_quartic = np.concatenate([eta, rho, sigma])

    if parameterization.uses_reduced_quartic_chart:
        full_cubic, reduced_quartic_values = parameterization.quartic_reduction.reduce_full(
            full_cubic,
            full_quartic,
        )
    else:
        reduced_quartic_values = None

    cubic_onebody_phase = np.zeros(parameterization.norb, dtype=np.float64)
    if parameterization.uses_reduced_cubic_chart:
        reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = (
            parameterization.cubic_reduction.reduce_full(full_pair_values, full_cubic)
        )
    else:
        reduced_pair_values = None
        reduced_cubic_values = None

    phase_vec = _restricted_left_phase_vector(d.full_double(), parameterization.nocc) + cubic_onebody_phase
    left_eff = np.asarray(ansatz.left, dtype=np.complex128) @ _diag_unitary(phase_vec)
    left_chart = parameterization._left_orbital_chart
    if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
        left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(left_eff)
    else:
        left_params = left_chart.parameters_from_unitary(left_eff)
        right_phase = np.zeros(parameterization.norb, dtype=np.float64)

    right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)

    out = np.zeros(parameterization.n_params, dtype=np.float64)
    idx = 0

    n = parameterization.n_left_orbital_rotation_params
    out[idx : idx + n] = left_params
    idx += n

    n = parameterization.n_pair_params
    if parameterization.uses_reduced_cubic_chart:
        out[idx : idx + n] = reduced_pair_values
    else:
        out[idx : idx + n] = np.asarray(
            [pair_eff[p, q] for p, q in parameterization.pair_indices],
            dtype=np.float64,
        )
    idx += n

    if parameterization.uses_reduced_cubic_chart:
        n = parameterization.n_tau_params
        out[idx : idx + n] = reduced_cubic_values
        idx += n
    else:
        n = parameterization.n_tau_params
        out[idx : idx + n] = _values_from_ordered_matrix(tau, parameterization.tau_indices)
        idx += n

        n = parameterization.n_omega_params
        full_omega = {triple: value for value, triple in zip(omega, d.omega_indices)}
        out[idx : idx + n] = np.asarray(
            [full_omega[t] for t in parameterization.omega_indices],
            dtype=np.float64,
        )
        idx += n

    if parameterization.uses_reduced_quartic_chart:
        n = parameterization.n_rho_params
        out[idx : idx + n] = reduced_quartic_values
        idx += n
    else:
        n = parameterization.n_eta_params
        full_eta = {pair: value for value, pair in zip(eta, d.eta_indices)}
        out[idx : idx + n] = np.asarray(
            [full_eta[t] for t in parameterization.eta_indices],
            dtype=np.float64,
        )
        idx += n

        n = parameterization.n_rho_params
        full_rho = {triple: value for value, triple in zip(rho, d.rho_indices)}
        out[idx : idx + n] = np.asarray(
            [full_rho[t] for t in parameterization.rho_indices],
            dtype=np.float64,
        )
        idx += n

        n = parameterization.n_sigma_params
        full_sigma = {quad: value for value, quad in zip(sigma, d.sigma_indices)}
        out[idx : idx + n] = np.asarray(
            [full_sigma[t] for t in parameterization.sigma_indices],
            dtype=np.float64,
        )
        idx += n

    n = parameterization.n_right_orbital_rotation_params
    left_param_unitary = parameterization._left_orbital_chart.unitary_from_parameters(
        left_params,
        parameterization.norb,
    )
    final_eff = _final_unitary_for_transfer(
        left_param_unitary,
        right_eff,
        parameterization.right_orbital_chart,
        parameterization.nocc,
    )
    out[idx : idx + n] = parameterization.right_orbital_chart.parameters_from_unitary(final_eff)
    return parameterization._public_parameters_from_native(out)


def _convert_ansatz_for_product_transfer(parameterization, ansatz):
    if isinstance(parameterization, IGCR2SpinRestrictedParameterization):
        if isinstance(ansatz, IGCR2Ansatz):
            return ansatz
    if isinstance(parameterization, IGCR3SpinRestrictedParameterization):
        if isinstance(ansatz, IGCR3Ansatz):
            return ansatz
        if isinstance(ansatz, IGCR2Ansatz):
            return IGCR3Ansatz.from_igcr2_ansatz(ansatz)
    if isinstance(parameterization, IGCR4SpinRestrictedParameterization):
        if isinstance(ansatz, IGCR4Ansatz):
            return ansatz
        if isinstance(ansatz, IGCR3Ansatz):
            return IGCR4Ansatz.from_igcr3_ansatz(ansatz)
        if isinstance(ansatz, IGCR2Ansatz):
            return IGCR4Ansatz.from_igcr2_ansatz(ansatz)
    raise TypeError(f"Unsupported ansatz transfer to {type(parameterization)!r} from {type(ansatz)!r}")


def _product_transfer_parameters_from_ansatz(parameterization, ansatz):
    if isinstance(parameterization, IGCR4SpinRestrictedParameterization):
        return _igcr4_product_transfer_parameters_from_ansatz(parameterization, ansatz)
    if isinstance(parameterization, IGCR3SpinRestrictedParameterization):
        return _igcr3_product_transfer_parameters_from_ansatz(parameterization, ansatz)
    if isinstance(parameterization, IGCR2SpinRestrictedParameterization):
        return _igcr2_product_transfer_parameters_from_ansatz(parameterization, ansatz)
    raise TypeError(f"Unsupported product transfer parameterization: {type(parameterization)!r}")


def _product_seed_from_ansatz(n_reference_params: int, parameterization, ansatz) -> np.ndarray:
    return _combined_seed(
        np.zeros(n_reference_params, dtype=np.float64),
        _product_transfer_parameters_from_ansatz(parameterization, ansatz),
    )


def _relabel_product_transfer_ansatz(ansatz, old_for_new, phases):
    if isinstance(ansatz, IGCR4Ansatz):
        return relabel_igcr4_ansatz_orbitals(ansatz, old_for_new, phases)
    if isinstance(ansatz, IGCR3Ansatz):
        return relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases)
    if isinstance(ansatz, IGCR2Ansatz):
        return relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
    raise TypeError(f"Unsupported ansatz type for transfer: {type(ansatz)!r}")


def _copy_direct_block(out, dst_start, params, src_start, n):
    out[dst_start : dst_start + n] = params[src_start : src_start + n]


def _direct_relabel_full_left_chart_params(params, relabel, norb):
    kappa = _zero_diag_antihermitian_from_parameters(params, norb)
    transformed = relabel.conj().T @ kappa @ relabel
    np.fill_diagonal(transformed, 0.0)
    return _parameters_from_zero_diag_antihermitian(transformed)


def _patch_direct_relabel_orbital_blocks(parameterization, previous_parameterization, ansatz_params, previous_ansatz_params, old_for_new, phases):
    source = previous_parameterization.ansatz_parameterization
    target = parameterization.ansatz_parameterization
    if (
        not isinstance(source._left_orbital_chart, IGCR2LeftUnitaryChart)
        or not isinstance(target._left_orbital_chart, IGCR2LeftUnitaryChart)
        or not isinstance(source.right_orbital_chart, IGCR2LeftUnitaryChart)
        or not isinstance(target.right_orbital_chart, IGCR2LeftUnitaryChart)
        or source.n_left_orbital_rotation_params != target.n_left_orbital_rotation_params
        or source.n_right_orbital_rotation_params != target.n_right_orbital_rotation_params
    ):
        return ansatz_params

    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    source_native = source._native_parameters_from_public(np.asarray(previous_ansatz_params, dtype=np.float64))
    target_native = target._native_parameters_from_public(np.asarray(ansatz_params, dtype=np.float64))

    n_left = source.n_left_orbital_rotation_params
    target_native[:n_left] = _direct_relabel_full_left_chart_params(
        source_native[:n_left],
        relabel,
        source.norb,
    )

    n_right = source.n_right_orbital_rotation_params
    target_native[
        target._right_orbital_rotation_start : target._right_orbital_rotation_start + n_right
    ] = _direct_relabel_full_left_chart_params(
        source_native[
            source._right_orbital_rotation_start : source._right_orbital_rotation_start + n_right
        ],
        relabel,
        source.norb,
    )
    return target._public_parameters_from_native(target_native)


def _direct_nested_product_ansatz_params(parameterization, previous_parameterization, previous_ansatz_params):
    if parameterization.norb != previous_parameterization.norb or parameterization.nocc != previous_parameterization.nocc:
        return None

    source = previous_parameterization.ansatz_parameterization
    target = parameterization.ansatz_parameterization
    params = np.asarray(previous_ansatz_params, dtype=np.float64)
    if params.shape != (source.n_params,):
        raise ValueError(f"Expected {(source.n_params,)}, got {params.shape}.")
    if (
        source.n_left_orbital_rotation_params != target.n_left_orbital_rotation_params
        or source.n_pair_params != target.n_pair_params
        or source.n_right_orbital_rotation_params != target.n_right_orbital_rotation_params
        or getattr(source, "pair_indices", None) != getattr(target, "pair_indices", None)
    ):
        return None

    out = np.zeros(target.n_params, dtype=np.float64)
    _copy_direct_block(out, 0, params, 0, source.n_left_orbital_rotation_params)
    _copy_direct_block(
        out,
        target.n_left_orbital_rotation_params,
        params,
        source.n_left_orbital_rotation_params,
        source.n_pair_params,
    )

    if isinstance(source, IGCR3SpinRestrictedParameterization) and isinstance(target, IGCR4SpinRestrictedParameterization):
        if (
            source.n_tau_params != target.n_tau_params
            or source.n_omega_params != target.n_omega_params
            or getattr(source, "tau_indices", None) != getattr(target, "tau_indices", None)
            or getattr(source, "omega_indices", None) != getattr(target, "omega_indices", None)
        ):
            return None
        _copy_direct_block(
            out,
            target.n_left_orbital_rotation_params + target.n_pair_params,
            params,
            source.n_left_orbital_rotation_params + source.n_pair_params,
            source.n_tau_params + source.n_omega_params,
        )
    elif not (
        isinstance(source, IGCR2SpinRestrictedParameterization)
        and isinstance(target, (IGCR3SpinRestrictedParameterization, IGCR4SpinRestrictedParameterization))
    ):
        return None

    _copy_direct_block(
        out,
        target._right_orbital_rotation_start,
        params,
        source._right_orbital_rotation_start,
        source.n_right_orbital_rotation_params,
    )
    return out


def _transfer_product_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal):
    if previous_parameterization is None:
        previous_parameterization = self
    if orbital_overlap is not None:
        if old_for_new is not None or phases is not None:
            raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
        old_for_new, phases = orbital_relabeling_from_overlap(
            orbital_overlap,
            nocc=self.nocc,
            block_diagonal=block_diagonal,
        )

    prev = np.asarray(previous_parameters, dtype=np.float64)
    reference_params = np.zeros(self.n_reference_params, dtype=np.float64)
    if hasattr(previous_parameterization, "split_parameters") and hasattr(previous_parameterization, "ansatz_parameterization"):
        if prev.shape != (previous_parameterization.n_params,):
            raise ValueError(f"Expected {(previous_parameterization.n_params,)}, got {prev.shape}.")
        if (
            isinstance(previous_parameterization, type(self))
            and previous_parameterization.norb == self.norb
            and previous_parameterization.nocc == self.nocc
            and _is_trivial_relabel(self.norb, old_for_new, phases)
        ):
            return np.array(prev, copy=True)

        prev_reference, prev_ansatz = previous_parameterization.split_parameters(prev)
        reference_params = _transfer_reference_params(
            self,
            prev_reference,
            previous_parameterization,
            old_for_new,
            phases,
        )
        if _is_trivial_relabel(self.norb, old_for_new, phases):
            ansatz_params = _direct_nested_product_ansatz_params(
                self,
                previous_parameterization,
                prev_ansatz,
            )
            if ansatz_params is not None:
                return np.concatenate([
                    reference_params,
                    ansatz_params,
                ])

        ansatz = previous_parameterization.ansatz_parameterization.ansatz_from_parameters(prev_ansatz)
        if old_for_new is not None:
            ansatz = _relabel_product_transfer_ansatz(ansatz, old_for_new, phases)
        ansatz = _convert_ansatz_for_product_transfer(self.ansatz_parameterization, ansatz)
        ansatz_params = _product_transfer_parameters_from_ansatz(self.ansatz_parameterization, ansatz)
        if old_for_new is not None:
            ansatz_params = _patch_direct_relabel_orbital_blocks(
                self,
                previous_parameterization,
                ansatz_params,
                prev_ansatz,
                old_for_new,
                phases,
            )
        return np.concatenate([
            reference_params,
            np.asarray(ansatz_params, dtype=np.float64),
        ])

    ansatz = previous_parameterization.ansatz_from_parameters(prev)
    if old_for_new is not None:
        ansatz = _relabel_product_transfer_ansatz(ansatz, old_for_new, phases)
    ansatz = _convert_ansatz_for_product_transfer(self.ansatz_parameterization, ansatz)
    ansatz_params = _product_transfer_parameters_from_ansatz(self.ansatz_parameterization, ansatz)
    return np.concatenate([
        reference_params,
        np.asarray(ansatz_params, dtype=np.float64),
    ])


@dataclass(frozen=True)
class GCR2ProductPairUCCDParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

    @property
    def reference_parameterization(self) -> ProductPairUCCDStateParameterization:
        return ProductPairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

    @property
    def ansatz_parameterization(self) -> IGCR2SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR2SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            interaction_pairs=self.interaction_pairs,
            left_orbital_chart=self.left_orbital_chart,
            right_orbital_chart_override=self.right_orbital_chart_override,
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def _composite(self):
        return _make_composite(self.reference_parameterization, self.ansatz_parameterization, (self.nocc, self.nocc))

    @property
    def n_reference_params(self) -> int:
        return self.reference_parameterization.n_params

    @property
    def n_pair_reference_params(self) -> int:
        return self.n_reference_params

    @property
    def n_ansatz_params(self) -> int:
        return self.ansatz_parameterization.n_params

    @property
    def n_params(self) -> int:
        return self._composite.n_params

    @property
    def pair_reference_indices(self) -> tuple[tuple[int, int], ...]:
        return self.reference_parameterization.pair_indices

    @property
    def pair_indices(self):
        return self.ansatz_parameterization.pair_indices

    def split_parameters(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._composite.split_parameters(params)

    def reference_state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self.reference_parameterization.state_from_parameters(params)

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.ansatz_parameterization.ansatz_from_parameters(params)

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self._composite.state_from_parameters(params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return _make_composite_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        return _product_seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return self.parameters_from_ansatz(_igcr2_product_ansatz_from_ucj(ansatz, self.nocc))

    def reference_parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return self.reference_parameterization.parameters_from_t2(t2, scale=scale)

    def parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=scale),
            np.zeros(self.n_ansatz_params, dtype=np.float64),
        )

    def parameters_from_t2_and_ucj_ansatz(self, t2: np.ndarray, ansatz: UCJAnsatz, *, pair_scale: float = 0.5) -> np.ndarray:
        ansatz_params = self.parameters_from_ucj_ansatz(ansatz)[self.n_reference_params :]
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=pair_scale),
            ansatz_params,
        )

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_product_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)


@dataclass(frozen=True)
class GCR3ProductPairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR3SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None
    tau_seed_scale: float = 0.0
    omega_seed_scale: float = 0.0

    @property
    def reference_parameterization(self) -> ProductPairUCCDStateParameterization:
        return ProductPairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

    @property
    def ansatz_parameterization(self) -> IGCR3SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR3SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            left_orbital_chart=self.left_orbital_chart,
            right_orbital_chart_override=self.right_orbital_chart_override,
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def _composite(self):
        return _make_composite(self.reference_parameterization, self.ansatz_parameterization, (self.nocc, self.nocc))

    @property
    def n_reference_params(self) -> int:
        return self.reference_parameterization.n_params

    @property
    def n_pair_reference_params(self) -> int:
        return self.n_reference_params

    @property
    def n_ansatz_params(self) -> int:
        return self.ansatz_parameterization.n_params

    @property
    def n_params(self) -> int:
        return self._composite.n_params

    @property
    def pair_reference_indices(self) -> tuple[tuple[int, int], ...]:
        return self.reference_parameterization.pair_indices

    @property
    def pair_indices(self):
        return getattr(self.ansatz_parameterization, "pair_indices", ())

    def split_parameters(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._composite.split_parameters(params)

    def reference_state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self.reference_parameterization.state_from_parameters(params)

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.ansatz_parameterization.ansatz_from_parameters(params)

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self._composite.state_from_parameters(params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return _make_composite_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        return _product_seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        seeded = IGCR3Ansatz.from_igcr2_ansatz(
            _igcr2_product_ansatz_from_ucj(ansatz, self.nocc),
            tau_scale=self.tau_seed_scale,
            omega_scale=self.omega_seed_scale,
        )
        return self.parameters_from_ansatz(seeded)

    def reference_parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return self.reference_parameterization.parameters_from_t2(t2, scale=scale)

    def parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=scale),
            np.zeros(self.n_ansatz_params, dtype=np.float64),
        )

    def parameters_from_t2_and_ucj_ansatz(self, t2: np.ndarray, ansatz: UCJAnsatz, *, pair_scale: float = 0.5) -> np.ndarray:
        ansatz_params = self.parameters_from_ucj_ansatz(ansatz)[self.n_reference_params :]
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=pair_scale),
            ansatz_params,
        )

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_product_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)


@dataclass(frozen=True)
class GCR4ProductPairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR4SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None
    tau_seed_scale: float = 0.0
    omega_seed_scale: float = 0.0
    eta_seed_scale: float = 0.0
    rho_seed_scale: float = 0.0
    sigma_seed_scale: float = 0.0

    @property
    def reference_parameterization(self) -> ProductPairUCCDStateParameterization:
        return ProductPairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

    @property
    def ansatz_parameterization(self) -> IGCR4SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR4SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            left_orbital_chart=self.left_orbital_chart,
            right_orbital_chart_override=self.right_orbital_chart_override,
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def _composite(self):
        return _make_composite(self.reference_parameterization, self.ansatz_parameterization, (self.nocc, self.nocc))

    @property
    def n_reference_params(self) -> int:
        return self.reference_parameterization.n_params

    @property
    def n_pair_reference_params(self) -> int:
        return self.n_reference_params

    @property
    def n_ansatz_params(self) -> int:
        return self.ansatz_parameterization.n_params

    @property
    def n_params(self) -> int:
        return self._composite.n_params

    @property
    def pair_reference_indices(self) -> tuple[tuple[int, int], ...]:
        return self.reference_parameterization.pair_indices

    @property
    def pair_indices(self):
        return getattr(self.ansatz_parameterization, "pair_indices", ())

    def split_parameters(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._composite.split_parameters(params)

    def reference_state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self.reference_parameterization.state_from_parameters(params)

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.ansatz_parameterization.ansatz_from_parameters(params)

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self._composite.state_from_parameters(params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return _make_composite_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        return _product_seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        seeded = IGCR4Ansatz.from_igcr2_ansatz(
            _igcr2_product_ansatz_from_ucj(ansatz, self.nocc),
            tau_scale=self.tau_seed_scale,
            omega_scale=self.omega_seed_scale,
            eta_scale=self.eta_seed_scale,
            rho_scale=self.rho_seed_scale,
            sigma_scale=self.sigma_seed_scale,
        )
        return self.parameters_from_ansatz(seeded)

    def reference_parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return self.reference_parameterization.parameters_from_t2(t2, scale=scale)

    def parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=scale),
            np.zeros(self.n_ansatz_params, dtype=np.float64),
        )

    def parameters_from_t2_and_ucj_ansatz(self, t2: np.ndarray, ansatz: UCJAnsatz, *, pair_scale: float = 0.5) -> np.ndarray:
        ansatz_params = self.parameters_from_ucj_ansatz(ansatz)[self.n_reference_params :]
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=pair_scale),
            ansatz_params,
        )

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_product_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)
