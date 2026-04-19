from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.gcr.commutator_gcr2 import _diag2_features, _validate_pairs
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2SpinRestrictedParameterization,
    _diag_unitary,
    _restricted_left_phase_vector,
    reduce_spin_restricted,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj._unitary import (
    antihermitian_from_parameters,
    parameters_from_antihermitian,
)
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz


def _phase_diag2(
    vec: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> None:
    pair_params = np.asarray(pair_params, dtype=np.float64)
    if pair_params.shape != (len(pairs),):
        raise ValueError("pair_params has the wrong shape")
    phases = _diag2_features(norb, nelec, pairs) @ pair_params
    vec *= np.exp(1j * phases)


@dataclass(frozen=True)
class GCR2FullUnitaryChart:
    """Raw anti-Hermitian chart for a full spin-restricted orbital rotation."""

    def n_params(self, norb: int) -> int:
        return norb**2

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return np.asarray(
            scipy.linalg.expm(antihermitian_from_parameters(params, norb)),
            dtype=np.complex128,
        )

    def parameters_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
        unitary = np.asarray(unitary, dtype=np.complex128)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError("unitary must be square")
        if not np.allclose(unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-10):
            raise ValueError("unitary must be unitary")
        generator = scipy.linalg.logm(unitary)
        generator = 0.5 * (generator - generator.conj().T)
        return parameters_from_antihermitian(generator)


@dataclass(frozen=True)
class GCR2SplitBridgeAnsatz:
    """Circuit-friendly split-bridge GCR-2 ansatz.

    The state preparation is

        U_L exp(iD/2) U_mid exp(iD/2) U_R |phi>.

    ``U_mid`` is a spin-restricted orbital rotation.  With a two-layer UCJ seed,
    the initializer projects the layer product as ``F U_2 D_2 (U_2^dag U_1)
    D_1 U_1^dag``.
    """

    pair_params: np.ndarray
    middle: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(
            vec,
            self.right,
            self.norb,
            nelec,
            copy=copy,
        )
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        out = apply_orbital_rotation(
            out,
            self.middle,
            self.norb,
            nelec,
            copy=False,
        )
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2UntiedSplitBridgeAnsatz:
    """Untied two-diagonal split-bridge GCR-2 ansatz.

    The state preparation is

        U_L exp(iD_1) U_mid exp(iD_2) U_R |phi>.

    Setting ``D_1 = D_2 = D/2`` and ``U_mid = I`` recovers ordinary iGCR2.
    """

    left_pair_params: np.ndarray
    right_pair_params: np.ndarray
    middle: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(
            vec,
            self.right,
            self.norb,
            nelec,
            copy=copy,
        )
        _phase_diag2(out, self.right_pair_params, self.norb, nelec, self.pairs)
        out = apply_orbital_rotation(
            out,
            self.middle,
            self.norb,
            nelec,
            copy=False,
        )
        _phase_diag2(out, self.left_pair_params, self.norb, nelec, self.pairs)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2SplitBridgeParameterization:
    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    middle_orbital_chart: object = field(default_factory=GCR2FullUnitaryChart)
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")
            base_pairs = tuple(self.base_parameterization.pair_indices)
            if self.pairs is None:
                object.__setattr__(self, "_pairs", base_pairs)
            else:
                pairs = _validate_pairs(self.pairs, self.norb)
                if pairs != base_pairs:
                    raise ValueError("pairs do not match base_parameterization")
                object.__setattr__(self, "_pairs", pairs)
        else:
            object.__setattr__(self, "_pairs", _validate_pairs(self.pairs, self.norb))

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def _base(self) -> IGCR2SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR2SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            interaction_pairs=list(self._pairs),
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._base.n_left_orbital_rotation_params

    @property
    def n_pair_params(self) -> int:
        return len(self._pairs)

    @property
    def n_middle_orbital_rotation_params(self) -> int:
        return self.middle_orbital_chart.n_params(self.norb)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._base.n_right_orbital_rotation_params

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_orbital_rotation_params
        )

    def _split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_pair_params
        pair = params[idx : idx + n]
        idx += n
        n = self.n_middle_orbital_rotation_params
        middle = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, pair, middle, right

    def _base_params_from_split(
        self,
        left: np.ndarray,
        pair: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2SplitBridgeAnsatz:
        left, pair, middle, right = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(left, pair, right)
        )
        if not isinstance(base_ansatz, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz")
        pair_matrix = np.asarray(base_ansatz.diagonal.pair, dtype=np.float64)
        pair_values = np.asarray(
            [pair_matrix[p, q] for p, q in self._pairs],
            dtype=np.float64,
        )
        return GCR2SplitBridgeAnsatz(
            pair_params=pair_values,
            middle=np.asarray(
                self.middle_orbital_chart.unitary_from_parameters(middle, self.norb),
                dtype=np.complex128,
            ),
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
        )

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        base_params = self._base.parameters_from_ansatz(ansatz)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64),
                right,
            ]
        )

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.n_layers == 2 and ansatz.is_spin_restricted:
            return self.parameters_from_two_layer_ucj_ansatz(ansatz)
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        return self.parameters_from_igcr2(base_params, self._base)

    def _layer_pair_and_phase(
        self,
        diagonal: SpinRestrictedSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        pair = reduce_spin_restricted(diagonal).pair
        phase = _restricted_left_phase_vector(diagonal.double_params, self.nocc)
        return pair, phase

    def _middle_params_and_right_phase(
        self,
        unitary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._chart_params_and_right_phase(self.middle_orbital_chart, unitary)

    def _chart_params_and_right_phase(
        self,
        chart: object,
        unitary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(chart, "parameters_and_right_phase_from_unitary"):
            params, right_phase = (
                chart.parameters_and_right_phase_from_unitary(unitary)
            )
        else:
            params = chart.parameters_from_unitary(unitary)
            right_phase = np.zeros(self.norb, dtype=np.float64)
        return np.asarray(params, dtype=np.float64), _diag_unitary(right_phase)

    def _orbital_params_from_matrices(
        self,
        left: np.ndarray,
        middle: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_params, left_right_phase = self._chart_params_and_right_phase(
            self._base._left_orbital_chart,
            left,
        )
        left_unitary = self._base._left_orbital_chart.unitary_from_parameters(
            left_params,
            self.norb,
        )
        middle = left_right_phase @ middle
        middle_params, middle_right_phase = self._middle_params_and_right_phase(middle)
        right = middle_right_phase @ right
        final = left_unitary @ right
        right_params = self._base.right_orbital_chart.parameters_from_unitary(final)
        return left_params, middle_params, np.asarray(right_params, dtype=np.float64)

    def _pair_values_from_matrix(self, pair: np.ndarray) -> np.ndarray:
        pair = np.asarray(pair, dtype=np.float64)
        return np.asarray([pair[p, q] for p, q in self._pairs], dtype=np.float64)

    def parameters_from_two_layer_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.n_layers != 2:
            raise ValueError("expected a two-layer UCJ ansatz")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted UCJ ansatz")
        first, second = ansatz.layers
        if not isinstance(first.diagonal, SpinRestrictedSpec) or not isinstance(
            second.diagonal, SpinRestrictedSpec
        ):
            raise TypeError("expected spin-restricted UCJ layers")

        pair1, phase1 = self._layer_pair_and_phase(first.diagonal)
        pair2, phase2 = self._layer_pair_and_phase(second.diagonal)
        final = (
            np.eye(self.norb, dtype=np.complex128)
            if ansatz.final_orbital_rotation is None
            else np.asarray(ansatz.final_orbital_rotation, dtype=np.complex128)
        )
        u1 = np.asarray(first.orbital_rotation, dtype=np.complex128)
        u2 = np.asarray(second.orbital_rotation, dtype=np.complex128)

        left = final @ u2 @ _diag_unitary(phase2)
        middle_unitary = u2.conj().T @ u1 @ _diag_unitary(phase1)
        right = u1.conj().T
        pair = pair1 + pair2
        left_params, middle, right_params = self._orbital_params_from_matrices(
            left,
            middle_unitary,
            right,
        )
        pair_params = self._pair_values_from_matrix(pair)
        return np.concatenate([left_params, pair_params, middle, right_params])

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=nelec,
                copy=True,
            )

        return func


@dataclass(frozen=True)
class GCR2UntiedSplitBridgeParameterization(GCR2SplitBridgeParameterization):
    """Parameterization for ``U_L exp(iD1) U_mid exp(iD2) U_R``."""

    @property
    def n_right_pair_params(self) -> int:
        return len(self._pairs)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_pair_params
            + self.n_right_orbital_rotation_params
        )

    def _split(  # type: ignore[override]
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_pair_params
        left_pair = params[idx : idx + n]
        idx += n
        n = self.n_middle_orbital_rotation_params
        middle = params[idx : idx + n]
        idx += n
        n = self.n_right_pair_params
        right_pair = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, left_pair, middle, right_pair, right

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2UntiedSplitBridgeAnsatz:
        left, left_pair, middle, right_pair, right = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(left, np.zeros(self.n_pair_params), right)
        )
        if not isinstance(base_ansatz, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz")
        return GCR2UntiedSplitBridgeAnsatz(
            left_pair_params=np.asarray(left_pair, dtype=np.float64),
            right_pair_params=np.asarray(right_pair, dtype=np.float64),
            middle=np.asarray(
                self.middle_orbital_chart.unitary_from_parameters(middle, self.norb),
                dtype=np.complex128,
            ),
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
        )

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        base_params = self._base.parameters_from_ansatz(ansatz)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = 0.5 * base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64),
                pair,
                right,
            ]
        )

    def parameters_from_two_layer_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        if ansatz.n_layers != 2:
            raise ValueError("expected a two-layer UCJ ansatz")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted UCJ ansatz")
        first, second = ansatz.layers
        if not isinstance(first.diagonal, SpinRestrictedSpec) or not isinstance(
            second.diagonal, SpinRestrictedSpec
        ):
            raise TypeError("expected spin-restricted UCJ layers")

        pair1, phase1 = self._layer_pair_and_phase(first.diagonal)
        pair2, phase2 = self._layer_pair_and_phase(second.diagonal)
        final = (
            np.eye(self.norb, dtype=np.complex128)
            if ansatz.final_orbital_rotation is None
            else np.asarray(ansatz.final_orbital_rotation, dtype=np.complex128)
        )
        u1 = np.asarray(first.orbital_rotation, dtype=np.complex128)
        u2 = np.asarray(second.orbital_rotation, dtype=np.complex128)

        left = final @ u2 @ _diag_unitary(phase2)
        middle_unitary = u2.conj().T @ u1 @ _diag_unitary(phase1)
        right = u1.conj().T
        left_params, middle, right_params = self._orbital_params_from_matrices(
            left,
            middle_unitary,
            right,
        )
        left_pair = self._pair_values_from_matrix(pair2)
        right_pair = self._pair_values_from_matrix(pair1)
        return np.concatenate(
            [left_params, left_pair, middle, right_pair, right_params]
        )
