from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.igcr2 import IGCR2LeftUnitaryChart, IGCR2SpinRestrictedParameterization
from xquces.gcr.igcr3 import IGCR3SpinRestrictedParameterization
from xquces.gcr.igcr4 import IGCR4SpinRestrictedParameterization
from xquces.pair_uccd import PairUCCDStateParameterization
from xquces.state_parameterization import (
    CompositeReferenceAnsatzParameterization,
    make_composite_reference_ansatz_jacobian,
)


@dataclass(frozen=True)
class GCR2PairUCCDParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 1.0

    @property
    def reference_parameterization(self) -> PairUCCDStateParameterization:
        return PairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

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
    def _composite(self) -> CompositeReferenceAnsatzParameterization:
        return CompositeReferenceAnsatzParameterization(
            self.reference_parameterization,
            self.ansatz_parameterization,
            (self.nocc, self.nocc),
        )

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
        return make_composite_reference_ansatz_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()


@dataclass(frozen=True)
class GCR3PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR3SpinRestrictedParameterization | None = None

    @property
    def reference_parameterization(self) -> PairUCCDStateParameterization:
        return PairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

    @property
    def ansatz_parameterization(self) -> IGCR3SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR3SpinRestrictedParameterization(self.norb, self.nocc)

    @property
    def _composite(self) -> CompositeReferenceAnsatzParameterization:
        return CompositeReferenceAnsatzParameterization(
            self.reference_parameterization,
            self.ansatz_parameterization,
            (self.nocc, self.nocc),
        )

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
        return make_composite_reference_ansatz_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()


@dataclass(frozen=True)
class GCR4PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR4SpinRestrictedParameterization | None = None

    @property
    def reference_parameterization(self) -> PairUCCDStateParameterization:
        return PairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

    @property
    def ansatz_parameterization(self) -> IGCR4SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR4SpinRestrictedParameterization(self.norb, self.nocc)

    @property
    def _composite(self) -> CompositeReferenceAnsatzParameterization:
        return CompositeReferenceAnsatzParameterization(
            self.reference_parameterization,
            self.ansatz_parameterization,
            (self.nocc, self.nocc),
        )

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
        return make_composite_reference_ansatz_jacobian(self._composite)(params)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()
