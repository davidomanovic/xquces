from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.igcr2 import IGCR2LeftUnitaryChart, IGCR2SpinRestrictedParameterization
from xquces.gcr.pair_uccd_reference import (
    _combined_seed,
    _make_composite,
    _make_composite_jacobian,
    _seed_from_ansatz,
    _seed_from_ucj,
    _transfer_params,
)
from xquces.pair_uccd import ProductPairUCCDStateParameterization
from xquces.ucj.model import UCJAnsatz


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
        return _seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return _seed_from_ucj(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def reference_parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return self.reference_parameterization.parameters_from_t2(t2, scale=scale)

    def parameters_from_t2(self, t2: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=scale),
            np.zeros(self.n_ansatz_params, dtype=np.float64),
        )

    def parameters_from_t2_and_ucj_ansatz(self, t2: np.ndarray, ansatz: UCJAnsatz, *, pair_scale: float = 0.5) -> np.ndarray:
        return _combined_seed(
            self.reference_parameters_from_t2(t2, scale=pair_scale),
            self.ansatz_parameterization.parameters_from_ucj_ansatz(ansatz),
        )

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)
