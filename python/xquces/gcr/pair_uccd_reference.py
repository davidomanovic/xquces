from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.igcr2 import (
    IGCR2LeftUnitaryChart,
    IGCR2SpinRestrictedParameterization,
    orbital_relabeling_from_overlap,
)
from xquces.gcr.igcr3 import IGCR3SpinRestrictedParameterization
from xquces.gcr.igcr4 import IGCR4SpinRestrictedParameterization
from xquces.pair_uccd import PairUCCDStateParameterization
from xquces.ucj.model import UCJAnsatz


def _make_composite(reference_parameterization, ansatz_parameterization, nelec):
    from xquces.state_parameterization import CompositeReferenceAnsatzParameterization

    return CompositeReferenceAnsatzParameterization(
        reference_parameterization,
        ansatz_parameterization,
        nelec,
    )


def _make_composite_jacobian(composite):
    from xquces.state_parameterization import make_composite_reference_ansatz_jacobian

    return make_composite_reference_ansatz_jacobian(composite)


def _seed_from_ucj(reference_n_params: int, ansatz_parameterization, ansatz: UCJAnsatz) -> np.ndarray:
    ansatz_params = np.asarray(
        ansatz_parameterization.parameters_from_ucj_ansatz(ansatz),
        dtype=np.float64,
    )
    return np.concatenate([np.zeros(reference_n_params, dtype=np.float64), ansatz_params])


def _seed_from_ansatz(reference_n_params: int, ansatz_parameterization, ansatz) -> np.ndarray:
    ansatz_params = np.asarray(
        ansatz_parameterization.parameters_from_ansatz(ansatz),
        dtype=np.float64,
    )
    return np.concatenate([np.zeros(reference_n_params, dtype=np.float64), ansatz_params])


def _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal):
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
    if isinstance(previous_parameterization, type(self)):
        prev = np.asarray(previous_parameters, dtype=np.float64)
        if prev.shape != (previous_parameterization.n_params,):
            raise ValueError(f"Expected {(previous_parameterization.n_params,)}, got {prev.shape}.")
        if (
            previous_parameterization.norb == self.norb
            and previous_parameterization.nocc == self.nocc
            and old_for_new is None
            and phases is None
        ):
            return np.array(prev, copy=True)
        _, prev_ansatz = previous_parameterization.split_parameters(prev)
        ansatz_params = self.ansatz_parameterization.transfer_parameters_from(
            prev_ansatz,
            previous_parameterization=previous_parameterization.ansatz_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        return np.concatenate([
            np.zeros(self.n_reference_params, dtype=np.float64),
            np.asarray(ansatz_params, dtype=np.float64),
        ])
    ansatz_params = self.ansatz_parameterization.transfer_parameters_from(
        previous_parameters,
        previous_parameterization=previous_parameterization,
        old_for_new=old_for_new,
        phases=phases,
        orbital_overlap=None,
        block_diagonal=block_diagonal,
    )
    return np.concatenate([
        np.zeros(self.n_reference_params, dtype=np.float64),
        np.asarray(ansatz_params, dtype=np.float64),
    ])


@dataclass(frozen=True)
class GCR2PairUCCDParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

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

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)


@dataclass(frozen=True)
class GCR3PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR3SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

    @property
    def reference_parameterization(self) -> PairUCCDStateParameterization:
        return PairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

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
        return _seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return _seed_from_ucj(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)


@dataclass(frozen=True)
class GCR4PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR4SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=IGCR2LeftUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

    @property
    def reference_parameterization(self) -> PairUCCDStateParameterization:
        return PairUCCDStateParameterization(self.norb, (self.nocc, self.nocc))

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
        return _seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        return _seed_from_ucj(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: object | None = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)
