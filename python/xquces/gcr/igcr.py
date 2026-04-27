from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xquces.gcr.bridge_gcr2 import GCR2FullUnitaryChart
from xquces.gcr.igcr2 import IGCR2LeftUnitaryChart, IGCR2SpinRestrictedParameterization
from xquces.gcr.igcr3 import IGCR3SpinRestrictedParameterization
from xquces.gcr.igcr4 import IGCR4SpinRestrictedParameterization


_AUTO_RIGHT_CHART = "auto"


@dataclass(frozen=True)
class IGCRSpinRestrictedParameterization:
    """Order-selecting facade for spin-restricted iGCR ansatz parameterizations."""

    norb: int
    nocc: int
    order: int = 2
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    reduce_quartic_gauge: bool = True
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None | str = _AUTO_RIGHT_CHART
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

    def __post_init__(self):
        if self.order not in {2, 3, 4}:
            raise ValueError("order must be 2, 3, or 4")

    @property
    def implementation(self):
        return self._implementation(full_right=False)

    def _implementation(self, *, full_right: bool):
        right_chart = self.right_orbital_chart_override
        if isinstance(right_chart, str) and right_chart == _AUTO_RIGHT_CHART:
            right_chart = IGCR2LeftUnitaryChart() if full_right else None

        common = {
            "norb": self.norb,
            "nocc": self.nocc,
            "interaction_pairs": self.interaction_pairs,
            "left_orbital_chart": self.left_orbital_chart,
            "right_orbital_chart_override": right_chart,
            "real_right_orbital_chart": self.real_right_orbital_chart,
            "left_right_ov_relative_scale": self.left_right_ov_relative_scale,
        }
        if self.order == 2:
            return IGCR2SpinRestrictedParameterization(**common)
        if self.order == 3:
            return IGCR3SpinRestrictedParameterization(
                **common,
                tau_indices_=self.tau_indices_,
                omega_indices_=self.omega_indices_,
                reduce_cubic_gauge=self.reduce_cubic_gauge,
            )
        return IGCR4SpinRestrictedParameterization(
            **common,
            tau_indices_=self.tau_indices_,
            omega_indices_=self.omega_indices_,
            eta_indices_=self.eta_indices_,
            rho_indices_=self.rho_indices_,
            sigma_indices_=self.sigma_indices_,
            reduce_cubic_gauge=self.reduce_cubic_gauge,
            reduce_quartic_gauge=self.reduce_quartic_gauge,
        )

    def _uses_full_right_for_reference(
        self,
        reference: object,
        nelec: tuple[int, int],
    ) -> bool:
        from xquces.state_parameterization import reference_is_hartree_fock_state

        if not (
            isinstance(self.right_orbital_chart_override, str)
            and self.right_orbital_chart_override == _AUTO_RIGHT_CHART
        ):
            return False
        return not reference_is_hartree_fock_state(reference, self.norb, nelec)

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        from xquces.state_parameterization import apply_ansatz_parameterization

        parameterization = self._implementation(
            full_right=self._uses_full_right_for_reference(reference, nelec)
        )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int] | None = None
    ):
        return self.apply(reference_vec, nelec).params_to_vec()

    def __getattr__(self, name: str):
        return getattr(self.implementation, name)
