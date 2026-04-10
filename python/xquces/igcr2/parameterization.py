"""Irreducible single-layer GCR-2 parameterization (spin-balanced).

Fixes two bugs from the original implementation:

  1. Diagonal correlator is now SPIN-BALANCED (separate same-spin and mixed-spin
     pair parameters).  The spin-restricted form (single γ_{pq} for N_p N_q)
     cannot represent a spin-balanced UCJ seed — that's where ~1.5 Ha was lost.

  2. Orbital rotations use (Re, Im) of the anti-Hermitian generator, not polar
     (θ, φ).  Polar coords have a singularity at θ=0 → exact zero eigenvalues
     in the LM overlap matrix → cond ~1e16.

Runtime forward map (every optimization iteration):
    1. expm(K_L)       — left unitary from norb² anti-Hermitian params
    2. V @ x_reduced   — one mat-vec multiply for Jastrow (V precomputed at init)
    3. OV unitary       — right rotation from 2·nocc·nvirt params
    4. apply U_R → e^{iJ} → U_L

No coupled gauge maps at optimization time.  The coupling (phase absorption
between U_L and J) runs once during seed conversion only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.gcr import GCRSpinBalancedParameterization
from xquces.gcr.model import GCRAnsatz
from xquces.ucj.model import UCJAnsatz


@dataclass(frozen=True)
class IGCR2Parameterization:
    """Irreducible single-layer spin-balanced GCR-2."""

    norb: int
    nocc: int

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    @property
    def _gcr(self) -> GCRSpinBalancedParameterization:
        return GCRSpinBalancedParameterization(norb=self.norb, nocc=self.nocc)

    @property
    def n_left_params(self) -> int:
        return self._gcr.n_left_orbital_rotation_params

    @property
    def n_jastrow_params(self) -> int:
        return self._gcr.n_jastrow_params

    @property
    def n_right_params(self) -> int:
        return self._gcr.n_right_orbital_rotation_params

    @property
    def n_params(self) -> int:
        return self._gcr.n_params

    def ansatz_from_parameters(self, params: np.ndarray) -> GCRAnsatz:
        return self._gcr.ansatz_from_parameters(params)

    def apply(self, params: np.ndarray, vec: np.ndarray, nelec: tuple[int, int]) -> np.ndarray:
        return self._gcr.ansatz_from_parameters(params).apply(vec, nelec=nelec, copy=True)

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        ref = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.apply(params, ref, nelec)
        return func

    def parameters_from_ansatz(self, ansatz: GCRAnsatz) -> np.ndarray:
        return self._gcr.parameters_from_ansatz(ansatz)

    def parameters_from_ucj(self, ucj_ansatz: UCJAnsatz) -> np.ndarray:
        return self._gcr.parameters_from_ucj_ansatz(ucj_ansatz)


def igcr2_params_from_ucj(
    ucj_ansatz: UCJAnsatz,
    nocc: int,
    parameterization: IGCR2Parameterization | None = None,
) -> np.ndarray:
    if parameterization is None:
        parameterization = IGCR2Parameterization(norb=ucj_ansatz.norb, nocc=nocc)
    return parameterization.parameters_from_ucj(ucj_ansatz)