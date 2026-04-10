from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.gcr.model import GCRAnsatz
from xquces.gcr.parameterization import GCRSpinBalancedParameterization
from xquces.ucj.init import UCJBalancedDFSeed


@dataclass(frozen=True)
class GaugeFixedGCRBalancedDFSeed:
    t2: np.ndarray
    t1: np.ndarray | None = None
    n_reps: int | None = None
    tol: float = 1e-8
    optimize: bool = False
    method: str = "L-BFGS-B"
    callback: object = None
    options: dict | None = None
    regularization: float = 0.0
    multi_stage_start: int | None = None
    multi_stage_step: int | None = None

    def build_parameters(self) -> tuple[GCRAnsatz, GCRSpinBalancedParameterization, np.ndarray]:
        ucj_ansatz = UCJBalancedDFSeed(
            t2=self.t2,
            t1=self.t1,
            n_reps=self.n_reps,
            tol=self.tol,
            optimize=self.optimize,
            method=self.method,
            callback=self.callback,
            options=self.options,
            regularization=self.regularization,
            multi_stage_start=self.multi_stage_start,
            multi_stage_step=self.multi_stage_step,
        ).build_ansatz()

        param = GCRSpinBalancedParameterization(
            norb=ucj_ansatz.norb,
            nocc=np.asarray(self.t2).shape[0],
        )

        x0 = param.parameters_from_ucj_ansatz(ucj_ansatz)
        ansatz = param.ansatz_from_parameters(x0)
        return ansatz, param, x0

    def build_ansatz(self) -> GCRAnsatz:
        ansatz, _, _ = self.build_parameters()
        return ansatz