from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import ffsim

from xquces.orbitals import canonicalize_unitary
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz, UCJLayer
from xquces.ucj.parameterization import (
    GaugeFixedUCJSpinBalancedParameterization,
    UCJSpinBalancedParameterization,
    UCJSpinRestrictedParameterization,
    ov_params_from_unitary,
)


def _orbital_rotation_from_t1_amplitudes(t1: np.ndarray) -> np.ndarray:
    t1 = np.asarray(t1, dtype=np.complex128)
    if t1.ndim != 2:
        raise ValueError("t1 must have shape (nocc, nvirt)")
    nocc, nvirt = t1.shape
    norb = nocc + nvirt
    kappa = np.zeros((norb, norb), dtype=np.complex128)
    kappa[nocc:, :nocc] = t1.T
    kappa[:nocc, nocc:] = -t1.conj()
    return canonicalize_unitary(scipy.linalg.expm(kappa))


def heuristic_restricted_pair_params_from_t2(t2: np.ndarray) -> np.ndarray:
    t2 = np.asarray(t2, dtype=np.float64)
    if t2.ndim != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc1, nocc2, nvirt1, nvirt2 = t2.shape
    if nocc1 != nocc2 or nvirt1 != nvirt2:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc = nocc1
    nvirt = nvirt1
    norb = nocc + nvirt
    pair = np.zeros((norb, norb), dtype=np.float64)
    for i in range(nocc):
        for j in range(i + 1, nocc):
            v = np.linalg.norm(t2[i, j])
            pair[i, j] = v
            pair[j, i] = v
    for a in range(nvirt):
        for b in range(a + 1, nvirt):
            v = np.linalg.norm(t2[:, :, a, b])
            pair[nocc + a, nocc + b] = v
            pair[nocc + b, nocc + a] = v
    for i in range(nocc):
        for a in range(nvirt):
            v = np.linalg.norm(t2[i, :, a, :])
            pair[i, nocc + a] = v
            pair[nocc + a, i] = v
    np.fill_diagonal(pair, 0.0)
    mx = np.max(np.abs(pair))
    if mx > 0:
        pair /= mx
    return pair


def project_spin_balanced_to_spin_restricted(ansatz: UCJAnsatz) -> UCJAnsatz:
    if not ansatz.is_spin_balanced:
        raise TypeError("expected a spin-balanced ansatz")
    layers: list[UCJLayer] = []
    for layer in ansatz.layers:
        d = layer.diagonal
        avg = 0.5 * (d.same_spin_params + d.mixed_spin_params)
        double_params = np.diag(avg).copy()
        pair_params = avg.copy()
        np.fill_diagonal(pair_params, 0.0)
        layers.append(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=double_params,
                    pair_params=pair_params,
                ),
                orbital_rotation=layer.orbital_rotation,
            )
        )
    return UCJAnsatz(
        layers=tuple(layers),
        final_orbital_rotation=ansatz.final_orbital_rotation,
    )


@dataclass(frozen=True)
class UCJBalancedDFSeed:
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

    def build_ansatz(self) -> UCJAnsatz:
        t2 = np.asarray(self.t2, dtype=np.float64)
        if t2.ndim != 4:
            raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
        nocc, _, nvirt, _ = t2.shape
        norb = nocc + nvirt

        diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
            t2,
            tol=self.tol,
            max_terms=self.n_reps,
            optimize=self.optimize,
            method=self.method,
            callback=self.callback,
            options=self.options,
            diag_coulomb_indices=None,
            regularization=self.regularization,
            multi_stage_start=self.multi_stage_start,
            multi_stage_step=self.multi_stage_step,
            return_optimize_result=False,
        )

        n_terms = len(diag_coulomb_mats)
        if self.n_reps is not None and n_terms < self.n_reps:
            pad = self.n_reps - n_terms
            diag_coulomb_mats = np.concatenate(
                [diag_coulomb_mats, np.zeros((pad, norb, norb), dtype=np.float64)],
                axis=0,
            )
            orbital_rotations = np.concatenate(
                [orbital_rotations, np.tile(np.eye(norb)[None, :, :], (pad, 1, 1))],
                axis=0,
            )

        layers: list[UCJLayer] = []
        for z, u in zip(diag_coulomb_mats, orbital_rotations):
            z = np.asarray(z, dtype=np.float64)
            z = 0.5 * (z + z.T)
            layers.append(
                UCJLayer(
                    diagonal=SpinBalancedSpec(
                        same_spin_params=z.copy(),
                        mixed_spin_params=z.copy(),
                    ),
                    orbital_rotation=canonicalize_unitary(np.asarray(u, dtype=np.complex128)),
                )
            )

        final_orbital_rotation = None
        if self.t1 is not None:
            final_orbital_rotation = _orbital_rotation_from_t1_amplitudes(self.t1)

        return UCJAnsatz(
            layers=tuple(layers),
            final_orbital_rotation=final_orbital_rotation,
        )

    def build_parameters(self) -> tuple[UCJAnsatz, UCJSpinBalancedParameterization, np.ndarray]:
        ansatz = self.build_ansatz()
        param = UCJSpinBalancedParameterization(
            norb=ansatz.norb,
            n_layers=ansatz.n_layers,
            with_final_orbital_rotation=ansatz.final_orbital_rotation is not None,
        )
        x0 = param.parameters_from_ansatz(ansatz)
        return ansatz, param, x0


@dataclass(frozen=True)
class GaugeFixedUCJBalancedDFSeed:
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

    def build_ansatz(self) -> UCJAnsatz:
        ansatz, _, _ = self.build_parameters()
        return ansatz

    def build_parameters(self) -> tuple[UCJAnsatz, GaugeFixedUCJSpinBalancedParameterization, np.ndarray]:
        ansatz = UCJBalancedDFSeed(
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

        nocc = np.asarray(self.t2).shape[0]
        param = GaugeFixedUCJSpinBalancedParameterization(
            norb=ansatz.norb,
            nocc=nocc,
            n_layers=ansatz.n_layers,
            with_final_orbital_rotation=ansatz.final_orbital_rotation is not None,
        )

        x0 = param.parameters_from_ansatz(ansatz)
        ansatz_roundtrip = param.ansatz_from_parameters(x0)
        return ansatz_roundtrip, param, x0


@dataclass(frozen=True)
class UCJRestrictedHeuristicSeed:
    t2: np.ndarray
    t1: np.ndarray | None = None
    n_layers: int = 1
    pair_scale: float = 1.0
    double_scale: float = 0.0
    with_final_orbital_rotation: bool = True

    def build_ansatz(self) -> UCJAnsatz:
        t2 = np.asarray(self.t2, dtype=np.float64)
        if t2.ndim != 4:
            raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
        nocc, _, nvirt, _ = t2.shape
        norb = nocc + nvirt

        pair = self.pair_scale * heuristic_restricted_pair_params_from_t2(t2)
        double_params = np.zeros(norb, dtype=np.float64)
        if self.double_scale != 0.0:
            double_params[:] = self.double_scale

        orbital_rotation = np.eye(norb, dtype=np.complex128)
        final_orbital_rotation = None
        if self.with_final_orbital_rotation and self.t1 is not None:
            final_orbital_rotation = _orbital_rotation_from_t1_amplitudes(self.t1)

        layers = tuple(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=double_params.copy(),
                    pair_params=pair.copy() if ell == 0 else np.zeros_like(pair),
                ),
                orbital_rotation=orbital_rotation,
            )
            for ell in range(self.n_layers)
        )
        return UCJAnsatz(
            layers=layers,
            final_orbital_rotation=final_orbital_rotation,
        )

    def build_parameters(self) -> tuple[UCJAnsatz, UCJSpinRestrictedParameterization, np.ndarray]:
        ansatz = self.build_ansatz()
        param = UCJSpinRestrictedParameterization(
            norb=ansatz.norb,
            n_layers=ansatz.n_layers,
            with_final_orbital_rotation=ansatz.final_orbital_rotation is not None,
        )
        x0 = param.parameters_from_ansatz(ansatz)
        return ansatz, param, x0


@dataclass(frozen=True)
class UCJRestrictedProjectedDFSeed:
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

    def build_ansatz(self) -> UCJAnsatz:
        balanced = UCJBalancedDFSeed(
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
        return project_spin_balanced_to_spin_restricted(balanced)

    def build_parameters(self) -> tuple[UCJAnsatz, UCJSpinRestrictedParameterization, np.ndarray]:
        ansatz = self.build_ansatz()
        param = UCJSpinRestrictedParameterization(
            norb=ansatz.norb,
            n_layers=ansatz.n_layers,
            with_final_orbital_rotation=ansatz.final_orbital_rotation is not None,
        )
        x0 = param.parameters_from_ansatz(ansatz)
        return ansatz, param, x0


@dataclass(frozen=True)
class GaugeFixedUCJRestrictedProjectedDFSeed:
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

    def build_ansatz(self) -> UCJAnsatz:
        balanced = GaugeFixedUCJBalancedDFSeed(
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
        return project_spin_balanced_to_spin_restricted(balanced)

    def build_parameters(self) -> tuple[UCJAnsatz, GaugeFixedUCJSpinRestrictedParameterization, np.ndarray]:
        ansatz = self.build_ansatz()
        nocc = np.asarray(self.t2).shape[0]
        param = GaugeFixedUCJSpinRestrictedParameterization(
            norb=ansatz.norb,
            nocc=nocc,
            n_layers=ansatz.n_layers,
            with_final_orbital_rotation=ansatz.final_orbital_rotation is not None,
        )
        x0 = param.parameters_from_ansatz(ansatz)
        return ansatz, param, x0
