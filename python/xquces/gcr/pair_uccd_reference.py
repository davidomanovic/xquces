from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2LeftUnitaryChart,
    IGCR2SpinRestrictedParameterization,
    orbital_relabeling_from_overlap,
)
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
)
from xquces.gcr.igcr4 import (
    IGCR4Ansatz,
    IGCR4SpinRestrictedParameterization,
    IGCR4SpinRestrictedSpec,
)
from xquces.pair_uccd import PairUCCDStateParameterization
from xquces.ucj.model import UCJAnsatz
from xquces.gcr.bridge_gcr2 import GCR2FullUnitaryChart

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


def _make_composite_subspace_jacobian(composite):
    from xquces.state_parameterization import (
        make_composite_reference_ansatz_subspace_jacobian,
    )

    return make_composite_reference_ansatz_subspace_jacobian(composite)


def _energy_gradient_from_composite(composite, params: np.ndarray, H) -> tuple[float, np.ndarray]:
    from xquces.state_parameterization import make_composite_reference_ansatz_vjp

    psi = composite.state_from_parameters(params)
    Hpsi = H @ psi
    energy = float(np.vdot(psi, Hpsi).real)
    residual = Hpsi - energy * psi
    grad = make_composite_reference_ansatz_vjp(composite)(params, residual)
    return energy, grad


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


def _combined_seed(reference_params: np.ndarray, ansatz_params: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.asarray(reference_params, dtype=np.float64),
        np.asarray(ansatz_params, dtype=np.float64),
    ])


@dataclass(frozen=True)
class HigherOrderLiftSeed:
    params: np.ndarray
    weights: np.ndarray
    energy: float | None
    baseline_params: np.ndarray
    baseline_energy: float | None
    accepted: bool
    message: str
    optimizer_result: object | None = None


def _state_energy(parameterization, hamiltonian, params: np.ndarray) -> float:
    psi = parameterization.state_from_parameters(params)
    return float(np.vdot(psi, hamiltonian @ psi).real)


def _triples_lift_from_pair_matrix(pair_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    norb = pair.shape[0]
    tau = np.zeros((norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            if p != q:
                tau[p, q] = pair[p, q]

    omega = np.zeros(len(list(itertools.combinations(range(norb), 3))), dtype=np.float64)
    for k, (p, q, r) in enumerate(itertools.combinations(range(norb), 3)):
        omega[k] = (pair[p, q] + pair[p, r] + pair[q, r]) / 3.0
    return tau, omega


def _quartic_lift_from_pair_matrix(pair_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    norb = pair.shape[0]

    eta = np.zeros(len(list(itertools.combinations(range(norb), 2))), dtype=np.float64)
    for k, (p, q) in enumerate(itertools.combinations(range(norb), 2)):
        eta[k] = pair[p, q]

    rho_indices = []
    for p in range(norb):
        for q in range(norb):
            if q == p:
                continue
            for r in range(q + 1, norb):
                if r == p:
                    continue
                rho_indices.append((p, q, r))
    rho = np.zeros(len(rho_indices), dtype=np.float64)
    for k, (p, q, r) in enumerate(rho_indices):
        rho[k] = (pair[p, q] + pair[p, r] + pair[q, r]) / 3.0

    sigma = np.zeros(len(list(itertools.combinations(range(norb), 4))), dtype=np.float64)
    for k, (p, q, r, s) in enumerate(itertools.combinations(range(norb), 4)):
        sigma[k] = (
            pair[p, q]
            + pair[p, r]
            + pair[p, s]
            + pair[q, r]
            + pair[q, s]
            + pair[r, s]
        ) / 6.0
    return eta, rho, sigma


def _transferred_reference_and_source_ansatz(
    target_parameterization,
    previous_parameters: np.ndarray,
    previous_parameterization,
) -> tuple[np.ndarray, object]:
    if previous_parameterization is None:
        previous_parameterization = target_parameterization
    prev = np.asarray(previous_parameters, dtype=np.float64)

    reference_params = np.zeros(target_parameterization.n_reference_params, dtype=np.float64)
    if hasattr(previous_parameterization, "split_parameters") and hasattr(previous_parameterization, "ansatz_parameterization"):
        if prev.shape != (previous_parameterization.n_params,):
            raise ValueError(f"Expected {(previous_parameterization.n_params,)}, got {prev.shape}.")
        prev_reference, prev_ansatz = previous_parameterization.split_parameters(prev)
        reference_params = _transfer_reference_params(
            target_parameterization,
            prev_reference,
            previous_parameterization,
            old_for_new=None,
            phases=None,
        )
        source_ansatz = previous_parameterization.ansatz_parameterization.ansatz_from_parameters(prev_ansatz)
        return reference_params, source_ansatz

    source_ansatz = previous_parameterization.ansatz_from_parameters(prev)
    return reference_params, source_ansatz


def _pair_block_start(ansatz_parameterization) -> int:
    return int(ansatz_parameterization.n_left_orbital_rotation_params)


def _cubic_block_start(ansatz_parameterization) -> int:
    return (
        int(ansatz_parameterization.n_left_orbital_rotation_params)
        + int(ansatz_parameterization.n_pair_params)
    )


def _quartic_block_start(ansatz_parameterization) -> int:
    return (
        int(ansatz_parameterization.n_left_orbital_rotation_params)
        + int(ansatz_parameterization.n_pair_params)
        + int(ansatz_parameterization.n_tau_params)
        + int(ansatz_parameterization.n_omega_params)
    )


def _pair_values_from_native(ansatz_parameterization, native_params: np.ndarray) -> np.ndarray:
    start = _pair_block_start(ansatz_parameterization)
    values = native_params[start : start + ansatz_parameterization.n_pair_params]
    by_pair = {
        tuple(pair): float(value)
        for pair, value in zip(ansatz_parameterization.pair_indices, values)
    }
    return np.asarray(
        [by_pair.get(tuple(pair), 0.0) for pair in itertools.combinations(range(ansatz_parameterization.norb), 2)],
        dtype=np.float64,
    )


def _full_cubic_from_native(ansatz_parameterization, native_params: np.ndarray) -> np.ndarray:
    start = _cubic_block_start(ansatz_parameterization)
    if getattr(ansatz_parameterization, "uses_reduced_cubic_chart", False):
        n = ansatz_parameterization.n_tau_params
        return ansatz_parameterization.cubic_reduction.full_from_reduced(native_params[start : start + n])

    tau_values = native_params[start : start + ansatz_parameterization.n_tau_params]
    omega_start = start + ansatz_parameterization.n_tau_params
    omega_values = native_params[omega_start : omega_start + ansatz_parameterization.n_omega_params]
    tau_by_pair = {
        tuple(pair): float(value)
        for pair, value in zip(ansatz_parameterization.tau_indices, tau_values)
    }
    omega_by_triple = {
        tuple(triple): float(value)
        for triple, value in zip(ansatz_parameterization.omega_indices, omega_values)
    }
    norb = ansatz_parameterization.norb
    tau_full = [
        tau_by_pair.get((p, q), 0.0)
        for p in range(norb)
        for q in range(norb)
        if p != q
    ]
    omega_full = [
        omega_by_triple.get(tuple(triple), 0.0)
        for triple in itertools.combinations(range(norb), 3)
    ]
    return np.asarray(tau_full + omega_full, dtype=np.float64)


def _direct_zero_extension_params(
    target_parameterization,
    previous_parameters: np.ndarray,
    previous_parameterization,
) -> np.ndarray | None:
    if previous_parameterization is None or not hasattr(previous_parameterization, "split_parameters"):
        return None
    if not hasattr(previous_parameterization, "ansatz_parameterization"):
        return None

    prev = np.asarray(previous_parameters, dtype=np.float64)
    if prev.shape != (previous_parameterization.n_params,):
        return None

    source = previous_parameterization.ansatz_parameterization
    target = target_parameterization.ansatz_parameterization
    required = (
        "n_left_orbital_rotation_params",
        "n_pair_params",
        "n_right_orbital_rotation_params",
        "_right_orbital_rotation_start",
        "_native_parameters_from_public",
        "_public_parameters_from_native",
    )
    if not all(hasattr(source, name) for name in required):
        return None
    if not all(hasattr(target, name) for name in required):
        return None
    if source.norb != target.norb or source.nocc != target.nocc:
        return None
    if source.n_left_orbital_rotation_params != target.n_left_orbital_rotation_params:
        return None
    if source.n_right_orbital_rotation_params != target.n_right_orbital_rotation_params:
        return None

    prev_reference, prev_ansatz = previous_parameterization.split_parameters(prev)
    reference_params = _transfer_reference_params(
        target_parameterization,
        prev_reference,
        previous_parameterization,
        old_for_new=None,
        phases=None,
    )
    source_native = source._native_parameters_from_public(prev_ansatz)
    target_native = np.zeros(target.n_params, dtype=np.float64)

    n_left = source.n_left_orbital_rotation_params
    target_native[:n_left] = source_native[:n_left]

    source_pair_start = _pair_block_start(source)
    target_pair_start = _pair_block_start(target)
    source_pairs = {
        tuple(pair): float(value)
        for pair, value in zip(
            source.pair_indices,
            source_native[source_pair_start : source_pair_start + source.n_pair_params],
        )
    }
    for k, pair in enumerate(target.pair_indices):
        value = source_pairs.get(tuple(pair))
        if value is not None:
            target_native[target_pair_start + k] = value

    if hasattr(source, "uses_reduced_cubic_chart") and hasattr(target, "uses_reduced_cubic_chart"):
        source_cubic_len = source.n_tau_params + source.n_omega_params
        target_cubic_len = target.n_tau_params + target.n_omega_params
        if source_cubic_len and source_cubic_len == target_cubic_len:
            source_start = _cubic_block_start(source)
            target_start = _cubic_block_start(target)
            target_native[target_start : target_start + target_cubic_len] = (
                source_native[source_start : source_start + source_cubic_len]
            )

    source_right = source._right_orbital_rotation_start
    target_right = target._right_orbital_rotation_start
    n_right = source.n_right_orbital_rotation_params
    target_native[target_right : target_right + n_right] = source_native[source_right : source_right + n_right]
    return _combined_seed(reference_params, target._public_parameters_from_native(target_native))


def _optimize_scalar_lift_weights(
    parameterization,
    hamiltonian,
    params_from_weights: Callable[[np.ndarray], np.ndarray],
    n_weights: int,
    *,
    optimize_weights: bool = True,
    maxiter: int = 40,
    max_abs_weight: float = 2.0,
    accept_tol: float = 1e-12,
) -> HigherOrderLiftSeed:
    zero = np.zeros(n_weights, dtype=np.float64)
    baseline_params = params_from_weights(zero)
    baseline_energy = (
        _state_energy(parameterization, hamiltonian, baseline_params)
        if hamiltonian is not None
        else None
    )
    if hamiltonian is None or not optimize_weights or n_weights == 0:
        return HigherOrderLiftSeed(
            params=baseline_params,
            weights=zero,
            energy=baseline_energy,
            baseline_params=baseline_params,
            baseline_energy=baseline_energy,
            accepted=False,
            message="zero extension",
        )

    best_params = baseline_params
    best_weights = zero
    best_energy = float(baseline_energy)

    def objective(weights: np.ndarray) -> float:
        nonlocal best_params, best_weights, best_energy
        weights = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)):
            return float("inf")
        try:
            params = params_from_weights(weights)
            energy = _state_energy(parameterization, hamiltonian, params)
        except Exception:
            return float("inf")
        if energy < best_energy:
            best_energy = float(energy)
            best_weights = np.array(weights, copy=True)
            best_params = np.array(params, copy=True)
        return float(energy)

    opt_result = minimize(
        objective,
        zero,
        method="L-BFGS-B",
        bounds=[(-float(max_abs_weight), float(max_abs_weight))] * n_weights,
        options={
            "maxiter": int(maxiter),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "eps": 1e-3,
            "maxls": 20,
        },
    )
    if np.isfinite(float(opt_result.fun)):
        objective(np.asarray(opt_result.x, dtype=np.float64))

    accepted = best_energy < float(baseline_energy) - float(accept_tol)
    if not accepted:
        return HigherOrderLiftSeed(
            params=baseline_params,
            weights=zero,
            energy=baseline_energy,
            baseline_params=baseline_params,
            baseline_energy=baseline_energy,
            accepted=False,
            message="zero extension retained",
            optimizer_result=opt_result,
        )
    return HigherOrderLiftSeed(
        params=best_params,
        weights=best_weights,
        energy=best_energy,
        baseline_params=baseline_params,
        baseline_energy=baseline_energy,
        accepted=True,
        message="higher-order lift accepted",
        optimizer_result=opt_result,
    )


def _transfer_reference_params(self, prev_reference, previous_parameterization, old_for_new, phases):
    del phases
    out = np.zeros(self.n_reference_params, dtype=np.float64)
    if (
        not hasattr(previous_parameterization, "reference_parameterization")
        or not hasattr(previous_parameterization, "pair_reference_indices")
        or previous_parameterization.norb != self.norb
        or previous_parameterization.nocc != self.nocc
        or type(previous_parameterization.reference_parameterization) is not type(self.reference_parameterization)
        or previous_parameterization.reference_parameterization.n_params != self.n_reference_params
    ):
        return out

    prev_reference = np.asarray(prev_reference, dtype=np.float64)
    if old_for_new is None:
        if getattr(previous_parameterization, "pair_reference_indices", None) == getattr(self, "pair_reference_indices", None):
            return np.array(prev_reference, copy=True)
        return out

    prev_by_pair = {
        tuple(pair): float(value)
        for pair, value in zip(previous_parameterization.pair_reference_indices, prev_reference)
    }
    transferred = np.zeros(self.n_reference_params, dtype=np.float64)
    for k, pair in enumerate(self.pair_reference_indices):
        old_pair = tuple(int(old_for_new[p]) for p in pair)
        if old_pair not in prev_by_pair:
            return out
        transferred[k] = prev_by_pair[old_pair]
    return transferred


def _is_trivial_relabel(norb: int, old_for_new, phases) -> bool:
    if old_for_new is None:
        return phases is None
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if old_for_new.shape != (norb,) or not np.array_equal(old_for_new, np.arange(norb)):
        return False
    if phases is None:
        return True
    phases = np.asarray(phases, dtype=np.complex128)
    return phases.shape == (norb,) and np.allclose(phases, np.ones(norb, dtype=np.complex128), atol=1e-10)


def _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal):
    if previous_parameterization is None:
        previous_parameterization = self
    if orbital_overlap is not None:
        if old_for_new is not None or phases is not None:
            raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
    prev = np.asarray(previous_parameters, dtype=np.float64)
    reference_params = np.zeros(self.n_reference_params, dtype=np.float64)

    if hasattr(previous_parameterization, "split_parameters") and hasattr(previous_parameterization, "ansatz_parameterization"):
        if prev.shape != (previous_parameterization.n_params,):
            raise ValueError(f"Expected {(previous_parameterization.n_params,)}, got {prev.shape}.")
        if (
            orbital_overlap is None
            and isinstance(previous_parameterization, type(self))
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
            None if orbital_overlap is not None else old_for_new,
            None if orbital_overlap is not None else phases,
        )

        ansatz_params = self.ansatz_parameterization.transfer_parameters_from(
            prev_ansatz,
            previous_parameterization=previous_parameterization.ansatz_parameterization,
            old_for_new=None if orbital_overlap is not None else old_for_new,
            phases=None if orbital_overlap is not None else phases,
            orbital_overlap=orbital_overlap,
            block_diagonal=block_diagonal,
        )
        return np.concatenate([
            reference_params,
            np.asarray(ansatz_params, dtype=np.float64),
        ])

    ansatz_params = self.ansatz_parameterization.transfer_parameters_from(
        prev,
        previous_parameterization=previous_parameterization,
        old_for_new=None if orbital_overlap is not None else old_for_new,
        phases=None if orbital_overlap is not None else phases,
        orbital_overlap=orbital_overlap,
        block_diagonal=block_diagonal,
    )
    return np.concatenate([
        reference_params,
        np.asarray(ansatz_params, dtype=np.float64),
    ])


@dataclass(frozen=True)
class GCR2PairUCCDParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=GCR2FullUnitaryChart)
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

    def state_subspace_jacobian_from_parameters(
        self,
        params: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        return _make_composite_subspace_jacobian(self._composite)(params, directions)

    def energy_gradient_from_parameters(
        self, params: np.ndarray, H
    ) -> tuple[float, np.ndarray]:
        """Return (energy, gradient) via the adjoint method (O(1) H-applications).

        Computes r = (H-E)|ψ⟩ with a single H-application, then evaluates
        2 Re(J† r) without any further H-applications.
        """
        return _energy_gradient_from_composite(self._composite, params, H)

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

@dataclass(frozen=True)
class GCR3PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR3SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=GCR2FullUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None
    tau_seed_scale: float = 0.25
    omega_seed_scale: float = 0.10

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

    def state_subspace_jacobian_from_parameters(
        self,
        params: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        return _make_composite_subspace_jacobian(self._composite)(params, directions)

    def energy_gradient_from_parameters(
        self, params: np.ndarray, H
    ) -> tuple[float, np.ndarray]:
        """Return (energy, gradient) via the adjoint method (O(1) H-applications)."""
        return _energy_gradient_from_composite(self._composite, params, H)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        return _seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        seeded = IGCR3Ansatz.from_ucj_ansatz(
            ansatz,
            nocc=self.nocc,
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
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)

    def nested_lift_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: object,
        *,
        hamiltonian=None,
        optimize_weights: bool = True,
        maxiter: int = 40,
        max_abs_weight: float = 2.0,
        accept_tol: float = 1e-12,
        return_info: bool = False,
    ):
        reference_params, source_ansatz = _transferred_reference_and_source_ansatz(
            self,
            previous_parameters,
            previous_parameterization,
        )
        if isinstance(source_ansatz, IGCR3Ansatz):
            params = _combined_seed(
                reference_params,
                self.ansatz_parameterization.parameters_from_ansatz(source_ansatz),
            )
            energy = _state_energy(self, hamiltonian, params) if hamiltonian is not None else None
            info = HigherOrderLiftSeed(
                params=params,
                weights=np.zeros(0, dtype=np.float64),
                energy=energy,
                baseline_params=params,
                baseline_energy=energy,
                accepted=False,
                message="already GCR-3",
            )
            return info if return_info else info.params
        if not isinstance(source_ansatz, IGCR2Ansatz):
            raise TypeError(f"GCR-3 nested lift requires a GCR-2 source ansatz, got {type(source_ansatz)!r}.")

        pair = source_ansatz.diagonal.to_standard().pair_params
        tau_direction, omega_direction = _triples_lift_from_pair_matrix(pair)

        def params_from_weights(weights: np.ndarray) -> np.ndarray:
            weights = np.asarray(weights, dtype=np.float64)
            diagonal = IGCR3SpinRestrictedSpec.from_igcr2_diagonal(
                source_ansatz.diagonal,
                tau=weights[0] * tau_direction,
                omega_values=weights[1] * omega_direction,
            )
            ansatz = IGCR3Ansatz(
                diagonal=diagonal,
                left=np.asarray(source_ansatz.left, dtype=np.complex128),
                right=np.asarray(source_ansatz.right, dtype=np.complex128),
                nocc=source_ansatz.nocc,
            )
            return _combined_seed(
                reference_params,
                self.ansatz_parameterization.parameters_from_ansatz(ansatz),
            )

        info = _optimize_scalar_lift_weights(
            self,
            hamiltonian,
            params_from_weights,
            2,
            optimize_weights=optimize_weights,
            maxiter=maxiter,
            max_abs_weight=max_abs_weight,
            accept_tol=accept_tol,
        )
        return info if return_info else info.params


@dataclass(frozen=True)
class GCR4PairUCCDParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR4SpinRestrictedParameterization | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object = field(default_factory=GCR2FullUnitaryChart)
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None
    tau_seed_scale: float = 0.25
    omega_seed_scale: float = 0.10
    eta_seed_scale: float = 0.05
    rho_seed_scale: float = 0.02
    sigma_seed_scale: float = 0.01

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

    def state_subspace_jacobian_from_parameters(
        self,
        params: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        return _make_composite_subspace_jacobian(self._composite)(params, directions)

    def energy_gradient_from_parameters(
        self, params: np.ndarray, H
    ) -> tuple[float, np.ndarray]:
        """Return (energy, gradient) via the adjoint method (O(1) H-applications)."""
        return _energy_gradient_from_composite(self._composite, params, H)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._composite.params_to_vec()

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        return _seed_from_ansatz(self.n_reference_params, self.ansatz_parameterization, ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        seeded = IGCR4Ansatz.from_ucj_ansatz(
            ansatz,
            nocc=self.nocc,
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
        return _transfer_params(self, previous_parameters, previous_parameterization, old_for_new, phases, orbital_overlap, block_diagonal)

    def nested_lift_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: object,
        *,
        hamiltonian=None,
        optimize_weights: bool = True,
        maxiter: int = 40,
        max_abs_weight: float = 2.0,
        accept_tol: float = 1e-12,
        return_info: bool = False,
    ):
        reference_params, source_ansatz = _transferred_reference_and_source_ansatz(
            self,
            previous_parameters,
            previous_parameterization,
        )
        if isinstance(source_ansatz, IGCR4Ansatz):
            params = _combined_seed(
                reference_params,
                self.ansatz_parameterization.parameters_from_ansatz(source_ansatz),
            )
            energy = _state_energy(self, hamiltonian, params) if hamiltonian is not None else None
            info = HigherOrderLiftSeed(
                params=params,
                weights=np.zeros(0, dtype=np.float64),
                energy=energy,
                baseline_params=params,
                baseline_energy=energy,
                accepted=False,
                message="already GCR-4",
            )
            return info if return_info else info.params
        if isinstance(source_ansatz, IGCR2Ansatz):
            source_ansatz = IGCR3Ansatz.from_igcr2_ansatz(
                source_ansatz,
                tau_scale=0.0,
                omega_scale=0.0,
            )
        if not isinstance(source_ansatz, IGCR3Ansatz):
            raise TypeError(f"GCR-4 nested lift requires a GCR-2 or GCR-3 source ansatz, got {type(source_ansatz)!r}.")

        pair = source_ansatz.diagonal.pair_matrix()
        eta_direction, rho_direction, sigma_direction = _quartic_lift_from_pair_matrix(pair)

        def params_from_weights(weights: np.ndarray) -> np.ndarray:
            weights = np.asarray(weights, dtype=np.float64)
            diagonal = IGCR4SpinRestrictedSpec.from_igcr3_diagonal(
                source_ansatz.diagonal,
                eta_values=weights[0] * eta_direction,
                rho_values=weights[1] * rho_direction,
                sigma_values=weights[2] * sigma_direction,
            )
            ansatz = IGCR4Ansatz(
                diagonal=diagonal,
                left=np.asarray(source_ansatz.left, dtype=np.complex128),
                right=np.asarray(source_ansatz.right, dtype=np.complex128),
                nocc=source_ansatz.nocc,
            )
            return _combined_seed(
                reference_params,
                self.ansatz_parameterization.parameters_from_ansatz(ansatz),
            )

        info = _optimize_scalar_lift_weights(
            self,
            hamiltonian,
            params_from_weights,
            3,
            optimize_weights=optimize_weights,
            maxiter=maxiter,
            max_abs_weight=max_abs_weight,
            accept_tol=accept_tol,
        )
        return info if return_info else info.params
