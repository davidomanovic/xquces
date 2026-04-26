from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.optimize import OptimizeResult

from xquces.optimize.linear_method import (
    _compute_adaptive_tikhonov,
    _get_param_update,
    _linear_method_matrices,
    _orthogonalize_columns,
    _solve_linear_method_eigensystem,
)


ArrayFunc = Callable[[np.ndarray], np.ndarray]
EnergyGradientFunc = Callable[[np.ndarray], tuple[float, np.ndarray]]
SubspaceJacobianFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
SubspaceBuilder = Callable[
    [np.ndarray, np.ndarray, np.ndarray | None, int],
    np.ndarray,
]


def _orthonormalize_columns(
    columns: list[np.ndarray],
    *,
    n_params: int,
    atol: float = 1e-12,
) -> np.ndarray:
    basis: list[np.ndarray] = []
    for col in columns:
        vec = np.asarray(col, dtype=np.float64).reshape(n_params).copy()
        for q in basis:
            vec -= q * float(np.dot(q, vec))
        norm = float(np.linalg.norm(vec))
        if norm > atol:
            basis.append(vec / norm)
    if not basis:
        return np.zeros((n_params, 0), dtype=np.float64)
    return np.column_stack(basis)


def gradient_coordinate_subspace(
    params: np.ndarray,
    gradient: np.ndarray,
    previous_step: np.ndarray | None,
    max_dim: int,
    *,
    coordinate_dim: int | None = None,
    history_columns: list[np.ndarray] | None = None,
    include_gradient: bool = True,
    include_previous_step: bool = True,
) -> np.ndarray:
    """Build a small real parameter-space basis from gradient-selected directions.

    The basis contains, in order when available:

    * the steepest-descent direction ``-gradient``;
    * the previous accepted step;
    * coordinate directions with largest absolute gradient components.

    The columns are Euclidean-orthonormalized before returning.
    """
    params = np.asarray(params, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if params.shape != gradient.shape:
        raise ValueError(
            f"params and gradient shape mismatch: {params.shape}, {gradient.shape}"
        )

    n_params = params.size
    max_dim = min(max(0, int(max_dim)), n_params)
    if max_dim == 0:
        return np.zeros((n_params, 0), dtype=np.float64)

    columns: list[np.ndarray] = []
    grad_norm = float(np.linalg.norm(gradient))
    if include_gradient and grad_norm > 0:
        columns.append(-gradient / grad_norm)

    if include_previous_step and previous_step is not None:
        step = np.asarray(previous_step, dtype=np.float64)
        if step.shape == params.shape:
            step_norm = float(np.linalg.norm(step))
            if step_norm > 0:
                columns.append(step / step_norm)

    if history_columns:
        for item in history_columns:
            hist = np.asarray(item, dtype=np.float64)
            if hist.shape == params.shape:
                hist_norm = float(np.linalg.norm(hist))
                if hist_norm > 0:
                    columns.append(hist / hist_norm)

    remaining = max_dim - len(columns)
    if coordinate_dim is not None:
        remaining = min(remaining, max(0, int(coordinate_dim)))

    if remaining > 0:
        for index in np.argsort(-np.abs(gradient))[:remaining]:
            unit = np.zeros(n_params, dtype=np.float64)
            unit[int(index)] = 1.0
            columns.append(unit)

    return _orthonormalize_columns(columns, n_params=n_params)[:, :max_dim]


def _finite_difference_subspace_jacobian(
    params_to_vec: ArrayFunc,
    params: np.ndarray,
    basis: np.ndarray,
    epsilon: float,
    center_vec: np.ndarray | None = None,
) -> np.ndarray:
    if center_vec is None:
        dim = np.asarray(params_to_vec(params), dtype=np.complex128).size
    else:
        dim = np.asarray(center_vec, dtype=np.complex128).size
    jac = np.zeros((dim, basis.shape[1]), dtype=np.complex128)
    for k in range(basis.shape[1]):
        direction = basis[:, k]
        jac[:, k] = (
            params_to_vec(params + epsilon * direction)
            - params_to_vec(params - epsilon * direction)
        ) / (2.0 * epsilon)
    return jac


def _energy_from_state(hamiltonian, vec: np.ndarray) -> tuple[float, np.ndarray]:
    hvec = hamiltonian @ vec
    return float(np.vdot(vec, hvec).real), hvec


def _relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def minimize_subspace_linear_method(
    params_to_vec: ArrayFunc,
    hamiltonian,
    x0: np.ndarray,
    *,
    energy_gradient: EnergyGradientFunc,
    jac: ArrayFunc | None = None,
    jac_subspace: SubspaceJacobianFunc | None = None,
    subspace_builder: SubspaceBuilder | None = None,
    subspace_dim: int = 96,
    coordinate_dim: int | None = None,
    history_size: int = 6,
    subspace_jacobian: Literal["analytic", "finite-difference"] = "analytic",
    maxiter: int = 1000,
    lindep: float = 1e-8,
    epsilon: float = 1e-8,
    ftol: float = 1e-16,
    gtol: float = 1e-6,
    regularization: float = 1e-5,
    regularization_growth: float = 10.0,
    regularization_attempts: int = 5,
    variation: float = 0.5,
    tikhonov: float | str | None = "auto",
    tikhonov_target_cond: float = 1e8,
    tikhonov_max: float | None = None,
    max_step_norm: float | None = None,
    step_shrink: float = 0.5,
    max_backtracks: int = 8,
    exhaustive_line_search: bool = False,
    accept_tol: float = 1e-12,
    fallback_gradient: bool = True,
    gradient_step: float = 1.0,
    callback: Callable[[OptimizeResult], object] | None = None,
) -> OptimizeResult:
    """Safeguarded linear method in a small parameter subspace.

    This optimizer builds the LM matrices for ``psi + J B y`` where ``B`` is a
    low-dimensional real parameter-space basis. The accepted update is
    ``params <- params + B y``. Every candidate is evaluated with the true
    variational energy and is accepted only if it does not increase that energy.

    ``jac`` may be the existing full analytic state Jacobian. For larger systems,
    pass ``subspace_jacobian="finite-difference"`` or a custom ``jac_subspace``
    to avoid materialising the full ``(state_dim, n_params)`` Jacobian.
    """
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1. Got {maxiter}.")
    if subspace_dim < 1:
        raise ValueError(f"subspace_dim must be at least 1. Got {subspace_dim}.")
    if history_size < 0:
        raise ValueError(f"history_size must be nonnegative. Got {history_size}.")
    if regularization < 0:
        raise ValueError(f"regularization must be nonnegative. Got {regularization}.")
    if regularization_attempts < 1:
        raise ValueError("regularization_attempts must be at least 1.")
    if regularization_growth < 1:
        raise ValueError("regularization_growth must be at least 1.")
    if not 0 <= variation <= 1:
        raise ValueError(f"variation must be between 0 and 1. Got {variation}.")
    if not 0 < step_shrink < 1:
        raise ValueError("step_shrink must be in (0, 1).")
    if subspace_jacobian not in {"analytic", "finite-difference"}:
        raise ValueError("subspace_jacobian must be 'analytic' or 'finite-difference'.")
    if subspace_jacobian == "analytic" and jac is None and jac_subspace is None:
        raise ValueError(
            "subspace_jacobian='analytic' requires jac or jac_subspace. "
            "Use subspace_jacobian='finite-difference' otherwise."
        )
    if tikhonov is not None and tikhonov != "auto" and float(tikhonov) < 0:
        raise ValueError(f"tikhonov must be nonnegative. Got {tikhonov}.")

    params = np.asarray(x0, dtype=np.float64).copy()
    n_params = params.size
    previous_step: np.ndarray | None = None
    pending_history_step: np.ndarray | None = None
    pending_history_grad: np.ndarray | None = None
    step_history: list[np.ndarray] = []
    grad_diff_history: list[np.ndarray] = []
    builder = subspace_builder

    nfev = 0
    njev = 0
    nlinop = 0
    success = False
    message = "Stop: Total number of iterations reached limit."
    grad = np.zeros_like(params)
    energy = float("nan")

    def call_vec(p: np.ndarray) -> np.ndarray:
        nonlocal nfev
        nfev += 1
        return np.asarray(params_to_vec(p), dtype=np.complex128)

    def energy_at(p: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal nlinop
        vec = call_vec(p)
        value, _ = _energy_from_state(hamiltonian, vec)
        nlinop += 1
        return value, vec

    def subspace_jac(
        params_: np.ndarray,
        basis_: np.ndarray,
        center_vec: np.ndarray,
    ) -> np.ndarray:
        nonlocal njev
        if basis_.shape[1] == 0:
            return np.zeros((center_vec.size, 0), dtype=np.complex128)
        if jac_subspace is not None:
            njev += 1
            return np.asarray(jac_subspace(params_, basis_), dtype=np.complex128)
        if subspace_jacobian == "finite-difference":
            njev += 1
            return _finite_difference_subspace_jacobian(
                call_vec, params_, basis_, epsilon, center_vec=center_vec
            )
        if jac is None:
            raise ValueError("analytic subspace Jacobian requested without jac")
        njev += 1
        return np.asarray(jac(params_), dtype=np.complex128) @ basis_

    for iteration in range(1, maxiter + 1):
        vec = call_vec(params)
        energy, _ = _energy_from_state(hamiltonian, vec)
        nlinop += 1

        _, grad = energy_gradient(params)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != params.shape:
            raise ValueError(
                "energy_gradient returned gradient shape "
                f"{grad.shape}; expected {params.shape}"
            )

        if pending_history_step is not None and pending_history_grad is not None:
            grad_diff = grad - pending_history_grad
            if np.linalg.norm(grad_diff) > 0:
                step_history.append(pending_history_step)
                grad_diff_history.append(grad_diff)
                if history_size == 0:
                    step_history.clear()
                    grad_diff_history.clear()
                elif len(step_history) > history_size:
                    del step_history[: len(step_history) - history_size]
                    del grad_diff_history[: len(grad_diff_history) - history_size]
            pending_history_step = None
            pending_history_grad = None

        max_abs_grad = float(np.max(np.abs(grad))) if grad.size else 0.0
        if max_abs_grad <= gtol:
            success = True
            message = "Convergence: Norm of gradient <= gtol."
            break

        if builder is None:
            history_columns: list[np.ndarray] = []
            if history_size:
                for step_item, grad_diff_item in reversed(
                    list(zip(step_history, grad_diff_history))
                ):
                    history_columns.extend([step_item, grad_diff_item])
            basis = gradient_coordinate_subspace(
                params,
                grad,
                previous_step,
                subspace_dim,
                coordinate_dim=coordinate_dim,
                history_columns=history_columns,
            )
        else:
            previous = None if previous_step is None else previous_step.copy()
            basis = np.asarray(
                builder(params.copy(), grad.copy(), previous, subspace_dim),
                dtype=np.float64,
            )
            if basis.ndim != 2 or basis.shape[0] != n_params:
                raise ValueError(
                    f"subspace_builder returned shape {basis.shape}; expected ({n_params}, m)"
                )
            basis = _orthonormalize_columns(
                [basis[:, k] for k in range(basis.shape[1])],
                n_params=n_params,
            )[:, :subspace_dim]

        if basis.shape[1] == 0:
            success = False
            message = "Stop: Empty subspace."
            break

        jac_sub = _orthogonalize_columns(subspace_jac(params, basis, vec), vec)
        energy_mat, overlap_mat = _linear_method_matrices(vec, jac_sub, hamiltonian)
        nlinop += jac_sub.shape[1] + 1

        S_pp = overlap_mat[1:, 1:]
        svals = np.linalg.svd(S_pp, compute_uv=False) if S_pp.size else np.array([0.0])
        s_max = float(np.max(svals)) if svals.size else 0.0
        s_min = float(np.min(svals)) if svals.size else 0.0
        cond_S = float(s_max / s_min) if s_min > 0 else float("inf")
        rank_S = int(np.sum(svals > 1e-10 * s_max)) if s_max > 0 else 0

        if tikhonov == "auto":
            tikhonov_mu = _compute_adaptive_tikhonov(
                overlap_mat, target_cond=tikhonov_target_cond
            )
            if tikhonov_max is not None:
                tikhonov_mu = min(tikhonov_mu, float(tikhonov_max))
        elif tikhonov is not None and float(tikhonov) > 0:
            tikhonov_mu = float(tikhonov)
        else:
            tikhonov_mu = 0.0

        overlap_mat_reg = overlap_mat
        if tikhonov_mu > 0:
            overlap_mat_reg = overlap_mat.copy()
            overlap_mat_reg[1:, 1:] += tikhonov_mu * np.eye(
                overlap_mat.shape[0] - 1
            )

        best_energy = float("inf")
        best_vec: np.ndarray | None = None
        best_step = np.zeros_like(params)
        best_reg = regularization
        best_scale = 0.0
        best_lm_eig = float("nan")
        accepted = False
        used_fallback = False

        reg_value = regularization
        for _ in range(regularization_attempts):
            lm_eig, _ = _solve_linear_method_eigensystem(
                energy_mat,
                overlap_mat_reg,
                reg_value,
                lindep=lindep,
            )
            reduced_update = _get_param_update(
                energy_mat,
                overlap_mat_reg,
                reg_value,
                variation,
                lindep,
            )
            step = basis @ reduced_update
            step_norm = float(np.linalg.norm(step))
            if max_step_norm is not None and step_norm > float(max_step_norm) > 0:
                step *= float(max_step_norm) / step_norm

            for backtrack in range(max_backtracks + 1):
                scale = step_shrink**backtrack
                trial_step = scale * step
                if not np.all(np.isfinite(trial_step)):
                    continue
                trial_energy, trial_vec = energy_at(params + trial_step)
                if trial_energy <= energy + accept_tol and trial_energy < best_energy:
                    best_energy = trial_energy
                    best_vec = trial_vec
                    best_step = trial_step
                    best_reg = reg_value
                    best_scale = scale
                    best_lm_eig = lm_eig
                    accepted = True
                    if not exhaustive_line_search:
                        break

            if accepted and not exhaustive_line_search:
                break

            reg_value *= regularization_growth

        if fallback_gradient and not accepted:
            direction = -grad
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 0:
                if (
                    max_step_norm is not None
                    and direction_norm > float(max_step_norm) > 0
                ):
                    direction *= float(max_step_norm) / direction_norm
                for backtrack in range(max_backtracks + 1):
                    scale = gradient_step * step_shrink**backtrack
                    trial_step = scale * direction
                    trial_energy, trial_vec = energy_at(params + trial_step)
                    if trial_energy <= energy + accept_tol and trial_energy < best_energy:
                        best_energy = trial_energy
                        best_vec = trial_vec
                        best_step = trial_step
                        best_scale = scale
                        best_reg = float("nan")
                        used_fallback = True
                        accepted = True
                        if not exhaustive_line_search:
                            break

        reduction = energy - best_energy if accepted else 0.0
        step_norm = float(np.linalg.norm(best_step)) if accepted else 0.0
        result = OptimizeResult(
            x=params + best_step if accepted else params.copy(),
            fun=best_energy if accepted else energy,
            jac=grad.copy(),
            nfev=nfev,
            njev=njev,
            nlinop=nlinop,
            nit=iteration,
        )
        result.previous_energy = energy
        result.energy_reduction = reduction
        result.accepted = accepted
        result.used_fallback = used_fallback
        result.subspace_dim = int(basis.shape[1])
        result.regularization = best_reg
        result.variation = variation
        result.tikhonov_mu = tikhonov_mu
        result.cond_S = cond_S
        result.rank_S = rank_S
        result.step_scale = best_scale
        result.step_norm = step_norm
        result.max_abs_grad = max_abs_grad
        result.lm_pred_eig = best_lm_eig
        result.subspace_jacobian = (
            subspace_jacobian if jac_subspace is None else "custom"
        )

        if callback is not None:
            callback(result)

        if not accepted:
            success = False
            message = "Stop: No non-increasing subspace LM step found."
            break

        params = params + best_step
        previous_step = best_step
        pending_history_step = best_step.copy()
        pending_history_grad = grad.copy()
        energy = best_energy
        if np.isfinite(best_reg):
            regularization = best_reg
        if best_vec is not None:
            del best_vec

        if reduction <= accept_tol:
            success = True
            message = "Convergence: No further energy-decreasing step found."
            break
        if _relative_change(energy + reduction, energy) <= ftol:
            success = True
            message = "Convergence: Relative reduction of objective function <= ftol."
            break

    final_vec = call_vec(params)
    final_energy, _ = _energy_from_state(hamiltonian, final_vec)
    nlinop += 1

    final_result = OptimizeResult(
        x=params,
        success=success,
        message=message,
        fun=final_energy,
        jac=grad,
        nfev=nfev,
        njev=njev,
        nlinop=nlinop,
        nit=iteration,
    )
    final_result.subspace_dim = subspace_dim
    final_result.variation = variation
    final_result.regularization = regularization
    final_result.subspace_jacobian = (
        subspace_jacobian if jac_subspace is None else "custom"
    )

    if callback is not None:
        callback(final_result)

    return final_result


__all__ = [
    "gradient_coordinate_subspace",
    "minimize_subspace_linear_method",
]
