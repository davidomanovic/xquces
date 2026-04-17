from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize


ArrayFunc = Callable[[np.ndarray], np.ndarray]
ScalarFunc = Callable[[np.ndarray], float]


def real_jacobian(jacobian: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.complex128)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    return np.vstack([jacobian.real, jacobian.imag])


def state_energy_gradient(dpsi: np.ndarray, residual: np.ndarray) -> np.ndarray:
    dpsi = np.asarray(dpsi, dtype=np.complex128)
    residual = np.asarray(residual, dtype=np.complex128).reshape(-1)
    if dpsi.ndim != 2:
        raise ValueError(f"Expected 2D Jacobian array, got shape {dpsi.shape}")
    if dpsi.shape[0] == residual.shape[0]:
        gradient = 2.0 * np.real(dpsi.conj().T @ residual)
    elif dpsi.shape[1] == residual.shape[0]:
        gradient = 2.0 * np.real(dpsi.conj() @ residual)
    else:
        raise ValueError(
            f"Incompatible Jacobian shape {dpsi.shape} for state dimension {residual.shape[0]}"
        )
    return np.asarray(gradient, dtype=np.float64)


def make_state_objective(
    params_to_state: ArrayFunc,
    state_jacobian: ArrayFunc,
    hamiltonian,
) -> tuple[ScalarFunc, ArrayFunc, dict]:
    cache: dict[str, np.ndarray | float | None] = {"x": None, "f": None, "g": None}

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if cache["x"] is not None and np.array_equal(x, cache["x"]):
            return float(cache["f"]), np.asarray(cache["g"], dtype=np.float64)

        psi = np.asarray(params_to_state(x), dtype=np.complex128)
        hpsi = np.asarray(hamiltonian @ psi, dtype=np.complex128)
        energy = float(np.real(np.vdot(psi, hpsi)))
        gradient = state_energy_gradient(state_jacobian(x), hpsi - energy * psi)

        cache["x"] = x.copy()
        cache["f"] = energy
        cache["g"] = gradient
        return energy, gradient

    def fun(x: np.ndarray) -> float:
        return evaluate(x)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[1]

    return fun, jac, cache


def tangent_metric_preconditioner(
    state_jacobian: ArrayFunc,
    x0: np.ndarray,
    *,
    damping: float = 1e-6,
    max_scale: float = 30.0,
) -> tuple[np.ndarray, dict[str, float]]:
    x0 = np.asarray(x0, dtype=np.float64)
    J = real_jacobian(state_jacobian(x0))
    metric = J.T @ J
    metric = 0.5 * (metric + metric.T)
    evals, evecs = np.linalg.eigh(metric)

    lam_max = max(float(evals[-1]), 1.0)
    floor = float(damping) * lam_max
    raw_scales = 1.0 / np.sqrt(np.maximum(evals, floor))
    median_scale = float(np.median(raw_scales))
    scales = raw_scales / median_scale if median_scale > 0 else raw_scales
    scales = np.clip(scales, 1.0 / float(max_scale), float(max_scale))

    transform = evecs @ np.diag(scales)
    info = {
        "metric_min_eigenvalue": float(evals[0]),
        "metric_max_eigenvalue": lam_max,
        "metric_floor": floor,
        "metric_condition": lam_max / max(float(evals[0]), floor),
        "scale_min": float(scales.min()),
        "scale_median": float(np.median(scales)),
        "scale_max": float(scales.max()),
    }
    return transform, info


def tangent_svd_preconditioner(
    state_jacobian: ArrayFunc,
    x0: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_scale: float | None = 30.0,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    x0 = np.asarray(x0, dtype=np.float64)
    J = real_jacobian(state_jacobian(x0))
    metric = J.T @ J
    metric = 0.5 * (metric + metric.T)
    evals, evecs = np.linalg.eigh(metric)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    lam_max = max(float(evals[0]), 0.0) if evals.size else 0.0
    cutoff = max(float(atol), float(rtol) * lam_max)
    keep = evals > cutoff
    if not np.any(keep):
        raise ValueError(
            "No active tangent modes survived SVD pruning. "
            f"lam_max={lam_max:.3e}, cutoff={cutoff:.3e}."
        )

    active_evals = evals[keep]
    singular_values = np.sqrt(active_evals)
    scales = 1.0 / singular_values
    if normalize:
        median_scale = float(np.median(scales))
        if median_scale > 0.0:
            scales = scales / median_scale
    if max_scale is not None:
        scales = np.clip(scales, 1.0 / float(max_scale), float(max_scale))

    transform = evecs[:, keep] * scales[np.newaxis, :]
    smallest_kept = float(active_evals[-1])
    info = {
        "metric_min_eigenvalue": float(evals[-1]) if evals.size else 0.0,
        "metric_max_eigenvalue": lam_max,
        "metric_cutoff": cutoff,
        "metric_condition": lam_max / smallest_kept,
        "n_params": float(x0.size),
        "active_modes": float(active_evals.size),
        "dropped_modes": float(x0.size - active_evals.size),
        "scale_min": float(scales.min()),
        "scale_median": float(np.median(scales)),
        "scale_max": float(scales.max()),
    }
    return transform, info


def minimize_bfgs(
    fun: ScalarFunc,
    jac: ArrayFunc,
    x0: np.ndarray,
    *,
    callback=None,
    gtol: float = 1e-8,
    maxiter: int = 100000,
    **options,
) -> OptimizeResult:
    opts = {"gtol": gtol, "maxiter": maxiter}
    opts.update(options)
    return minimize(
        fun,
        x0=np.asarray(x0, dtype=np.float64),
        jac=jac,
        method="BFGS",
        callback=callback,
        options=opts,
    )


def minimize_metric_bfgs(
    fun: ScalarFunc,
    jac: ArrayFunc,
    x0: np.ndarray,
    state_jacobian: ArrayFunc,
    *,
    callback=None,
    damping: float = 1e-6,
    max_scale: float = 30.0,
    gtol: float = 1e-8,
    maxiter: int = 100000,
    **options,
) -> OptimizeResult:
    x0 = np.asarray(x0, dtype=np.float64)
    transform, info = tangent_metric_preconditioner(
        state_jacobian,
        x0,
        damping=damping,
        max_scale=max_scale,
    )

    def x_from_y(y: np.ndarray) -> np.ndarray:
        return x0 + transform @ np.asarray(y, dtype=np.float64)

    def fun_y(y: np.ndarray) -> float:
        return fun(x_from_y(y))

    def jac_y(y: np.ndarray) -> np.ndarray:
        return transform.T @ jac(x_from_y(y))

    result = minimize_bfgs(
        fun_y,
        jac_y,
        np.zeros_like(x0),
        callback=callback,
        gtol=gtol,
        maxiter=maxiter,
        **options,
    )
    x = x_from_y(result.x)
    result.x = x
    result.fun = fun(x)
    result.jac = jac(x)
    result.metric_preconditioner = info
    return result


def minimize_svd_metric_bfgs(
    fun: ScalarFunc,
    jac: ArrayFunc,
    x0: np.ndarray,
    state_jacobian: ArrayFunc,
    *,
    callback=None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_scale: float | None = 30.0,
    normalize: bool = True,
    gtol: float = 1e-8,
    maxiter: int = 100000,
    **options,
) -> OptimizeResult:
    x0 = np.asarray(x0, dtype=np.float64)
    transform, info = tangent_svd_preconditioner(
        state_jacobian,
        x0,
        rtol=rtol,
        atol=atol,
        max_scale=max_scale,
        normalize=normalize,
    )

    def x_from_y(y: np.ndarray) -> np.ndarray:
        return x0 + transform @ np.asarray(y, dtype=np.float64)

    def fun_y(y: np.ndarray) -> float:
        return fun(x_from_y(y))

    def jac_y(y: np.ndarray) -> np.ndarray:
        return transform.T @ jac(x_from_y(y))

    result = minimize_bfgs(
        fun_y,
        jac_y,
        np.zeros(transform.shape[1], dtype=np.float64),
        callback=callback,
        gtol=gtol,
        maxiter=maxiter,
        **options,
    )
    y = np.array(result.x, dtype=np.float64, copy=True)
    x = x_from_y(y)
    full_jac = jac(x)
    result.reduced_x = y
    result.reduced_jac = transform.T @ full_jac
    result.x = x
    result.fun = fun(x)
    result.jac = full_jac
    result.metric_preconditioner = info
    return result
