from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import LinearOperator


def _orthogonalize_columns(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    coeffs = vec.T.conj() @ mat
    return mat - vec.reshape((-1, 1)) * coeffs.reshape((1, -1))


def _as_column_jacobian(jac_mat: np.ndarray, dim: int) -> np.ndarray:
    jac_mat = np.asarray(jac_mat, dtype=np.complex128)
    if jac_mat.ndim != 2:
        raise ValueError(f"Expected 2D Jacobian array, got shape {jac_mat.shape}.")
    if jac_mat.shape[0] == dim:
        return jac_mat
    if jac_mat.shape[1] == dim:
        return jac_mat.T
    raise ValueError(
        f"Incompatible Jacobian shape {jac_mat.shape} for state dimension {dim}."
    )


def _apply_hamiltonian(operator: LinearOperator, vec: np.ndarray) -> np.ndarray:
    try:
        return operator @ vec
    except Exception:
        vec = np.asarray(vec, dtype=np.complex128)
        if vec.ndim != 2:
            raise
        return np.column_stack([operator @ vec[:, j] for j in range(vec.shape[1])])


def minimize_tangent_trust_region(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    hamiltonian: LinearOperator,
    x0: np.ndarray,
    *,
    jac: Callable[[np.ndarray], np.ndarray],
    expectation: Callable[[np.ndarray], float] | None = None,
    method: str = "trust-krylov",
    orthogonalize_jacobian: bool = True,
    hessp_mode: str = "gradient_fd",
    hessp_fd_epsilon: float = 1e-4,
    maxiter: int = 200,
    gtol: float = 1e-5,
    options: dict | None = None,
    callback: Callable[[OptimizeResult], Any] | None = None,
) -> OptimizeResult:
    if method not in {"trust-krylov", "trust-ncg"}:
        raise ValueError(
            f"Unsupported method {method!r}. Use 'trust-krylov' or 'trust-ncg'."
        )
    if hessp_mode not in {"gradient_fd", "tangent_space"}:
        raise ValueError(
            f"Unsupported hessp_mode {hessp_mode!r}. Use 'gradient_fd' or 'tangent_space'."
        )
    if hessp_fd_epsilon <= 0:
        raise ValueError(f"hessp_fd_epsilon must be positive. Got {hessp_fd_epsilon}.")

    if options is None:
        options = {}
    options = {"gtol": gtol, "maxiter": maxiter, **options}

    cache: dict[str, Any] = {"x": None}
    nfev = 0
    njev = 0
    nlinop = 0

    def evaluate(x: np.ndarray) -> dict[str, Any]:
        nonlocal nfev, njev, nlinop

        x = np.asarray(x, dtype=np.float64)

        if cache["x"] is not None and np.array_equal(x, cache["x"]):
            return cache

        psi = np.asarray(params_to_vec(x), dtype=np.complex128).reshape(-1)
        jac_mat = _as_column_jacobian(jac(x), psi.size)
        if orthogonalize_jacobian:
            jac_mat = _orthogonalize_columns(jac_mat, psi)

        h_psi = np.asarray(
            _apply_hamiltonian(hamiltonian, psi), dtype=np.complex128
        ).reshape(-1)
        h_jac = np.asarray(
            _apply_hamiltonian(hamiltonian, jac_mat), dtype=np.complex128
        )

        objective_energy = (
            float(expectation(psi))
            if expectation is not None
            else float(np.real(np.vdot(psi, h_psi)))
        )
        model_energy = float(np.real(np.vdot(psi, h_psi)))
        grad = 2.0 * np.real(jac_mat.conj().T @ h_psi)

        cache.clear()
        cache.update(
            {
                "x": x.copy(),
                "psi": psi,
                "jac_mat": jac_mat,
                "h_psi": h_psi,
                "h_jac": h_jac,
                "fun": objective_energy,
                "model_energy": model_energy,
                "jac": np.asarray(grad, dtype=np.float64),
            }
        )

        nfev += 1
        njev += 1
        nlinop += 1 + jac_mat.shape[1]

        return cache

    def fun(x: np.ndarray) -> float:
        return float(evaluate(x)["fun"])

    def grad_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate(x)["jac"], dtype=np.float64)

    def hessp_fn(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        data = evaluate(x)
        p = np.asarray(p, dtype=np.float64).reshape(-1)

        if hessp_mode == "tangent_space":
            jac_mat = data["jac_mat"]
            h_jac = data["h_jac"]
            model_energy = data["model_energy"]

            tangent_vec = jac_mat @ p
            h_tangent_vec = h_jac @ p
            overlap_vec = jac_mat.conj().T @ tangent_vec
            hv = 2.0 * np.real(
                jac_mat.conj().T @ h_tangent_vec - model_energy * overlap_vec
            )
            return np.asarray(hv, dtype=np.float64)

        pnorm = float(np.linalg.norm(p))
        if pnorm == 0.0:
            return np.zeros_like(p)

        step = hessp_fd_epsilon / max(1.0, pnorm)
        gp = grad_fn(x + step * p)
        gm = grad_fn(x - step * p)
        hv = (gp - gm) / (2.0 * step)
        return np.asarray(hv, dtype=np.float64)

    nit = 0

    def scipy_callback(xk: np.ndarray) -> None:
        nonlocal nit
        nit += 1
        data = evaluate(xk)
        result = OptimizeResult(
            x=np.asarray(data["x"], dtype=np.float64).copy(),
            fun=float(data["fun"]),
            jac=np.asarray(data["jac"], dtype=np.float64).copy(),
            nit=nit,
            nfev=nfev,
            njev=njev,
            nlinop=nlinop,
        )
        result.model_energy = float(data["model_energy"])
        result.hessian_model = hessp_mode
        result.using_analytic_jac = True
        if callback is not None:
            callback(result)

    result = minimize(
        fun,
        x0=np.asarray(x0, dtype=np.float64),
        jac=grad_fn,
        hessp=hessp_fn,
        method=method,
        callback=scipy_callback,
        options=options,
    )

    data = evaluate(result.x)
    result.fun = float(data["fun"])
    result.jac = np.asarray(data["jac"], dtype=np.float64)
    result.nfev = nfev
    result.njev = njev
    result.nlinop = nlinop
    result.model_energy = float(data["model_energy"])
    result.hessian_model = hessp_mode
    result.using_analytic_jac = True

    return result
